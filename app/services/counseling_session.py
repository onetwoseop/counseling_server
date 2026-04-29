"""
CounselingSession — 세션 단위 상담 오케스트레이터.
setup 데이터 → 플랜 생성(병렬) + 첫 발화 생성 → StepManager + HistoryManager 초기화.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any

from ai_modules.schemas import CounselingSetup, LLMContext, LLMResponse
from app.core.container import AIContainer
from app.services.plan_generator import DynamicPlanGenerator
from app.services.step_manager import StepManager
from app.services.history_manager import HistoryManager
from app.services.emotion_monitor import EmotionMonitor

logger = logging.getLogger(__name__)


class CounselingSession:
    """
    세션별 상담 상태 관리 + 초기 상담 플로우 오케스트레이션.

    초기 상담 플로우:
    1. setup 데이터 수신
    2. GPT-4o-mini 5-step 플랜 생성 + Qwen 첫 발화 병렬 실행
    3. StepManager + HistoryManager 초기화
    4. 클라이언트에 플랜 + 첫 발화 반환
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.plan_generator = DynamicPlanGenerator()
        self.emotion_monitor = EmotionMonitor()
        # 세션별 상태
        self._setups: Dict[str, CounselingSetup] = {}
        self._step_managers: Dict[str, StepManager] = {}
        self._history_managers: Dict[str, HistoryManager] = {}

    def init_session(self, session_id: str) -> None:
        self._history_managers[session_id] = HistoryManager(max_recent_turns=4)
        self.emotion_monitor.init_session(session_id)

    def cleanup_session(self, session_id: str) -> None:
        self._setups.pop(session_id, None)
        self._step_managers.pop(session_id, None)
        self._history_managers.pop(session_id, None)
        self.emotion_monitor.cleanup_session(session_id)

    def get_step_manager(self, session_id: str) -> Optional[StepManager]:
        return self._step_managers.get(session_id)

    def get_history_manager(self, session_id: str) -> Optional[HistoryManager]:
        return self._history_managers.get(session_id)

    async def start_counseling(
        self, session_id: str, topic: str, mood: str, content: str
    ) -> Dict[str, Any]:
        """
        초기 상담 시작: GPT 플랜 생성 + Qwen 첫 발화를 asyncio.gather로 병렬 실행.

        Returns: {
            "plan": [...],
            "analysis": {...},
            "first_message": "...",
            "step_status": {...}
        }
        """
        setup = CounselingSetup(topic=topic, mood=mood, content=content)
        self._setups[session_id] = setup

        loop = asyncio.get_event_loop()
        t0 = time.time()

        # GPT 플랜 생성 + Qwen 첫 발화 병렬 실행
        # - 첫 발화는 플랜 없이 즉시 생성 가능한 인트로 프롬프트 사용
        # - GPT 플랜이 완료되면 StepManager 초기화 → 2번째 턴부터 GPT 질문 + 분석 주입
        plan_future = loop.run_in_executor(
            None, self.plan_generator.generate, topic, mood, content
        )
        first_msg_coro = self._generate_quick_opening(session_id, setup)

        plan, first_message = await asyncio.gather(plan_future, first_msg_coro)
        logger.info(
            f"[Session] {session_id}: 플랜+첫발화 병렬 완료 ({time.time() - t0:.2f}초)"
        )

        # StepManager 초기화 (GPT 플랜 완료 후)
        step_mgr = StepManager(plan=plan, topic=topic)
        self._step_managers[session_id] = step_mgr
        logger.info(
            f"[Session] {session_id}: StepManager 초기화 완료 "
            f"(Step 1: '{step_mgr.current_step['name']}', "
            f"Q1: '{step_mgr.get_current_question()}')"
        )

        # HistoryManager에 첫 발화 기록
        history_mgr = self._history_managers.get(session_id)
        if history_mgr:
            history_mgr.add_user_message(
                f"[상담 시작] 주제: {topic}, 기분: {mood}, 내용: {content}"
            )
            history_mgr.add_assistant_message(first_message)

        return {
            "plan": [
                {
                    "step": s["step"],
                    "title": s["name"],
                    "goal": s["goal"],
                    "focus": s.get("focus", ""),
                    "questions": s.get("key_questions", []),
                }
                for s in plan["steps"]
            ],
            "analysis": plan.get("analysis", {}),
            "first_message": first_message,
            "step_status": step_mgr.get_status(),
        }

    async def _generate_quick_opening(self, session_id: str, setup: CounselingSetup) -> str:
        """
        GPT 플랜 없이 즉시 생성 가능한 첫 인사 (plan_generator와 병렬 실행용).
        인트로 프롬프트는 하드코딩 → 2번째 턴부터 GPT 플랜의 질문 + 분석이 주입됨.
        """
        INTRO_PROMPT = (
            "당신은 따뜻하고 공감적인 AI 심리상담사 '루나'입니다. "
            "내담자가 처음 상담실에 들어왔습니다. "
            "내담자의 상황을 읽고 CBT 상담사로서 따뜻하게 맞이하며, "
            "현재 감정을 탐색할 수 있는 열린 질문을 하나만 하세요. "
            "답변은 2~3문장으로 간결하게, 반드시 한국어로만 작성하고 영어 단어는 절대 사용하지 마세요."
        )
        user_text = (
            f"[상담 시작]\n"
            f"주제: {setup.topic}\n"
            f"현재 기분: {setup.mood}\n"
            f"호소 내용: {setup.content}\n\n"
            f"따뜻한 첫 인사와 감정 탐색 열린 질문을 한국어로 해주세요."
        )

        if self.container.llm is None:
            logger.warning(f"[Session] {session_id}: LLM 미로딩 → 기본 첫 인사 사용")
            return "안녕하세요, 오늘 어떤 이야기를 나눠볼까요?"

        llm_context = LLMContext(
            user_text=user_text,
            system_prompt=INTRO_PROMPT,
            history=[],
        )
        loop = asyncio.get_event_loop()
        t0 = time.time()
        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )
        logger.info(
            f"[Session] {session_id}: 첫 발화 완료 "
            f"('{response.reply_text[:50]}...', {time.time() - t0:.2f}초)"
        )
        return response.reply_text
