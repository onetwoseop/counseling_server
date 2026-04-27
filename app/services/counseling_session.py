"""
CounselingSession — 세션 단위 상담 오케스트레이터.
setup 데이터 → 플랜 생성 → StepManager 초기화 → 첫 발화 생성.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any

from ai_modules.schemas import CounselingSetup, LLMContext, LLMResponse
from app.core.container import AIContainer
from app.services.plan_generator import DynamicPlanGenerator
from app.services.step_manager import StepManager
from app.services.emotion_monitor import EmotionMonitor

logger = logging.getLogger(__name__)


class CounselingSession:
    """
    세션별 상담 상태를 관리하고, 초기 상담 플로우를 오케스트레이션한다.

    초기 상담 플로우:
    1. setup 데이터 수신
    2. GPT-4o-mini → 5-step 플랜 생성
    3. StepManager 초기화
    4. Step 1 시스템 프롬프트 → Qwen LLM → 첫 상담사 발화 생성
    5. 클라이언트에 플랜 + 첫 발화 반환
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.plan_generator = DynamicPlanGenerator()
        self.emotion_monitor = EmotionMonitor()
        # 세션별 상태
        self._setups: Dict[str, CounselingSetup] = {}
        self._step_managers: Dict[str, StepManager] = {}
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def init_session(self, session_id: str) -> None:
        self._histories[session_id] = []
        self.emotion_monitor.init_session(session_id)

    def cleanup_session(self, session_id: str) -> None:
        self._setups.pop(session_id, None)
        self._step_managers.pop(session_id, None)
        self._histories.pop(session_id, None)
        self.emotion_monitor.cleanup_session(session_id)

    def get_step_manager(self, session_id: str) -> Optional[StepManager]:
        return self._step_managers.get(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._histories.get(session_id, [])

    def add_to_history(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """매 턴 종료 후 대화 히스토리에 추가. 최대 20개(10턴) 유지."""
        history = self._histories.get(session_id)
        if history is None:
            return
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
        if len(history) > 20:
            # setup 첫 2개(user_setup + assistant_first)는 보존, 나머지 오래된 것 제거
            self._histories[session_id] = history[:2] + history[-(20 - 2):]

    async def start_counseling(
        self, session_id: str, topic: str, mood: str, content: str
    ) -> Dict[str, Any]:
        """
        초기 상담 시작: setup → 플랜 생성 → 첫 발화 생성.

        Returns: {
            "plan": [...],
            "first_message": "...",
            "step_status": {...}
        }
        """
        setup = CounselingSetup(topic=topic, mood=mood, content=content)
        self._setups[session_id] = setup

        # 1. GPT-4o-mini로 5-step 플랜 생성
        loop = asyncio.get_event_loop()
        t0 = time.time()
        plan = await loop.run_in_executor(
            None, self.plan_generator.generate, topic, mood, content
        )
        logger.info(f"[Session] {session_id}: 플랜 생성 완료 ({time.time() - t0:.2f}초)")

        # 2. StepManager 초기화
        step_mgr = StepManager(plan=plan, topic=topic)
        self._step_managers[session_id] = step_mgr

        # 3. Step 1 시스템 프롬프트로 첫 상담사 발화 생성
        first_message = await self._generate_step_opening(session_id, setup, step_mgr)

        # 4. 히스토리에 기록
        if session_id in self._histories:
            self._histories[session_id].append(
                {"role": "user", "content": f"[상담 시작] 주제: {topic}, 기분: {mood}, 내용: {content}"}
            )
            self._histories[session_id].append(
                {"role": "assistant", "content": first_message}
            )

        return {
            "plan": [
                {"step": s["step"], "title": s["title"], "goal": s["goal"]}
                for s in plan
            ],
            "first_message": first_message,
            "step_status": step_mgr.get_status(),
        }

    async def _generate_step_opening(
        self,
        session_id: str,
        setup: CounselingSetup,
        step_mgr: StepManager,
    ) -> str:
        """현재 단계의 시스템 프롬프트로 LLM 첫 발화 생성."""
        system_prompt = step_mgr.get_system_prompt()
        user_text = (
            f"[상담 시작]\n"
            f"주제: {setup.topic}\n"
            f"현재 기분: {setup.mood}\n"
            f"호소 내용: {setup.content}\n\n"
            f"위 내용을 바탕으로, 내담자에게 첫 인사와 함께 "
            f"감정을 탐색할 수 있는 열린 질문을 해주세요."
        )

        llm_context = LLMContext(
            user_text=user_text,
            system_prompt=system_prompt,
            history=[],
        )

        loop = asyncio.get_event_loop()
        t0 = time.time()

        if self.container.llm is None:
            logger.warning(f"[Session] {session_id}: LLM 미로딩")
            return "안녕하세요, 오늘 어떤 이야기를 나눠볼까요?"

        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )
        logger.info(
            f"[Session] {session_id}: 첫 발화 생성 완료 "
            f"('{response.reply_text[:50]}...', {time.time() - t0:.2f}초)"
        )
        return response.reply_text
