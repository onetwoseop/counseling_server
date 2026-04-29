"""
HistoryManager — 대화 히스토리 + GPT-4o-mini 단계 요약 관리.

스텝별 대화를 GPT-4o-mini로 요약하여 다음 단계 system_prompt에 누적 메모로 주입.
LLM에 들어갈 컨텍스트:
  system_prompt (이전 단계 요약 포함) + 현재 단계 최근 N턴 raw history.
"""

import logging
from typing import Dict, List, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


STEP_SUMMARY_PROMPT = """다음은 CBT 상담의 '{step_name}' 단계에서 나눈 대화입니다.
이 대화의 핵심 내용을 3~5줄로 요약해주세요.
반드시 포함할 내용:
- 내담자의 주요 발언/감정
- 상담사가 파악한 핵심 포인트
- 이 단계에서 도출된 결론이나 발견 (자동적 사고, 인지 왜곡 증거 등)

[대화 내용]
{conversation}

[요약]"""


class HistoryManager:
    """세션별 대화 히스토리 + 단계별 요약 관리자."""

    def __init__(self, max_recent_turns: int = 4, api_key: Optional[str] = None):
        self.max_recent_turns = max_recent_turns
        self.api_key = api_key or settings.openai_api_key

        # 단계별 요약 (단계 전환 시 채워짐)
        self.step_summaries: Dict[int, Dict[str, Any]] = {}
        # 현재 단계 대화
        self.current_step_history: List[Dict[str, str]] = []
        # 전체 원본 히스토리 (리포트용, 절대 잘리지 않음)
        self.full_history: List[Dict[str, str]] = []

    def add_user_message(self, text: str) -> None:
        self._add_message("user", text)

    def add_assistant_message(self, text: str) -> None:
        self._add_message("assistant", text)

    def _add_message(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self.current_step_history.append(msg)
        self.full_history.append(msg)

    def on_step_transition(self, completed_step_num: int, step_name: str) -> None:
        """단계 전환 시 호출 — 현재 단계 대화를 GPT로 요약하고 초기화."""
        summary = self._summarize_step(step_name)
        self.step_summaries[completed_step_num] = {
            "step_num": completed_step_num,
            "step_name": step_name,
            "summary": summary,
            "turn_count": len(self.current_step_history) // 2,
        }
        logger.info(
            f"[History] Step {completed_step_num} ({step_name}) 요약: '{summary[:100]}...'"
        )
        self.current_step_history = []

    def get_recent_turns(self) -> List[Dict[str, str]]:
        """LLM 컨텍스트에 들어갈 최근 N턴 메시지 (현재 단계 내)."""
        max_messages = self.max_recent_turns * 2
        if len(self.current_step_history) <= max_messages:
            return list(self.current_step_history)
        return list(self.current_step_history[-max_messages:])

    def get_step_summaries(self) -> Dict[int, Dict[str, Any]]:
        return dict(self.step_summaries)

    def get_full_history(self) -> List[Dict[str, str]]:
        return list(self.full_history)

    def _summarize_step(self, step_name: str) -> str:
        if not self.current_step_history:
            return f"{step_name}: 대화 없음"

        if not self.api_key:
            return self._fallback_summary(step_name)

        conversation = self._format_conversation(self.current_step_history)
        prompt = STEP_SUMMARY_PROMPT.format(
            step_name=step_name,
            conversation=conversation[:3000],
        )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 CBT 상담 대화를 간결하게 요약하는 전문가입니다. 한국어로만 답하세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[History] GPT 요약 오류: {e} → 폴백 사용")
            return self._fallback_summary(step_name)

    def _fallback_summary(self, step_name: str) -> str:
        """GPT 실패 시 키워드 기반 간단 요약."""
        user_msgs = [m["content"] for m in self.current_step_history if m["role"] == "user"]
        if not user_msgs:
            return f"{step_name}: 대화 없음"
        first = user_msgs[0][:80]
        last = user_msgs[-1][:80] if len(user_msgs) > 1 else ""
        result = f"{step_name} ({len(user_msgs)}턴): \"{first}\""
        if last:
            result += f" → \"{last}\""
        return result

    @staticmethod
    def _format_conversation(messages: List[Dict[str, str]]) -> str:
        lines = []
        for msg in messages:
            role = "내담자" if msg["role"] == "user" else "상담사"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)
