"""
StepManager — 5단계 CBT 상담 플랜의 진행 상태를 관리.
plan 스키마: {"analysis": {core_problem, cognitive_pattern}, "steps": [...]}
질문 단위로 스텝을 진행. 현재 스텝의 모든 질문이 소화되면 다음 스텝으로 전환.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class StepManager:
    """5단계 CBT 상담 플랜 진행 관리자. 질문 단위로 스텝을 진행."""

    def __init__(self, plan: Dict[str, Any], topic: str):
        self.plan = plan
        self.analysis: Dict[str, Any] = plan.get("analysis", {})
        self.steps: List[Dict[str, Any]] = plan["steps"]
        self.topic = topic
        self.current_step_idx = 0
        self.current_question_idx = 0

    @property
    def current_step(self) -> Dict[str, Any]:
        return self.steps[self.current_step_idx]

    @property
    def step_number(self) -> int:
        """1-based 단계 번호."""
        return self.current_step_idx + 1

    @property
    def is_last_step(self) -> bool:
        return self.current_step_idx >= len(self.steps) - 1

    @property
    def is_complete(self) -> bool:
        return self.current_step_idx >= len(self.steps)

    def get_current_question(self) -> Optional[str]:
        if self.is_complete:
            return None
        questions = self.current_step.get("key_questions", [])
        if self.current_question_idx < len(questions):
            return questions[self.current_question_idx]
        return None

    def get_questions(self) -> List[str]:
        if self.is_complete:
            return []
        return self.current_step.get("key_questions", [])

    def advance_question(self) -> Optional[str]:
        """
        질문 하나 소화 완료 → 다음 질문으로.
        현재 스텝 질문이 모두 소화되면 다음 스텝으로 전환.

        Returns:
            "step_changed"        — 다음 스텝으로 전환됨
            "counseling_complete" — 모든 스텝 완료
            None                  — 현재 스텝 내 다음 질문으로 이동
        """
        questions = self.current_step.get("key_questions", [])
        self.current_question_idx += 1

        if self.current_question_idx >= len(questions):
            return self._advance_step()

        logger.info(
            f"[StepMgr] 질문 진행: Step {self.step_number} "
            f"Q{self.current_question_idx + 1}/{len(questions)} "
            f"'{self.get_current_question()}'"
        )
        return None

    def _advance_step(self) -> str:
        if self.is_last_step:
            self.current_step_idx = len(self.steps)
            logger.info("[StepMgr] 모든 단계 완료")
            return "counseling_complete"

        prev_name = self.current_step["name"]
        self.current_step_idx += 1
        self.current_question_idx = 0
        logger.info(
            f"[StepMgr] 단계 전환: {prev_name} → "
            f"Step {self.step_number}: {self.current_step['name']} "
            f"(질문 {len(self.get_questions())}개)"
        )
        return "step_changed"

    def get_status(self) -> Dict[str, Any]:
        """현재 진행 상태 (클라이언트 전송용). 프론트 호환을 위해 'title' 키 유지."""
        if self.is_complete:
            return {
                "step": len(self.steps),
                "title": "상담 완료",
                "total_steps": len(self.steps),
                "complete": True,
            }
        questions = self.get_questions()
        return {
            "step": self.step_number,
            "title": self.current_step["name"],
            "goal": self.current_step["goal"],
            "question_idx": self.current_question_idx,
            "total_questions": len(questions),
            "current_question": self.get_current_question(),
            "total_steps": len(self.steps),
            "complete": False,
        }
