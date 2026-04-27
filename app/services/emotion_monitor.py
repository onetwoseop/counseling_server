"""
EmotionMonitor — 모달리티별 부정 감정 하이라이트 저장.
AI에 전달하는 로직은 빼고, 감지 결과만 기록해둔다. (추후 활용)
"""

import logging
from typing import Dict, List, Any

from ai_modules.schemas import EmotionResult
from app.core.config import settings

logger = logging.getLogger(__name__)

NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}


class EmotionMonitor:
    """모달리티별(텍스트/음성/얼굴) 부정 감정 감지 → 하이라이트 저장."""

    def __init__(self, threshold: float = settings.negative_emotion_threshold):
        self.threshold = threshold
        # session_id → list of highlight dicts
        self._highlights: Dict[str, List[Dict[str, Any]]] = {}

    def init_session(self, session_id: str) -> None:
        self._highlights[session_id] = []

    def cleanup_session(self, session_id: str) -> None:
        self._highlights.pop(session_id, None)

    def check(
        self,
        session_id: str,
        modality: str,
        result: EmotionResult,
        step: int = 0,
        turn: int = 0,
    ) -> bool:
        """
        감정 결과를 체크하고, 부정 감정이 threshold 이상이면 하이라이트에 저장.
        Returns: True면 부정 감정 감지됨.
        """
        for emotion in NEGATIVE_EMOTIONS:
            prob = result.probabilities.get(emotion, 0.0)
            if prob >= self.threshold:
                highlight = {
                    "modality": modality,
                    "emotion": emotion,
                    "probability": prob,
                    "step": step,
                    "turn": turn,
                }
                if session_id in self._highlights:
                    self._highlights[session_id].append(highlight)
                logger.info(
                    f"[EmoMon] {session_id}: {modality} 부정감정 감지 "
                    f"({emotion}={prob:.3f}, step={step}, turn={turn})"
                )
                return True
        return False

    def get_highlights(self, session_id: str) -> List[Dict[str, Any]]:
        return self._highlights.get(session_id, [])
