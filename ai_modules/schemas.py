from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# 데이터 포맷 정리 코드(입력값, 출력값 둘 다 정의한 것)
# 보고 어떤 포맷이 더 올바른지 정리해서 알려주기 바람

# --- 초기 상담 설정 ---
class CounselingSetup(BaseModel):
    topic: str            # 상담 주제 (예: "직장 스트레스", "불면증 케어")
    mood: str             # 현재 기분 (예: "happy", "sad", "anxious")
    content: str          # 상담 내용 자유 입력 텍스트

# --- 공통 결과 포맷 ---
class EmotionResult(BaseModel):
    primary_emotion: str = Field(..., description="가장 확률이 높은 감정 (예: happy)")
    probabilities: Dict[str, float] = Field(..., description="전체 감정 확률 분포") # ex) "sad": 0.9

# --- 1. VAD (음성 감지) ---
class VADInput(BaseModel):
    audio_chunk: bytes = Field(..., description="실시간 오디오 바이너리 청크")

class VADOutput(BaseModel):
    is_speech: bool
    confidence: float

# --- 2. STT (Whisper) ---
class STTInput(BaseModel):
    audio_data: bytes = Field(..., description="발화가 완료된 오디오 전체")
    language: str = "ko"

class STTOutput(BaseModel):
    text: str
    language: str

# --- 3. Vision (Face Emotion) ---
class FaceInput(BaseModel):
    video_frame: Any = Field(..., description="OpenCV 이미지 배열 혹은 바이너리")

# --- 4. LLM (상담 생성) ---
class LLMContext(BaseModel):
    user_text: str
    system_prompt: Optional[str] = None   # 커스텀 시스템 프롬프트 (None이면 기본 프롬프트)
    face_emotions: List[EmotionResult] = []
    voice_emotions: List[EmotionResult] = []
    text_emotion: Optional[str] = None    # STT 텍스트 감정 (primary label)
    fused_emotion: Optional[str] = None   # 3모달 융합 최종 감정 → LoRA 선택에 사용
    history: List[Dict[str, str]] = []    # [{"role": "user"|"assistant", "content": "..."}]

class LLMResponse(BaseModel):
    reply_text: str
    suggested_action: Optional[str] = None