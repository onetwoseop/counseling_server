import abc
import numpy as np
from typing import Any
from .schemas import (
    VADInput, VADOutput, STTInput, STTOutput,
    EmotionResult, LLMContext, LLMResponse, FaceInput
)

# VAD(음성감지)
class BaseVADModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def process(self, input_data: VADInput) -> VADOutput: pass

# STT(받아쓰기)
class BaseSTTModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def transcribe(self, input_data: STTInput) -> STTOutput: pass
    
    
class BaseEmotionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def analyze(self, input_data: Any) -> EmotionResult: pass

    
# Emotion Analysis (음성 감정 분석)
class DummyAudioEmotionModel(BaseEmotionModel):
    # ──────────────────────────────────────────────────────────────
    # [AI 개발자 교체 가이드 - 음성 감정]
    #
    # 현재 전달 형식:
    #   input_data.audio_data : float32 PCM bytes (16kHz, mono)
    #   → np.frombuffer(input_data.audio_data, dtype=np.float32) 로 numpy 변환 가능
    #
    # AudioEmotionEstimator(ai_core) 가 요구하는 형식:
    #   audio_chunk : List[float]
    #   → audio_list = np.frombuffer(input_data.audio_data, dtype=np.float32).tolist()
    #   → self._estimator.infer(audio_list) 로 호출
    #
    # 교체 시 이 클래스를 삭제하고 BaseEmotionModel 을 상속한 새 클래스를 작성하세요.
    # container.py 에서 DummyAudioEmotionModel → 새 클래스명 으로 변경하면 적용됩니다.
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        print("[Audio Emo] Dummy SpeechBrain Loaded")

    def analyze(self, input_data: STTInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="fear",  # 목소리가 떨린다고 가정
            probabilities={"fear": 0.7, "sad": 0.3}
        )


# Emotion Analysis (안면 감정 분석)
class DummyFaceEmotionModel(BaseEmotionModel):
    # ──────────────────────────────────────────────────────────────
    # [AI 개발자 교체 가이드 - 얼굴 감정]
    #
    # 현재 전달 형식:
    #   input_data.video_frame : JPEG bytes (Any 타입)
    #   → WebSocket 에서 base64 decode 된 JPEG 원본 바이너리
    #
    # FaceEmotionEstimator(ai_core) 가 요구하는 형식:
    #   frame_bgr : np.ndarray  (OpenCV BGR, shape: H×W×3)
    #   → import cv2, numpy as np
    #   → arr = np.frombuffer(input_data.video_frame, dtype=np.uint8)
    #   → frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #   → self._estimator.infer(frame_bgr) 로 호출
    #
    # 교체 시 이 클래스를 삭제하고 BaseEmotionModel 을 상속한 새 클래스를 작성하세요.
    # container.py 에서 DummyFaceEmotionModel → 새 클래스명 으로 변경하면 적용됩니다.
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        print("[Face] Dummy DeepFace Loaded")

    def analyze(self, input_data: FaceInput) -> EmotionResult:
        return EmotionResult(
            primary_emotion="sad",
            probabilities={"sad": 0.8, "neutral": 0.2}
        )


# VAD
class SileroVADModel(BaseVADModel):
    """
    Silero VAD 실제 구현체.
    입력: float32 PCM 16kHz, 512샘플(32ms) 단위 chunk
    """
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512  # Silero VAD 요구 chunk 크기 (32ms @ 16kHz)

    def __init__(self, speech_threshold: float = 0.5):
        self.speech_threshold = speech_threshold

    def load_model(self):
        import torch
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()
        print("[VAD] Silero VAD Loaded")

    def process(self, input_data: VADInput) -> VADOutput:
        import torch
        audio_array = np.frombuffer(input_data.audio_chunk, dtype=np.float32)
        # chunk가 512샘플보다 짧으면 패딩
        if len(audio_array) < self.CHUNK_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.CHUNK_SAMPLES - len(audio_array)))
        tensor = torch.from_numpy(audio_array[:self.CHUNK_SAMPLES])
        confidence = float(self.model(tensor, self.SAMPLE_RATE).item())
        return VADOutput(is_speech=confidence >= self.speech_threshold, confidence=confidence)


# STT
class FasterWhisperSTTModel(BaseSTTModel):
    """
    faster-whisper 실제 구현체.
    입력: STTInput.audio_data = float32 PCM bytes (16kHz, mono)
    """
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None

    def load_model(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        print(f"[STT] faster-whisper ({self.model_size}) Loaded")

    def transcribe(self, input_data: STTInput) -> STTOutput:
        audio_array = np.frombuffer(input_data.audio_data, dtype=np.float32)
        segments, info = self.model.transcribe(audio_array, language=input_data.language)
        text = " ".join(seg.text.strip() for seg in segments)
        return STTOutput(text=text, language=info.language)


# ──────────────────────────────────────────────────────────────
# [FFmpeg 변환 유틸 - 실제 브라우저(webm/opus) 연결 시 사용]
#
# 브라우저 MediaRecorder가 webm/opus로 보내면
# faster-whisper/VAD가 요구하는 float32 PCM 16kHz로 변환 필요.
#
# 사용 위치: session_manager.py audio 처리 블록 또는 pipeline.append_audio_chunk()
#
# def webm_to_float32_pcm(webm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
#     import ffmpeg
#     out, _ = (
#         ffmpeg
#         .input("pipe:0")
#         .output("pipe:1", format="f32le", ac=1, ar=sample_rate)
#         .run(input=webm_bytes, capture_stdout=True, capture_stderr=True)
#     )
#     return np.frombuffer(out, dtype=np.float32)
# ──────────────────────────────────────────────────────────────


# 4. LLM
class BaseLLMModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def generate_response(self, context: LLMContext) -> LLMResponse: pass

class DummyLLMModel(BaseLLMModel):
    """테스트용 가짜 구현. 실제 모델 연결 시 이 클래스를 교체하면 됩니다."""
    # ──────────────────────────────────────────────────────────────
    # [AI 개발자 교체 가이드 - LLM]
    #
    # 현재 전달 형식 (LLMContext):
    #   context.user_text      : str   - 이번 발화 전체 텍스트 (STT 결과 합산)
    #   context.face_emotions  : List[EmotionResult] - 프레임별 얼굴 감정 (primary_emotion, probabilities)
    #   context.voice_emotions : List[EmotionResult] - 발화별 음성 감정 (primary_emotion, probabilities)
    #   context.history        : List[Dict[str, str]] - 이전 대화 기록 (현재 비어있음)
    #   context.text_emotion   : Optional[str] - 텍스트 기반 감정 (현재 미사용)
    #
    # VLLMOpenAIClient / TransformersLLMClient(ai_core) 가 요구하는 형식:
    #   단일 턴:     llm.chat(system_prompt, user_prompt)
    #   멀티 턴:     llm.chat_multiturn(system_prompt, history_messages, current_user_prompt)
    #   history_messages 형식: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    #   → context.history 는 이미 동일한 List[Dict] 구조이므로 그대로 전달 가능
    #
    # 반환 형식:
    #   LLMResponse(reply_text=str, suggested_action=Optional[str])
    #   → llm.chat(...) 의 반환값(str)을 reply_text 에 담으면 됩니다.
    #
    # 교체 시 이 클래스를 삭제하고 BaseLLMModel 을 상속한 새 클래스를 작성하세요.
    # container.py 에서 DummyLLMModel → 새 클래스명 으로 변경하면 적용됩니다.
    # ──────────────────────────────────────────────────────────────

    def load_model(self):
        print("[LLM] Dummy Qwen2 Model Loaded")

    def generate_response(self, context: LLMContext) -> LLMResponse:
        # context에서 사용 가능한 데이터:
        #   context.user_text      : 이번 발화 전체 텍스트 (STT 결과 합산)
        #   context.face_emotions  : List[EmotionResult] - 프레임별 얼굴 감정
        #   context.voice_emotions : List[EmotionResult] - 발화별 음성 감정
        face_summary = context.face_emotions[0].primary_emotion if context.face_emotions else "N/A"
        voice_summary = context.voice_emotions[0].primary_emotion if context.voice_emotions else "N/A"
        return LLMResponse(
            reply_text=(
                f"사용자님, '{context.user_text}'라고 하셨군요. "
                f"(얼굴: {face_summary}, 목소리: {voice_summary}) 많이 속상하셨겠습니다."
            ),
            suggested_action="심호흡 하기"
        )