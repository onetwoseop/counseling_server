import abc
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)
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

# 감정 분석 (음성/얼굴)
class BaseEmotionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def analyze(self, input_data: Any) -> EmotionResult: pass

# 텍스트 감정 분석
class BaseTextEmotionModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def analyze(self, text: str) -> EmotionResult: pass

# 감정 융합
class BaseEmotionFusionModel(abc.ABC):
    @abc.abstractmethod
    def fuse(
        self,
        text_result: EmotionResult,
        voice_result: EmotionResult,
        face_result: EmotionResult,
    ) -> EmotionResult: pass

# LLM
class BaseLLMModel(abc.ABC):
    @abc.abstractmethod
    def load_model(self): pass
    @abc.abstractmethod
    def generate_response(self, context: LLMContext) -> LLMResponse: pass


# ──────────────────────────────────────────────────────────────
# VAD - Silero VAD
# 입력: float32 PCM 16kHz, 512샘플(32ms) 단위 chunk
# ──────────────────────────────────────────────────────────────
class SileroVADModel(BaseVADModel):
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512

    def __init__(self, speech_threshold: float = 0.5):
        self.speech_threshold = speech_threshold

    def load_model(self):
        import torch
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.model.eval()
        logger.info("[VAD] Silero VAD Loaded")

    def process(self, input_data: VADInput) -> VADOutput:
        import torch
        audio_array = np.frombuffer(input_data.audio_chunk, dtype=np.float32)
        if len(audio_array) < self.CHUNK_SAMPLES:
            audio_array = np.pad(audio_array, (0, self.CHUNK_SAMPLES - len(audio_array)))
        tensor = torch.from_numpy(audio_array[:self.CHUNK_SAMPLES].copy())
        confidence = float(self.model(tensor, self.SAMPLE_RATE).item())
        return VADOutput(is_speech=confidence >= self.speech_threshold, confidence=confidence)


# ──────────────────────────────────────────────────────────────
# STT - faster-whisper
# 입력: STTInput.audio_data = float32 PCM bytes (16kHz, mono)
# ──────────────────────────────────────────────────────────────
class FasterWhisperSTTModel(BaseSTTModel):
    def __init__(self, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self):
        import torch
        from faster_whisper import WhisperModel
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[STT] CUDA 사용 불가 → CPU로 폴백")
            self.device = "cpu"
            self.compute_type = "int8"
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        logger.info(f"[STT] faster-whisper ({self.model_size}) Loaded on {self.device} ({self.compute_type})")

    def transcribe(self, input_data: STTInput) -> STTOutput:
        audio_array = np.frombuffer(input_data.audio_data, dtype=np.float32)
        segments, info = self.model.transcribe(audio_array, language=input_data.language)
        text = " ".join(seg.text.strip() for seg in segments)
        return STTOutput(text=text, language=info.language)


# ──────────────────────────────────────────────────────────────
# FFmpeg 변환 유틸 - 브라우저 MediaRecorder webm/opus → float32 PCM 16kHz
# ──────────────────────────────────────────────────────────────
def webm_to_float32_pcm(webm_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    import ffmpeg
    out, _ = (
        ffmpeg
        .input("pipe:0")
        .output("pipe:1", format="f32le", ac=1, ar=sample_rate)
        .run(input=webm_bytes, capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# 얼굴 감정 - DeepFace
# 입력: FaceInput.video_frame = JPEG bytes
# ──────────────────────────────────────────────────────────────
class DeepFaceFaceEmotionModel(BaseEmotionModel):
    def load_model(self):
        import numpy as np
        from deepface import DeepFace
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False, silent=True)
        except Exception:
            pass
        logger.info("[Face] DeepFace Loaded")

    def analyze(self, input_data: FaceInput) -> EmotionResult:
        import cv2
        import numpy as np
        from deepface import DeepFace
        try:
            arr = np.frombuffer(input_data.video_frame, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("이미지 디코딩 실패")
            result = DeepFace.analyze(
                frame, actions=["emotion"],
                enforce_detection=False, silent=True
            )
            emotions = result[0]["emotion"]
            primary = result[0]["dominant_emotion"]
            return EmotionResult(
                primary_emotion=primary,
                probabilities={k: round(v / 100, 3) for k, v in emotions.items()},
            )
        except Exception as e:
            logger.error(f"[Face] 분석 오류: {e}")
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
