from ai_modules.interfaces import (
    SileroVADModel, FasterWhisperSTTModel, DeepFaceFaceEmotionModel,
)
from ai_modules.models import (
    TextEmotionModel, Wav2VecEmotionModel,
    EmotionFusionModel, CBTLLMModel,
)
from app.core.config import settings


class AIContainer:
    """모든 AI 모델을 한곳에서 관리하는 싱글톤 컨테이너."""

    def __init__(self):
        self.vad = None
        self.stt = None
        self.text_emotion = None
        self.audio_emotion = None
        self.face_emotion = None
        self.fusion = None
        self.llm = None

    def load_models(self):
        print(">>> 컨테이너 모델 로딩.....")

        self.vad = SileroVADModel(speech_threshold=settings.vad_speech_threshold)
        self.vad.load_model()

        self.stt = FasterWhisperSTTModel(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        self.stt.load_model()

        self.text_emotion = TextEmotionModel(
            model_path=settings.text_emotion_model_path,
            device=settings.text_emotion_device,
        )
        self.text_emotion.load_model()

        self.audio_emotion = Wav2VecEmotionModel(
            model_path=settings.voice_emotion_model_path,
            device=settings.audio_emotion_device,
        )
        self.audio_emotion.load_model()

        self.face_emotion = DeepFaceFaceEmotionModel()
        self.face_emotion.load_model()

        self.fusion = EmotionFusionModel()

        self.llm = CBTLLMModel(
            cbt_adapter_path=settings.cbt_adapter_path,
            lora_dir=settings.cbt_lora_dir,
            device=settings.cbt_llm_device,
        )
        self.llm.load_model()

        print(">>> 모든 모델 로딩 완료")


# 전역 인스턴스
ai_container = AIContainer()
