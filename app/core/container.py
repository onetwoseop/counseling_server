from ai_modules.interfaces import (
    SileroVADModel, FasterWhisperSTTModel,
    DummyAudioEmotionModel, DummyFaceEmotionModel, DummyLLMModel
)

from app.core.config import settings

# 모든 AI 모델을 한곳에서 관리하는 싱글톤(서버 전체에서 딱 하나의 인스턴스만 존재) 컨테이너
class AIContainer:
    def __init__(self): # 클래스 초기화
        # 개발 용이성 위해 비워둠, 추후 구현에 따라 로딩 시점 변환 가능
        self.vad = None
        self.stt = None
        self.audio_emotion = None
        self.face_emotion = None
        self.llm = None

    def load_models(self):
        print(">>> AI 모델 로딩을 시작합니다.....")
        
        # VAD (음성 감지)
        self.vad = SileroVADModel()
        self.vad.load_model()

        # STT (음성 -> 텍스트)
        self.stt = FasterWhisperSTTModel(model_size=settings.whisper_model_size)
        self.stt.load_model()

        # Emotion (음성 감정)
        self.audio_emotion = DummyAudioEmotionModel()
        self.audio_emotion.load_model()
        
        # Emotion (얼굴 감정)
        self.face_emotion = DummyFaceEmotionModel() 
        self.face_emotion.load_model()
        
        # LLM (상담 답변)
        self.llm = DummyLLMModel() 
        self.llm.load_model()
        
        print(">>> 모든 모델 로딩 완료")

# 전역 인스턴스 생성 -> main.py에서 사용
ai_container = AIContainer()