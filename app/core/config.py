from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # VAD 설정
    vad_silence_threshold: float = 1.5
    vad_speech_threshold: float = 0.5
    vad_sample_rate: int = 16000
    vad_chunk_samples: int = 512

    # STT 설정
    whisper_model_size: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # CBT LLM 설정 (Qwen2.5-3B + CBT LoRA)
    cbt_llm_device: str = "cuda"
    cbt_adapter_path: str = "models/cbt-counselor-final"
    cbt_lora_dir: str = "models/lora"

    # 텍스트 감정 설정 (klue/bert) — fp16 GPU 사용 시 ~200MB
    text_emotion_model_path: str = "models/text-emotion-final"
    text_emotion_device: str = "cuda"

    # 음성 감정 설정 (wav2vec2)
    audio_emotion_device: str = "cpu"
    voice_emotion_model_path: str = "models/voice-emotion-final"

    # 더미 모드
    # use_dummy_llm: bool = False  # USE_DUMMY_LLM=true 로 활성화

    # OpenAI API (GPT-4o-mini 플랜 생성용)
    openai_api_key: str = ""  # OPENAI_API_KEY 환경변수로 설정

    # 감정 임계치
    negative_emotion_threshold: float = 0.65

    # 서버 설정
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

settings = Settings()
