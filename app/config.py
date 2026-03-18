from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # VAD 설정
    vad_silence_threshold: float = 0.5   # pipeline.py의 SILENCE_THRESHOLD_SEC
    vad_sample_rate: int = 16000          # pipeline.py의 VAD_SAMPLE_RATE
    vad_chunk_samples: int = 512          # pipeline.py의 VAD_CHUNK_SAMPLES

    # STT 설정
    whisper_model_size: str = "base"      # container.py의 "base"

    # 서버 설정
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

settings = Settings()