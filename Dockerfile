FROM --platform=linux/amd64 python:3.11-slim


# 환경변수 설정
# 로그 즉시 출력, 불필요한 .pyc 생성 제한
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Non-root 사용자 생성
RUN useradd -m appuser

# 소스 코드 복사
COPY --chown=appuser:appuser . .

# 사용자 전환
USER appuser

# 컨테이너가 사용할 포트 명시
EXPOSE 8000

# FastAPI 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]