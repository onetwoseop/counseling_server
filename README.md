# Attune Counseling Server

AI 기반 실시간 심리상담 WebSocket 서버입니다.
클라이언트로부터 음성 청크와 영상 프레임을 WebSocket으로 수신하여
**VAD → STT → 텍스트/음성/얼굴 감정 분석 → 감정 융합 → CBT LLM 응답 생성** 파이프라인을 처리합니다.

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | FastAPI + Uvicorn |
| 음성 감지 (VAD) | Silero VAD (PyTorch) |
| 음성 인식 (STT) | faster-whisper (CUDA) |
| 텍스트 감정 | klue/bert (로컬 `models/text-emotion-final`) |
| 음성 감정 | Wav2Vec2 (로컬 `models/voice-emotion-final`) |
| 얼굴 감정 | DeepFace |
| 감정 융합 | 가중치 평균 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25) |
| LLM | Qwen2.5-3B-Instruct + CBT LoRA (로컬 `models/cbt-counselor-final`) |

---

## 프로젝트 구조

```
counseling_server/
├── app/
│   ├── main.py                  # FastAPI 앱, WebSocket 엔드포인트
│   ├── schemas.py               # WebSocket 메시지 스키마
│   ├── core/
│   │   ├── config.py            # 환경변수 설정 (pydantic-settings)
│   │   └── container.py         # AI 모델 싱글턴 컨테이너
│   └── services/
│       ├── audio_processor.py   # VAD 침묵 감지 + 증분 STT 워커
│       ├── pipeline.py          # 상담 데이터 흐름 오케스트레이터
│       └── session_manager.py   # WebSocket 연결 관리 및 데이터 라우팅
├── ai_modules/
│   ├── interfaces.py            # AI 모델 베이스 클래스 + 더미 구현체
│   ├── models.py                # 실제 AI 모델 구현체 (로컬 모델 연동)
│   └── schemas.py               # AI 모델 입출력 스키마
├── models/                      # ⚠️ Git 제외 (팀 내부 공유) — 아래 참고
│   ├── cbt-counselor-final/     # CBT 상담 LoRA 어댑터 (Qwen2.5-3B 기반)
│   ├── text-emotion-final/      # 텍스트 감정 분류 (klue/bert)
│   ├── voice-emotion-final/     # 음성 감정 분류 (Wav2Vec2)
│   └── lora/                    # 감정별 LoRA (angry/sad/happy/fear/disgust/surprise/neutral)
├── tests/
│   ├── test_pipeline.py         # 파이프라인 구조 검증 (서버 불필요)
│   └── test_ws_client.py        # WebSocket 통합 테스트 (서버 필요, mp4 사용)
├── Dockerfile                   # GPU 서버 배포용 (nvidia/cuda:12.4.1 베이스)
├── docker-compose.example.yml   # Docker 구성 참고용 (실제 사용 시 복사 후 수정)
├── .env.example                 # 환경변수 템플릿
├── requirements.txt
└── README.md
```

---

## 시작하기

### 1. 파이썬 가상환경 생성 및 활성화

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

> PyTorch (CUDA 12.4 빌드)와 AI 모델 패키지가 포함되어 있어 초기 설치에 수분이 걸릴 수 있습니다.

### 3. AI 모델 파일 배치

`models/` 폴더는 Git에 포함되지 않습니다. 팀 내부 공유 스토리지(Google Drive 등)에서 다운로드 후 아래 구조로 배치하세요.

```
counseling_server/
└── models/
    ├── cbt-counselor-final/
    ├── text-emotion-final/
    ├── voice-emotion-final/
    └── lora/
        ├── angry/
        ├── sad/
        ├── happy/
        ├── fear/
        ├── disgust/
        ├── surprise/
        └── neutral/
```

> LLM 베이스 모델(`Qwen/Qwen2.5-3B-Instruct`)은 서버 최초 실행 시 HuggingFace에서 자동 다운로드됩니다 (~6GB).

### 4. 환경변수 설정

```bash
cp .env.example .env
```

주요 설정값:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `WHISPER_MODEL_SIZE` | `small` | Whisper 모델 크기 |
| `WHISPER_DEVICE` | `cuda` | STT 디바이스 |
| `CBT_LLM_USE_REAL` | `true` | CBT LLM 활성화 여부 |
| `CBT_LLM_DEVICE` | `cuda` | LLM 디바이스 |
| `TEXT_EMOTION_DEVICE` | `cpu` | 텍스트 감정 디바이스 |
| `AUDIO_EMOTION_DEVICE` | `cpu` | 음성 감정 디바이스 |
| `FACE_EMOTION_USE_REAL` | `true` | DeepFace 활성화 여부 |
| `APP_PORT` | `8000` | 서버 포트 |

> GPU 클라우드 서버에서 VRAM 여유가 있을 경우 `TEXT_EMOTION_DEVICE=cuda`, `AUDIO_EMOTION_DEVICE=cuda`로 변경 가능합니다.

### 5. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> `--reload` 옵션은 파일 변경 시 서버가 재시작되어 WebSocket 연결이 끊깁니다. 개발 중에도 사용하지 않는 것을 권장합니다.

서버가 정상 실행되면 아래 메시지가 출력됩니다:

```
>>> AI 모델 로딩을 시작합니다.....
[VAD] Silero VAD Loaded
[STT] faster-whisper (small) Loaded on cuda
[TextEmo] 로딩 완료: models/text-emotion-final on cpu
[VoiceEmo] Wav2Vec2 로딩 완료: models/voice-emotion-final on cpu
[Face] DeepFace Loaded
[CBT LLM] 로딩 완료. 감정 LoRA 로드: [angry, disgust, fear, happy, sad, surprise, neutral]
>>> 모든 모델 로딩 완료
INFO:     Application startup complete.
```

서버 상태 확인: `http://localhost:8000/`

---

## WebSocket 연결

```
ws://localhost:8000/ws/counseling/{client_id}
```

### 클라이언트 → 서버 메시지 형식

```json
// 초기 상담 설정 (연결 직후 1회 전송)
{
  "type": "setup",
  "data": {
    "topic": "직장 스트레스",
    "mood": "anxious",
    "content": "업무가 너무 많아서 힘들어요",
    "style": "empathy"
  }
}

// 음성 청크 (webm/opus, MediaRecorder 출력 그대로)
{ "type": "audio", "data": "<base64 webm bytes>" }

// 영상 프레임 (JPEG)
{ "type": "video", "data": "<base64 JPEG>" }

// 발화 종료 신호
{ "type": "control", "data": "END_OF_SPEECH" }

// 세션 종료
{ "type": "control", "data": "END_OF_SESSION" }
```

### 서버 → 클라이언트 메시지 형식

```json
{ "status": "connected",         "message": "상담실에 입장하였습니다." }
{ "status": "initial_questions", "message": "AI가 생성한 초기 CBT 질문" }
{ "status": "processing",        "message": "답변 생성 중..." }
{ "status": "stt_done",          "text": "STT 변환 결과 텍스트" }
{ "status": "response",          "message": "AI 상담사 응답 텍스트" }
```

### 처리 흐름

```
END_OF_SPEECH 수신
  → ffmpeg: webm/opus → float32 PCM 16kHz 변환
  → Whisper STT → 텍스트 추출
  → 텍스트 감정 분석 (klue/bert)
  → 음성 감정 분석 (Wav2Vec2, PCM 사용)
  → 얼굴 감정 평균 (누적 프레임 기반)
  → 3모달 감정 융합 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25)
  → CBT LLM (감정 기반 LoRA 전환 → 응답 생성)
  → response 전송
```

---

## Docker 배포 (GPU 클라우드 서버)

`docker-compose.example.yml`을 복사해서 사용하세요.

```bash
cp docker-compose.example.yml docker-compose.yml
# docker-compose.yml 내용을 서버 환경에 맞게 수정

# NVIDIA Container Toolkit 설치 필요 (1회)
# sudo apt install nvidia-container-toolkit

docker compose up --build -d
```

> `Dockerfile`은 `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` 베이스 이미지를 사용합니다.
> CUDA 버전이 다른 서버라면 Dockerfile 첫 줄의 태그를 변경하세요.

---

## 테스트

### test_pipeline.py — 파이프라인 구조 검증 (서버 불필요)

```bash
python -m tests.test_pipeline
```

### test_ws_client.py — WebSocket 통합 테스트 (서버 필요)

`test_files/test_video1.mp4` 파일이 필요합니다.

```bash
# 터미널 1
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 터미널 2
python -m tests.test_ws_client
```

---

## VRAM 사용량 참고 (GTX 1660 Super 6GB 기준)

| 모델 | 디바이스 | 사용량 |
|------|----------|--------|
| Whisper small | CUDA | ~0.5GB |
| Qwen2.5-3B (8bit 양자화) | CUDA | ~3.5GB |
| 텍스트 감정 (BERT) | CPU | - |
| 음성 감정 (Wav2Vec2) | CPU | - |
| **합계** | | **~4.0GB** |
