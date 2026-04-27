# Attune Counseling Server

AI 기반 실시간 CBT 심리상담 WebSocket 서버입니다.
클라이언트로부터 음성 청크와 영상 프레임을 WebSocket으로 수신하여
**VAD → STT → 텍스트/음성/얼굴 감정 분석 → 감정 융합 → 5-Step CBT LLM 응답 생성** 파이프라인을 처리합니다.

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
| 상담 플랜 생성 | GPT-4o-mini (OpenAI API) |
| LLM | Qwen2.5-3B-Instruct + CBT LoRA (로컬 `models/cbt-counselor-final`) |

---

## 프로젝트 구조

```
counseling_server/
├── app/
│   ├── main.py                    # FastAPI 앱, WebSocket 엔드포인트, 로깅 설정
│   ├── schemas.py                 # WebSocket 메시지 스키마
│   ├── core/
│   │   ├── config.py              # 환경변수 설정 (pydantic-settings)
│   │   └── container.py           # AI 모델 싱글턴 컨테이너
│   └── services/
│       ├── audio_processor.py     # VAD 침묵 감지 + 배치 STT 처리
│       ├── pipeline.py            # 상담 데이터 흐름 오케스트레이터
│       ├── session_manager.py     # WebSocket 연결 관리 및 데이터 라우팅
│       ├── counseling_session.py  # 세션 오케스트레이터 (플랜 생성 → 첫 발화)
│       ├── plan_generator.py      # GPT-4o-mini 5-Step CBT 플랜 생성
│       ├── step_manager.py        # 스텝별 질문 진행 및 전환 관리
│       └── emotion_monitor.py     # 모달리티별 부정 감정 감지 및 하이라이트 저장
├── ai_modules/
│   ├── interfaces.py              # AI 모델 베이스 클래스 (VAD, STT, DeepFace)
│   ├── models.py                  # 실제 AI 모델 구현체 (로컬 모델 연동)
│   └── schemas.py                 # AI 모델 입출력 스키마
├── models/                        # ⚠️ Git 제외 (팀 내부 공유) — 아래 참고
│   ├── cbt-counselor-final/       # CBT 상담 LoRA 어댑터 (Qwen2.5-3B 기반)
│   ├── text-emotion-final/        # 텍스트 감정 분류 (klue/bert)
│   ├── voice-emotion-final/       # 음성 감정 분류 (Wav2Vec2)
│   └── lora/                      # 감정별 LoRA (angry/sad/happy/fear/disgust/surprise/neutral)
├── test_e2e.py                    # E2E 통합 테스트 (in-process, 실제 모델 사용)
├── .env.example                   # 환경변수 템플릿
├── requirements.txt
└── README.md
```

---

## 상담 흐름

```
[연결]
  WebSocket 접속 → 세션 초기화 → "connected" 응답

[초기 상담 설정]
  클라이언트: {"type": "setup", "data": {"topic", "mood", "content"}}
  서버:
    1. GPT-4o-mini → 내담자 맞춤 5-Step CBT 플랜 생성
    2. StepManager 초기화 (Step 1, Q1)
    3. Qwen LLM → Step 1 시스템 프롬프트 + 첫 질문으로 첫 발화 생성
    4. "counseling_ready" 응답 (플랜 + 첫 발화 + 단계 상태 포함)

[멀티턴 상담]
  오디오 청크 (0x01 헤더) → VAD 필터 → STT 버퍼 누적
  영상 프레임 (0x02 헤더) → DeepFace → 얼굴 감정 버퍼 누적
  청크마다 → Wav2Vec2 백그라운드 → 음성 감정 버퍼 누적

  END_OF_SPEECH →
    Whisper STT → 텍스트 추출
    텍스트 감정 분석 (klue/bert)
    감정 융합 (텍스트 + 음성 + 얼굴)
    StepManager에서 현재 질문 주입 → Qwen LLM 응답 생성
    질문 소화 → advance_question() → 다음 질문 or 스텝 전환

[세션 종료]
  {"type": "control", "data": "END_OF_SESSION"} → 세션 정리
```

### 5-Step CBT 플랜 구조

| 단계 | 기본 내용 | 질문 수 |
|------|-----------|---------|
| Step 1 | 감정 탐색 및 공감 | 2~4개 |
| Step 2 | 상황 분석 | 2~4개 |
| Step 3 | 인지 탐색 | 2~4개 |
| Step 4 | 대안적 사고 연습 | 2~4개 |
| Step 5 | 정리 및 마무리 | 2~3개 |

> 질문이 모두 소화됐을 때만 다음 스텝으로 전환 (강제 턴 제한 없음)

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

> PyTorch (CUDA 12.1 빌드)와 AI 모델 패키지가 포함되어 있어 초기 설치에 수분이 걸릴 수 있습니다.

### 3. AI 모델 파일 배치

`models/` 폴더는 Git에 포함되지 않습니다. 팀 내부 공유 스토리지에서 다운로드 후 아래 구조로 배치하세요.

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
| `OPENAI_API_KEY` | _(필수)_ | GPT-4o-mini 플랜 생성용 |
| `WHISPER_MODEL_SIZE` | `small` | Whisper 모델 크기 |
| `WHISPER_DEVICE` | `cuda` | STT 디바이스 |
| `CBT_LLM_DEVICE` | `cuda` | LLM 디바이스 |
| `TEXT_EMOTION_DEVICE` | `cpu` | 텍스트 감정 디바이스 |
| `AUDIO_EMOTION_DEVICE` | `cpu` | 음성 감정 디바이스 |
| `NEGATIVE_EMOTION_THRESHOLD` | `0.65` | 부정 감정 감지 임계값 |
| `APP_PORT` | `8000` | 서버 포트 |

### 5. 서버 실행

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

서버 상태 확인: `http://localhost:8000/`

---

## WebSocket API

```
ws://localhost:8000/ws/counseling/{client_id}
```

### 클라이언트 → 서버

```json
// 초기 상담 설정 (연결 직후 1회)
{
  "type": "setup",
  "data": {
    "topic": "가정사",
    "mood": "우울",
    "content": "엄마와 사이 안좋음"
  }
}

// 음성 바이너리: bytes([0x01]) + float32 PCM 16kHz
// 영상 바이너리: bytes([0x02]) + JPEG bytes

// 발화 종료
{ "type": "control", "data": "END_OF_SPEECH" }

// 세션 종료
{ "type": "control", "data": "END_OF_SESSION" }
```

### 서버 → 클라이언트

```json
{ "status": "connected",        "message": "상담실에 입장하였습니다." }
{ "status": "counseling_ready", "message": "첫 상담사 발화",
  "plan": [{"step": 1, "title": "감정 탐색 및 공감", "goal": "..."}, ...],
  "step_status": {"step": 1, "title": "...", "question_idx": 0, "total_questions": 3, ...}
}
{ "status": "processing",       "message": "답변 생성 중..." }
{ "status": "stt_done",         "text": "STT 변환 결과 텍스트" }
{ "status": "response",         "message": "AI 상담사 응답 텍스트" }
```

---

## 테스트

```bash
python test_e2e.py
```

- starlette TestClient로 in-process 실행 (서버 별도 기동 불필요)
- `test_audio.raw` (float32 PCM 16kHz mono), `input_video.mp4` 파일 필요
- 실제 모델 전체 사용 (VAD / STT / 감정 / LLM)

---

## VRAM 사용량 참고 (GTX 1660 Super 6GB 기준)

| 모델 | 디바이스 | 사용량 |
|------|----------|--------|
| Whisper small | CUDA | ~0.5GB |
| Qwen2.5-3B (8bit 양자화) | CUDA | ~3.5GB |
| 텍스트 감정 (BERT) | CPU | - |
| 음성 감정 (Wav2Vec2) | CPU | - |
| **합계** | | **~4.0GB** |
