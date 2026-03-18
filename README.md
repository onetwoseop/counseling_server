# Attune Counseling Server

AI 기반 실시간 상담 WebSocket 서버입니다.
클라이언트로부터 음성 청크와 영상 프레임을 WebSocket으로 수신하여 VAD → STT → 감정 분석 → LLM 응답 생성 파이프라인을 처리합니다.

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| 웹 프레임워크 | FastAPI + Uvicorn |
| 음성 감지 (VAD) | Silero VAD (PyTorch) |
| 음성 인식 (STT) | faster-whisper |
| 감정 분석 | Dummy 모델 (추후 DeepFace / SpeechBrain 교체 예정) |
| LLM | Dummy 모델 (추후 Qwen2 교체 예정) |

---

## 프로젝트 구조

```
counseling_server/
├── app/
│   ├── main.py                  # FastAPI 앱, WebSocket 엔드포인트
│   ├── config.py                # 환경변수 설정 (pydantic-settings)
│   ├── schemas.py               # WebSocket 메시지 스키마
│   ├── core/
│   │   └── container.py         # AI 모델 싱글턴 컨테이너
│   └── services/
│       ├── pipeline.py          # VAD·STT·감정·LLM 처리 파이프라인
│       └── session_manager.py   # WebSocket 연결 관리 및 데이터 라우팅
├── ai_modules/
│   ├── interfaces.py            # AI 모델 구현체 (실제 + 더미)
│   └── schemas.py               # AI 모델 입출력 스키마
├── tests/
│   ├── test_pipeline.py         # 파이프라인 구조 검증 (서버 불필요)
│   └── test_ws_client.py        # WebSocket 통합 테스트 (서버 필요, mp4 사용)
├── .env.example                 # 환경변수 템플릿
├── requirements.txt
└── README.md
```

---

## 시작하기

### 1. 파이썬 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 활성화 (macOS / Linux)
source venv/bin/activate
```

### 2. 의존성 설치

> PyTorch는 용량이 크므로 처음 설치에 시간이 걸릴 수 있습니다.

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

`.env.example`을 복사해 `.env` 파일을 만들고 필요에 따라 값을 수정합니다.

```bash
cp .env.example .env
```

주요 설정값:

| 키 | 기본값 | 설명 |
|----|--------|------|
| `WHISPER_MODEL_SIZE` | `base` | Whisper 모델 크기 (`tiny` / `base` / `small` / `medium` / `large`) |
| `VAD_SILENCE_THRESHOLD` | `0.5` | 발화 종료로 판단할 침묵 시간 (초) |
| `APP_PORT` | `8000` | 서버 포트 |

### 4. 서버 실행

```bash
# 개발 모드 (코드 변경 시 자동 재시작)
uvicorn app.main:app --reload

# 특정 포트 지정
uvicorn app.main:app --reload --port 8000

# 외부 접속 허용
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 정상 실행되면 아래 메시지가 출력됩니다:
```
>>> AI 모델 로딩을 시작합니다.....
[VAD] Silero VAD Loaded
[STT] faster-whisper (base) Loaded
...
>>> 모든 모델 로딩 완료
INFO:     Application startup complete.
```

서버 상태 확인:
```
http://localhost:8000/
```

---

## WebSocket 연결

```
ws://localhost:8000/ws/counseling/{client_id}
```

### 클라이언트 → 서버 메시지 형식

```json
// 음성 청크
{ "type": "audio", "data": "<base64 float32 PCM 16kHz>" }

// 영상 프레임
{ "type": "video", "data": "<base64 JPEG>" }

// 발화 종료 신호 (수동)
{ "type": "control", "data": "END_OF_SPEECH" }

// 세션 종료
{ "type": "control", "data": "END_OF_SESSION" }
```

### 서버 → 클라이언트 메시지 형식

```json
{ "status": "connected",   "message": "상담실에 입장하였습니다." }
{ "status": "processing",  "message": "답변 생성 중..." }
{ "status": "response",    "message": "AI 상담사 응답 텍스트" }
```

---

## 테스트

### test_pipeline.py — 파이프라인 구조 검증

서버 실행 없이 더미 데이터로 전체 파이프라인 흐름(모델 로딩 → 얼굴 감정 → 오디오 버퍼링 → STT → LLM 응답)을 검증합니다.

```bash
cd counseling_server
python -m tests.test_pipeline
```

정상 출력 예시:
```
STEP 0  AI 모델 로딩 (Dummy)
STEP 1  이미지 프레임 3개 → 얼굴 감정 분석
STEP 2  오디오 청크 수신 (발화 중 시뮬레이션)
STEP 3  END_OF_SPEECH → STT + 음성 감정 추출
STEP 4-5  END_OF_SESSION → 감정 종합 → LLM 응답 생성
전체 파이프라인 구조 검증 완료
```

---

### test_ws_client.py — WebSocket 통합 테스트

실제 mp4 파일에서 오디오/영상을 추출해 WebSocket으로 서버에 전송하는 통합 테스트입니다.
**서버가 먼저 실행 중이어야 합니다.**

**사전 준비:** `test_files/test_video1.mp4` 파일이 필요합니다.

```bash
# 터미널 1 — 서버 실행
uvicorn app.main:app --reload

# 터미널 2 — 테스트 실행
cd counseling_server
python -m tests.test_ws_client
```

동작 방식:
- mp4에서 오디오를 `float32 PCM 16kHz` 로 변환 후 `32ms` 단위 청크로 전송
- `90프레임(약 3초)` 마다 영상 프레임 1장을 JPEG로 변환하여 전송
- `15초` 마다 `END_OF_SPEECH` 신호를 보내 STT + LLM 응답 수신 대기
- 추출된 프레임은 `temp_frames/` 폴더에 저장되어 육안 확인 가능

---

## AI 모델 교체 가이드

현재 더미 모델로 동작하는 항목은 `ai_modules/interfaces.py`에서 실제 모델 클래스로 교체할 수 있습니다.
`requirements.txt`의 주석 처리된 패키지를 함께 해제하세요.

| 모듈 | 현재 | 교체 대상 |
|------|------|-----------|
| 음성 감정 | `DummyAudioEmotionModel` | SpeechBrain |
| 얼굴 감정 | `DummyFaceEmotionModel` | DeepFace |
| LLM | `DummyLLMModel` | Qwen2-7B-Instruct |
