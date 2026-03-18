import asyncio
import logging
from typing import Dict, List, Optional

from ai_modules.schemas import (
    EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer

from app.config import settings

logger = logging.getLogger(__name__)

# CounselingPipeline
# 역할: WebSocket에서 수신한 데이터를 버퍼링하고, AI 모델을 순서대로 호출하는 처리 파이프라인.

# config.py 세팅
SILENCE_THRESHOLD_SEC = settings.vad_silence_threshold
VAD_SAMPLE_RATE = settings.vad_sample_rate
VAD_CHUNK_SAMPLES = settings.vad_chunk_samples
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 4 # float32 = 4bytes


class CounselingPipeline:

    # 데이터들 버퍼
    def __init__(self, container: AIContainer):
        self.container = container
        # 세션별 버퍼
        self._audio_buffers: Dict[str, bytearray] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}
        # VAD 침묵 감지용
        self._vad_chunk_buffer: Dict[str, bytearray] = {}   # VAD chunk 조각 모음
        self._silence_samples: Dict[str, int] = {}           # 연속 침묵 샘플 수
        self._stt_running: Dict[str, bool] = {}              # STT 중복 실행 방지 플래그

    # 세션 수명 주기

    # 버퍼, 세션 초기화
    def init_session(self, session_id: str) -> None:
        self._audio_buffers[session_id] = bytearray()
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        self._vad_chunk_buffer[session_id] = bytearray()
        self._silence_samples[session_id] = 0
        self._stt_running[session_id] = False
        logger.info(f"세션 초기화: {session_id}")

    # 버퍼, 세션 삭제
    def cleanup_session(self, session_id: str) -> None:
        for buf in (
            self._audio_buffers,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
            self._vad_chunk_buffer,
            self._silence_samples,
            self._stt_running,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")


    # 오디오 청크 누적 + VAD 자동 발화 종료 감지
    def append_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        """
        클라이언트에서 실시간으로 들어오는 오디오 청크를 버퍼에 누적.
        동시에 VAD로 침묵을 감지하여 SILENCE_THRESHOLD_SEC 이상 침묵이면
        True를 반환 (발화 종료 신호). session_manager가 이 신호를 받아 on_speech_end 호출.
        chunk는 float32 PCM 16kHz bytes여야 함.
        """
        self._audio_buffers[session_id].extend(chunk)
        total = len(self._audio_buffers[session_id])
        logger.info(f"[Audio] {session_id}: +{len(chunk)}B (누적: {total}B)")

        # VAD chunk 버퍼에 이어붙이고 512샘플(2048bytes) 단위로 처리
        self._vad_chunk_buffer[session_id].extend(chunk)
        while len(self._vad_chunk_buffer[session_id]) >= VAD_CHUNK_BYTES:
            vad_chunk = bytes(self._vad_chunk_buffer[session_id][:VAD_CHUNK_BYTES])
            self._vad_chunk_buffer[session_id] = self._vad_chunk_buffer[session_id][VAD_CHUNK_BYTES:]

            from ai_modules.schemas import VADInput
            vad_result = self.container.vad.process(VADInput(audio_chunk=vad_chunk))

            if vad_result.is_speech:
                self._silence_samples[session_id] = 0
            else:
                self._silence_samples[session_id] += VAD_CHUNK_SAMPLES
                silence_sec = self._silence_samples[session_id] / VAD_SAMPLE_RATE
                if silence_sec >= SILENCE_THRESHOLD_SEC:
                    self._silence_samples[session_id] = 0
                    # STT가 이미 실행 중이면 중복 호출 방지
                    if not self._stt_running.get(session_id, False):
                        logger.info(f"[VAD] {session_id}: {silence_sec:.1f}초 침묵 감지 → 발화 종료")
                        return True  # 발화 종료 신호 - session_manager가 처리

        return False  # 아직 발화 중

    # 이미지 프레임 → 얼굴 감정 분석
    def process_face_frame(self, session_id: str, image_bytes: bytes) -> EmotionResult:
        """JPEG bytes를 받아 얼굴 감정을 추출하고 버퍼에 저장."""
        face_input = FaceInput(video_frame=image_bytes)
        result = self.container.face_emotion.analyze(face_input)
        self._face_emotion_buffer[session_id].append(result)
        logger.info(
            f"[Face] {session_id}: "
            f"{result.primary_emotion} {result.probabilities}"
        )
        return result

    # 발화 종료 → STT + 음성 감정
    async def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        """
        END_OF_SPEECH 신호 수신 시 호출.
        버퍼된 오디오 전체를 STT와 음성 감정 모델에 넘김.
        STT는 CPU 집중 작업이므로 스레드풀에서 실행해 이벤트 루프를 블로킹하지 않음.
        """
        if session_id not in self._audio_buffers:
            return None

        audio_data = bytes(self._audio_buffers[session_id])
        if not audio_data:
            logger.warning(f"[SpeechEnd] {session_id}: 오디오 버퍼 비어있음, 건너뜀")
            return None

        self._stt_running[session_id] = True
        logger.info(f"[SpeechEnd] {session_id}: 버퍼 {len(audio_data)}B → 처리 시작")
        stt_input = STTInput(audio_data=audio_data)
        self._audio_buffers[session_id].clear()

        loop = asyncio.get_event_loop()

        try:
            # STT - 스레드풀에서 실행 (블로킹 방지)
            stt_result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)

            # STT 완료 후 세션이 이미 정리됐을 수 있음 (연결 종료 등)
            if session_id not in self._stt_text_buffer:
                logger.warning(f"[SpeechEnd] {session_id}: 세션 정리됨, 결과 버림")
                return None

            self._stt_text_buffer[session_id].append(stt_result.text)
            logger.info(f"[STT] {session_id}: '{stt_result.text}'")

            # 음성 감정
            voice_emotion = await loop.run_in_executor(None, self.container.audio_emotion.analyze, stt_input)
            if session_id in self._voice_emotion_buffer:
                self._voice_emotion_buffer[session_id].append(voice_emotion)
            logger.info(
                f"[VoiceEmotion] {session_id}: "
                f"{voice_emotion.primary_emotion} {voice_emotion.probabilities}"
            )

            return stt_result
        finally:
            if session_id in self._stt_running:
                self._stt_running[session_id] = False

    # 세션 종료 → 감정 종합 → LLM 응답

    def generate_response(self, session_id: str) -> Optional[LLMResponse]:
        """
        상담 싱글턴 종료 신호 수신 시 호출.
        누적된 텍스트와 감정 목록을 그대로 AI에 전달.
        """
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer[session_id]
        voice_emotions = self._voice_emotion_buffer[session_id]

        logger.info(
            f"[Generate] {session_id}: "
            f"face {len(face_emotions)}건, voice {len(voice_emotions)}건 → AI 전달"
        )
        logger.info(f"[Generate] {session_id}: 누적 텍스트='{accumulated_text}'")

        llm_context = LLMContext(
            user_text=accumulated_text,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
        )
        response = self.container.llm.generate_response(llm_context)
        logger.info(f"[LLM] {session_id}: '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"[LLM] 추천 행동: '{response.suggested_action}'")

        return response


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
