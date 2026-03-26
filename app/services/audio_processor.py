import asyncio
import logging
from typing import Dict, Optional

from ai_modules.schemas import STTInput, VADInput
from app.core.container import AIContainer
from app.core.config import settings

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD_SEC = settings.vad_silence_threshold
VAD_SAMPLE_RATE = settings.vad_sample_rate
VAD_CHUNK_SAMPLES = settings.vad_chunk_samples
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 4  # float32 = 4bytes


class AudioProcessor:
    """
    VAD(음성 감지) + 증분 STT 처리를 담당하는 클래스.
    CounselingPipeline에서 오디오 관련 버퍼와 로직을 위임받아 처리한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self._audio_buffers: Dict[str, bytearray] = {}
        self._vad_chunk_buffer: Dict[str, bytearray] = {}
        self._silence_samples: Dict[str, int] = {}
        self._stt_running: Dict[str, bool] = {}
        self._accumulated_text: Dict[str, str] = {}
        self._transcription_queue: Dict[str, asyncio.Queue] = {}
        self._transcription_tasks: Dict[str, asyncio.Task] = {}
        self._last_audio_snapshot: Dict[str, bytes] = {}

    def init_session(self, session_id: str) -> None:
        self._audio_buffers[session_id] = bytearray()
        self._vad_chunk_buffer[session_id] = bytearray()
        self._silence_samples[session_id] = 0
        self._stt_running[session_id] = False
        self._accumulated_text[session_id] = ""
        self._transcription_queue[session_id] = asyncio.Queue(maxsize=64)
        self._last_audio_snapshot[session_id] = b""

    def cleanup_session(self, session_id: str) -> None:
        task = self._transcription_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        for buf in (
            self._audio_buffers,
            self._vad_chunk_buffer,
            self._silence_samples,
            self._stt_running,
            self._accumulated_text,
            self._transcription_queue,
            self._last_audio_snapshot,
        ):
            buf.pop(session_id, None)

    # 증분 STT 워커 시작 (세션 연결 시 호출)
    async def start_worker(self, session_id: str) -> None:
        task = asyncio.create_task(self._worker(session_id))
        self._transcription_tasks[session_id] = task

    # 증분 STT 워커 - 큐에서 오디오를 꺼내 STT 실행 후 텍스트 누적
    async def _worker(self, session_id: str) -> None:
        queue = self._transcription_queue[session_id]
        loop = asyncio.get_event_loop()
        try:
            while True:
                audio_bytes = await queue.get()
                try:
                    if session_id not in self._accumulated_text:
                        continue
                    self._stt_running[session_id] = True
                    stt_input = STTInput(audio_data=audio_bytes)
                    stt_result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
                    if session_id in self._accumulated_text and stt_result.text.strip():
                        prev = self._accumulated_text[session_id]
                        self._accumulated_text[session_id] = (prev + " " + stt_result.text.strip()).strip()
                        logger.info(f"[IncrSTT] {session_id}: +'{stt_result.text}' → 누적: '{self._accumulated_text[session_id]}'")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[IncrSTT] {session_id}: STT 실패: {e}")
                finally:
                    queue.task_done()
                    if session_id in self._stt_running:
                        self._stt_running[session_id] = False
        except asyncio.CancelledError:
            logger.info(f"[IncrSTT] {session_id}: 워커 종료")

    # 오디오 청크 누적 + VAD 침묵 감지 → 침묵 시 STT 큐 등록
    def append_chunk(self, session_id: str, chunk: bytes) -> bool:
        self._audio_buffers[session_id].extend(chunk)
        total = len(self._audio_buffers[session_id])
        logger.info(f"[Audio] {session_id}: +{len(chunk)}B (누적: {total}B)")

        self._vad_chunk_buffer[session_id].extend(chunk)
        while len(self._vad_chunk_buffer[session_id]) >= VAD_CHUNK_BYTES:
            vad_chunk = bytes(self._vad_chunk_buffer[session_id][:VAD_CHUNK_BYTES])
            self._vad_chunk_buffer[session_id] = self._vad_chunk_buffer[session_id][VAD_CHUNK_BYTES:]

            vad_result = self.container.vad.process(VADInput(audio_chunk=vad_chunk))

            if vad_result.is_speech:
                self._silence_samples[session_id] = 0
            else:
                self._silence_samples[session_id] += VAD_CHUNK_SAMPLES
                silence_sec = self._silence_samples[session_id] / VAD_SAMPLE_RATE
                if silence_sec >= SILENCE_THRESHOLD_SEC:
                    self._silence_samples[session_id] = 0
                    audio_snapshot = bytes(self._audio_buffers[session_id])
                    if audio_snapshot and session_id in self._transcription_queue:
                        self._last_audio_snapshot[session_id] = audio_snapshot
                        self._audio_buffers[session_id].clear()
                        try:
                            self._transcription_queue[session_id].put_nowait(audio_snapshot)
                            logger.info(f"[VAD] {session_id}: {silence_sec:.1f}초 침묵 → STT 큐 등록 ({len(audio_snapshot)}B)")
                        except asyncio.QueueFull:
                            logger.warning(f"[VAD] {session_id}: STT 큐 가득 참, 세그먼트 버림")
                    return True  # 발화 종료 신호

        return False  # 아직 발화 중

    # STT 큐 완료 대기 후 누적 텍스트 반환 (없으면 폴백 직접 STT)
    async def wait_and_get_text(self, session_id: str) -> Optional[str]:
        if session_id not in self._transcription_queue:
            return None

        queue = self._transcription_queue[session_id]

        if not queue.empty() or self._stt_running.get(session_id, False):
            logger.info(f"[SpeechEnd] {session_id}: 증분 STT 완료 대기 중...")
            await queue.join()

        accumulated = self._accumulated_text.get(session_id, "").strip()

        # 폴백: 큐 처리가 안 됐고 버퍼에 오디오가 남아있는 경우
        if not accumulated:
            audio_data = bytes(self._audio_buffers[session_id])
            if not audio_data:
                return None
            logger.info(f"[SpeechEnd] {session_id}: 폴백 STT ({len(audio_data)}B)")
            stt_input = STTInput(audio_data=audio_data)
            self._audio_buffers[session_id].clear()
            loop = asyncio.get_event_loop()
            self._stt_running[session_id] = True
            try:
                stt_result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
                accumulated = stt_result.text.strip()
            finally:
                if session_id in self._stt_running:
                    self._stt_running[session_id] = False

        # 다음 발화를 위해 초기화
        if session_id in self._accumulated_text:
            self._accumulated_text[session_id] = ""

        return accumulated or None

    def get_last_audio_snapshot(self, session_id: str) -> bytes:
        """음성 감정 분석용 마지막 오디오 스냅샷 반환"""
        return self._last_audio_snapshot.get(session_id, b"")
