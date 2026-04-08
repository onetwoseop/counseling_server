import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional

from ai_modules.schemas import (
    CounselingSetup, EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer
from app.services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class CounselingPipeline:
    """
    WebSocket에서 수신한 데이터를 버퍼링하고 AI 모델을 순서대로 호출하는 오케스트레이터.
    오디오(VAD/STT) 처리는 AudioProcessor에 위임한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.audio = AudioProcessor(container)
        # 세션별 비오디오 버퍼
        self._counseling_setup: Dict[str, Optional[CounselingSetup]] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}
        # 16kHz Float32 PCM 청크 누적 버퍼 (배치 STT 폴백용)
        self._raw_audio_buffer: Dict[str, bytearray] = {}
        # 청크별 실시간 STT 누적 텍스트 (VAD 미사용 경로 폴백용)
        self._chunk_stt_text: Dict[str, str] = {}
        # 대화 히스토리 (멀티턴): [{"role": "user"|"assistant", "content": "..."}]
        self._conversation_history: Dict[str, List[Dict[str, str]]] = {}
        # 음성 감정 분석용 마지막 PCM 스냅샷
        self._last_pcm_audio: Dict[str, bytes] = {}
        # 음성 감정 분석 백그라운드 태스크 (LLM과 병렬 실행)
        self._voice_emotion_tasks: Dict[str, asyncio.Task] = {}

    # 세션 수명 주기

    def init_session(self, session_id: str) -> None:
        self.audio.init_session(session_id)
        self._counseling_setup[session_id] = None
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        self._raw_audio_buffer[session_id] = bytearray()
        self._chunk_stt_text[session_id] = ""
        self._conversation_history[session_id] = []
        self._last_pcm_audio[session_id] = b""
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        self.audio.cleanup_session(session_id)
        # 음성 감정 태스크가 남아있으면 취소
        task = self._voice_emotion_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        for buf in (
            self._counseling_setup,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
            self._raw_audio_buffer,
            self._chunk_stt_text,
            self._conversation_history,
            self._last_pcm_audio,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")

    # 실시간 오디오 처리를 위한 stt 백그라운드 실행 (지금 안 씀)
    async def start_transcription_worker(self, session_id: str) -> None:
        await self.audio.start_worker(session_id)

    # 초기 상담 설정 저장
    def setup_counseling(self, session_id: str, topic: str, mood: str, content: str) -> None:
        self._counseling_setup[session_id] = CounselingSetup(
            topic=topic,
            mood=mood,
            content=content
        )
        logger.info(f"[Setup] {session_id}: topic={topic} / mood={mood}")

    # 초기 CBT 질문 생성 (setup 데이터 → LLM → 초기 질문 반환)
    def generate_initial_questions(self, session_id: str) -> Optional[LLMResponse]:
        setup = self._counseling_setup.get(session_id)
        if not setup:
            logger.warning(f"[InitialQ] {session_id}: 초기 설정 없음")
            return None
        if self.container.llm is None:
            logger.warning(f"[InitialQ] {session_id}: LLM 미로딩, 건너뜀")
            return None
        setup_text = f"주제: {setup.topic}, 기분: {setup.mood}, 내용: {setup.content}"
        llm_context = LLMContext(user_text=setup_text)
        response = self.container.llm.generate_response(llm_context)
        logger.info(f"[InitialQ] {session_id}: '{response.reply_text}'")

        # 초기 질문을 히스토리에 저장 (이후 대화에서 맥락 유지)
        if session_id in self._conversation_history:
            self._conversation_history[session_id].append(
                {"role": "user", "content": "[상담 시작] " + setup_text}
            )
            self._conversation_history[session_id].append(
                {"role": "assistant", "content": response.reply_text}
            )
        return response

    # 오디오 청크 → VAD 필터 → 음성 구간만 _audio_buffers에 누적 (AudioProcessor에 위임)
    def append_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        return self.audio.append_chunk(session_id, chunk)

    # 16kHz Float32 PCM 청크 누적 (배치 STT 폴백용)
    def append_raw_audio_chunk(self, session_id: str, chunk: bytes) -> None:
        if session_id in self._raw_audio_buffer:
            self._raw_audio_buffer[session_id].extend(chunk)

    # Float32 PCM 청크 → numpy → Whisper STT + 음성 감정 추출 (VAD 미사용 경로)
    async def transcribe_audio_chunk(self, session_id: str, chunk: bytes) -> None:
        if len(chunk) < 2000 or session_id not in self._chunk_stt_text:
            return
        try:
            loop = asyncio.get_event_loop()
            audio_array = np.frombuffer(chunk, dtype=np.float32)
            if audio_array.size < 100:
                return
            pcm_bytes = audio_array.tobytes()
            self._last_pcm_audio[session_id] = pcm_bytes
            stt_input = STTInput(audio_data=pcm_bytes)

            result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
            if session_id not in self._chunk_stt_text:
                return
            if result.text.strip():
                self._chunk_stt_text[session_id] = (
                    self._chunk_stt_text[session_id] + " " + result.text.strip()
                ).strip()
                logger.info(f"[ChunkSTT] {session_id}: +'{result.text.strip()}' → 누적: '{self._chunk_stt_text[session_id]}'")

            try:
                voice_emotion = await loop.run_in_executor(
                    None, self.container.audio_emotion.analyze, stt_input
                )
                if session_id in self._voice_emotion_buffer:
                    self._voice_emotion_buffer[session_id].append(voice_emotion)
                    logger.info(f"[ChunkVoiceEmo] {session_id}: {voice_emotion.primary_emotion}")
            except Exception as e:
                logger.error(f"[ChunkVoiceEmo] {session_id}: 오류: {e}")

        except Exception as e:
            logger.error(f"[ChunkSTT] {session_id}: 오류: {e}")

    # PCM 누적 버퍼 → Whisper 배치 STT (폴백용)
    async def _transcribe_raw_audio(self, session_id: str) -> Optional[STTOutput]:
        raw = bytes(self._raw_audio_buffer.get(session_id, bytearray()))
        if len(raw) < 2000:
            logger.warning(f"[RawSTT] {session_id}: 버퍼 너무 작음 ({len(raw)}B), 건너뜀")
            return None
        try:
            logger.info(f"[RawSTT] {session_id}: PCM 변환 시작 ({len(raw)}B)")
            loop = asyncio.get_event_loop()
            audio_array = np.frombuffer(raw, dtype=np.float32)
            if audio_array.size < 100:
                logger.warning(f"[RawSTT] {session_id}: PCM 변환 결과 너무 짧음")
                return None
            logger.info(f"[RawSTT] {session_id}: PCM 변환 완료 ({audio_array.size}샘플) → Whisper 실행")
            pcm_bytes = audio_array.tobytes()
            self._last_pcm_audio[session_id] = pcm_bytes
            stt_input = STTInput(audio_data=pcm_bytes)
            result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
            self._raw_audio_buffer[session_id] = bytearray()
            logger.info(f"[RawSTT] {session_id}: 결과='{result.text}'")
            return result
        except Exception as e:
            logger.error(f"[RawSTT] {session_id}: 오류: {e}")

    # 이미지 프레임 → 얼굴 감정 분석
    def process_face_frame(self, session_id: str, image_bytes: bytes) -> EmotionResult:
        face_input = FaceInput(video_frame=image_bytes)
        result = self.container.face_emotion.analyze(face_input)
        self._face_emotion_buffer[session_id].append(result)
        logger.info(f"[Face] {session_id}: {result.primary_emotion} {result.probabilities}")
        return result

    # 발화 종료 → VAD 누적 음성 일괄 STT → 음성 감정 백그라운드 시작 → 결과 반환
    async def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        # 1. VAD 누적 음성 일괄 STT 처리 (주 경로)
        accumulated = await self.audio.wait_and_get_text(session_id)

        # 2. 폴백: VAD 음성 없을 때 청크별 직접 STT 누적 텍스트 사용
        # (transcribe_audio_chunk가 호출된 경우에만 유효, 현재 미사용 경로)
        if not accumulated:
            accumulated = self._chunk_stt_text.get(session_id, "").strip()
            self._chunk_stt_text[session_id] = ""
            if accumulated:
                logger.info(f"[SpeechEnd] {session_id}: 청크 STT 누적 텍스트 사용: '{accumulated}'")

        # 3. 폴백: PCM 누적 버퍼 배치 STT
        # (append_raw_audio_chunk가 호출된 경우에만 유효, 현재 미사용 경로)
        if not accumulated:
            logger.info(f"[SpeechEnd] {session_id}: 청크 STT 없음 → 배치 STT 폴백")
            raw_result = await self._transcribe_raw_audio(session_id)
            if raw_result and raw_result.text.strip():
                accumulated = raw_result.text.strip()
            else:
                logger.warning(f"[SpeechEnd] {session_id}: 배치 STT도 텍스트 없음, 건너뜀")
                return None

        if session_id not in self._stt_text_buffer:
            return None

        # 음성 감정 분석 — LLM과 병렬 실행하기 위해 백그라운드 태스크로 분리
        # generate_response() 초반에 결과를 join함
        voice_pcm = self.audio.get_last_audio_snapshot(session_id)
        if voice_pcm and session_id in self._voice_emotion_buffer:
            self._voice_emotion_tasks[session_id] = asyncio.create_task(
                self._analyze_voice_emotion(session_id, voice_pcm)
            )

        self._stt_text_buffer[session_id].append(accumulated)
        logger.info(f"[SpeechEnd] {session_id}: 최종 텍스트 = '{accumulated}'")
        return STTOutput(text=accumulated, language="ko")

    # 음성 감정 분석 백그라운드 태스크
    async def _analyze_voice_emotion(self, session_id: str, voice_pcm: bytes) -> None:
        try:
            loop = asyncio.get_event_loop()
            voice_emotion = await loop.run_in_executor(
                None, self.container.audio_emotion.analyze, STTInput(audio_data=voice_pcm)
            )
            if session_id in self._voice_emotion_buffer:
                self._voice_emotion_buffer[session_id].append(voice_emotion)
                logger.info(f"[VoiceEmo] {session_id}: {voice_emotion.primary_emotion} {voice_emotion.probabilities}")
        except Exception as e:
            logger.error(f"[VoiceEmo] {session_id}: 오류: {e}")

    # 여러 EmotionResult 리스트에서 확률 평균 → 대표 감정 1개 반환
    @staticmethod
    def _average_emotion(results: List[EmotionResult]) -> EmotionResult:
        if not results:
            return EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})
        if len(results) == 1:
            return results[0]
        from collections import defaultdict
        avg: dict = defaultdict(float)
        for r in results:
            for emotion, prob in r.probabilities.items():
                avg[emotion] += prob
        n = len(results)
        prob_dict = {k: round(v / n, 3) for k, v in avg.items()}
        primary = max(prob_dict, key=prob_dict.get)
        return EmotionResult(primary_emotion=primary, probabilities=prob_dict)

    # STT 누적 텍스트 + 3모달 감정 융합 + 히스토리 → LLM 응답 생성
    async def generate_response(self, session_id: str) -> Optional[LLMResponse]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        # 음성 감정 백그라운드 태스크 완료 대기 (on_speech_end에서 시작, LLM과 병렬 실행됨)
        voice_task = self._voice_emotion_tasks.pop(session_id, None)
        if voice_task:
            await voice_task

        face_emotions = self._face_emotion_buffer.get(session_id, [])
        voice_emotions = self._voice_emotion_buffer.get(session_id, [])
        history = self._conversation_history.get(session_id, [])
        loop = asyncio.get_event_loop()

        # ── 텍스트 감정 분석 (run_in_executor → 이벤트 루프 비점유) ──
        try:
            text_emo = await loop.run_in_executor(
                None, self.container.text_emotion.analyze, accumulated_text
            )
            logger.info(f"[TextEmo] {session_id}: {text_emo.primary_emotion} {text_emo.probabilities}")
        except Exception as e:
            logger.error(f"[TextEmo] {session_id}: 오류: {e}")
            text_emo = EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})

        # ── 감정 리스트 평균화 ────────────────────────────────
        face_emo = self._average_emotion(face_emotions)
        voice_emo = self._average_emotion(voice_emotions)

        # ── 3모달 감정 융합 ───────────────────────────────────
        try:
            fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)
            logger.info(
                f"[Fusion] {session_id}: text={text_emo.primary_emotion} "
                f"voice={voice_emo.primary_emotion} face={face_emo.primary_emotion} "
                f"→ fused={fused_emo.primary_emotion}"
            )
        except Exception as e:
            logger.error(f"[Fusion] {session_id}: 오류: {e}")
            fused_emo = text_emo

        logger.info(
            f"[Generate] {session_id}: face {len(face_emotions)}건, "
            f"voice {len(voice_emotions)}건, history {len(history)//2}턴 → AI 전달"
        )
        logger.info(f"[Generate] {session_id}: 누적 텍스트='{accumulated_text}'")

        # 첫 번째 턴에만 [상담 설정] 접두어 포함
        setup = self._counseling_setup.get(session_id)
        if setup and not history:
            context_prefix = (
                f"[상담 설정] 주제: {setup.topic} / 기분: {setup.mood}"
                f" / 내용: {setup.content}\n\n"
                f"[사용자 발화] "
            )
            full_text = context_prefix + accumulated_text
        else:
            full_text = accumulated_text

        llm_context = LLMContext(
            user_text=full_text,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
            text_emotion=text_emo.primary_emotion,
            fused_emotion=fused_emo.primary_emotion,
            history=history,
        )
        # LLM 추론 (run_in_executor → GPU 연산 중에도 이벤트 루프 유지)
        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )
        logger.info(f"[LLM] {session_id}: '{response.reply_text}'")
        if response.suggested_action:
            logger.info(f"[LLM] 추천 행동: '{response.suggested_action}'")

        # 히스토리에 이번 턴 추가 (clean 텍스트, 설정 접두어 없이)
        if session_id in self._conversation_history:
            self._conversation_history[session_id].append(
                {"role": "user", "content": accumulated_text}
            )
            self._conversation_history[session_id].append(
                {"role": "assistant", "content": response.reply_text}
            )
            # 최대 20개 메시지(10턴) 유지 → 토큰 오버플로우 방지
            if len(self._conversation_history[session_id]) > 20:
                self._conversation_history[session_id] = self._conversation_history[session_id][-20:]

        # 다음 발화를 위해 감정/STT 버퍼 초기화
        self._stt_text_buffer[session_id] = []
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []

        return response


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
