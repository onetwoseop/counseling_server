import asyncio
import logging
from typing import Dict, List, Optional

from ai_modules.schemas import (
    CounselingSetup, EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from ai_modules.interfaces import webm_to_float32_pcm
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
        # 브라우저 webm/opus 청크를 그대로 누적 (END_OF_SPEECH 시 배치 변환)
        self._raw_audio_buffer: Dict[str, bytearray] = {}
        # 대화 히스토리 (멀티턴): [{"role": "user"|"assistant", "content": "..."}]
        self._conversation_history: Dict[str, List[Dict[str, str]]] = {}
        # webm→PCM 변환 후 저장 (음성 감정 분석용)
        self._last_pcm_audio: Dict[str, bytes] = {}

    # 세션 수명 주기

    def init_session(self, session_id: str) -> None:
        self.audio.init_session(session_id)
        self._counseling_setup[session_id] = None
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        self._raw_audio_buffer[session_id] = bytearray()
        self._conversation_history[session_id] = []
        self._last_pcm_audio[session_id] = b""
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        self.audio.cleanup_session(session_id)
        for buf in (
            self._counseling_setup,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
            self._raw_audio_buffer,
            self._conversation_history,
            self._last_pcm_audio,
        ):
            buf.pop(session_id, None)
        logger.info(f"세션 정리: {session_id}")

    async def start_transcription_worker(self, session_id: str) -> None:
        await self.audio.start_worker(session_id)

    # 초기 상담 설정 저장
    def setup_counseling(self, session_id: str, topic: str, mood: str, content: str, style: str = None) -> None:
        self._counseling_setup[session_id] = CounselingSetup(
            topic=topic,
            mood=mood,
            content=content,
            style=style
        )
        logger.info(f"[Setup] {session_id}: topic={topic} / mood={mood}")

    # 초기 CBT 질문 생성 (setup 데이터 → LLM → 초기 질문 반환)
    # TODO: AI 개발자 - LLMContext.user_text 대신 CounselingSetup을 직접 활용하는 방식으로 교체 예정
    def generate_initial_questions(self, session_id: str) -> Optional[LLMResponse]:
        setup = self._counseling_setup.get(session_id)
        if not setup:
            logger.warning(f"[InitialQ] {session_id}: 초기 설정 없음, 건너뜀")
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

    # 오디오 청크 → AudioProcessor에 위임 (float32 PCM 입력 시 사용)
    def append_audio_chunk(self, session_id: str, chunk: bytes) -> bool:
        return self.audio.append_chunk(session_id, chunk)

    # 브라우저 webm/opus 청크 누적 (END_OF_SPEECH 시 배치 STT 처리)
    def append_raw_audio_chunk(self, session_id: str, chunk: bytes) -> None:
        if session_id in self._raw_audio_buffer:
            self._raw_audio_buffer[session_id].extend(chunk)

    # webm 누적 버퍼 → ffmpeg PCM 변환 → Whisper 배치 STT
    async def _transcribe_raw_audio(self, session_id: str) -> Optional[STTOutput]:
        raw = bytes(self._raw_audio_buffer.get(session_id, bytearray()))
        if len(raw) < 2000:
            logger.warning(f"[RawSTT] {session_id}: 버퍼 너무 작음 ({len(raw)}B), 건너뜀")
            return None
        try:
            logger.info(f"[RawSTT] {session_id}: ffmpeg 변환 시작 ({len(raw)}B webm/opus)")
            loop = asyncio.get_event_loop()
            audio_array = await loop.run_in_executor(None, webm_to_float32_pcm, raw)
            if audio_array.size < 100:
                logger.warning(f"[RawSTT] {session_id}: PCM 변환 결과 너무 짧음")
                return None
            logger.info(f"[RawSTT] {session_id}: PCM 변환 완료 ({audio_array.size}샘플) → Whisper 실행")
            pcm_bytes = audio_array.tobytes()
            self._last_pcm_audio[session_id] = pcm_bytes  # 음성 감정 분석용 저장
            stt_input = STTInput(audio_data=pcm_bytes)
            result = await loop.run_in_executor(None, self.container.stt.transcribe, stt_input)
            # 다음 발화를 위해 버퍼 초기화
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

    # 발화 종료 → STT 완료 대기 → 음성 감정 → 결과 반환
    async def on_speech_end(self, session_id: str) -> Optional[STTOutput]:
        # 1. 증분 STT 결과 대기 (VAD 기반 - float32 PCM 입력 시 동작)
        accumulated = await self.audio.wait_and_get_text(session_id)

        # 2. 증분 STT 결과가 없으면 → raw webm 버퍼 배치 STT 시도
        if not accumulated:
            logger.info(f"[SpeechEnd] {session_id}: 증분 STT 없음 → 배치 STT 시도 (webm→PCM)")
            raw_result = await self._transcribe_raw_audio(session_id)
            if raw_result and raw_result.text.strip():
                accumulated = raw_result.text.strip()
            else:
                logger.warning(f"[SpeechEnd] {session_id}: 배치 STT도 텍스트 없음, 건너뜀")
                return None

        if session_id not in self._stt_text_buffer:
            return None

        self._stt_text_buffer[session_id].append(accumulated)
        logger.info(f"[SpeechEnd] {session_id}: 최종 텍스트 = '{accumulated}'")

        # 음성 감정 - VAD 스냅샷 우선, 없으면 webm→PCM 변환 결과 사용
        audio_for_emotion = self.audio.get_last_audio_snapshot(session_id)
        if not audio_for_emotion:
            audio_for_emotion = self._last_pcm_audio.get(session_id, b"")
        if audio_for_emotion:
            loop = asyncio.get_event_loop()
            try:
                voice_emotion = await loop.run_in_executor(
                    None, self.container.audio_emotion.analyze, STTInput(audio_data=audio_for_emotion)
                )
                if session_id in self._voice_emotion_buffer:
                    self._voice_emotion_buffer[session_id].append(voice_emotion)
                logger.info(f"[VoiceEmotion] {session_id}: {voice_emotion.primary_emotion} {voice_emotion.probabilities}")
            except Exception as e:
                logger.error(f"[VoiceEmotion] {session_id}: 오류: {e}")

        return STTOutput(text=accumulated, language="ko")

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
    def generate_response(self, session_id: str) -> Optional[LLMResponse]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer.get(session_id, [])
        voice_emotions = self._voice_emotion_buffer.get(session_id, [])
        history = self._conversation_history.get(session_id, [])

        # ── 텍스트 감정 분석 ──────────────────────────────────
        try:
            text_emo = self.container.text_emotion.analyze(accumulated_text)
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
            style_text = f" / 상담 방식: {setup.style}" if setup.style else ""
            context_prefix = (
                f"[상담 설정] 주제: {setup.topic} / 기분: {setup.mood}"
                f" / 내용: {setup.content}{style_text}\n\n"
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
        response = self.container.llm.generate_response(llm_context)
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
