import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional

from ai_modules.schemas import (
    CounselingSetup, EmotionResult, FaceInput, LLMContext, LLMResponse, STTInput, STTOutput
)
from app.core.container import AIContainer
from app.services.audio_processor import AudioProcessor
from app.services.counseling_session import CounselingSession

logger = logging.getLogger(__name__)


class CounselingPipeline:
    """
    WebSocket에서 수신한 데이터를 버퍼링하고 AI 모델을 순서대로 호출하는 오케스트레이터.
    오디오(VAD/STT) 처리는 AudioProcessor에 위임한다.
    """

    def __init__(self, container: AIContainer):
        self.container = container
        self.audio = AudioProcessor(container)
        self.session = CounselingSession(container)
        # 세션별 비오디오 버퍼
        self._counseling_setup: Dict[str, Optional[CounselingSetup]] = {}
        self._face_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._voice_emotion_buffer: Dict[str, List[EmotionResult]] = {}
        self._stt_text_buffer: Dict[str, List[str]] = {}
        # 16kHz Float32 PCM 청크 누적 버퍼 (배치 STT 폴백용)
        self._raw_audio_buffer: Dict[str, bytearray] = {}
        # 청크별 실시간 STT 누적 텍스트 (VAD 미사용 경로 폴백용)
        self._chunk_stt_text: Dict[str, str] = {}
        # 음성 감정 분석용 마지막 PCM 스냅샷
        self._last_pcm_audio: Dict[str, bytes] = {}
        # 음성 감정 분석 throttle용 세션별 Lock (동시 실행 1개로 제한, 폭주 방지)
        self._voice_emotion_locks: Dict[str, asyncio.Lock] = {}

    # 세션 수명 주기

    def init_session(self, session_id: str) -> None:
        self.audio.init_session(session_id)
        self.session.init_session(session_id)
        self._counseling_setup[session_id] = None
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []
        self._stt_text_buffer[session_id] = []
        self._raw_audio_buffer[session_id] = bytearray()
        self._chunk_stt_text[session_id] = ""
        self._last_pcm_audio[session_id] = b""
        self._voice_emotion_locks[session_id] = asyncio.Lock()
        logger.info(f"세션 초기화: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        self.audio.cleanup_session(session_id)
        self.session.cleanup_session(session_id)
        for buf in (
            self._counseling_setup,
            self._face_emotion_buffer,
            self._voice_emotion_buffer,
            self._stt_text_buffer,
            self._raw_audio_buffer,
            self._chunk_stt_text,
            self._last_pcm_audio,
            self._voice_emotion_locks,
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

    # 이미지 프레임 → 얼굴 감정 분석 (비동기, 이벤트 루프 블로킹 방지)
    async def process_face_frame(self, session_id: str, image_bytes: bytes) -> None:
        try:
            face_input = FaceInput(video_frame=image_bytes)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.container.face_emotion.analyze, face_input
            )
            # cleanup race 방어: 분석 완료 시점에 세션이 종료됐을 수 있음
            if session_id in self._face_emotion_buffer:
                self._face_emotion_buffer[session_id].append(result)
                logger.info(f"[Face] {session_id}: {result.primary_emotion} {result.probabilities}")
        except Exception as e:
            logger.error(f"[Face] {session_id}: 분석 오류: {e}")

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

        # 발화 중 청크별 음성 감정이 이미 _voice_emotion_buffer에 누적되어 있으므로
        # 전체 버퍼 재분석 태스크는 생략 (17초 절약)
        self._stt_text_buffer[session_id].append(accumulated)
        logger.info(f"[SpeechEnd] {session_id}: 최종 텍스트 = '{accumulated}'")
        return STTOutput(text=accumulated, language="ko")

    # 음성 감정 분석 백그라운드 태스크 (실제 추론)
    async def _analyze_voice_emotion(self, session_id: str, voice_pcm: bytes) -> None:
        try:
            t0 = time.time()
            loop = asyncio.get_event_loop()
            voice_emotion = await loop.run_in_executor(
                None, self.container.audio_emotion.analyze, STTInput(audio_data=voice_pcm)
            )
            if session_id in self._voice_emotion_buffer:
                self._voice_emotion_buffer[session_id].append(voice_emotion)
                logger.info(
                    f"[VoiceEmo] {session_id}: {voice_emotion.primary_emotion} "
                    f"{voice_emotion.probabilities} ({time.time() - t0:.2f}초)"
                )
        except Exception as e:
            logger.error(f"[VoiceEmo] {session_id}: 오류: {e}")

    # 음성 감정 분석 throttled wrapper — 세션당 동시 실행 1개로 제한 (폭주 방지)
    # 이미 분석 중이면 이 청크는 드롭. session_manager에서 매 청크마다 호출.
    async def analyze_voice_emotion_throttled(self, session_id: str, voice_pcm: bytes) -> None:
        lock = self._voice_emotion_locks.get(session_id)
        if lock is None or lock.locked():
            return
        async with lock:
            await self._analyze_voice_emotion(session_id, voice_pcm)

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

    @staticmethod
    def _build_dynamic_system_prompt(step_mgr, current_q, history_mgr) -> str:
        """
        분석(인지 왜곡) + 현재 단계(목표/집중) + 이전 단계 요약 + 응답 지시를 조립.
        """
        KOREAN_RULE = "반드시 한국어로만 답변하세요. 영어 단어는 절대 사용하지 마세요."
        analysis = step_mgr.analysis or {}
        step = step_mgr.current_step

        parts = [
            "[당신의 역할]",
            "당신은 따뜻하고 공감적인 CBT(인지행동치료) 전문 상담사 '루나'입니다.",
            KOREAN_RULE,
            "",
            "[내담자 분석]",
            f"- 핵심 문제: {analysis.get('core_problem', '미분석')}",
            f"- 예상 인지 왜곡: {analysis.get('cognitive_pattern', '미분석')}",
            "",
            f"[현재 단계: Step {step_mgr.step_number} - {step['name']}]",
            f"- 목표: {step.get('goal', '')}",
            f"- 집중 포인트(CBT 기법): {step.get('focus', '')}",
        ]

        # 이전 단계 요약 주입
        if history_mgr:
            summaries = history_mgr.get_step_summaries()
            if summaries:
                parts.append("")
                parts.append("[이전 단계에서 발견한 것]")
                for step_num in sorted(summaries.keys()):
                    info = summaries[step_num]
                    parts.append(f"- Step {step_num} ({info['step_name']}): {info['summary']}")

        if current_q:
            parts.extend([
                "",
                "[이번 응답 지시 - 반드시 준수]",
                "① 사용자의 말에 1~2문장으로 진심 어린 공감을 표현하세요.",
                "② 위 [집중 포인트]의 CBT 기법을 활용하여, 아래 질문을 자연스럽게 이어서 마지막에 물어보세요:",
                f"   {current_q}",
                "③ 이 질문 외에 다른 질문은 절대 추가하지 마세요.",
                "④ 감정에 함몰되지 말고, 사용자의 사고 패턴과 사건 분석에 집중하세요.",
            ])

        return "\n".join(parts)

    # STT 누적 텍스트 + 3모달 감정 융합 + 스텝별 시스템 프롬프트 + 히스토리 → LLM 응답 생성
    # 반환: {"llm_response": LLMResponse, "transition": str|None, "step_status": dict, "next_step_status": dict}
    async def generate_response(self, session_id: str) -> Optional[dict]:
        accumulated_text = " ".join(self._stt_text_buffer.get(session_id, []))
        if not accumulated_text:
            logger.warning(f"[Generate] {session_id}: 누적 텍스트 없음, 건너뜀")
            return None

        face_emotions = self._face_emotion_buffer.get(session_id, [])
        voice_emotions = self._voice_emotion_buffer.get(session_id, [])
        history_mgr = self.session.get_history_manager(session_id)
        history = history_mgr.get_recent_turns() if history_mgr else []
        loop = asyncio.get_event_loop()

        # ── 현재 단계 + 질문 정보 (StepManager) ──────────────
        step_mgr = self.session.get_step_manager(session_id)
        user_text_for_llm = accumulated_text  # LLM에 전달할 텍스트 (질문 힌트 포함 가능)
        if step_mgr:
            current_q = step_mgr.get_current_question()
            q_idx = step_mgr.current_question_idx
            total_q = len(step_mgr.get_questions())
            # 동적 system_prompt 조립 (분석 + 단계 + 이전 단계 요약 + 응답 지시)
            system_prompt = self._build_dynamic_system_prompt(step_mgr, current_q, history_mgr)
            if current_q:
                # 소형 LLM은 system보다 최근 user 메시지를 더 강하게 따름
                # → 질문을 user_text 끝에도 명시하여 이중 강조 (히스토리엔 원본만 저장)
                user_text_for_llm = (
                    f"{accumulated_text}\n\n"
                    f"[지금 반드시 이 질문으로 마무리하세요: {current_q}]"
                )
            logger.info(
                f"[Generate] {session_id}: Step {step_mgr.step_number} "
                f"'{step_mgr.current_step['name']}' "
                f"Q{q_idx + 1}/{total_q}: '{current_q}'"
            )
        else:
            system_prompt = None
            logger.warning(f"[Generate] {session_id}: StepManager 없음, 기본 프롬프트 사용")

        # ── 텍스트 감정 분석 (run_in_executor → 이벤트 루프 비점유) ──
        try:
            t0 = time.time()
            text_emo = await loop.run_in_executor(
                None, self.container.text_emotion.analyze, accumulated_text
            )
            logger.info(
                f"[TextEmo] {session_id}: {text_emo.primary_emotion} "
                f"{text_emo.probabilities} ({time.time() - t0:.2f}초)"
            )
        except Exception as e:
            logger.error(f"[TextEmo] {session_id}: 오류: {e}")
            text_emo = EmotionResult(primary_emotion="neutral", probabilities={"neutral": 1.0})

        # ── 감정 리스트 평균화 ────────────────────────────────
        face_emo = self._average_emotion(face_emotions)
        voice_emo = self._average_emotion(voice_emotions)

        # ── 3모달 감정 융합 ───────────────────────────────────
        try:
            t0 = time.time()
            fused_emo = self.container.fusion.fuse(text_emo, voice_emo, face_emo)
            logger.info(
                f"[Fusion] {session_id}: text={text_emo.primary_emotion} "
                f"voice={voice_emo.primary_emotion} face={face_emo.primary_emotion} "
                f"→ fused={fused_emo.primary_emotion} ({time.time() - t0:.2f}초)"
            )
        except Exception as e:
            logger.error(f"[Fusion] {session_id}: 오류: {e}")
            fused_emo = text_emo

        logger.info(
            f"[Generate] {session_id}: face {len(face_emotions)}건, "
            f"voice {len(voice_emotions)}건, history {len(history)//2}턴 → LLM 전달"
        )

        llm_context = LLMContext(
            user_text=user_text_for_llm,   # 질문 힌트 포함 버전 (LLM 전용)
            system_prompt=system_prompt,
            face_emotions=face_emotions,
            voice_emotions=voice_emotions,
            text_emotion=text_emo.primary_emotion,
            fused_emotion=fused_emo.primary_emotion,
            history=history,
        )

        # LLM 추론 (run_in_executor → GPU 연산 중에도 이벤트 루프 유지)
        t0 = time.time()
        response = await loop.run_in_executor(
            None, self.container.llm.generate_response, llm_context
        )
        logger.info(
            f"[LLM] {session_id}: '{response.reply_text}' "
            f"({time.time() - t0:.2f}초, 어댑터: {self.container.llm._active_adapter})"
        )
        if response.suggested_action:
            logger.info(f"[LLM] 추천 행동: '{response.suggested_action}'")

        # 히스토리에 이번 턴 추가 (HistoryManager에 저장)
        if history_mgr:
            history_mgr.add_user_message(accumulated_text)
            history_mgr.add_assistant_message(response.reply_text)

        # advance_question() 전 상태 스냅샷 (Qwen이 방금 다룬 질문 정보)
        pre_advance_status = step_mgr.get_status() if step_mgr else None

        # 질문 소화 → 다음 질문 or 다음 스텝 전환
        transition: Optional[str] = None
        if step_mgr:
            transition = step_mgr.advance_question()
            if transition == "step_changed":
                # 완료된 단계 요약 → HistoryManager에 저장 (다음 단계 system_prompt에 주입됨)
                if history_mgr and pre_advance_status:
                    completed_step_num = pre_advance_status["step"]
                    completed_step_name = pre_advance_status["title"]
                    # GPT 요약은 동기 호출이지만 짧음(~1-2초). 별도 thread로 실행
                    await loop.run_in_executor(
                        None,
                        history_mgr.on_step_transition,
                        completed_step_num,
                        completed_step_name,
                    )
                logger.info(
                    f"[StepMgr] {session_id}: → Step {step_mgr.step_number} "
                    f"'{step_mgr.current_step['name']}' 시작 "
                    f"(Q1: '{step_mgr.get_current_question()}')"
                )
            elif transition == "counseling_complete":
                # 마지막 단계 요약도 저장 (리포트용)
                if history_mgr and pre_advance_status:
                    await loop.run_in_executor(
                        None,
                        history_mgr.on_step_transition,
                        pre_advance_status["step"],
                        pre_advance_status["title"],
                    )
                logger.info(f"[StepMgr] {session_id}: 전체 상담 완료")

        # advance_question() 후 상태 스냅샷 (다음에 다룰 질문 정보)
        post_advance_status = step_mgr.get_status() if step_mgr else None

        # 다음 발화를 위해 감정/STT 버퍼 초기화
        self._stt_text_buffer[session_id] = []
        self._face_emotion_buffer[session_id] = []
        self._voice_emotion_buffer[session_id] = []

        return {
            "llm_response": response,
            "transition": transition,          # None | "step_changed" | "counseling_complete"
            "step_status": pre_advance_status, # Qwen이 방금 다룬 질문 기준 상태
            "next_step_status": post_advance_status,  # 다음에 다룰 질문 기준 상태
        }


# 전역 인스턴스 (session_manager에서 import해서 사용)
from app.core.container import ai_container
pipeline = CounselingPipeline(ai_container)
