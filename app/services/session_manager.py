import asyncio
import json
import base64
import logging
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict

from app.schemas import InputTest, ServerResponse
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)

# 접속자를 관리하고 데이터를 전달하는 역할, 비동기 처리

class ConnectionManager:
    def __init__(self):
        # 활성화된 상담 세션들을 저장하는 장부
        self.active_connections: Dict[str, WebSocket] = {}
        # 세션별 수신 카운터 (로그 확인용)
        self._audio_counts: Dict[str, int] = {}
        self._video_counts: Dict[str, int] = {}

    # [초기 상담 생성] 웹소캣 연결 수락
    async def connect(self, websocket: WebSocket, ticket_id: str):
        await websocket.accept()
        self.active_connections[ticket_id] = websocket
        pipeline.init_session(ticket_id)
        await pipeline.start_transcription_worker(ticket_id)  # 증분 STT 워커 (현재 미사용, 향후 전환 대비)
        self._audio_counts[ticket_id] = 0
        self._video_counts[ticket_id] = 0
        # 연결 성공 로깅으로 바꿈
        logger.info(f"--- [Session] {ticket_id} 연결 (현재 접속자: {len(self.active_connections)}명)")

        # 연결 성공 메시지 전송
        await self.send_personal_message(
            ServerResponse(status="connected", message="상담실에 입장하였습니다.").model_dump(),
            ticket_id
        )

    # 연결 해제 처리
    async def disconnect(self, ticket_id: str):
        if ticket_id in self.active_connections:
            del self.active_connections[ticket_id]
            pipeline.cleanup_session(ticket_id)
            self._audio_counts.pop(ticket_id, None)
            self._video_counts.pop(ticket_id, None)
            logger.info(f"[Session] {ticket_id} 연결 해제")

    # JSON 변환 메시지 전송을 위한 함수 (오류나 예외 상황에 의한 연결 끊김은 조용히 무시)
    async def send_personal_message(self, message: dict, ticket_id: str):
        if ticket_id not in self.active_connections:
            return
        try:
            ws = self.active_connections[ticket_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            # WebSocketDisconnect, ClientDisconnected 등 모두 조용히 처리
            await self.disconnect(ticket_id)

    # VAD 누적 음성 일괄 STT → 음성 감정(백그라운드) + LLM 처리 → 완료 시 클라이언트에 결과 전송
    async def _process_speech_end(self, ticket_id: str):
        stt_result = await pipeline.on_speech_end(ticket_id)
        if not stt_result:
            return

        # STT 결과 전송
        await self.send_personal_message(
            {"status": "stt_done", "text": stt_result.text},
            ticket_id
        )

        # LLM 응답 생성 및 전송
        llm_response = await pipeline.generate_response(ticket_id)
        if llm_response:
            await self.send_personal_message(
                {"status": "response", "message": llm_response.reply_text},
                ticket_id
            )

    # [데이터 처리 파이프라인 1]텍스트 프레임 처리
    async def process_text_data(self, ticket_id: str, raw_text: str):
        try:
            data_dict = json.loads(raw_text) # 데이터 Dictionary 형태로 변환
            input_obj = InputTest(**data_dict) # 데이터 규격 검사 틀

            # [초기 상담 설정 처리]
            if input_obj.type == "setup":
                d = input_obj.data
                pipeline.setup_counseling(
                    ticket_id,
                    topic=d["topic"],
                    mood=d["mood"],
                    content=d["content"],
                ) 
                initial_response = pipeline.generate_initial_questions(ticket_id) # 초기 질문 생성
                if initial_response: # 답변을 만든 경우에만 작동
                    await self.send_personal_message( # 답변 전송
                        {"status": "initial_questions", "message": initial_response.reply_text},
                        ticket_id
                    )

            # [발화 신호 처리]
            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH": # 발화 종료
                    a = self._audio_counts.get(ticket_id, 0)
                    v = self._video_counts.get(ticket_id, 0)
                    buf_size = len(pipeline.audio._audio_buffers.get(ticket_id, b""))
                    logger.info(
                        f"[Control] {ticket_id} 발화 종료 -> "
                        f"오디오 {a}청크 / VAD 누적 {buf_size/1024:.1f}kb / 영상 {v}프레임 수신 확인"
                    )
                    # 다음 턴을 위해 테이블 초기화
                    self._audio_counts[ticket_id] = 0
                    self._video_counts[ticket_id] = 0
                    await self.send_personal_message( # 프론트 화면에 로딩 스피너 생성
                        {"status": "processing", "message": "답변 생성 중..."},
                        ticket_id
                    )
                    # 비동기 처리, 답변 생성
                    asyncio.create_task(self._process_speech_end(ticket_id))

                elif input_obj.data == "END_OF_SESSION": # 세션 종료 신호
                    logger.info(f"[Session] {ticket_id} 세션 종료 신호 수신")
                    await self.disconnect(ticket_id)
            
            else:
                logger.warning(f"[Session] 알 수 없는 텍스트 타입: {input_obj.type}")
        
        except json.JSONDecodeError:
            logger.error(f"[Session] JSON 파싱 실패: {raw_text[:100]}")
        except Exception as e:
            logger.error(f"[Session] 텍스트 처리 중 오류: {e}")
    
    # [데이터 처리 파이프라인 2] 바이너리 프레임 전용 (순수 오디오, 비디오 데이터 입력으로 전환)
    async def process_binary_data(self, ticket_id: str, raw_bytes: bytes):
        try:
            # 프론트엔드가 붙인 명찰 확인
            header = raw_bytes[0] # 바이너리 종류
            payload = raw_bytes[1:] # 데이터 본체

            # 음성 바이너리 처리
            if header == 1:
                pipeline.append_audio_chunk(ticket_id, payload)

                self._audio_counts[ticket_id] = self._audio_counts.get(ticket_id, 0) + 1
                if self._audio_counts[ticket_id] % 50 == 0:
                    buf_size = len(pipeline.audio._audio_buffers.get(ticket_id, b""))
                    logger.info(f"[Audio] {ticket_id}: {self._audio_counts[ticket_id]}청크 수신 / VAD 버퍼 {buf_size//1024}kb 누적")

            # 영상 바이너리 처리
            elif header == 2:
                pipeline.process_face_frame(ticket_id, payload)
                self._video_counts[ticket_id] = self._video_counts.get(ticket_id, 0) + 1

            else:
                logger.warning(f"[Session] 알 수 없는 바이너리 헤더: {header}")

        except Exception as e:
            logger.error(f"[Session] 바이너리 처리 중 오류: {e}")

manager = ConnectionManager()
