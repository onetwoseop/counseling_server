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
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        pipeline.init_session(client_id)
        await pipeline.start_transcription_worker(client_id)
        self._audio_counts[client_id] = 0
        self._video_counts[client_id] = 0
        # 연결 성공 로깅으로 바꿈
        logger.info(f"--- [Session] {client_id} 연결 (현재 접속자: {len(self.active_connections)}명)")

        # 연결 성공 메시지 전송
        await self.send_personal_message(
            ServerResponse(status="connected", message="상담실에 입장하였습니다.").model_dump(),
            client_id
        )

    # 연결 해제 처리
    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            pipeline.cleanup_session(client_id)
            self._audio_counts.pop(client_id, None)
            self._video_counts.pop(client_id, None)
            logger.info(f"[Session] {client_id} 연결 해제")

    # JSON 변환 메시지 전송을 위한 함수 (오류나 예외 상황에 의한 연결 끊김은 조용히 무시)
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id not in self.active_connections:
            return
        try:
            ws = self.active_connections[client_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            # WebSocketDisconnect, ClientDisconnected 등 모두 조용히 처리
            await self.disconnect(client_id)

    # STT + LLM 백그라운드 처리 → 완료 시 클라이언트에 결과 전송
    async def _process_speech_end(self, client_id: str):
        stt_result = await pipeline.on_speech_end(client_id)
        if not stt_result:
            return

        # STT 결과 전송
        await self.send_personal_message(
            {"status": "stt_done", "text": stt_result.text},
            client_id
        )

        # LLM 응답 생성 및 전송
        llm_response = await pipeline.generate_response(client_id)
        if llm_response:
            await self.send_personal_message(
                {"status": "response", "message": llm_response.reply_text},
                client_id
            )

    # [데이터 처리 파이프라인] 들어온 데이터 분류 및 처리
    async def process_data(self, client_id: str, raw_data: str):
        try:
            # JSON 데이터 파싱
            data_dict = json.loads(raw_data)
            input_obj = InputTest(**data_dict)

            # 타입별 처리 분기

            # [초기 상담 설정 처리]
            if input_obj.type == "setup":
                d = input_obj.data
                pipeline.setup_counseling(
                    client_id,
                    topic=d["topic"],
                    mood=d["mood"],
                    content=d["content"],
                )
                initial_response = pipeline.generate_initial_questions(client_id)
                if initial_response:
                    await self.send_personal_message(
                        {"status": "initial_questions", "message": initial_response.reply_text},
                        client_id
                    )

            # [음성 데이터 처리]
            # 브라우저 MediaRecorder는 webm/opus를 전송 → raw 버퍼에 누적
            # END_OF_SPEECH 수신 시 ffmpeg으로 PCM 변환 후 배치 STT 처리
            elif input_obj.type == "audio":
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    raw_bytes = base64.b64decode(base64_data)
                    pipeline.append_raw_audio_chunk(client_id, raw_bytes)
                    asyncio.create_task(pipeline.transcribe_audio_chunk(client_id, raw_bytes))
                    self._audio_counts[client_id] = self._audio_counts.get(client_id, 0) + 1
                    count = self._audio_counts[client_id]
                    # 50청크마다 수신 현황 로그 (100ms×50 = 5초마다)
                    if count % 50 == 0:
                        buf_size = len(pipeline._raw_audio_buffer.get(client_id, b""))
                        logger.info(f"[Audio] {client_id}: {count}청크 수신 / 버퍼 {buf_size//1024}KB 누적")
                except Exception as e:
                    logger.error(f"[Error] 오디오 처리 실패: {e}")

            # [이미지 데이터 처리]
            elif input_obj.type == "video":
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    pipeline.process_face_frame(client_id, image_bytes)
                    self._video_counts[client_id] = self._video_counts.get(client_id, 0) + 1
                except Exception as e:
                    logger.error(f"[Error] 이미지 변환 실패: {e}")

            # [발화 신호 처리] - 제어 신호에만 ACK 응답
            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH":
                    a = self._audio_counts.get(client_id, 0)
                    v = self._video_counts.get(client_id, 0)
                    buf_size = len(pipeline._raw_audio_buffer.get(client_id, b""))
                    logger.info(
                        f"[Control] {client_id} 발화 종료 → "
                        f"오디오 {a}청크({buf_size//1024}KB) / 영상 {v}프레임 수신 확인"
                    )
                    # 다음 발화를 위해 카운터 초기화
                    self._audio_counts[client_id] = 0
                    self._video_counts[client_id] = 0
                    # 즉시 응답 후 STT는 백그라운드에서 처리 (WebSocket keepalive 유지)
                    await self.send_personal_message(
                        {"status": "processing", "message": "답변 생성 중..."},
                        client_id
                    )
                    # 말하기 종료 및 STT, LLM 로직 추가
                    asyncio.create_task(self._process_speech_end(client_id))

                elif input_obj.data == "END_OF_SESSION":
                    logger.info(f"[Session] {client_id} 세션 종료 신호 수신")
                    await self.disconnect(client_id)

            else:
                logger.warning(f"[Session] 알 수 없는 타입: {input_obj.type}")

            # audio/video는 fire-and-forget (ACK 없음), control 신호만 ACK

            


        except json.JSONDecodeError:
            logger.error(f"[Session] JSON 파싱 실패: {raw_data[:100]}")
        except Exception as e:
            logger.error(f"[Session] 처리 중 오류: {e}")


manager = ConnectionManager()
