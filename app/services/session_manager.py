import asyncio
import json
import base64
import logging
from fastapi import WebSocket
from typing import Dict

from app.schemas import InputTest, ServerResponse
from app.services.pipeline import pipeline

logger = logging.getLogger(__name__)

# 접속자를 관리하고 데이터를 전달하는 역할, 비동기 처리

class ConnectionManager:
    def __init__(self):
        # 활성화된 상담 세션들을 저장하는 장부
        self.active_connections: Dict[str, WebSocket] = {}

    # [초기 상담 생성] 웹소캣 연결 수락
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        pipeline.init_session(client_id)
        await pipeline.start_transcription_worker(client_id)
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
            logger.info(f"[Session] {client_id} 연결 해제")

    # 특정 사용자에게 JSON 변환 메시지 전송
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            ws = self.active_connections[client_id]
            await ws.send_text(json.dumps(message, ensure_ascii=False)) # 파이썬 딕셔너리를 문자열로 변환

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

        # LLM 응답 생성 및 전송 (DummyLLM 포함, 실제 모델 연결 시 자동 반영)
        llm_response = pipeline.generate_response(client_id)
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
                    style=d.get("style")
                )
                initial_response = pipeline.generate_initial_questions(client_id)
                if initial_response:
                    await self.send_personal_message(
                        {"status": "initial_questions", "message": initial_response.reply_text},
                        client_id
                    )

            # [음성 데이터 처리]
            elif input_obj.type == "audio":
                task = None
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    speech_ended = pipeline.append_audio_chunk(client_id, base64.b64decode(base64_data))
                    # VAD가 침묵을 감지하면 백그라운드에서 STT+LLM 처리
                    if speech_ended:
                        # 백그라운드 stt 처리
                        task = asyncio.create_task(self._process_speech_end(client_id))
                except Exception as e:
                    logger.error(f"[Error] 오디오 처리 실패: {e}")
                    if task is not None:
                        task.cancel()

            # [이미지 데이터 처리]
            elif input_obj.type == "video":
                # 표정기반 감정추출 로직 추가 필요
                try:
                    base64_data = str(input_obj.data)
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    image_bytes = base64.b64decode(base64_data)
                    pipeline.process_face_frame(client_id, image_bytes)
                except Exception as e:
                    logger.error(f"[Error] 이미지 변환 실패: {e}")

            # [발화 신호 처리] - 제어 신호에만 ACK 응답
            elif input_obj.type == "control":
                if input_obj.data == "END_OF_SPEECH":
                    logger.info(f"[Control] {client_id}의 발화 종료, 처리 시작 ...")
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
