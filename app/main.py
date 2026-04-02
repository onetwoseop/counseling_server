import asyncio # 비동기 처리 라이브러리
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager # 수명 주기 관리
from app.core.container import ai_container # 모델 정보(전역 인스턴스) 가져오기
from app.services.session_manager import manager

# 수명 주기 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> AI 모델 로딩을 시작합니다.....")
    ai_container.load_models() # 모델들을 메모리에 올려둠
    yield # 서버 실행 시점
    print(">>> 서버 종료.")

# 앱 생성
app = FastAPI(title="AI 상담소 서버", lifespan=lifespan)

# 작동 확인
@app.get("/")
async def health_check():
    return JSONResponse(
        content={"status":"ok", "message":"상담 서버 정상 작동 중"},
        media_type="application/json; charset=utf-8" # 한글 깨져서 형식 지정
    )

# 웹소켓 엔드포인트
@app.websocket("/ws/counseling/{client_id}")
async def counseling_endpoint(websocket: WebSocket, client_id: str):
    try:
        # [초기 생성] 연결 요청 처리 (connect 안에서도 끊길 수 있으므로 try 안에 포함)
        await manager.connect(websocket, client_id)

        # [실시간 파이프라인] 무한 루프로 데이터 대기
        while True:
            data = await websocket.receive_text()
            await manager.process_data(client_id, data)

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"웹소켓 에러 발생: {e}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    # FastAPI를 실행하는 ASGI 서버 엔진