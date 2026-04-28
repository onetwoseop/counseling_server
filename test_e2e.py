"""
E2E 상담 파이프라인 테스트
- starlette TestClient로 in-process 실행 → server_pipeline 객체 직접 공유 가능
- test_audio.raw  : float32 PCM 16kHz mono
- input_video.mp4 : 테스트 영상

실행:
    python test_e2e.py
"""
import time
import numpy as np
import cv2

from starlette.testclient import TestClient
from app.main import app
from app.services.pipeline import pipeline as server_pipeline

AUDIO_FILE  = "test_audio.raw"
VIDEO_FILE  = "input_video.mp4"
SESSION_ID  = "test-001"
CHUNK_SEC   = 2
SAMPLE_RATE = 16000


def run_test():
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/counseling/{SESSION_ID}") as ws:

            # ── 1. 연결 확인 ──────────────────────────────────────
            msg = ws.receive_json()
            assert msg["status"] == "connected", f"연결 실패: {msg}"
            print(f"[1] 연결 성공: {msg['message']}")

            # ── 2. 초기 상담 설정 전송 ────────────────────────────
            ws.send_json({
                "type": "setup",
                "data": {"topic": "가정사", "mood": "우울", "content": "엄마와 사이 안좋음"}
            })
            print("[2] setup 전송 (GPT 플랜 생성 + 첫 발화 대기 중...)")

            # ── 3. 플랜 생성 + 첫 상담사 발화 확인 ────────────────
            msg = ws.receive_json()
            assert msg["status"] == "counseling_ready", f"예상 counseling_ready, 실제: {msg}"
            print(f"[3] 첫 상담사 발화: \"{msg['message']}\"")
            print(f"    플랜: {[s['title'] for s in msg.get('plan', [])]}")
            print(f"    단계 상태: {msg.get('step_status', {})}")

            # ── 4. 오디오 청크 전송 (0x01 헤더) ──────────────────
            raw_audio = np.fromfile(AUDIO_FILE, dtype=np.float32)
            samples_per_chunk = SAMPLE_RATE * CHUNK_SEC
            total_chunks = len(raw_audio) // samples_per_chunk

            print(f"\n[4] 오디오 전송 시작 — 총 {total_chunks}청크 (2초 간격)")
            for i in range(total_chunks):
                seg = raw_audio[i * samples_per_chunk : (i + 1) * samples_per_chunk]
                ws.send_bytes(bytes([0x01]) + seg.tobytes())
                print(f"  [Audio] {i+1}/{total_chunks}  ({len(seg.tobytes())}B)")
                time.sleep(CHUNK_SEC)

            remainder = raw_audio[total_chunks * samples_per_chunk:]
            if len(remainder) > 0:
                ws.send_bytes(bytes([0x01]) + remainder.tobytes())
                print(f"  [Audio] 잔여 청크 ({len(remainder.tobytes())}B)")

            # ── 4b. 영상 프레임 전송 (0x02 헤더) ─────────────────
            cap = cv2.VideoCapture(VIDEO_FILE)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = int(fps * CHUNK_SEC)
            frame_idx, sent_frames = 0, 0

            print(f"\n[4b] 영상 프레임 전송 시작 (fps={fps:.1f}, {frame_interval}프레임마다)")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        ws.send_bytes(bytes([0x02]) + buf.tobytes())
                        sent_frames += 1
                        print(f"  [Video] {sent_frames}번째 프레임 전송 (frame#{frame_idx})")
                frame_idx += 1
            cap.release()
            print(f"  [Video] 전송 완료 — 총 {sent_frames}프레임")

            # ── 5. END_OF_SPEECH 전송 ─────────────────────────────
            ws.send_json({"type": "control", "data": "END_OF_SPEECH"})
            print("\n[5] END_OF_SPEECH 전송")

            # ── 6. stt_done 대기 ──────────────────────────────────
            print("[6] stt_done 대기 중...")
            while True:
                msg = ws.receive_json()
                if msg["status"] == "processing":
                    print(f"  → {msg['message']}")
                elif msg["status"] == "stt_done":
                    print(f"[6] STT 완료: \"{msg['text']}\"")
                    break

            # ── 7. 파이프라인 감정 버퍼 직접 검사 ────────────────
            face_buf  = server_pipeline._face_emotion_buffer.get(SESSION_ID, [])
            voice_buf = server_pipeline._voice_emotion_buffer.get(SESSION_ID, [])

            print(f"\n[7] ── 감정 버퍼 검사 ──")
            print(f"  얼굴 감정 버퍼: {len(face_buf)}건")
            for i, e in enumerate(face_buf):
                print(f"    [{i}] primary={e.primary_emotion:10s}  {e.probabilities}")
            print(f"  음성 감정 버퍼: {len(voice_buf)}건")
            for i, e in enumerate(voice_buf):
                print(f"    [{i}] primary={e.primary_emotion:10s}  {e.probabilities}")

            if len(face_buf) == 0:
                print("  !! 얼굴 감정 버퍼 비어있음 — 영상 프레임 전송 or DeepFace 로드 확인 필요")
            if len(voice_buf) == 0:
                print("  !! 음성 감정 버퍼 비어있음 — END_OF_SPEECH 이후 채워짐")

            # ── 8. LLM 응답 확인 (턴 1) ───────────────────────────
            msg = ws.receive_json()
            assert msg["status"] == "response", f"예상 response, 실제: {msg}"
            print(f"\n[8] 턴1 LLM 응답: \"{msg['message']}\"")
            print(f"    step_status: {msg.get('step_status', '없음')}")

            # StepManager 상태 직접 확인
            step_mgr = server_pipeline.session.get_step_manager(SESSION_ID)
            if step_mgr:
                print(f"    StepManager: Step {step_mgr.step_number}, Q{step_mgr.current_question_idx + 1}/{len(step_mgr.get_questions())}")
                print(f"    다음 질문: '{step_mgr.get_current_question()}'")

            # ── 9. 멀티턴 반복 (최대 3턴 추가) ───────────────────
            MULTITURN_REPLIES = [
                "요즘 너무 힘들어요. 엄마랑 말을 안 하게 됐어요.",
                "저도 잘 모르겠는데, 사소한 말다툼이 커진 것 같아요.",
                "맞아요, 예전에도 비슷한 일이 있었던 것 같아요.",
            ]

            raw_audio = np.fromfile(AUDIO_FILE, dtype=np.float32)
            samples_per_chunk = SAMPLE_RATE * CHUNK_SEC

            for turn_idx, reply_text in enumerate(MULTITURN_REPLIES, start=2):
                step_mgr = server_pipeline.session.get_step_manager(SESSION_ID)
                if step_mgr and step_mgr.is_complete:
                    print(f"\n[턴{turn_idx}] 모든 단계 완료 → 멀티턴 종료")
                    break

                print(f"\n[9-{turn_idx}] 턴{turn_idx} 시작 — 사용자 발화: \"{reply_text}\"")

                # 오디오 청크 전송
                for i in range(min(2, len(raw_audio) // samples_per_chunk)):
                    seg = raw_audio[i * samples_per_chunk : (i + 1) * samples_per_chunk]
                    ws.send_bytes(bytes([0x01]) + seg.tobytes())
                    time.sleep(0.5)

                # END_OF_SPEECH
                ws.send_json({"type": "control", "data": "END_OF_SPEECH"})
                print(f"  → END_OF_SPEECH 전송")

                # stt_done + response 대기
                while True:
                    msg = ws.receive_json()
                    if msg["status"] == "processing":
                        print(f"  → 처리 중...")
                    elif msg["status"] == "stt_done":
                        print(f"  → STT: \"{msg['text']}\"")
                    elif msg["status"] == "response":
                        print(f"  → LLM 응답: \"{msg['message']}\"")
                        print(f"     step_status: {msg.get('step_status', '없음')}")
                        step_mgr = server_pipeline.session.get_step_manager(SESSION_ID)
                        if step_mgr and not step_mgr.is_complete:
                            print(f"     StepManager: Step {step_mgr.step_number}, Q{step_mgr.current_question_idx + 1}/{len(step_mgr.get_questions())}")
                        break

            # ── 10. 세션 종료 ─────────────────────────────────────
            ws.send_json({"type": "control", "data": "END_OF_SESSION"})
            print("\n[10] END_OF_SESSION → 테스트 종료")
            print("\n✓ 멀티턴 전체 테스트 완료")


if __name__ == "__main__":
    run_test()
