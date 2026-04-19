# ─────────────────────────────────────────────
# detector/alert.py
# FastAPI 서버로 위험 신호와 heartbeat를 전송하는 모듈
#
# 역할:
#   1. 위험 감지 시 스냅샷 이미지와 함께 서버로 HTTP 전송
#   2. 30초마다 서버로 heartbeat(생존 신호) 전송
#
# main.py에서 감지 로직과 전송 로직을 분리하기 위해
# 별도 파일로 관리합니다.
# ─────────────────────────────────────────────

import os
import cv2          # 스냅샷 이미지를 JPEG로 인코딩
import requests     # FastAPI 서버로 HTTP 요청 전송
import numpy as np  # 이미지 데이터 타입 처리

from dotenv import load_dotenv  # .env에서 서버 주소 읽기

# ── 환경변수 로드 ──────────────────────────
# .env 파일에서 SERVER_URL 읽기
# 기본값: http://localhost:8000 (로컬 개발 환경)
load_dotenv()
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")


def send_alert(level: int, iou: float, frame: np.ndarray = None):
    """
    서버로 위험 신화와 스냅샷 이미지를 전송

    입력:
        level:  위험 단계 (1: 경고, 2: 위험, 3: 긴급)
        iou:    피복 중첩률 (0.0 ~ 1.0)
        frame:  감지 순간의 이미지 프레임 (없으면 이미지 없이 전송)

    동작:
        HTTP POST /alert 엔드포인트로 multipart/form-data 전송
        이미지가 있으면 JPEG로 인코딩해서 함께 전송
    """
    try:
        # 기본 데이터 (위험 단계, 중첩률)
        # Form 데이터로 전송 - 이미지와 함께 보낼 수 있는 포맷
        data = {
            "level": str(level), #Form은 문자열로 전송, 서버에서 int로 변환
            "iou": str(round(iou, 4)), # 소수점 4자리로 반올림
        }

        files = {}

        # 스냅샷 이미지가 있으면 JPEG로 인코딩해서 함께 전송

        if frame is not None:
            # cv2.imencode: numpy 배열(이미지)을 JPEG 바이트로 변환
            # [1]은 인코딩 성공 여부([0])를 제외한 실제 데이터
            _, jpge = cv2.imencode(".jpg", frame)

            # tobytes(): numpy 배열을 bytes로 변환
            # 서버의 UploadFile이 bytes 형식을 요구하기 때문
            files["snapshot"] = ("snapshot.jpg", jpge.tobytes(), "image/jpeg")

        # 서버로 POST 요청 전송
        # timeout=5: 5초안에 응답 없으면 포기 (감지 loop가 멈추지 않도록)
        response = requests.post(
            f"{SERVER_URL}/alert",
            data=data,
            files=files if files else None,
            timeout=5
        )

        # 응답 확인
        if response.status_code == 200:
            print(f"✅ 알림 전송 완료 — level: {level}, IoU: {iou:.2f}")
        else:
            print(f"⚠️ 알림 전송 실패 — 상태코드: {response.status_code}")

    except requests.exceptions.ConnectionError:
        # 서버가 꺼져있거나 네트워크 문제
        # 감지는 계속 되어야 하므로 에러를 무시하고 진행
        print(f"❌ 서버 연결 실패 — {SERVER_URL}에 접속할 수 없습니다.")

    except requests.exceptions.Timeout:
        # 서버 응답이 5초 초과
        # 마찬가지로 감지 루프를 멈추지 않기 위해 무시
        print("❌ 알림 전송 타임아웃 — 서버 응답이 너무 느립니다.")

    except Exception as e:
        # 그 외 예상치 못한 에러
        print(f"❌ 알림 전송 중 오류 발생: {e}")


def send_heartbeat():
    """
    서버로 생존 신호(heartbeat) 전송

    동작:
        HTTP POST /heartbeat 엔드포인트 호출
        서버는 이 신호를 받아 마지막 수신 시각을 갱신함
        60초 이상 신호가 없으면 서버가 기기 오프라인으로 판단

    에러처리:
        heartbeat 실패는 조용히 넘어감
        실패 로그만 출력하고 감지 루프를 방해하지 않음
    """

    try:
        response = requests.post(
            f"{SERVER_URL}/heartbeat",
            timeout=3   # heartbeat는 3초 타임아웃 (alert보다 짧게)
        )

        if response.status_code == 200:
            print("💓 Heartbeat 전송 완료")
        else:
            print(f"⚠️ Heartbeat 전송 실패 — 상태코드: {response.status_code}")

    except Exception:
        # heartbeat 실패는 조용히 넘어감
        # 알림과 달리 heartbeat 실패는 치명적이지 않음
        print("❌ Heartbeat 전송 실패 — 서버에 연결할 수 없습니다.")