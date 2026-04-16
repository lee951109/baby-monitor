# ─────────────────────────────────────────────
# server/main.py
# FastAPI 서버 — 감지 신호 수신 및 FCM 푸시 알림 발송
#
# 역할:
#   1. detector(RPi5/노트북)에서 보내는 위험 신호를 HTTP로 수신
#   2. 위험 수준에 따라 FCM으로 보호자 핸드폰에 푸시 알림 발송
#   3. 30초마다 오는 heartbeat 신호로 기기 연결 상태 확인
# ─────────────────────────────────────────────

import os
import time
from contextlib import asynccontextmanager

# FastAPI: Python 웹 서버 프레임워크
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

# Firebase Admin SDK: FCM 푸시 알림 방송
import firebase_admin
from firebase_admin import credentials, messaging

# .env 파일에서 환경변수 읽기
from dotenv import load_dotenv


# ── 환경변수 로드 ──────────────────────────
# .env 파일에서 FIREBASE_PROJECT_ID, FIREBASE_CREDENTIALS_PATH 읽기
load_dotenv();

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

# ── Firebase 초기화 ────────────────────────
# 서비스 계정 JSON 파일로 Firebase Admin SDK 인증
# 이 초기화는 서버 시작 시 딱 한 번만 실행됨
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)

# ── 기기 연결 상태 추적 ────────────────────
# detector에서 heartbeat 신호가 마지막으로 온 시각 저장
# 60초 이상 신호 없으면 기기 오프라인으로 판단
last_heartbeat = {"timestamp": time.time()}

# ── FCM 토큰 저장소 ────────────────────────
# 실제 서비스에서는 DB에 저장하지만
# MVP 단계에서는 메모리에 임시 저장
# 앱에서 로그인 시 토큰을 등록하는 방식으로 사용
fcm_tokens: list[str] = []


# ── FastAPI 앱 초기화 ──────────────────────
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행
    print("✅ Baby Monitor 서버 시작")
    print(f"✅ Firebase 프로젝트: {FIREBASE_PROJECT_ID}")
    yield
    # 서버 종료 시 실행
    print("🛑 Baby Monitor 서버 종료")

app = FastAPI(
    title="Baby Monitor API"
    , description="신생아 질식 감지 알림 서비스"
    , lifespan=lifespan
)

# CORS 설정
# Next.js 앱에서 이 서버로 요청할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발 중에는 전체 허용, 배포 시 특정 도메인으로 변경
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── 엔드포인트 ─────────────────────────────

@app.get("/")
async def root():
    """서버 상태 확인용 엔드포인트"""
    return {"status": "running", "project": FIREBASE_PROJECT_ID}


@app.post("/heartbeat")
async def heartbeat():
    """
    detector에서 30초마다 보내는 생존 신호 수신
    이 신호가 60초 이상 오지 않는다면 기기 오프라인으로 판단
    """
    last_heartbeat["timestamp"] = time.time()
    return {"status": "ok", "timestamp": last_heartbeat["timestamp"]}


@app.get("/device-status")
async def device_status():
    """
    앱에서 기기 연결 상태를 확인하는 엔드포인트
    마지막 heartbeat로부터 60초 이상 지났으면 오프라인
    """
    elapsed = time.time() - last_heartbeat["timestamp"]
    is_online = elapsed < 60

    return {
        "online": is_online,
        "last_heartbeat_seconds_ago": round(elapsed)
    }


@app.post("/register-token")
async def register_token(token: str = Form(...)):
    """
    앱에서 FCM 토큰을 등록하는 엔드포인트
    보호자가 앱을 설치하면 이 토큰을 서버에 등록해야 알림을 받을 수 있음
    """
    if token not in fcm_tokens:
        fcm_tokens.append(token)
        print(f"✅ FCM 토큰 등록: {token[:20]}...")

    return {"status": "registered", "total_tokens": len(fcm_tokens)}


@app.post("/alert")
async def receive_alert(
    level: int = Form(...),    # 위험 단계 (1: 경고, 2: 위험, 3:긴급) 
    iou: float = Form(...),    # 피복 중첩률 (0.0 ~ 1.0)
    snapshot: UploadFile = File(None) # 감지 순간 스냅샷 이미지 
):
    """
    detector에서 위험 감지 시 보내는 알림 수신 엔드포인트

    level 1 (경고): 피복이 얼굴 방향으로 접근 중
    level 2 (위험): 피복이 얼굴 근처까지 접근
    level 3 (긴급): 피복이 얼굴을 덮음
    """
    print(f"🚨 위험 신호 수신 — level: {level}, IoU: {iou:.2f}")

    # 위험 단계별 알림 메시지 설정
    alert_messages = {
        1: {
            "title": "⚠️ 주의: 피복 접근 감지",
            "body": f"이불/베개가 아기 얼굴 방향으로 접근 중입니다. (중첩률 {iou*100:.0f}%)"
        },
        2: {
            "title": "🔶 위험: 피복 근접 감지",
            "body": f"이불/베개가 아기 얼굴 근처에 있습니다. 즉시 확인하세요. (중첩률 {iou*100:.0f}%)"
        },
        3: {
            "title": "🚨 긴급: 얼굴 피복 감지",
            "body": f"아기 얼굴에 이불/베개가 덮였습니다. 즉시 확인하세요! (중첩률 {iou*100:.0f}%)"
        }
    }

    message_data = alert_messages.get(level, alert_messages[3])

    # 등록된 FCM 토큰이 없으면 알림 발송 불가
    if not fcm_tokens:
        print("⚠️ 등록된 FCM 토큰 없음 — 알림 발송 건너뜀")
        return {"status": "no_tokens", "level": level}
    
    # 등록된 모든 보호자에게 동시 알림 발송
    sent_count = 0
    for token in fcm_tokens:
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=message_data["title"],
                    body=message_data["body"],
                ),
                # Android 설정: 최고 우선순위로 잠금화면 표시
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        channel_id="baby_emergency",
                        priority="max",
                        visibility="public",
                    )
                ),
                # IOS 설정: 집중모드 무시
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            sound="default",
                            badge=1,
                        )
                    )
                ),
                token=token,
            )
            messaging.send(message)
            sent_count += 1
            print(f"✅ 알림 발송 완료: {token[:20]}...")
        
        except Exception as e:
            print(f"❌ 알림 발송 실패: {e}")

    return {
        "status": "sent",
        "level": level,
        "iou": iou,
        "sent_count": sent_count
    }