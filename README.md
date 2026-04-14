# 👶 Baby Monitor — 신생아 질식 감지 앱

> 아이가 자는 동안 이불이 얼굴을 덮는 순간을 감지해 보호자에게 즉시 알림을 보내는 시스템

---

## 프로젝트 배경

신생아 질식 사고는 이불, 베개, 수건 등 평범한 물건이 원인이 되는 경우가 많습니다.
기존 베이비 모니터는 영상을 보여줄 뿐, 위험 상황을 자동으로 감지하지 못합니다.
이 프로젝트는 AI 영상 감지 기술을 활용해 **덮이기 전에** 보호자에게 알림을 보내는 것을 목표로 합니다.

---

## 시스템 구조

[Edge]   라즈베리파이 5 + NoIR 카메라
→ 로컬에서 영상 감지 (영상이 외부로 나가지 않음)

[Hub]    FastAPI 서버
→ 위험 신호 수신 후 FCM 푸시 알림 발송

[Client] Next.js + Capacitor 앱
→ 보호자 핸드폰에서 알림 수신 및 상황 확인

---

## 폴더 구조

baby-monitor/
├── detector/         # YOLO + MediaPipe 감지 서버 (RPi5에서 실행)
│   ├── main.py       # 웹캠 → 감지 → 위험 판단
│   ├── alert.py      # FastAPI 서버로 HTTP 전송
│   └── requirements.txt
├── server/           # FastAPI 알림 서버
│   ├── main.py       # 신호 수신 → FCM 알림 발송
│   └── requirements.txt
└── mobile/           # Next.js + Capacitor 앱
└── app/

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 감지 | YOLOv8 · MediaPipe · OpenCV |
| 서버 | FastAPI · Python |
| 앱 | Next.js · Capacitor · TypeScript |
| 알림 | Firebase FCM |
| 하드웨어 | Raspberry Pi 5 · NoIR 카메라 |

---

## 개발 로드맵

- [x] 프로젝트 초기 세팅
- [ ] 1단계 (MVP): 위험 감지 → 스냅샷 → 푸시 알림
- [ ] 2단계: 슬라이드쇼 라이브 뷰 (1~2fps)
- [ ] 3단계: WebRTC 실시간 스트리밍

---

## 개발 환경 세팅

### detector

```bash
cd detector
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### server

```bash
cd server
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### mobile

```bash
cd mobile
npm installㅂ
npm run dev
```

---

## 관련 블로그

- [곧 아빠가 되는 개발자가 직접 만드는 신생아 질식 감지 앱](https://velog.io)
