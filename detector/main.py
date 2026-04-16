# ─────────────────────────────────────────────
# detector/main.py
# 감지 메인 — 웹캠 영상에서 얼굴과 피복을 감지해 위험 판단
#
# 역할:
#   1. 웹캠에서 실시간 영상 읽기
#   2. MediaPipe로 아기 얼굴 랜드마크 감지
#   3. YOLO로 이불/베개 등 피복 객체 감지
#   4. 피복이 얼굴 방향으로 접근 중인지 궤적 분석
#   5. 위험 수준 판단 후 서버로 알림 전송
#   6. 30초마다 서버로 heartbeat 전송 (생존 신호)
# ─────────────────────────────────────────────

import cv2          # 웹캠 영상 읽기 및 이미지 처리
import numpy as np  # 좌표 계산 (거리, 벡터)
import time         # heartbeat 타이머
import threading    # heartbeat를 별도 스레드로 실행
from collections import deque # 피복 이동 궤적 저장 (최근 N프레임)

# MediaPipe: 얼굴 랜드마크 468점 실시간 감지
import mediapipe as mp

# YOLOv8: 이불/배개 등 피복 객체 감지
from ultralytics import YOLO

# 서버로 위험 신호 전송하는 모듈 (alert.py에서 정의)
from alert import send_alert, send_heartbeat


# ── 설정값 ─────────────────────────────────
# 위험 단계별 임계값 (픽셀 단위)
CAUTION_DISTANCE = 150      # 경고: 피복이 얼굴로부터 150px 이내 접근
DANGER_DISTANCE = 60        # 위험: 피복이 얼굴로부터 60px 이내 접근
EMERGENCY_IOU = 0.3         # 긴급: 피복이 얼굴 bbox가 30% 이상 중첩

# 피복 이동 궤적 분석용 프레임 수
# 최근 10프레임의 피복 위치를 저장해 이동 방향 계산
TRAJECTORY_FRAMES = 10

# heartbeat 전송 주기 (초)
HEARTBEAT_INTERVAL = 30

# ── 모델 초기화 ──────────────────────────
# YOLOv8n: 가장 가벼운 nano 모델
# CPU에서도 실시간(10~15fps) 동작 가능
# 첫 실행 시 모델 파일(yolov8n.pt) 자동 다운로드
model = YOLO("yolo8n.pt")

# MediaPipe FaceMesh 초기화
# min_detection_confidence: 0.5 미만 신뢰도면 얼굴로 인식 안 함
mp_face_mesh = mp.solutions.face_mesh
face_mesh= mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 피복 객체 이동 궤적 저장
# deque: 최대 TRAJECTORY_FRAMES개만 유지, 오래된 것은 자동 삭제
cover_history = deque(maxlen=TRAJECTORY_FRAMES)


def get_face_bbox(frame):
    """
    MediaPipe로 얼굴 랜드마크를 감지하고 bbox 반환

    입력: BGR 이미지 (OpenCV 기본 포맷)
    출력: (x, y, w, h) 튜플 또는 None (얼굴 미감지 시)

    MediaPipe는 RGB를 요구하므로 BGR->RGB 변환 필요
    """

    #OpenCV는 BGR, MediaPipe는 RGB 사용 -> 변환 필요
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None # 얼굴 미감지
    
    h, w = frame.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    # 랜드마크 468개의 x, y 좌표로 얼굴 bbox 계산
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def get_cover_bboxes(frame):
    """
    YOLO로 이불/배개 등 피복 객체를 감지하고 bbox 목록 반환

    입력: BGR 이미지
    출력: [(x, y, w, h), ...] 리스트 (감지된 피복 객체들)

    COCO 데이터셋 기준 피복 관련 클래스:
    - 57: handbag → 이불/베개 근사값으로 활용
    - 58: tie
    - 65: bed
    - 67: cell phone
    실제 서비스에서는 신생아 전용 데이터셋으로 파인튜닝 필요
    """

    # verbose=False: 로그 출력 억제
    results = model(frame, verbose=False)
    bboxes = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        # 임시: 모든 객체를 피복으로 감지 (추후 클래스 필터링 필요)
        # TODO: 파인튜닝 후 특정 클래스만 필터링
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return bboxes


def calculate_iou(bbox1, bbox2):
    """
    두 bbox의 IoU(교집합/합집합 비율) 계산

    입력: (x, y, w, h) 형식의 두 bbox
    출력: 0.0 ~ 1.0 사이의 IoU 값
          0.0 = 전혀 겹치지 않음
          1.0 = 완전히 겹침

    질식 위험 판단의 핵심 지표
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 교집합 영역 계산
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy

    if intersection == 0:
        return 0.0
    
    # 합집합 = 두 영역으 합 - 교집합
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def calculate_distance(bbox1, bbox2):
    """
    두 bbox 중심점 간의 픽셀 거리 계산

    입력: (x, y, w, h) 형식의 두 bbox
    출력: 픽셀 단위 거리 (float)

    피복이 얼굴에 얼마나 가까이 있는지 판단
    """
    cx1 = bbox1[0] + bbox1[2] / 2
    cy1 = bbox1[1] + bbox1[3] / 2
    cx2 = bbox2[0] + bbox2[2] / 2
    cy2 = bbox2[1] + bbox2[3] / 2

    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def is_approaching_face(face_bbox, cover_history):
    """
    피복이 얼굴 방향으로 접근 중인지 분석

    입력:
        face_bbox: 현재 얼굴 bbox
        cover_history: 최근 N프레입의 피복 중심점 좌표 목록

    출력: True = 접근 중 / False = 멀어지거나 정지

    원리: 
        이전 프레임 대비 이동 벡터(velocity)와
        피복->얼굴 방향 벡터의 내적이 양수면 접근 중
    """

    # 궤적이 너무 짧으면 판단 불가
    if len(cover_history) < 3:
        return False  
    
    face_center = np.array([
        face_bbox[0] + face_bbox[2] / 2,
        face_bbox[1] + face_bbox[3] / 2,
    ])

    # 최근 이동 백터 (현재 위치 - 3프레임 전 위치)
    current = np.array(cover_history[-1])
    prev = np.array(cover_history[-3])
    velocity = current - prev

    # 피복에서 얼굴 방향 벡터
    to_face = face_center - current
    
    # 내적 > 0 이면 같은 방향 = 얼굴 방향으로 접근 중
    return np.dot(velocity, to_face) > 0


def analyze_threat(face_bbox, cover_bboxes):
    """
    얼굴과 피복 정보를 종합해 위험 단계 판단

    입력:
        face_bbox: 얼굴 bbox
        cover_bboxes: 감지된 피복 bbox 목록

    출력: (위험단계, IoU값) 튜플
        0 = 안전
        1 = 경고 (피복 접근 중)
        2 = 위험 (피복 근접)
        3 = 긴급 (피복 중첩)
    """

    # 피복 미감지 = 안전
    if not cover_bboxes:
        return 0, 0.0 
    
    max_iou = 0.0
    min_distance = float('inf')

    for cover_bbox in cover_bboxes:
        iou = calculate_iou(face_bbox, cover_bbox)
        distance = calculate_distance(face_bbox, cover_bbox)

        # 피복 중심점을 궤적에 추가
        cover_cx = cover_bbox[0] + cover_bbox[2] / 2
        cover_cy = cover_bbox[1] + cover_bbox[3] / 2
        cover_history.append((cover_cx, cover_cy))

        max_iou = max(max_iou, iou)
        min_distance = min(min_distance, distance)

    approaching = is_approaching_face(face_bbox, cover_history)

    # 위험 단계 판단 (높은 단계부터 체크)
    if max_iou >= EMERGENCY_IOU:
        return 3, max_iou   # 긴급: 이미 덮임
    elif min_distance < DANGER_DISTANCE and approaching:
        return 2, max_iou   # 위험: 빠르게 접근 중
    elif min_distance < CAUTION_DISTANCE and approaching:
        return 1, max_iou   # 경고: 접근 중
    else:
        return 0, max_iou   # 안전
    

def draw_debug(frame, face_bbox, cover_bboxes, threat_level, iou):
    """
    개발 중 디버깅용 - 화면에 감지 결과를 시각적으로 표시
    
    실제 제품에서는 이 함수 제거 또는 비활성화 
    """

    # 위험 단계별 색상 (BGR 포맷)
    colors = {
        0: (0, 255, 0),    # 초록: 안전
        1: (0, 255, 255),  # 노랑: 경고
        2: (0, 165, 255),  # 주황: 위험
        3: (0, 0, 255),    # 빨강: 긴급
    }
    color = colors[threat_level]


    # 얼굴 bbox 표시
    if face_bbox:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "FACE", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


    # 피복 bbox 표시
    for bbox in cover_bboxes:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "COVER", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 위험 단계 텍스트 표시
    labels = {0: "SAFE", 1: "CAUTION", 2: "DANGER", 3: "EMERGENCY"}
    cv2.putText(frame, f"{labels[threat_level]} (IoU: {iou:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def heartbeat_loop():
    """
    30초마다 서버로 생존 신호 전송
    별도 스레드에서 실행되어 감지 루프를 방해하지 않음
    """
    while True:
        send_heartbeat()
        time.sleep(HEARTBEAT_INTERVAL)


def main():
    """
    메인 감지 루프
    웹캠에서 영상을 읽어 매 프레임마다 감지 실행
    """

    # 웹캠 초기화 (0 = 기본 웹캠)
    # RPi5로 전호나 시 이 줄만 변경하면 됨
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return
    
    print("✅ 감지 시작 — 종료하려면 'q' 키를 누르세요.")

    # heartbeat를 별도 스레드로 실행
    # daemon=True: 메인 프로그램 종료 시 같이 종료
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    # 알림 중복 발송 방지용 타이머
    # 같은 위험 상황에서 3초마다 한 번만 알림 발송
    last_alert_time = 0
    ALERT_COOLDOWN = 3 # 초

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break

        # 얼굴 감지
        face_bbox = get_face_bbox(frame)

        if face_bbox is None:
            # 얼굴이 감지되지 않으면 다음 프레임으로
            cv2.putText(frame, "얼굴 미감지", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 (128, 128, 128), 2)
            cv2.imshow("Baby Monitor", frame)
        else:
            # 피복 감지
            cover_bboxes = get_cover_bboxes(frame)

            # 위험 단계 판단
            threat_level, iou = analyze_threat(face_bbox, cover_bboxes)

            # 디버그 화면 표시
            frame = draw_debug(frame, face_bbox, cover_bboxes, threat_level, iou)
            cv2.imshow("Baby Monitor", frame)

            # 위험 감지 시 알림 발송 (쿨다운 적용)
            if threat_level > 0:
                now = time.time()
                if now - last_alert_time > ALERT_COOLDOWN:
                    print(f"🚨 위험 감지 — level: {threat_level}, IoU: {iou:.2f}")
                    send_alert(level=threat_level, iou=iou, frame=frame)
                    last_alert_time = now

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    print("🛑 감지 종료")


if __name__ == "__main__":
    main()