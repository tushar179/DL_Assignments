import cv2
import torch
import time
from ultralytics import YOLO

# ===== SETTINGS =====
MODEL_NAME = "yolov8n.pt"  # nano model = fastest
VIDEO_SOURCE = 0           # 0 = webcam, or "video.mp4"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# =====================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_half = device == 'cuda'
print(f"[INFO] Device: {device} | FP16: {use_half}")

# Load YOLO model
model = YOLO(MODEL_NAME)
model.to(device)

cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=device,
        half=use_half,
        verbose=False
    )

    # Draw boxes on frame
    annotated_frame = results[0].plot()

    # FPS calculation
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
