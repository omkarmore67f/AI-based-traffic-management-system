import time
from collections import deque
import numpy as np
import cv2
from scipy.signal import find_peaks
from ultralytics import YOLO

# Load model ONCE (important for speed)
MODEL = YOLO("yolov8n.pt")   # nano = best for CPU

# COCO class ids
VEHICLE_CLASS_IDS = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

def detect_cars(video_file, show=False, debug=False):
    FRAME_SKIP = 6
    CONF_THRESHOLD = 0.30
    WINDOW_SECONDS = 6

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video:", video_file)
        return 0.0

    frame_counter = 0
    car_counts = deque()
    mean_peak_value = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        # YOLOv8 inference
        results = MODEL(frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

        # 🔹 draw bounding boxes
        frame = results[0].plot()

        car_count = 0
        for r in results:
            if r.boxes is None:
                continue
            for cls in r.boxes.cls:
                if int(cls) == VEHICLE_CLASS_IDS["car"]:
                    car_count += 1

        now = time.time()
        car_counts.append((now, car_count))

        # Rolling window
        while car_counts and car_counts[0][0] < now - WINDOW_SECONDS:
            car_counts.popleft()

        values = [c for _, c in car_counts]
        if values:
            peaks, _ = find_peaks(values)
            mean_peak_value = float(
                np.mean([values[i] for i in peaks]) if len(peaks) else max(values)
            )

        if show:
            cv2.putText(frame, f'Cars: {car_count}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if debug:
        print(f"Processed {frame_counter} frames in {time.time()-start_time:.2f}s")
        print("Final mean peak:", mean_peak_value)

    return float(mean_peak_value)
