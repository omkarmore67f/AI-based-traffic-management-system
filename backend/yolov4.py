import cv2 as cv
import time
from collections import deque
import numpy as np
from scipy.signal import find_peaks

# ================= GLOBAL YOLO LOAD (IMPORTANT FOR SPEED) =================
try:
    with open('classes.txt', 'r') as f:
        CLASS_NAMES = [c.strip().lower() for c in f.readlines()]
except:
    CLASS_NAMES = []

NET = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
NET.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
NET.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

MODEL = cv.dnn_DetectionModel(NET)
MODEL.setInputParams(size=(320, 320), scale=1/255, swapRB=True)
# ========================================================================


def detect_cars(video_file, show=False, debug=False):
    """FAST CPU traffic car counter (COMMON MODE)"""

    # ===== SPEED / ACCURACY BALANCE =====
    FRAME_SKIP = 6          # BIG speed gain (safe for traffic)
    CONF_THRESHOLD = 0.30
    NMS_THRESHOLD = 0.30
    WINDOW_SECONDS = 6
    # ==================================

    if not CLASS_NAMES:
        print("ERROR: classes.txt not loaded")
        return 0.0

    cap = cv.VideoCapture(video_file)
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

        # 🔥 REAL SPEED FIX — SKIP FRAMES
        if frame_counter % FRAME_SKIP != 0:
            continue

        try:
            classes, scores, boxes = MODEL.detect(
                frame, CONF_THRESHOLD, NMS_THRESHOLD
            )
        except:
            continue

        car_count = 0
        for classid in classes:
            try:
                cid = int(classid) if isinstance(classid, (int, np.integer)) else int(classid[0])
            except:
                continue

            if 0 <= cid < len(CLASS_NAMES) and CLASS_NAMES[cid] == "car":
                car_count += 1

        now = time.time()
        car_counts.append((now, car_count))

        # Short rolling window
        while car_counts and car_counts[0][0] < now - WINDOW_SECONDS:
            car_counts.popleft()

        values = [c for _, c in car_counts]
        if values:
            peaks, _ = find_peaks(values)
            mean_peak_value = float(
                np.mean([values[i] for i in peaks]) if len(peaks) else max(values)
            )

        # Optional display
        if show:
            cv.putText(frame, f'Cars: {car_count}', (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv.destroyAllWindows()

    if debug:
        print(f"Processed {frame_counter} frames in {time.time()-start_time:.2f}s")
        print("Final mean peak:", mean_peak_value)

    return float(mean_peak_value)