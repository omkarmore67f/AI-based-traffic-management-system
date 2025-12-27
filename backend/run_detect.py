import os
import sys

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(__file__))

from yolov8_detect import detect_cars


def main():
    uploads = os.path.join(os.path.dirname(__file__), "uploads")
    if not os.path.isdir(uploads):
        print("No uploads directory found at", uploads)
        return

    videos = sorted([f for f in os.listdir(uploads) if f.lower().endswith(".mp4")])
    if not videos:
        print("No mp4 videos found in", uploads)
        return

    results = {}
    for v in videos:
        path = os.path.join(uploads, v)
        print("Processing:", path)
        try:
            res = detect_cars(path, show=True, debug=True)
        except Exception as e:
            res = {"error": str(e)}
        print("Result:", res)
        results[v] = res

    print("SUMMARY_JSON")
    import json
    print(json.dumps(results))

if __name__ == "__main__":
    main()