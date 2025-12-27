from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from yolov4 import detect_cars
from algo import optimize_traffic

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_files():
    print("\n====== UPLOAD REQUEST RECEIVED ======\n")

    # Accept both: videos[] or file
    files = request.files.getlist('videos')
    if len(files) == 0:
        single = request.files.get('file')
        if single:
            files = [single]

    print("Number of files received:", len(files))

    # Allow single-video testing
    if len(files) not in [1, 4]:
        return jsonify({'error': 'Upload 1 video for test or 4 videos for full system'}), 400

    os.makedirs("uploads", exist_ok=True)

    video_paths = []
    for i, file in enumerate(files):
        filename = f"video_{i}.mp4" if len(files) > 1 else "test_video.mp4"
        path = os.path.join("uploads", filename)
        file.save(path)
        print(f"Saved: {path}")
        video_paths.append(path)

    # RUN DETECTION
    num_cars_list = []
    for vid in video_paths:
        print("Running detect_cars on:", vid)
        try:
            count = detect_cars(vid, show=False, debug=True)
        except Exception as e:
            print("detect_cars FAILED:", str(e))
            return jsonify({'error': 'detect_cars failed', 'detail': str(e)}), 500

        print("Cars detected:", count)
        num_cars_list.append(float(count))

    print("Final car counts list:", num_cars_list)

    # Single-video test mode → return simple green signal
    if len(num_cars_list) == 1:
        val = num_cars_list[0]
        green = max(10, int(10 + 2 * val))   # simple logic
        print("Sending single-video result:", {"cars": val, "green_seconds": green})
        return jsonify({"cars": val, "green_seconds": green})

    # Full 4-video optimization
    try:
        result = optimize_traffic(num_cars_list)
        print("Optimization result:", result)
    except Exception as e:
        print("OPTIMIZATION FAILED:", str(e))
        return jsonify({'error': 'optimization failed', 'detail': str(e)}), 500

    # Ensure valid output for frontend
    if not isinstance(result, dict) or len(result) == 0:
        print("Invalid result from optimize_traffic:", result)
        result = {"error": "optimize_traffic returned invalid output"}

    print("Sending final JSON:", result)
    return jsonify(result)


if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)