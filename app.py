import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, request, jsonify
from flask_cors import CORS
from recognize import recognize, enroll
import datetime
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Warmup
print("⏳ Loading AI models...")
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', test_img)
recognize(buf.tobytes())
print("✅ Models ready\n")

@app.route("/recognize", methods=["POST"])
def api_recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_bytes = request.files["image"].read()
    result      = recognize(image_bytes)
    result["timestamp"] = datetime.datetime.now().isoformat()
    print(f"[{result['timestamp']}] {result['status']} | "
          f"{result.get('name', '—')} | {result.get('confidence', 0)}%")
    return jsonify(result), 200

@app.route("/enroll", methods=["POST"])
def api_enroll():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    if "name" not in request.form:
        return jsonify({"error": "No name provided"}), 400
    name        = request.form["name"].strip().lower().replace(" ", "_")
    image_bytes = request.files["image"].read()
    result      = enroll(name, image_bytes)
    if result["success"]:
        return jsonify(result), 200
    else:
        return jsonify(result), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "running",
        "model":    "FaceNet512",
        "detector": "opencv"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)