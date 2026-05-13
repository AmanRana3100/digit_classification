"""
app.py — Digit Classifier Inference API (Flask)

Endpoints:
    POST /predict      — JSON body: { "image": "<base64 PNG>" }
    GET  /health       — health check

Usage:
    1. Train first:   python train.py
    2. Run server:    python app.py
    3. Open browser:  http://localhost:5000
"""

import io
import os
import base64
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
# At the top of app.py, before loading the model:
if not os.path.exists(MODEL_PATH):
    print("Model not found — training now...")
    os.system("python train.py")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH", "digit_model.keras")
HOST        = os.environ.get("HOST", "0.0.0.0")
PORT        = int(os.environ.get("PORT", 5000))
DEBUG       = os.environ.get("DEBUG", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

# ─── Load Model ───────────────────────────────────────────────────────────────
log.info(f"Loading model from {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Run `python train.py` first to generate digit_model.keras"
    )
model = keras.models.load_model(MODEL_PATH)
log.info("Model loaded successfully.")


# ─── Image Preprocessing ──────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert any PIL image to a 28×28 grayscale array suitable for the model.
    Handles white-on-black (canvas) or black-on-white (scanned) inputs.
    """
    # Convert to grayscale
    img = img.convert("L")

    # Resize to 28×28 using high-quality downsampling
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # Auto-invert: MNIST digits are white on black.
    # If the image looks like black-on-white (mean > 127), invert it.
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Add batch + channel dims → (1, 28, 28, 1)
    return arr.reshape(1, 28, 28, 1)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "image": "<base64-encoded PNG/JPEG>" }
    Returns:      { "digit": int, "confidence": float, "probabilities": [float × 10] }
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' field in JSON body"}), 400

    try:
        # Decode base64 → PIL Image
        img_data = data["image"]
        # Strip data-URL prefix if present (e.g. "data:image/png;base64,...")
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))

        # Preprocess
        arr = preprocess_image(img)

        # Inference
        probs = model.predict(arr, verbose=0)[0]   # shape: (10,)
        digit      = int(np.argmax(probs))
        confidence = float(probs[digit])

        log.info(f"Prediction → digit={digit}  confidence={confidence:.3f}")

        return jsonify({
            "digit":         digit,
            "confidence":    round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in probs],
        })

    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Serve the web UI if it exists."""
    if os.path.exists("static/index.html"):
        return send_from_directory("static", "index.html")
    return jsonify({"message": "Digit Classifier API is running. POST /predict to classify."}), 200


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Starting server on {HOST}:{PORT}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
