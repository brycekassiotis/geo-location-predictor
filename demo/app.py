"""
The demo web app
COMMAND: python demo/app.py
WEBSITE: http://127.0.0.1:5000

Model is loaded once when this file starts, then every upload reuses it.
"""
import base64
import io
import math
import threading
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from predictor import Predictor, model_input_preview
from src.geo.metrics import geoscore, haversine_np

# Upload Limits
MAX_UPLOAD_MB = 20
MIN_SIDE_REJECT = 100
MIN_SIDE_WARN = 224
MAX_PIXELS = 100000000
NARROW_WARN = 0.6

app = Flask(__name__)  # Static files are served from demo/static/
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

predictor = Predictor()
predictor.predict(Image.new("RGB", (224, 224))) # Sample prediction to warm up the GPU and load the model into memory
predict_lock = threading.Lock() # Only one prediction at a time (since on gpu)


def error(message, status):
    return jsonify({"error": message}), status

@app.errorhandler(413)
def too_large(_):
    return error(f"That file is too big (the limit is {MAX_UPLOAD_MB} MB).", 413)

# GET requests to / return the static index.html page
@app.get("/")
def index():
    return app.send_static_file("index.html")

# GET requests to /api/info return the model info and validation metrics
@app.get("/api/info")
def info():
    m = predictor.meta["val_metrics"]
    val = {k: float(m[k]) for k in ("geoscore", "median_km", "mean_km") if k in m}
    return jsonify({"checkpoint": predictor.checkpoint_dir.name, "num_cells": predictor.meta["num_cells"],
                    "device": predictor.device.type, "val": val})

# POST requests to /api/score return the distance and GeoScore for each guess compared to the true location
@app.post("/api/score")
def score():
    """
    Compares model guesses with the true location the user typed in.
    Request JSON:  {"truth": [lat, lon], "points": [[lat, lon], ...]}
    Response JSON: {"distance_km": [...], "geoscore": [...]}, one entry per point
    """
    body = request.get_json(silent=True) or {}
    # Validate the input: truth (the true location) must be a lat/lon pair, points (the guesses) must be a list of lat/lon pairs, all finite and in range.
    try:
        truth = [float(v) for v in body["truth"]]
        points = [[float(v) for v in p] for p in body["points"]]
        valid = (len(truth) == 2 and 0 < len(points) <= 50 and all(len(p) == 2 for p in points))
        coords = truth + [v for p in points for v in p]
        valid = valid and all(math.isfinite(v) for v in coords)
        valid = valid and all(abs(v) <= (90 if i % 2 == 0 else 180) for i, v in enumerate(coords))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        return error("Enter the location as latitude, longitude (latitude -90 to 90, longitude -180 to 180).", 400)

    pts = np.asarray(points)
    dist = haversine_np(pts[:, 0], pts[:, 1], truth[0], truth[1])
    return jsonify({"distance_km": [float(d) for d in dist], "geoscore": [float(s) for s in geoscore(dist)]})

# POST requests to /predict return the model's guess for the uploaded image
@app.post("/predict")
def predict():
    # Check if the request has a file part
    f = request.files.get("image")
    if f is None or f.filename == "":
        return error("No file was uploaded.", 400)

    # Check if the file is a valid image and image is not too big
    try:
        img = Image.open(f.stream)
        w, h = img.size
        if w * h > MAX_PIXELS:
            return error(f"That image is too large ({w}x{h} pixels). Please upload a smaller one.", 413)
        img.load()
    except Exception:
        return error("Couldn't read that file as an image. Try a JPEG or PNG.", 400)

    # Check if the image is too small to be useful
    if min(w, h) < MIN_SIDE_REJECT:
        return error(f"That image is too small ({w}x{h}). The shorter side must be at least " f"{MIN_SIDE_REJECT} pixels.", 422)

    # Warn if the image is small or narrow, but still try to predict
    warnings = []
    if min(w, h) < MIN_SIDE_WARN:
        warnings.append(f"This image is small ({w}x{h}). It gets enlarged to 224 pixels, "
                        "so the guess is less reliable.")
    kept = min(w, h) / max(w, h)
    if kept < NARROW_WARN:
        warnings.append(f"The model only sees the middle square of the photo (about {kept:.0%} of it); "
                        "the rest is cut off.")

    # Run the model on the image
    t0 = time.time()
    with predict_lock:
        out = predictor.predict(img)
    elapsed_ms = round(1000 * (time.time() - t0))

    # Save the model's 224x224 input crop as a base64-encoded PNG for the web page to display
    buf = io.BytesIO()
    model_input_preview(img).save(buf, format="PNG")
    crop = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return jsonify({"guess": out["guess"], "top": out["top"], "crop": crop,
                    "size": [w, h], "warnings": warnings, "elapsed_ms": elapsed_ms})

# Run the app on localhost:5000
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)