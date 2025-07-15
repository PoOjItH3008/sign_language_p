from flask import Flask, render_template, request, jsonify
import base64, numpy as np, cv2
from ultralytics import YOLO

app = Flask(__name__)

# ─── Load models ───────────────────────────────────────────────────
model_detect  = YOLO("best.pt")      # object detection
model_segment = YOLO("actions.pt")   # instance segmentation (-seg)

# ─── Utility: base64 data-URL → BGR image ──────────────────────────
def data_url_to_bgr(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_np    = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

# ─── Detection inference ───────────────────────────────────────────
def run_detection(img_bgr):
    res = model_detect(img_bgr, verbose=False)[0]
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": float(box.conf[0]),
            "label":      model_detect.names[int(box.cls[0])]
        })
    return dets

# ─── Mask helper ───────────────────────────────────────────────────
def encode_mask(mask: np.ndarray) -> str:
    _, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    return base64.b64encode(png).decode("utf-8")

# ─── Segmentation inference ────────────────────────────────────────
def run_segmentation(img_bgr):
    res   = model_segment(img_bgr, verbose=False)[0]
    masks = res.masks.data.cpu().numpy() if res.masks else []
    dets  = []
    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": float(box.conf[0]),
            "label":      model_segment.names[int(box.cls[0])],
            "mask":       encode_mask(masks[i]) if len(masks) > i else None
        })
    return dets

# ─── Shared prediction logic ───────────────────────────────────────
def _common_predict(task: str):
    payload = request.get_json(force=True)
    if "image" not in payload:
        return jsonify({"error": "image field missing"}), 400

    try:
        frame = data_url_to_bgr(payload["image"])
    except Exception as e:
        return jsonify({"error": f"decode error: {e}"}), 400

    out = {}
    if task in ("detect", "both"):
        out["detections"] = run_detection(frame)
    if task in ("segment", "both"):
        out["segmentations"] = run_segmentation(frame)
    return jsonify(out)

# ─── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    task = request.get_json(force=True).get("task", "detect").lower()
    if task not in ("detect", "segment", "both"):
        return jsonify({"error": "task must be detect | segment | both"}), 400
    return _common_predict(task)

# ─── Run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)          # add host/port if needed
