import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "ultrasonic-boom-clmei/8"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

# ---- anti-false-positive tuning (all overridable via Railway Variables) ----
CONFIDENCE     = 20    # detection threshold (%). Lower = detects from farther / more easily.
TARGET_AREA    = 0.08  # sensor must fill this fraction of frame to "stop". Lower = stops from farther.
CONFIRM_FRAMES = 2     # must be seen this many frames in a row before acting (kills flicker false-positives).

# tracks how many frames in a row we've seen the target
streak = 0


@app.route('/health')
def health():
    return jsonify({
        "status":         "ok",
        "model":          ROBOFLOW_MODEL,
        "api_key_set":    bool(ROBOFLOW_API_KEY),
        "confidence":     CONFIDENCE,
        "target_area":    TARGET_AREA,
        "confirm_frames": CONFIRM_FRAMES
    })


@app.route('/detect', methods=['POST'])
def detect():
    global streak
    try:
        b64 = request.get_json().get('image')
        if not b64:
            return jsonify({"error": "no image"}), 400

        # decode the incoming base64 frame into a PIL image for the SDK
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            streak = 0
            print(f"Bad image data: {e}")
            return jsonify({
                "command": "STOP", "target_found": False,
                "reason": f"Bad image data: {e}",
                "confidence": 0, "bbox": None
            })

        # call Roboflow via the inference-sdk (required for YOLO26 models —
        # the old raw REST endpoint returns 404 for this model type)
        try:
            d = CLIENT.infer(img, model_id=ROBOFLOW_MODEL)
        except Exception as e:
            streak = 0
            print(f"ROBOFLOW ERROR: {e}")
            return jsonify({
                "command": "STOP", "target_found": False,
                "reason": f"Roboflow error: {e}",
                "confidence": 0, "bbox": None
            })

        if not isinstance(d, dict) or "predictions" not in d:
            streak = 0
            print(f"Unexpected Roboflow response shape: {d}")
            return jsonify({
                "command": "STOP", "target_found": False,
                "reason": "Unexpected response from Roboflow",
                "confidence": 0, "bbox": None
            })

        preds = d.get("predictions", [])
        iw    = d.get("image", {}).get("width",  img.width)
        ih    = d.get("image", {}).get("height", img.height)
        all_labels = [p.get("class", "?") for p in preds]
        print(f"Detections: {all_labels}")

        # apply our own confidence threshold (the SDK call above doesn't take
        # one as a URL param like the old endpoint did, so filter here instead)
        preds = [p for p in preds if p.get("confidence", 0) * 100 >= CONFIDENCE]

        if not preds:
            streak = 0   # nothing this frame -> reset the streak
            return jsonify({
                "command": "FORWARD", "target_found": False,
                "reason": "No sensor detected — searching",
                "confidence": 0, "bbox": None, "all_labels": all_labels
            })

        best = max(preds, key=lambda p: p["confidence"])
        cx   = best["x"]
        bw   = best["width"]
        bh   = best["height"]
        area = (bw * bh) / (iw * ih)
        conf = round(best["confidence"] * 100)
        bbox = {
            "x": round((cx - bw / 2) / iw, 3),
            "y": round((best["y"] - bh / 2) / ih, 3),
            "w": round(bw / iw, 3),
            "h": round(bh / ih, 3)
        }

        # count consecutive detections; require CONFIRM_FRAMES before we trust it
        streak += 1
        confirmed = streak >= CONFIRM_FRAMES
        print(f"Best: {best.get('class')} conf={conf}% area={round(area,3)} streak={streak} confirmed={confirmed}")

        if not confirmed:
            return jsonify({
                "command": "FORWARD", "target_found": False,
                "reason": f"Maybe sensor ({conf}%) — confirming {streak}/{CONFIRM_FRAMES}",
                "confidence": best["confidence"], "bbox": bbox, "all_labels": all_labels
            })

        if area > TARGET_AREA:
            cmd    = "TARGET_FOUND"
            reason = f"Sensor close ({conf}%) — stopping!"
        else:
            offset = cx - (iw / 2)
            if offset > iw * 0.2:
                cmd, reason = "RIGHT", f"Sensor on right ({conf}%) — turning"
            elif offset < -iw * 0.2:
                cmd, reason = "LEFT", f"Sensor on left ({conf}%) — turning"
            else:
                cmd, reason = "FORWARD", f"Sensor ahead ({conf}%) — moving closer"

        return jsonify({
            "command": cmd, "target_found": cmd == "TARGET_FOUND",
            "reason": reason, "confidence": best["confidence"],
            "bbox": bbox, "all_labels": all_labels
        })

    except Exception as e:
        streak = 0
        print(f"Error: {e}")
        return jsonify({
            "command": "STOP", "target_found": False,
            "reason": str(e), "confidence": 0, "bbox": None
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
