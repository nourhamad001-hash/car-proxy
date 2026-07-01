import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "ultrasonic-boom-clmei/6"

# ---- anti-false-positive tuning (all overridable via Railway Variables) ----
CONFIDENCE     = 30   # detection threshold. Lower = detects from farther / more easily.
CONFIRM_FRAMES = 2     # must be seen this many frames in a row before acting (kills flicker false-positives).

# ---- distance estimation (camera-based, no ultrasonic sensor needed) ----
# distance_cm = (SENSOR_WIDTH_CM * FOCAL_LENGTH_PX) / bbox_width_px
# SENSOR_WIDTH_CM: real-world width of the HC-SR04 board, measure with a ruler.
# FOCAL_LENGTH_PX: calibrate once -> take a photo with the sensor at a KNOWN
#   distance (e.g. 30cm), note the bbox width (pixels) the model reports for
#   that photo, then: FOCAL_LENGTH_PX = (bw_at_30cm * 30) / SENSOR_WIDTH_CM
#   The value below is a rough placeholder until you calibrate it.
SENSOR_WIDTH_CM  = 4.5
FOCAL_LENGTH_PX  = 500
STOP_DISTANCE_CM = 21   # stop when estimated distance is closer than this

# tracks how many frames in a row we've seen the target
streak = 0


@app.route('/health')
def health():
    return jsonify({
        "status":           "ok",
        "model":            ROBOFLOW_MODEL,
        "api_key_set":      bool(ROBOFLOW_API_KEY),
        "confidence":       CONFIDENCE,
        "confirm_frames":   CONFIRM_FRAMES,
        "stop_distance_cm": STOP_DISTANCE_CM,
        "focal_length_px":  FOCAL_LENGTH_PX
    })


@app.route('/detect', methods=['POST'])
def detect():
    global streak
    try:
        b64 = request.get_json().get('image')
        if not b64:
            return jsonify({"error": "no image"}), 400

        r = requests.post(
            f"https://serverless.roboflow.com/{ROBOFLOW_MODEL}",
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": CONFIDENCE,
                "format": "json"
            },
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )

        try:
            d = r.json()
        except Exception:
            d = {}

        if r.status_code != 200 or "predictions" not in d:
            streak = 0
            msg = d.get("message") or d.get("error") or r.text[:200] or "unknown error"
            print(f"ROBOFLOW ERROR {r.status_code}: {msg}")
            return jsonify({
                "command": "STOP", "target_found": False,
                "reason": f"Roboflow error ({r.status_code}): {msg}",
                "confidence": 0, "bbox": None
            })

        preds = d.get("predictions", [])
        iw    = d.get("image", {}).get("width",  320)
        ih    = d.get("image", {}).get("height", 240)
        all_labels = [p.get("class", "?") for p in preds]
        print(f"Detections: {all_labels}")

        if not preds:
            streak = 0
            return jsonify({
                "command": "FORWARD", "target_found": False,
                "reason": "No sensor detected — searching",
                "confidence": 0, "bbox": None, "all_labels": []
            })

        best = max(preds, key=lambda p: p["confidence"])
        cx   = best["x"]
        bw   = best["width"]
        bh   = best["height"]
        conf = round(best["confidence"] * 100)
        distance_cm = round((SENSOR_WIDTH_CM * FOCAL_LENGTH_PX) / bw, 1) if bw > 0 else None
        bbox = {
            "x": round((cx - bw / 2) / iw, 3),
            "y": round((best["y"] - bh / 2) / ih, 3),
            "w": round(bw / iw, 3),
            "h": round(bh / ih, 3)
        }

        streak += 1
        confirmed = streak >= CONFIRM_FRAMES
        print(f"Best: {best.get('class')} conf={conf}% distance_cm={distance_cm} streak={streak} confirmed={confirmed}")

        # Stop immediately if close enough, regardless of confirm streak -
        # waiting for confirmation here only risks overshooting/crashing into
        # the target while frames are still "confirming".
        if distance_cm is not None and distance_cm <= STOP_DISTANCE_CM:
            return jsonify({
                "command": "TARGET_FOUND", "target_found": True,
                "reason": f"Sensor close ({distance_cm}cm) — stopping!",
                "confidence": best["confidence"], "bbox": bbox, "all_labels": all_labels,
                "distance_cm": distance_cm
            })

        # Forward-only navigation: no left/right steering at all, just drive
        # straight at whatever's detected until it's close enough to stop.
        if not confirmed:
            reason = f"Maybe sensor ({conf}%) — confirming {streak}/{CONFIRM_FRAMES}"
        else:
            reason = f"Sensor ahead ({distance_cm}cm) — moving closer"

        return jsonify({
            "command": "FORWARD", "target_found": False,
            "reason": reason, "confidence": best["confidence"],
            "bbox": bbox, "all_labels": all_labels, "distance_cm": distance_cm
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
