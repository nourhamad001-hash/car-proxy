import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "ultrasonic-boom-clmei/6"

# ---- anti-false-positive tuning (all overridable via Railway Variables) ----
CONFIDENCE     = 20    # detection threshold. Lower = detects from farther / more easily.
CONFIRM_FRAMES = 1     # must be seen this many frames in a row before trusting a detection at all.

# ---- simple fixed-count approach (no distance measurement) ----
# Once the sensor is confirmed, drive FORWARD this many times, then stop.
# No camera calibration needed, just a fixed number of "steps" toward it.
FORWARD_STEPS_BEFORE_STOP = 2

# tracks how many frames in a row we've seen the target
streak = 0
# tolerant miss tracking - one flaky missed frame doesn't fully reset streak
miss_streak = 0
MISS_TOLERANCE = 2
# how many confirmed FORWARD moves we've made toward the target so far
approach_count = 0


@app.route('/health')
def health():
    return jsonify({
        "status":                    "ok",
        "model":                     ROBOFLOW_MODEL,
        "api_key_set":               bool(ROBOFLOW_API_KEY),
        "confidence":                CONFIDENCE,
        "confirm_frames":            CONFIRM_FRAMES,
        "forward_steps_before_stop": FORWARD_STEPS_BEFORE_STOP
    })


@app.route('/detect', methods=['POST'])
def detect():
    global streak, miss_streak, approach_count
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
            approach_count = 0
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
            miss_streak += 1
            if miss_streak >= MISS_TOLERANCE:
                streak = 0
                approach_count = 0
            print(f"No detection this frame (miss {miss_streak}/{MISS_TOLERANCE}), streak={streak}")
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
        bbox = {
            "x": round((cx - bw / 2) / iw, 3),
            "y": round((best["y"] - bh / 2) / ih, 3),
            "w": round(bw / iw, 3),
            "h": round(bh / ih, 3)
        }

        streak += 1
        miss_streak = 0
        confirmed = streak >= CONFIRM_FRAMES
        print(f"Best: {best.get('class')} conf={conf}% streak={streak} confirmed={confirmed} approach_count={approach_count}")

        if not confirmed:
            return jsonify({
                "command": "FORWARD", "target_found": False,
                "reason": f"Maybe sensor ({conf}%) — confirming {streak}/{CONFIRM_FRAMES}",
                "confidence": best["confidence"], "bbox": bbox, "all_labels": all_labels
            })

        # Confirmed: take one more forward step. Once we've taken
        # FORWARD_STEPS_BEFORE_STOP steps toward it, stop.
        approach_count += 1
        if approach_count >= FORWARD_STEPS_BEFORE_STOP:
            approach_count = 0
            return jsonify({
                "command": "TARGET_FOUND", "target_found": True,
                "reason": f"Found it! ({conf}%) — reached, stopping!",
                "confidence": best["confidence"], "bbox": bbox, "all_labels": all_labels
            })

        return jsonify({
            "command": "FORWARD", "target_found": False,
            "reason": f"Found it! Driving to it ({approach_count}/{FORWARD_STEPS_BEFORE_STOP})...",
            "confidence": best["confidence"], "bbox": bbox, "all_labels": all_labels
        })

    except Exception as e:
        streak = 0
        approach_count = 0
        print(f"Error: {e}")
        return jsonify({
            "command": "STOP", "target_found": False,
            "reason": str(e), "confidence": 0, "bbox": None
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
