import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "ultrasonic-boom-clmei/7"

# ---- anti-false-positive tuning (all overridable via Railway Variables) ----
CONFIDENCE     = int(os.environ.get("CONFIDENCE", "55"))      # was 40. Higher = fewer false hits.
TARGET_AREA    = float(os.environ.get("TARGET_AREA", "0.10")) # was 0.05. Sensor must fill >=10% of frame to "stop".
CONFIRM_FRAMES = int(os.environ.get("CONFIRM_FRAMES", "2"))   # must be seen this many times in a row before acting.

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

        r = requests.post(
            f"https://serverless.roboflow.com/{ROBOFLOW_MODEL}"
            f"?api_key={ROBOFLOW_API_KEY}&confidence={CONFIDENCE}",
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )

        try:
            d = r.json()
        except Exception:
            d = {}

        # surface real Roboflow errors instead of hiding them as "searching"
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
            streak = 0   # nothing this frame -> reset the streak
            return jsonify({
                "command": "FORWARD", "target_found": False,
                "reason": "No sensor detected — searching",
                "confidence": 0, "bbox": None, "all_labels": []
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
