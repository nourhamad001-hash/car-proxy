import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "sensor-dvgxe/2"

# ─── Health check ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model":  ROBOFLOW_MODEL,
        "api_key_set": bool(ROBOFLOW_API_KEY)
    })

# ─── Main detection endpoint ──────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        b64  = data.get('image')

        if not b64:
            return jsonify({"error": "no image provided"}), 400

        if not ROBOFLOW_API_KEY:
            return jsonify({"error": "ROBOFLOW_API_KEY not set"}), 500

        # Forward to Roboflow
        r = requests.post(
            f"https://serverless.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}",
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )

        d      = r.json()
        preds  = d.get("predictions", [])
        iw     = d.get("image", {}).get("width",  320)
        ih     = d.get("image", {}).get("height", 240)

        # No detection
        if not preds:
            return jsonify({
                "command":      "FORWARD",
                "target_found": False,
                "reason":       "No HC-SR04 detected — searching",
                "confidence":   0,
                "bbox":         None,
                "count":        0
            })

        # Best detection by confidence
        best = max(preds, key=lambda p: p["confidence"])
        cx   = best["x"]
        bw   = best["width"]
        bh   = best["height"]
        area = (bw * bh) / (iw * ih)
        conf = round(best["confidence"] * 100)

        # Bounding box as fractions 0.0-1.0
        bbox = {
            "x": round((cx  - bw / 2) / iw, 3),
            "y": round((best["y"] - bh / 2) / ih, 3),
            "w": round(bw / iw, 3),
            "h": round(bh / ih, 3)
        }

        # Decide driving command
        if area > 0.10:
            command = "TARGET_FOUND"
            reason  = f"HC-SR04 very close ({conf}%) — stopping!"
        else:
            offset = cx - (iw / 2)
            if offset > iw * 0.2:
                command = "RIGHT"
                reason  = f"HC-SR04 on right ({conf}%) — turning right"
            elif offset < -iw * 0.2:
                command = "LEFT"
                reason  = f"HC-SR04 on left ({conf}%) — turning left"
            else:
                command = "FORWARD"
                reason  = f"HC-SR04 centered ({conf}%) — moving closer"

        return jsonify({
            "command":      command,
            "target_found": command == "TARGET_FOUND",
            "reason":       reason,
            "confidence":   round(best["confidence"], 3),
            "bbox":         bbox,
            "count":        len(preds)
        })

    except requests.Timeout:
        return jsonify({
            "command":      "STOP",
            "target_found": False,
            "reason":       "Roboflow timeout — stopping for safety",
            "confidence":   0,
            "bbox":         None
        }), 504

    except Exception as e:
        return jsonify({
            "command":      "STOP",
            "target_found": False,
            "reason":       f"Server error: {str(e)}",
            "confidence":   0,
            "bbox":         None
        }), 500

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}")
    print(f"Model: {ROBOFLOW_MODEL}")
    print(f"API key set: {bool(ROBOFLOW_API_KEY)}")
    app.run(host='0.0.0.0', port=port)
