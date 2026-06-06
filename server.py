import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL   = "ultrasonic-boom-clmei/7"  # latest model version

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model": ROBOFLOW_MODEL,
        "api_key_set": bool(ROBOFLOW_API_KEY)
    })

@app.route('/detect', methods=['POST'])
def detect():
    try:
        b64 = request.get_json().get('image')
        if not b64:
            return jsonify({"error": "no image"}), 400

        r = requests.post(
            f"https://serverless.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}&confidence=40",
            data=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )

        d     = r.json()
        preds = d.get("predictions", [])
        iw    = d.get("image", {}).get("width",  320)
        ih    = d.get("image", {}).get("height", 240)

        all_labels = [p.get("class", "?") for p in preds]
        print(f"Detections: {all_labels}")

        if not preds:
            return jsonify({
                "command":      "FORWARD",
                "target_found": False,
                "reason":       "No HC-SR04 detected — searching",
                "confidence":   0,
                "bbox":         None
            })

        best = max(preds, key=lambda p: p["confidence"])
        cx   = best["x"]
        bw   = best["width"]
        bh   = best["height"]
        area = (bw * bh) / (iw * ih)
        conf = round(best["confidence"] * 100)

        bbox = {
            "x": round((cx  - bw / 2) / iw, 3),
            "y": round((best["y"] - bh / 2) / ih, 3),
            "w": round(bw / iw, 3),
            "h": round(bh / ih, 3)
        }

        print(f"Best: {best.get('class')} conf={conf}% area={round(area,3)}")

        if area > 0.05:
            cmd    = "TARGET_FOUND"
            reason = f"HC-SR04 close ({conf}%) — stopping!"
        else:
            offset = cx - (iw / 2)
            if offset > iw * 0.2:
                cmd    = "RIGHT"
                reason = f"HC-SR04 on right ({conf}%) — turning"
            elif offset < -iw * 0.2:
                cmd    = "LEFT"
                reason = f"HC-SR04 on left ({conf}%) — turning"
            else:
                cmd    = "FORWARD"
                reason = f"HC-SR04 ahead ({conf}%) — moving closer"

        return jsonify({
            "command":      cmd,
            "target_found": cmd == "TARGET_FOUND",
            "reason":       reason,
            "confidence":   best["confidence"],
            "bbox":         bbox,
            "all_labels":   all_labels
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "command":      "STOP",
            "target_found": False,
            "reason":       str(e),
            "confidence":   0,
            "bbox":         None
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
