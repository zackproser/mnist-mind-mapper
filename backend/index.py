from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from model_setup import run_inference

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')

    # Make prediction using ONNX Runtime
    predicted = run_inference(image)
    
    return jsonify({"prediction": int(predicted)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5328)
