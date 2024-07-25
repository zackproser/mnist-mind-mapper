from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
from model_setup import model, device, transform

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
    
    return jsonify({"predicted": int(predicted.item())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5328, debug=True)
