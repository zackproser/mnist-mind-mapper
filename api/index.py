import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify

# Simplified NN class
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cpu")  # Use CPU to reduce complexity
model = SimpleNN().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device, weights_only=True))
model.eval()

# Simplified transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    image_b64 = data['image']
    if ',' in image_b64:
        # Handle data URL format
        image_b64 = image_b64.split(',')[1]
    
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data)).convert('L')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
    
    return jsonify({"predicted": int(predicted.item())})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5328)
