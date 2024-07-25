import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Define the SimpleNN class (same as before)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
model.eval()

# Define image transformation
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
    
    # Decode the base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
    
    return jsonify({"predicted": int(predicted.item())})

@app.route("/api/test_images", methods=["GET"])
def test_images():
    # Load test dataset
    from torchvision import datasets
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Get first batch of images
    images, labels = next(iter(test_loader))

    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Sample Test Images with Predictions')

    for i, (image, label) in enumerate(zip(images, labels)):
        output = model(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output.data, 1)
        
        ax = axs[i // 5, i % 5]
        ax.imshow(image.squeeze().numpy(), cmap='gray')
        ax.set_title(f'Pred: {predicted.item()}, True: {label.item()}')
        ax.axis('off')

    plt.tight_layout()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5328, debug=True)
