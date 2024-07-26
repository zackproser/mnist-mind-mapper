import onnxruntime as ort
from torchvision import transforms
import numpy as np

# Create an ONNX Runtime inference session
ort_session = ort.InferenceSession("mnist_model.onnx")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Function to run inference using ONNX Runtime
def run_inference(image):
    # Apply transformations
    tensor = transform(image).unsqueeze(0).numpy()
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Get prediction
    return np.argmax(ort_outputs[0])
