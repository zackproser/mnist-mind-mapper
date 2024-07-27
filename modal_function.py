import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import io
import numpy as np
import base64

app = modal.App("mnist-mind-mapper")

image = (
    modal.Image.debian_slim()
    .pip_install("onnxruntime", "fastapi", "uvicorn", "torch", "Pillow", "torchvision")
    .copy_local_file("mnist_model.onnx", "/root/mnist_model.onnx")
)

@app.cls(image=image)
class Model:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.load_model()

    def load_model(self):
        model_path = "/root/mnist_model.onnx"
        self.ort_session = ort.InferenceSession(model_path)

    def predict(self, image_data: bytes):
        img = Image.open(io.BytesIO(image_data))
        img = self.transform(img).unsqueeze(0).numpy()
        ort_inputs = {self.ort_session.get_inputs()[0].name: img}
        ort_outs = self.ort_session.run(None, ort_inputs)
        prediction = np.argmax(ort_outs[0])
        return int(prediction)

web_app = FastAPI()

@web_app.post("/predict")
async def predict(request: Request):
    model = Model()
    data = await request.json()
    if 'image' not in data:
        raise HTTPException(status_code=400, detail="No image part")
    
    image_data = data['image']
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    try:
        image_data = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    try:
        prediction = model.predict(image_data)
        return JSONResponse(content={'prediction': prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    modal.run(fastapi_app)
