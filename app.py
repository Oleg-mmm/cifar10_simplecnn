import asyncio
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from cifar10 import SimpleCNN
from cifar100 import ResNet18_CIFAR


app = FastAPI(title="CIFAR Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Загрузка моделей")
model_cifar10 = SimpleCNN(num_classes=10).to(device)
model_cifar10.load_state_dict(torch.load("best_cifar10.pth", map_location=device))
model_cifar10.eval()

model_cifar100 = ResNet18_CIFAR(num_classes=100).to(device)
model_cifar100.load_state_dict(torch.load("best_cifar100.pth", map_location=device))
model_cifar100.eval()
print("Модели загружены")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Названия классов из torchvision
classes_cifar10 = torchvision.datasets.CIFAR10.classes
classes_cifar100 = torchvision.datasets.CIFAR100.classes

def opencv_preprocessing(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Не удалось прочитать изображение')
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    
    return transform(img).unsqueeze(0).to(device)


def run_inference(model_name, image_bytes):
    tensor = opencv_preprocessing(image_bytes)
    with torch.no_grad():
        if model_name == 'cifar10':
            outputs = model_cifar10(tensor)
            classes = classes_cifar10
        else:
            outputs = model_cifar100(tensor)
            classes = classes_cifar100
        prob = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(prob, dim=1)
        
    return {
            "model": model_name,
            "class": classes[pred_idx.item()],
            "confidence": round(conf.item(), 4)
            }


@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in ["cifar10", "cifar100"]:
        return JSONResponse(content={"error": "Модель не найдена. Используйте 'cifar10' или 'cifar100'"}, status_code=400)
        
    image_bytes = await file.read()
        
    try:
        result = await asyncio.to_thread(run_inference, model_name, image_bytes)
    except ValueError as val_e:
        return JSONResponse(content={"error": str(val_e)}, status_code=400)

    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok"}