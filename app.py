import io
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
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

@app.post("/predict/{model_name}")
async def predict(model_name: str, file: UploadFile = File(...)):
    if model_name not in ["cifar10", "cifar100"]:
        return JSONResponse(content={"error": "Модель не найдена. Используй 'cifar10' или 'cifar100'"}, status_code=400)
        
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        if model_name == "cifar10":
            outputs = model_cifar10(tensor)
            classes = classes_cifar10
        else:
            outputs = model_cifar100(tensor)
            classes = classes_cifar100
            
        prob = torch.nn.functional.softmax(outputs, dim=1)
        pred, predict_idx = torch.max(prob, 1)
        
    return JSONResponse(content={
        "model": model_name,
        "class": classes[predict_idx.item()],
        "confidence": round(pred.item(), 4)
    })

@app.get("/health")
async def health():
    return {"status": "ok"}