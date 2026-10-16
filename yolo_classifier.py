from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

results = model.train(
    data = 'cifar10',
    epochs = 10,
    imgsz = 32,
    batch = 64,
    name = 'yolov8n_model',
    pretrained = True
)

metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")