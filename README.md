# CIFAR Classification

Три эксперимента по классификации изображений на PyTorch:
- **CIFAR-10** — собственная CNN с нуля
- **CIFAR-100** — ResNet18 с нуля, с усиленной аугментацией под малые классы
- **CIFAR-10 (YOLOv8-cls)** — сравнение с `YOLOv8n-cls` из Ultralytics (transfer learning)

Основные модели обучаются с нуля (`weights=None`), без pretrained весов.
Seed фиксирован (123) для воспроизводимости.

## Что демонстрирует проект

- Обучение CNN с нуля
- Адаптация ResNet18 под малые изображения (32×32)
- Аугментация данных через `torchvision.transforms`
- Контроль переобучения: EarlyStopping, ReduceLROnPlateau, Dropout
- Сравнительный анализ архитектур: CNN vs `YOLOv8n-cls`
- Инференс-пайплайн: OpenCV (BGR→RGB, resize) → PyTorch → FastAPI → JSON
- Контейнеризация через Docker

## CIFAR-10 — SimpleCNN

### Архитектура
`SimpleCNN` — 3 свёрточных блока по 2 свёртки каждый, BatchNorm после каждой
свёртки, MaxPool после каждого блока. Голова: `AdaptiveAvgPool → Dropout(0.5)
→ Linear(256, 10)`.

### Обучение
- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW` (lr=1e-3, weight_decay=0.05)
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2)
- EarlyStopping: patience=5, delta=0.01
- Аугментация: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`

### Результаты
| Модель    | Best Val Acc | Best Val Loss | Эпох |
|-----------|--------------|---------------|------|
| SimpleCNN | 0.895        | 0.347         | 30   |

Цифры из обучения в Google Colab на 40 эпох с `patience=20`. При `patience=5`
обучение останавливается примерно на 30-й эпохе с близкими значениями.

Лучший чекпоинт: `best_cifar10.pth`.

## CIFAR-100 — ResNet18

### Архитектура
ResNet18 из `torchvision`, адаптированный под вход 32×32:
- `conv1` заменён на 3×3, stride=1 (вместо 7×7, stride=2)
- `maxpool` убран (`nn.Identity()`) — иначе теряется разрешение на картинках 32×32
- `fc` заменён на `Linear(512, 100)`

Обучение с нуля, `weights=None`.

### Обучение
- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW` (lr=1e-3, weight_decay=0.05)
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2, threshold=0.01)
- EarlyStopping: patience=5, delta=0.01 (по val_loss)
- Аугментация: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`,
  `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- Batch size: 64, epochs: до 100 (early stopping на 30)
- Device: CUDA (Google Colab, Tesla T4)

Аугментация усилена относительно CIFAR-10, потому что в CIFAR-100 в 10 раз
меньше примеров на класс (500 против 5000).

### Результаты
| Модель   | Best Val Acc | Best Val Loss | Эпоха останова | Best эпоха |
|----------|--------------|---------------|----------------|------------|
| ResNet18 | 0.7117       | 1.1691        | 30             | 25         |

Train acc на эпохе останова: 0.9548. Gap train/val ≈ 24% — модель
переобучается после ~25-й эпохи, что ожидаемо для ResNet18 (~11M параметров)
на 50k примеров с 500 примерами на класс.

Лучший чекпоинт: `best_cifar100.pth`.

## Сравнение с YOLOv8-cls (Ultralytics)

Для знакомства с библиотекой Ultralytics была обучена модель `yolov8n-cls` на CIFAR-10 (10 эпох, pretrained ImageNet weights).

**Код:** `yolo_classifier.py`

| Модель | Архитектура | Обучение | Top-1 Acc | Top-5 Acc | Примечание |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleCNN** | Своя CNN, 3 блока | С нуля, ~30 эпох | **0.895** | — | Специализирована под 32×32 |
| **YOLOv8n-cls** | Ultralytics (nano) | Transfer learning, 10 эпох | 0.778 | 0.989 | Универсальный backbone |

**Вывод:**
Моя `SimpleCNN`, спроектированная специально под размер 32×32, показала точность **89.5%**, превзойдя `YOLOv8n-cls` (**77.8%**). Это демонстрирует, что:
- **Transfer learning** даёт быстрый старт, но упирается в архитектурный потолок. Рост Top-1 Accuracy замедлился после 6-й эпохи, что говорит о пределе модели для этого датасета.
- Специализированные архитектуры могут достичь более высокой точности на конкретной задаче, чем универсальные модели.

## Inference API (FastAPI + Docker)

Для демонстрации полного цикла ML-решения (train -> save -> serve) модель обернута в REST API.

**Стек:** FastAPI, Uvicorn, Docker.

**Препроцессинг через OpenCV:**
- `cv2.imdecode` — чтение байтов в ndarray
- `cv2.cvtColor` — BGR → RGB (OpenCV читает в BGR, PyTorch ждёт RGB)
- `cv2.resize` до 32×32 с `INTER_AREA` (модели обучались на этом размере)

**Запуск:**
```bash
docker build -t cifar-api .
docker run -p 8000:8000 cifar-api