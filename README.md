# CIFAR-10 SimpleCNN

Классификация CIFAR-10 с помощью собственной CNN на PyTorch.

## Архитектура

SimpleCNN — 3 свёрточных блока по 2 свёртки каждый,
BatchNorm после каждой свёртки, MaxPool после каждого блока.
Голова: AdaptiveAvgPool → Dropout(0.5) → Linear(256, 10).

## Обучение

- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=1e-3, weight_decay=0.05)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- EarlyStopping: patience=5, delta=0.01
- Аугментация: RandomCrop(32, padding=4), RandomHorizontalFlip

## Результаты

| Модель    | Best Val Acc | Best Val Loss | Эпох |
|-----------|--------------|---------------|------|
| SimpleCNN |    0.895     |     0.347     |  30  |

Цифры из обучения в google collab на 40 эпох с `patience=20`. При `patience=5`
обучение останавливается примерно на 30-й эпохе с близкими значениями.

Лучший чекпоинт сохраняется в `best_cifar10.pth`.
