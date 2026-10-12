from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from training import EarlyStopping, train, device

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "best_cifar100.pth"


class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


def main():
    torch.manual_seed(123)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=str(BASE_DIR / "data"), train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=str(BASE_DIR / "data"), train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    model = ResNet18_CIFAR(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=0.01
    )

    train(
        train_loader, val_loader, model, criterion, optimizer,
        epochs=100,
        early_stop=EarlyStopping(patience=5, delta=0.01),
        scheduler=scheduler,
        ckpt_path=CKPT_PATH,
    )


if __name__ == "__main__":
    main()