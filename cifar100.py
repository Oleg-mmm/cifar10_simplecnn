from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "best_cifar100.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class EarlyStopping:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_loss_score = float("inf")
        self.counter = 0
        self.earlystop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss_score - self.delta:
            self.best_loss_score = current_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.earlystop = True
        return False


def validate(dataloader, model, criterion):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predict = model(X)
            loss = criterion(predict, y)
            total_loss += loss.item()
            correct += (predict.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(dataloader), correct / total


def train(train_loader, val_loader, model, criterion, optimizer,
          epochs=100, early_stop=None, scheduler=None, ckpt_path=CKPT_PATH):
    for epoch in range(epochs):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            predict = model(X)
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total += y.size(0)
            correct += (predict.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        val_loss, val_acc = validate(val_loader, model, criterion)
        print(
            f"Epoch {epoch + 1:>3} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if early_stop is not None:
            improved = early_stop(val_loss)
            if improved:
                torch.save(model.state_dict(), ckpt_path)
            if early_stop.earlystop:
                print("Early stopping")
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()


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
    )


if __name__ == "__main__":
    main()