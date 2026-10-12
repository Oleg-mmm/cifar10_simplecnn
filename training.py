import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
          epochs=100, early_stop=None, scheduler=None, ckpt_path=None):
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
            if improved and ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
            if early_stop.earlystop:
                print("Early stopping")
                break

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()