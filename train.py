import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from config import config as cfg
from model import Net  # Assuming you name your model file `model.py` and class `Net`
from torchsummary import summary

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} - Training")
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        progress_bar.set_postfix(
            loss=f"{running_loss / total:.4f}", accuracy=f"{correct / total:.4f}"
        )

    return running_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} - Validation")
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            progress_bar.set_postfix(
                loss=f"{running_loss / total:.4f}", accuracy=f"{correct / total:.4f}"
            )

    return running_loss / total, correct / total


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(
        batch_size=cfg.batch_size,
        valid_split=cfg.valid_split,
        num_workers=cfg.num_workers
    )

    model = Net(base_channels=16).to(device)
    summary(model, input_size=(3, 32, 32))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.001,
                                                     threshold_mode='abs', eps=0.001, verbose=True)

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg.epochs)
        valid_loss, valid_acc = validate_one_epoch(model, test_loader, criterion, device, epoch, cfg.epochs)

        print(f"\nEpoch {epoch+1}/{cfg.epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

        scheduler.step(valid_acc)

    torch.save(model.state_dict(), "model.pth")
    print("Training complete. Model saved to 'model.pth'.")


if __name__ == "__main__":
    main()
