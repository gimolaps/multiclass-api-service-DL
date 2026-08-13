import torch
import torch.nn as nn

from src.config import CHECKPOINTS_DIR, DEVICE
from src.model import CNN
from src.dataset import get_train_val_dataloaders


def evaluate():
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_train_val_dataloaders()

    model = CNN().to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / "best_model.pth", map_location=DEVICE)
    )
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    val_current_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            val_current_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_current_loss / len(val_dataloader)
    val_acc = 100 * val_correct / val_total

    print(f"Val loss: {val_loss:.4f}")
    print(f"Val accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    evaluate()