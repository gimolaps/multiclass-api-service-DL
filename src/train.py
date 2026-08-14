import json
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    DEVICE,
    EPOCHS,
    LR,
)

from src.model import CNN
from src.dataset import get_train_val_dataloaders

use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def train():
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, train_dataloader, val_dataloader = get_train_val_dataloaders()

    model = CNN().to(DEVICE)

    resume_path = CHECKPOINTS_DIR / "last_model.pth"

    if resume_path.exists():
        model.load_state_dict(
            torch.load(resume_path, map_location=DEVICE)
        )
        print(f"Loaded checkpoint: {resume_path}")
    else:
        print("No checkpoint found. Training from scratch.")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    history = []

    for epoch in range(EPOCHS):
        print(f"Training epoch {epoch + 1}....")

        # =====================
        # train loop
        # =====================
        model.train()

        current_loss = 0.0
        correct = 0
        total = 0

        for num_batch, batch in enumerate(train_dataloader):
            images, labels = batch

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = current_loss / len(train_dataloader)
        train_acc = 100 * correct / total

        # =====================
        # validation loop
        # =====================
        model.eval()

        val_current_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for num_batch, batch in enumerate(val_dataloader):
                images, labels = batch

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

        scheduler.step()

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        }

        history.append(epoch_result)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best_model.pth")
            print("Best model saved")

        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.2f}%")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val accuracy: {val_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 40)

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "last_model.pth")

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_epochs": EPOCHS
    }

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(REPORTS_DIR / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("Training finished")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()
