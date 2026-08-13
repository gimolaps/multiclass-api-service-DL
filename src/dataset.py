from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from src.config import (
    TRAIN_DATA,
    VAL_DATA,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    WORDS_FILE,
)


class TinyImageNetValDataset(Dataset):
    def __init__(self, val_dir, transform=None, class_to_idx=None):
        self.val_dir = Path(val_dir)
        self.images_dir = self.val_dir / "images"
        self.annotations_path = self.val_dir / "val_annotations.txt"
        self.transform = transform
        self.class_to_idx = class_to_idx

        self.samples = []

        with open(self.annotations_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                image_name = parts[0]
                wnid = parts[1]

                label = self.class_to_idx[wnid]
                image_path = self.images_dir / image_name

                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_val_and_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_train_val_dataloaders():
    train_transform = get_train_transform()
    val_transform = get_val_and_test_transform()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DATA,
        transform=train_transform
    )

    val_dataset = TinyImageNetValDataset(
        val_dir=VAL_DATA,
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def load_wnid_to_name():
    wnid_to_name = {}

    with open(WORDS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            wnid, class_name = line.strip().split("\t", 1)
            wnid_to_name[wnid] = class_name

    return wnid_to_name