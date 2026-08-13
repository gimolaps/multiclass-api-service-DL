import random
import torch
from PIL import Image

from src.config import CHECKPOINTS_DIR, TEST_DATA, DEVICE
from src.model import CNN
from src.dataset import get_val_and_test_transform, get_train_val_dataloaders, load_wnid_to_name


def load_test_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)
    return image


def predict_random_images(num_images=5):
    train_dataset, _, _, _ = get_train_val_dataloaders()

    idx_to_wnid = train_dataset.classes
    wnid_to_name = load_wnid_to_name()

    transform = get_val_and_test_transform()

    image_paths = list(TEST_DATA.glob("*.JPEG"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No JPEG images found in {TEST_DATA}")

    image_paths = random.sample(image_paths, min(num_images, len(image_paths)))

    model = CNN().to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINTS_DIR / "best_model.pth", map_location=DEVICE)
    )
    model.eval()

    with torch.no_grad():
        for image_path in image_paths:
            image = load_test_image(image_path, transform)

            logits = model(image)
            predicted = torch.argmax(logits, dim=1)

            predicted_idx = predicted.item()
            predicted_wnid = idx_to_wnid[predicted_idx]
            predicted_name = wnid_to_name[predicted_wnid]

            print(f"Image: {image_path.name}")
            print(f"Predicted wnid: {predicted_wnid}")
            print(f"Prediction: {predicted_name}")
            print("-" * 40)


if __name__ == "__main__":
    predict_random_images(num_images=5)