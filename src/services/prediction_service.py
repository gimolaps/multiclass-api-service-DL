import io
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image

from src.config import CHECKPOINTS_DIR, DEVICE
from src.model import CNN
from src.dataset import (
    get_val_and_test_transform,
    load_wnid_to_name,
)


@lru_cache(maxsize=1)
def load_model():
    model = CNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINTS_DIR / "best_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()
    return model


@lru_cache(maxsize=1)
def load_classes():
    wnids_path = "data/wnids.txt"

    with open(wnids_path, "r", encoding="utf-8") as file:
        idx_to_wnid = sorted(line.strip() for line in file if line.strip())

    wnid_to_name = load_wnid_to_name()

    return idx_to_wnid, wnid_to_name


def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = get_val_and_test_transform()
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    return image


def predict_image(image_bytes):
    model = load_model()
    idx_to_wnid, wnid_to_name = load_classes()

    image = prepare_image(image_bytes)

    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_index = idx.item()
        confidence = prob.item() * 100

        wnid = idx_to_wnid[class_index]
        class_name = wnid_to_name[wnid]

        results.append({
            "class_name": class_name,
            "confidence": round(confidence, 2)
        })

    return results
