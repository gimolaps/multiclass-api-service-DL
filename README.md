# Tiny ImageNet-200 CNN Classifier with FastAPI

Image classification project based on **Tiny ImageNet-200**.

The project contains a custom CNN model trained on 200 image classes and a simple **FastAPI web interface** where a user can upload an image and receive a predicted class with confidence score.

---

## Project Preview

The web app allows the user to:

- upload an image in `.jpg`, `.jpeg`, `.png`, or `.webp` format;
- run prediction using the trained CNN model;
- view the main predicted class;
- view top-5 predicted classes with confidence percentages.

---

## Dataset

Dataset used:

**Tiny ImageNet-200**

Original Kaggle dataset:

```text
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet/data
```

Dataset format:

```text
200 classes
64x64 RGB images
JPEG format
train / val / test splits
```

Expected local structure:

```text
data/
├── train/
├── val/
│   ├── images/
│   └── val_annotations.txt
├── test/
│   └── images/
├── wnids.txt
└── words.txt
```

Raw dataset files are not included in the repository.  
Download the dataset manually and place it inside the `data/` directory.

---

## Project Structure

```text
multiclass_api_service/
├── checkpoints/
│   └── best_model.pth
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── wnids.txt
│   └── words.txt
├── notebooks/
│   └── notebooks.ipynb
├── reports/
│   ├── history.json
│   └── metrics.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── README.md
└── requirements.txt
```

---

## Model

The project uses a custom Convolutional Neural Network.

Architecture:

```text
Input: 3 x 64 x 64

Block 1:
Conv2d -> BatchNorm2d -> ReLU
Conv2d -> BatchNorm2d -> ReLU
MaxPool2d

Block 2:
Conv2d -> BatchNorm2d -> ReLU
Conv2d -> BatchNorm2d -> ReLU
MaxPool2d

Block 3:
Conv2d -> BatchNorm2d -> ReLU
Conv2d -> BatchNorm2d -> ReLU
MaxPool2d

Classifier:
Flatten
Linear
ReLU
Dropout
Linear -> 200 classes
```

Output shape:

```text
[batch_size, 200]
```

The model returns raw logits.  
`CrossEntropyLoss` is used during training.

---

## Training Result

Best validation result:

```text
Best validation accuracy: 34.26%
Best epoch: 49
```

This result was achieved using a custom CNN trained from scratch on Tiny ImageNet-200.

Random guessing baseline:

```text
1 / 200 = 0.5%
```

So the model performs significantly better than random prediction, but it is still a simple educational CNN. Better results can be achieved with stronger architectures such as ResNet18, EfficientNet, or transfer learning.

---

## Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet:

```bash
pip freeze > requirements.txt
```

Required main libraries:

```text
torch
torchvision
fastapi
uvicorn
python-multipart
pillow
```

---

## Run FastAPI Web App

Run from the project root:

```bash
uvicorn src.main:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000
```

Upload an image and click **Predict class**.

---

## API Endpoint

### `GET /`

Returns the upload page.

### `POST /predict`

Accepts image file and returns prediction result.

Supported formats:

```text
.jpg
.jpeg
.png
.webp
```

---

## Prediction Output

The app returns:

```text
Main predicted class
Confidence percentage
Top-5 predictions
```

Example:

```text
Prediction: drumstick
Confidence: 42.18%
```

---

## Training

To train the model again:

```bash
python src/train.py
```

Training saves:

```text
checkpoints/best_model.pth
checkpoints/last_model.pth
reports/history.json
reports/metrics.json
```

Best model is selected by validation accuracy.

---

## Evaluation

To evaluate the saved best model on validation data:

```bash
python src/evaluate.py
```

Validation data uses:

```text
data/val/images/
data/val/val_annotations.txt
```

---

## Prediction from Script

To run prediction on random test images without the web app:

```bash
python src/predict.py
```

The test split does not contain labels, so this script only prints predicted classes.  
It does not calculate test accuracy.

---

## Notes

Tiny ImageNet validation labels are stored in:

```text
val_annotations.txt
```

The validation images are not stored in class folders, so a custom `Dataset` class is used for validation.

The `words.txt` file is used to convert WordNet IDs into readable class names.

Example:

```text
n01443537 -> goldfish
```

---

## Current Limitations

The current model is a custom CNN trained from scratch.

Main limitations:

- 200 classes is a difficult classification task;
- 64x64 images contain limited visual detail;
- the model is much smaller than modern pretrained CNNs;
- test accuracy cannot be calculated because the Tiny ImageNet test split has no labels.

Possible improvements:

- stronger data augmentation;
- deeper CNN;
- transfer learning with ResNet18;
- fine-tuning pretrained models;
- top-5 accuracy metric;
- confusion matrix on validation data;
- image preview and better UI in FastAPI.

---

## Author

Project created as a learning project for computer vision, PyTorch, CNN training, and FastAPI deployment.