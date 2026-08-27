# Multiclass Image Classification API — Tiny ImageNet CNN

A Dockerized FastAPI service for multiclass image classification using a trained PyTorch CNN model.

The application allows users to upload an image through a web interface or API endpoint and receive the top-5 predicted classes with confidence scores.

---

## Project Overview

This project demonstrates the full path from a trained deep learning model to a usable web service:

- CNN model inference with PyTorch
- REST API built with FastAPI
- Separate frontend interface
- Image upload and preprocessing
- Top-5 class prediction
- Dockerized deployment setup
- Clean project structure with routers and services

The goal of the project is to show not only model training, but also how a deep learning model can be served as a real application.

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| API | FastAPI |
| Image Processing | Pillow |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker |
| Model Artifact | `.pth` checkpoint |

---

## Features

- Upload image from browser
- Preview selected image before prediction
- Predict image class using a trained CNN model
- Return top-5 predictions
- Show confidence score for every prediction
- Validate allowed image formats
- Provide Swagger API documentation
- Run locally or inside Docker container

Supported image formats:

```text
JPG, JPEG, PNG, WEBP
