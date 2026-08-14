import io
import base64
from html import escape
from functools import lru_cache

import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image

from src.config import CHECKPOINTS_DIR, DEVICE
from src.model import CNN
from src.dataset import (
    get_val_and_test_transform,
    get_train_val_dataloaders,
    load_wnid_to_name,
)


app = FastAPI(title="Tiny ImageNet CNN Classifier")


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
    train_dataset, _, _, _ = get_train_val_dataloaders()

    idx_to_wnid = train_dataset.classes
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
            "confidence": confidence
        })

    return results


def image_to_base64(image_bytes, content_type):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tiny ImageNet Classifier</title>
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                min-height: 100vh;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #111827, #1f2937, #374151);
                color: #111827;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .container {
                width: 100%;
                max-width: 620px;
                background: #ffffff;
                padding: 36px;
                border-radius: 18px;
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
            }

            h1 {
                margin: 0 0 10px;
                font-size: 30px;
                text-align: center;
            }

            .subtitle {
                text-align: center;
                color: #6b7280;
                margin-bottom: 30px;
                font-size: 15px;
            }

            .upload-box {
                border: 2px dashed #9ca3af;
                border-radius: 14px;
                padding: 28px;
                text-align: center;
                background: #f9fafb;
            }

            input[type="file"] {
                margin: 18px 0;
                width: 100%;
                padding: 12px;
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 10px;
            }

            button {
                width: 100%;
                padding: 14px 20px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            .hint {
                margin-top: 15px;
                color: #6b7280;
                font-size: 13px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tiny ImageNet Classifier</h1>
            <p class="subtitle">Upload an image and the CNN model will predict the class.</p>

            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="upload-box">
                    <b>Select image file</b>
                    <input type="file" name="file" accept=".jpg,.jpeg,.png,.webp" required>
                    <button type="submit">Predict class</button>
                </div>
            </form>

            <p class="hint">Supported formats: JPG, JPEG, PNG, WEBP</p>
        </div>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wrong file format</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #111827;
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-height: 100vh;
                }

                .box {
                    background: #1f2937;
                    padding: 30px;
                    border-radius: 14px;
                    text-align: center;
                }

                a {
                    color: #93c5fd;
                }
            </style>
        </head>
        <body>
            <div class="box">
                <h2>Wrong file format</h2>
                <p>Allowed formats: jpg, jpeg, png, webp</p>
                <a href="/">Back</a>
            </div>
        </body>
        </html>
        """

    image_bytes = await file.read()
    results = predict_image(image_bytes)
    image_preview = image_to_base64(image_bytes, file.content_type)

    top_prediction = results[0]

    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                min-height: 100vh;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #111827, #1f2937, #374151);
                color: #111827;
                padding: 40px 20px;
            }}

            .container {{
                max-width: 900px;
                margin: auto;
                background: white;
                padding: 34px;
                border-radius: 18px;
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
            }}

            h1 {{
                margin-top: 0;
                text-align: center;
                font-size: 30px;
            }}

            .content {{
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 28px;
                align-items: start;
            }}

            .image-card {{
                background: #f3f4f6;
                padding: 16px;
                border-radius: 14px;
                text-align: center;
            }}

            .image-card img {{
                max-width: 100%;
                border-radius: 12px;
                border: 1px solid #d1d5db;
            }}

            .filename {{
                margin-top: 12px;
                font-size: 13px;
                color: #6b7280;
                word-break: break-all;
            }}

            .main-result {{
                padding: 20px;
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 14px;
                margin-bottom: 22px;
            }}

            .main-result h2 {{
                margin-top: 0;
                color: #1d4ed8;
            }}

            .class-name {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }}

            .confidence {{
                font-size: 18px;
                color: #374151;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 12px;
                overflow: hidden;
            }}

            th {{
                background: #f3f4f6;
                color: #374151;
            }}

            th, td {{
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
                text-align: left;
            }}

            tr:last-child td {{
                border-bottom: none;
            }}

            .back {{
                display: inline-block;
                margin-top: 26px;
                padding: 12px 18px;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                font-weight: bold;
            }}

            .back:hover {{
                background: #1d4ed8;
            }}

            @media (max-width: 760px) {{
                .content {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Result</h1>

            <div class="content">
                <div class="image-card">
                    <img src="{image_preview}" alt="Uploaded image">
                    <div class="filename">{escape(file.filename)}</div>
                </div>

                <div>
                    <div class="main-result">
                        <h2>Main prediction</h2>
                        <div class="class-name">{escape(top_prediction["class_name"])}</div>
                        <div class="confidence">Confidence: {top_prediction["confidence"]:.2f}%</div>
                    </div>

                    <h2>Top 5 predictions</h2>

                    <table>
                        <tr>
                            <th>#</th>
                            <th>Class</th>
                            <th>Confidence</th>
                        </tr>
    """

    for i, result in enumerate(results, start=1):
        result_html += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{escape(result["class_name"])}</td>
                            <td>{result["confidence"]:.2f}%</td>
                        </tr>
        """

    result_html += """
                    </table>

                    <a class="back" href="/">Upload another image</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return result_html
