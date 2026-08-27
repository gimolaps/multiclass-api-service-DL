from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.routers.predict_router import router as predict_router


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Tiny ImageNet CNN Classifier")

app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

app.include_router(predict_router)


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "OK"}
