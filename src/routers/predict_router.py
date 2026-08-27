from fastapi import APIRouter, UploadFile, File, HTTPException

from src.services.prediction_service import predict_image


router = APIRouter(tags=["prediction"])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/webp"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Wrong file format. Allowed formats: jpg, jpeg, png, webp"
        )

    image_bytes = await file.read()
    predictions = predict_image(image_bytes)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "top_prediction": predictions[0],
        "predictions": predictions
    }
