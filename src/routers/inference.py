from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
from model_utils.model_loader import load_model

router = APIRouter()


# Define class_names
class_names = ['class1', 'class2', 'class3']


@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    model = load_model()

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image type.")

    # perform predictions
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))


    #NOTE: add logic to make an LLM call passing the result followed by a prompt

    return JSONResponse(content={
        "predicted_class":predicted_class,
        "confidence": confidence
        })
