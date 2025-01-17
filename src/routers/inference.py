from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from routers.diagnosis import diagnose_image
from pathlib import Path
from config import MODEL_DATA_DIR
import tensorflow as tf
import numpy as np

router = APIRouter()

# Function to determine the classes from the MODEL_DATA_DIR folder
def get_classes():
  ''' Determine the classes from subdirectory structure'''
  classes = []
  dir_items = Path(MODEL_DATA_DIR).iterdir()
  for item in dir_items:
    if item.is_dir():
      classes.append(item.name)

  return classes
# Define class_names
class_names = get_classes()

def get_model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return model

@router.post("/predict")
async def predict(processed_image, request:Request):
    print(f"CLASSNAMES: {class_names}")

    model = request.app.state.model 

    # perform predictions
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    print(f"Predictions: {predictions}")
    print(f"Predicted Class: {predicted_class}")


    #NOTE: add logic to make an LLM call passing the result followed by a prompt
    result = await diagnose_image(predicted_class)
    # print(f"RECEIVED RESULT FROM DIAGNOSIS: {result['message']}")

    return result