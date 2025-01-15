from fastapi import APIRouter, UploadFile, File
import os

@router.post("/diagnosis")
async def diagnose_image(file:UploadFile = File(...)):
    """
    Accepts the results of the ML Prediction on diseases and passes to llm call then returns the result for display on the front-end
    """
