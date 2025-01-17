from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from routers.processImage import process_image
from pathlib import Path
import os

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"

## Create an "uploads" directory if doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_file_type(file:UploadFile):
    if file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, or JPEG are allowed.")
    else:
        print(f"File {file} received and ready for processing.")

router = APIRouter()

@router.post("/upload")
async def upload_file(request:Request, file:UploadFile = File(...)):
    """
    Receives a single file upload and validates file type.

    Saves image to folder 'uploads'
    """
    
    # Check for valid file type
    validate_file_type(file)
    
    # Create a filepath to save the file to
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # read the file contents as bytes
    image_bytes = await file.read()

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(image_bytes)

    result = await process_image(image_bytes, request)
    # print(f"Results from the /process-image route: {result.shape}")
    print(f"RECEIVED RESULT FROM PROCESSING: {result['message']}")

    response = {"filename":file.filename, "filepath": file_path, "result":result['message']}

    print(response)

    return JSONResponse(content=response) 

