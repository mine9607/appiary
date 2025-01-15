from fastapi import APIRouter, UploadFile, File
import os

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_DIR = "uploads"

## Create an "uploads" directory if doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def validate_file_type(file:UploadFile):
    if file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, or JPEG are allowed.")

router = APIRouter()

@router.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    """
    Receives a single file upload and validates file type.

    Saves image to folder 'uploads'
    """
    
    # Check for valid file type
    validate_file_type(file)
    
    # Create a filepath to save the file to
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename, "filepath":file_path}

