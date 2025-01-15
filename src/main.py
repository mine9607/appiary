from fastapi import FastAPI 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tensorflow as tf
import os

# Import routers
from routers import uploads, processImage, inference, diagnosis

app = FastAPI()

# Mount the static directory for serving JS and other assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define the root route to serve the HTML form
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

# Initialize a global variable for the ML Model:
model = None

# Load the saved model from disk on app startup:
@app.on_event("startup")
def load_model():
    global model
    model_path = Path(__file__).parent / "models" / "my_model"
    if model_path.is_dir():
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully.")
    else:
        print(f"Model directory {model_path} does not exist.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file
    with open(os.path.join(templates_dir, "index.html")) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Include routers for modular functionality
app.include_router(uploads.router, prefix="", tags=["Uploads"])
#app.include_router(processImage.router)
app.include_router(inference.router, prefix="", tags=["Predictions"])
app.include_router(diagnosis.router, prefix="", tags=["Diagnosis"])

'''
@app.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Handles multiple file uploads.
    - `files`: A list of uploaded files.
    """
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save each file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        saved_files.append({"filename": file.filename, "filepath":file_path})
    
    return {"files": saved_files}
'''

if __name__=="__main__":
            import uvicorn
            uvicorn.run(app, host="127.0.0.1", port=8000)
