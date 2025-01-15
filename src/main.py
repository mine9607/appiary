from fastapi import FastAPI 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model_utils import load_model
from pathlib import Path
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
def load_model_event():
    global model
    model = load_model()



@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file
    with open(os.path.join(templates_dir, "index.html")) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Include routers for modular functionality
app.include_router(uploads.router, prefix="", tags=["Uploads"])
app.include_router(processImage.router, prefix="", tags=["ProcessImages"])
app.include_router(inference.router, prefix="", tags=["Predictions"])
app.include_router(diagnosis.router, prefix="", tags=["Diagnosis"])


if __name__=="__main__":
            import uvicorn
            uvicorn.run("main:app", port=8000, log_level="info")
