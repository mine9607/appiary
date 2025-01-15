from fastapi import FastAPI 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model_utils.model_loader import load_model 
from config import STATIC_DIR_PATH, TEMPLATE_DIR_PATH, PORT

# Import routers
from routers import uploads, processImage, inference, diagnosis

app = FastAPI()

# Mount the static directory for serving JS and other assets
app.mount("/static", StaticFiles(directory=STATIC_DIR_PATH), name="static")

# Initialize a global variable for the ML Model:
model = load_model()

# Load the saved model from disk on app startup:
@app.on_event("startup")
def load_model_event():
    global model
    from model_utils.model_loader import load_model
    model = load_model()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file from the template directory
    with open(TEMPLATE_DIR_PATH / "index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Include routers for modular functionality
app.include_router(uploads.router, prefix="", tags=["Uploads"])
app.include_router(processImage.router, prefix="", tags=["ProcessImages"])
app.include_router(inference.router, prefix="", tags=["Predictions"])
app.include_router(diagnosis.router, prefix="", tags=["Diagnosis"])


if __name__=="__main__":
            import uvicorn
            uvicorn.run("main:app", port=PORT, log_level="info")
