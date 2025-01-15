from fastapi import FastAPI 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Import routers
from routers import uploads, processImage, inference

app = FastAPI()

# Mount the static directory for serving JS and other assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define the root route to serve the HTML form
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file
    with open(os.path.join(templates_dir, "index.html")) as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Include routers for modular functionality
app.include_router(uploads.router, prefix="", tags=["Uploads"])
#app.include_router(processImage.router)
#app.include_router(inference.router)

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
