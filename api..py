from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from typing import Optional

app = FastAPI()

# Path to your ASL virtual camera script
SCRIPT_PATH = r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\asl_virtual_cam.py"

# Track the running virtual cam process
running_process: Optional[subprocess.Popen] = None


class ModelRequest(BaseModel):
    model_name: str  # "Resnet" | "MobileNet" | "CustomCNN"


@app.post("/start-cam")
def start_cam(request: ModelRequest):
    global running_process

    # Check if script exists
    if not os.path.exists(SCRIPT_PATH):
        raise HTTPException(status_code=500, detail="ASL virtual cam script not found.")

    # If already running, return status
    if running_process is not None and running_process.poll() is None:
        return {
            "status": "already_running",
            "message": "Virtual camera is already active."
        }

    # Map model names to actual model files
    model_map = {
        "Resnet": "best_resnet_model.pth",
        "MobileNet": "best_mobilenet_model.pth",
        "CustomCNN": "best_customcnn_model.pth"
    }
    model_file = model_map.get(request.model_name)

    # Validate model choice
    if model_file is None:
        raise HTTPException(status_code=400, detail="Invalid model name provided.")

    model_path = os.path.join(os.path.dirname(SCRIPT_PATH), model_file)

    # Check if model file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail=f"Model file '{model_file}' not found.")

    try:
        # Start the virtual cam process (non-blocking)
        running_process = subprocess.Popen(
            ["python", SCRIPT_PATH, model_file],
            cwd=os.path.dirname(SCRIPT_PATH),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start virtual cam: {str(e)}")

    return {
        "status": "started",
        "model": request.model_name,
        "message": f"Virtual camera started using {request.model_name} model."
    }


@app.post("/stop-cam")
def stop_cam():
    global running_process

    if running_process is None or running_process.poll() is not None:
        return {"status": "not_running", "message": "Virtual camera is not running."}

    running_process.terminate()
    running_process = None

    return {"status": "stopped", "message": "Virtual camera has been stopped."}


@app.get("/status")
def status():
    """Check if the virtual camera process is running"""
    global running_process
    is_running = running_process is not None and running_process.poll() is None
    return {
        "status": "running" if is_running else "stopped"
    }
