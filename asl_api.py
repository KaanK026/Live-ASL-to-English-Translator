# asl_api.py

import multiprocessing
import logging
from logging.handlers import QueueHandler
from typing import Optional

from fastapi import FastAPI, HTTPException,Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.models.model_resnet import get_resnet18
from asl_worker import start_virtual_cam
import torch
import os
# ---------------------------
# FastAPI Setup
# ---------------------------

app = FastAPI(title="ASL Virtual Camera API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Global Variables
# ---------------------------

camera_process = None
log_listener_process = None
log_queue = None


# ---------------------------
# Request Models
# ---------------------------

class StartCamRequest(BaseModel):
    model_path: str
    device_preference: Optional[str] = "auto"  # "auto", "cpu", or "cuda"
    camera_index: Optional[int] = 0


# ---------------------------
# Logger Process
# ---------------------------

def logger_process(queue: multiprocessing.Queue):
    """Listener process to write logs from camera process."""
    handler = logging.FileHandler('camera_log.txt', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger('camera_logger')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger.handle(record)
        except Exception:
            import sys, traceback
            traceback.print_exc(file=sys.stderr)


# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/start-cam")
async def start_camera(request: StartCamRequest=Body(...)):
    global camera_process, log_listener_process, log_queue

    if camera_process and camera_process.is_alive():
        raise HTTPException(status_code=400, detail="Camera is already running.")

    log_queue = multiprocessing.Queue()
    log_listener_process = multiprocessing.Process(target=logger_process, args=(log_queue,))
    log_listener_process.start()

    # Device selection
    if request.device_preference.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    if os.path.basename(request.model_path) == "Resnet":
        model = get_resnet18(num_classes=29, pretrained=False)
        model.load_state_dict(
            torch.load(
                r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\best_resnet_model.pth",
                map_location=device
            ),
            strict=True
        )

    # Start camera process
    ctx = multiprocessing.get_context("spawn")
    camera_process = ctx.Process(target=start_virtual_cam, args=(model, device))
    camera_process.start()

    return {"message": "Virtual camera started."}


@app.post("/stop-cam")
async def stop_camera():
    global camera_process, log_listener_process, log_queue

    if not camera_process or not camera_process.is_alive():
        raise HTTPException(status_code=400, detail="Camera is not running.")

    camera_process.terminate()
    camera_process.join()

    if log_queue:
        log_queue.put(None)  # Stop logging
    if log_listener_process:
        log_listener_process.join()

    return {"message": "Virtual camera stopped."}


@app.get("/status")
async def get_status():
    if camera_process and camera_process.is_alive():
        return {"status": "running"}
    return {"status": "stopped"}


@app.get("/logs")
async def get_logs(tail: int = 2000):
    try:
        with open("camera_log.txt", "r") as f:
            lines = f.read()[-tail:]
        return {"logs": lines}
    except FileNotFoundError:
        return {"logs": ""}


# ---------------------------
# Main Entry
# ---------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=8000)
