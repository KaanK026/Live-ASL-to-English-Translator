import subprocess
import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to hold the running subprocess
camera_process = None

class StartCamRequest(BaseModel):
    model_name: str

@app.post("/start-cam")
async def start_cam_endpoint(request: StartCamRequest):
    global camera_process
    if camera_process and camera_process.poll() is None:
        raise HTTPException(status_code=400, detail="Camera is already running.")

    print("\n--- [API] ATTEMPTING TO START CAMERA WORKER ---")

    try:
        # --- 1. VERIFY ALL PATHS ---
        # Use sys.executable, which is the most reliable way to get the current python interpreter
        python_executable = sys.executable
        worker_script_path = os.path.abspath("camera_worker.py")
        model_path = os.path.abspath(r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\best_resnet_model.pth")

        print(f"[API] Python Executable Path: {python_executable}")
        print(f"[API] Worker Script Path:     {worker_script_path}")
        print(f"[API] Model Path:             {model_path}")

        # --- 2. CHECK IF FILES EXIST ---
        if not os.path.exists(python_executable):
            print("[API] FATAL ERROR: Python executable not found at the path above.")
            raise HTTPException(status_code=500, detail="Server misconfiguration: Python executable not found.")
        if not os.path.exists(worker_script_path):
            print("[API] FATAL ERROR: camera_worker.py not found at the path above.")
            raise HTTPException(status_code=500, detail="Server misconfiguration: camera_worker.py not found.")

        # --- 3. CONSTRUCT COMMAND AND LAUNCH ---
        command = [python_executable, worker_script_path, model_path, request.model_name]
        print(f"[API] Executing command: {' '.join(command)}")

        # Launch the subprocess and CAPTURE its output streams
        camera_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Decodes the output streams as text
        )

        # --- 4. CHECK FOR INSTANT CRASH ---
        # Wait a very short moment to see if the process terminated immediately
        time.sleep(1.0)
        if camera_process.poll() is not None:
            # The process has terminated. Read the error.
            print("[API] FATAL: Subprocess terminated immediately after launch.")
            stdout_output, stderr_output = camera_process.communicate()
            print("--- Captured STDOUT from worker ---")
            print(stdout_output)
            print("--- Captured STDERR from worker ---")
            print(stderr_output)
            print("---------------------------------")
            raise HTTPException(status_code=500, detail=f"Camera worker failed to start. Check API console for errors. Error: {stderr_output}")

        print(f"[API] SUCCESS: Camera worker started successfully with PID: {camera_process.pid}")
        return {"message": "Virtual camera started."}

    except Exception as e:
        print(f"[API] An unexpected error occurred in the endpoint itself: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-cam")
async def stop_cam_endpoint():
    # This function remains the same
    global camera_process
    if camera_process is None or camera_process.poll() is not None:
        raise HTTPException(status_code=400, detail="Camera is not running.")
    print(f"API: Terminating camera worker PID: {camera_process.pid}")
    camera_process.terminate()
    camera_process.wait(timeout=5)
    camera_process = None
    return {"message": "Virtual camera stopped."}

@app.get("/status")
async def get_status():
    # This function remains the same
    if camera_process and camera_process.poll() is None:
        return {"status": "running"}
    return {"status": "stopped"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

### Your Action Plan

