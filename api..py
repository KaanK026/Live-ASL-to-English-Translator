# api.py
import asyncio
import uuid
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.utils import load_idx_to_class
from src.Datasets import transform
from torchvision import transforms
from textblob import TextBlob
import numpy as np
from PIL import Image
import cv2  # for client-uploaded video frames

# Load class mapping
idx_to_class = load_idx_to_class()

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available models
MODEL_PATHS = {
    "Resnet Model": "best_resnet_model.pth",
    "MobileNetV3 Model": "best_mobileNetV3_model.pth",
    "Custom CNN Model": "best_CNN_model.pth"
}

# Client sessions
client_data = {}  # key: client_id, value: {"latest_text": str, "model": TorchModel}

# Stability parameters
STABILITY_THRESHOLD = 5

# --- Utility functions ---
def predict_frame(model, frame, previous_prediction, same_count, text):
    """Run model inference on a single frame and update text buffer"""
    frame = transforms.ToPILImage()(frame)
    frame = transform(frame)
    input_tensor = frame.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()

    predicted_class = idx_to_class.get(str(predicted), "Unknown")

    # Stability logic
    if predicted_class == previous_prediction:
        same_count += 1
    else:
        previous_prediction = predicted_class
        same_count = 0

    if same_count >= STABILITY_THRESHOLD:
        if predicted_class == "space":
            if text and not text.endswith(" "):
                last_word = text.split()[-1]
                corrected = str(TextBlob(last_word).correct())
                text = text[:-(len(last_word))] + corrected
                text += " "
        elif predicted_class == "del":
            text = text[:-1]
        elif predicted_class != "nothing":
            text += predicted_class

    return text, previous_prediction, same_count

# --- Endpoints ---
@app.post("/set_model/{model_name}")
async def set_model(model_name: str, client_id: str):
    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=404, detail="Model not found")
    try:
        loaded_model = torch.load(MODEL_PATHS[model_name], map_location=device)
        loaded_model.to(device)
        loaded_model.eval()
        if client_id not in client_data:
            client_data[client_id] = {"latest_text": "", "model": loaded_model}
        else:
            client_data[client_id]["model"] = loaded_model
        return {"status": "success", "current_model.": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

# --- WebSocket for video frames ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    client_data[client_id] = {"latest_text": "", "model": None}

    previous_prediction = None
    same_count = 0
    text = ""

    try:
        while True:
            data = await websocket.receive_bytes()
            # Convert bytes to numpy array (assuming client sends raw frames)
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            client_model = client_data[client_id]["model"]
            if client_model is None:
                await asyncio.sleep(0.01)
                continue

            text, previous_prediction, same_count = predict_frame(
                client_model, frame, previous_prediction, same_count, text
            )

            client_data[client_id]["latest_text"] = text
            await websocket.send_json({"text": text})
    except WebSocketDisconnect:
        client_data.pop(client_id, None)
