import cv2
import torch
import numpy as np
from torchvision import transforms
from textblob import TextBlob
from src.models.model_resnet import get_resnet18
import pyvirtualcam
import mediapipe as mp


# ---------------------------
# Constants & Configurations
# ---------------------------

LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "space", "nothing"
]
idx_to_class = {i: letter for i, letter in enumerate(LETTERS)}

STABILITY_THRESHOLD = 25   # frames to wait before accepting prediction
CONF_THRESHOLD = 0.9       # minimum confidence required

transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Mediapipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


# ---------------------------
# Helper Functions
# ---------------------------

def crop_hand(frame, hand_landmarks):
    """Crop the region of the detected hand."""
    h, w, _ = frame.shape
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    x_min = int(max(min(x_coords) * w - 20, 0))
    x_max = int(min(max(x_coords) * w + 20, w))
    y_min = int(max(min(y_coords) * h - 20, 0))
    y_max = int(min(max(y_coords) * h + 20, h))

    return frame[y_min:y_max, x_min:x_max]


def start_virtual_cam(model, device):
    """Starts the virtual camera and processes ASL predictions."""
    model.to(device)
    model.eval()

    text = ""
    previous_prediction = None
    same_count = 0

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
        print(f"[INFO] Virtual camera started ({width}x{height} at {fps}fps)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                predicted_class = 'nothing'

                # If hand detected, crop and predict
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_crop = crop_hand(frame_rgb, hand_landmarks)

                    if hand_crop.size > 0:
                        pil_frame = transforms.ToPILImage()(hand_crop)
                        input_tensor = transform_inference(pil_frame).unsqueeze(0).to(device)

                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            conf, pred = torch.max(probs, 1)

                            if conf.item() >= CONF_THRESHOLD:
                                predicted_class = idx_to_class.get(pred.item(), 'nothing')

                # Stability check for letters
                if predicted_class not in ['nothing', 'del', 'space']:
                    if predicted_class == previous_prediction:
                        same_count += 1
                    else:
                        previous_prediction = predicted_class
                        same_count = 1

                    if same_count >= STABILITY_THRESHOLD:
                        text += predicted_class
                        same_count = 0

                # Handle special commands
                if predicted_class == 'space' and text and not text.endswith(" "):
                    last_word = text.split()[-1]
                    corrected = str(TextBlob(last_word).correct())
                    text = text[:-(len(last_word))] + corrected + " "
                elif predicted_class == 'del' and text:
                    text = text[:-1]

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Overlay predicted letter and full text
                cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Text: {text}', (10, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Send frame to virtual camera
                cam.send(cv2.flip(frame, 1))
                cam.sleep_until_next_frame()

        except KeyboardInterrupt:
            print("[INFO] Exiting...")

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=29, pretrained=False)
    model.load_state_dict(
        torch.load(
            r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\best_resnet_model.pth",
            map_location=device
        ),
        strict=True
    )
    start_virtual_cam(model, device)
