from src.models.model_resnet import get_resnet18
from src.Datasets import transform, ImageDataset, DataLoader
import torch
import sys
import os

# Paths
TEST_FILE = r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\datas\asl_alphabet_test\asl_alphabet_test"
MODEL_PATH = r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\best_resnet_model.pth"
TRAIN_CLASS_ORDER_FILE = r"C:\Users\PC\PycharmProjects\Live-ASL-to-English-Translator\train_class_order.txt"
# Optional: a text file containing class names in the exact order used during training, one per line

# Load test dataset
test_data = ImageDataset(root_dir=TEST_FILE, transform=transform)
num_workers = 1 if sys.platform == 'darwin' else 6
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

# Load training class order to align indices
if os.path.exists(TRAIN_CLASS_ORDER_FILE):
    with open(TRAIN_CLASS_ORDER_FILE, 'r') as f:
        train_letters = [line.strip() for line in f.readlines()]
else:
    # Fallback: assume alphabetical + del, space, nothing
    train_letters = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "del", "space", "nothing"
    ]

# Map training indices to test labels
# Get dataset mapping
dataset_class_to_idx = test_data.class_to_idx
# Create mapping from training index -> dataset index
# Filter training letters to only those present in test dataset
train_idx_to_test_idx = []
filtered_train_letters = []

for letter in train_letters:
    if letter in dataset_class_to_idx:
        train_idx_to_test_idx.append(dataset_class_to_idx[letter])
        filtered_train_letters.append(letter)

train_letters = filtered_train_letters


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_resnet18(num_classes=29, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0

if __name__=="__main__":
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Map predicted training index to dataset index
            predicted_dataset_idx = torch.tensor([train_idx_to_test_idx[p.item()] for p in predicted]).to(device)

            # Compare
            correct += (predicted_dataset_idx == labels).sum().item()
            total += labels.size(0)

            # Optional: print predictions
            predicted_letters = [train_letters[p.item()] for p in predicted]
            actual_letters = [list(dataset_class_to_idx.keys())[list(dataset_class_to_idx.values()).index(l.item())] for l in labels]
            print(f"Predicted: {predicted_letters}, Actual: {actual_letters}")

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
