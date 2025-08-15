from models.model_cnn import Model
import torch
from Datasets import transform, ImageDataset, DataLoader
import sys
TEST_FILE='/Users/kaankocaer/PycharmProjects/pythonProject2/datas/asl_alphabet_test'

model=Model()

model.load_state_dict(torch.load('model.pth'))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

model.eval()

test_data= ImageDataset(root_dir=TEST_FILE, transform=transform)
class_to_letter={idx: letter for letter, idx in test_data.class_to_idx.items()}
num_workers=1 if sys.platform == 'darwin' else 6

test_loader=DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

correct=0
total=0

with torch.no_grad():
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)

        outputs= model(images)

        _,predicted=torch.max(outputs,1)

        predicted_letters=[class_to_letter[idx.item()] for idx in predicted]

        total= labels.size(0)
        correct= (predicted == labels).sum().item()

        print(f"Predicted letters: {predicted_letters}, Actual labels: {[class_to_letter[labels.item()]]}")

accuracy=correct/total
print(f"Accuracy: {accuracy * 100:.2f}%")