from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from src.Datasets import ImageDataset, transform
from src.models.model_cnn import Model
import sys
import torch.nn.functional as F
import torch
import json
from src.utils import _class_to_idx_json
TRAIN_FILE= '/datas/asl_alphabet_train'

full_data= ImageDataset(root_dir=TRAIN_FILE, transform=transform)

_class_to_idx_json(full_data,output_path='class_to_idx.json')
with open('class_to_idx',"w") as f:
    json.dump(full_data.class_to_idx, f)

indices= list(range(len(full_data)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_dataset=Subset(full_data, train_indices)
validation_dataset=Subset(full_data, val_indices)

is_mac = sys.platform == 'darwin'  # Check if the platform is macOS
num_workers=1 if is_mac else 6

train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
validation_loader=DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=num_workers)


CNN_model = Model()

num_epochs=20

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0

    for images, labels in data_loader:
        images= images.to(device)
        labels= labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted= torch.max(outputs, 1)
        correct+=(predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss=running_loss/ total
    accuracy=correct / total
    return epoch_loss, accuracy

def validate(model, data_loader, device):
    model.eval()
    running_loss=0
    correct=0
    total=0

    with torch.no_grad():
        for images,labels in data_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=F.cross_entropy(outputs, labels)

            running_loss+= loss.item() * images.size(0)


            _,predicted=torch.max(outputs,1)
            correct+= (predicted == labels).sum().item()
            total+= labels.size(0)

    epoch_loss= running_loss / total
    accuracy= correct / total
    return epoch_loss, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= CNN_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss=float('inf')

for epoch in range(num_epochs):
    train_loss, train_accuracy=train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy = validate(model, validation_loader, device)
    print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, ")
    print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_CNN_model.pth')
        print(f"Best model saved with validation loss: {best_val_loss:.4f} at epoch {epoch}")
