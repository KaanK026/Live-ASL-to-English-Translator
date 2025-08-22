from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import sys
from src.models.model_resnet import get_resnet18
from src.models.model_mobileNetV3 import get_mobileNetV3
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



full_dataset=ImageFolder(root='/Users/kaankocaer/PycharmProjects/pythonProject2/datas/asl_alphabet_train/asl_alphabet_train', transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

num_workers= 1 if sys.platform=='darwin' else 6

train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

num_classes= len(train_dataset.dataset.classes)
model=get_resnet18(num_classes=num_classes, pretrained=True)
#get_mobileNetV3(num_classes=num_classes, pretrained=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)

optimizer=optim.Adam(model.parameters(), lr=0.001)
scheduler=StepLR(optimizer, step_size=3, gamma=0.95)
loss_function=nn.CrossEntropyLoss()


num_epochs= 20


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    running_loss=0.0
    correct=0
    total=0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss= running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

def validation(model,data_loader, device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    accuracy = correct / total
    return epoch_loss, accuracy

bes_val_loss=float('inf')

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy = validation(model, val_loader, device)

    scheduler.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the model if validation loss improves
    if val_loss < bes_val_loss:
        bes_val_loss = val_loss
        torch.save(model.state_dict(), 'best_resnet_model.pth')
        #torch.save(model.state.dict(), 'best_mobileNetV3_model.pth')
        print("Model saved!")