from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.images = []  # List to hold image file paths
        self.labels= []
        self.transform = transform

        # Traverse the root directory to find all image files
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg'):
                        self.images.append(os.path.join(folder_path, filename))
                        self.labels.append(folder)
        self.class_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}




    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.class_to_idx[self.labels[idx]]
        try:
            image=Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image= Image.new('RGB', (28, 28))#returning black placeholder
        if self.transform:
            image = self.transform(image)
        return image, label

transform=transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



