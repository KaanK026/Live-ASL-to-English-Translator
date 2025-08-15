from src.infer_live import start_process
import torch

model_type=None#TODO frontend
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

if model_type == 'resnet':
    model = torch.load('best_resnet_model.pth', map_location=device)
elif model_type == 'cnn':
    model = torch.load('best_CNN_model.pth', map_location=device)
elif model_type=='mobileNetV3':
    model= torch.load('best_mobileNetV3_model.pth', map_location=device)

start_process(model, device)