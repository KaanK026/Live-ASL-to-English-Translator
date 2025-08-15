import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # Replace the final layer with your number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

