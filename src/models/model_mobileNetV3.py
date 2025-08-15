import torch.nn as nn
from torchvision.models import mobilenet_v3_large


def get_mobileNetV3(num_classes, pretrained=True):
    model = mobilenet_v3_large(pretrained=pretrained)
    # Replace the final classifier layer with your number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model