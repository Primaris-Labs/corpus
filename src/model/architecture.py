"""
Model architecture - MobileNetV2 with a custom classification head.
Swap to resnet50 by changing the base model.
"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
