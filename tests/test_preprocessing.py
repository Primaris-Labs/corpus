"""
Unit tests for image preprocessing.
"""
import torch
from PIL import Image
from src.data.preprocessing import get_train_transforms, get_val_transforms


def test_train_transforms_output_shape():
    img = Image.new("RGB", (300, 400))
    transform = get_train_transforms(image_size=224)
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_output_shape():
    img = Image.new("RGB", (100, 100))
    transform = get_val_transforms(image_size=224)
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_output_is_tensor():
    img = Image.new("RGB", (224, 224))
    transform = get_val_transforms()
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
