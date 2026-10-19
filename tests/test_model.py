"""
Unit tests for model architecture.
"""
import torch
from src.model.architecture import build_model


def test_model_output_shape():
    model = build_model(num_classes=10, pretrained=False)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (1, 10)


def test_model_accepts_batch():
    model = build_model(num_classes=10, pretrained=False)
    model.eval()
    dummy = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    assert output.shape == (8, 10)
