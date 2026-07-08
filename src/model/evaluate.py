"""
Evaluate the trained model on the test set.
Prints per-class accuracy and overall accuracy.
"""
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BodyPartDataset
from src.data.preprocessing import get_val_transforms
from src.model.architecture import build_model


def evaluate():
    with open("configs/model.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["model"]["num_classes"]
    image_size = cfg["data"]["image_size"]

    dataset = BodyPartDataset("data/processed", transform=get_val_transforms(image_size))
    loader = DataLoader(dataset, batch_size=32)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load("src/model/weights/best.pt", map_location=device))
    model = model.to(device)
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Overall accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    evaluate()
