"""
FastAPI application — POST /identify endpoint.
Accepts a base64-encoded image and returns the predicted body part.
"""
import base64
import io
import json
import os

import torch
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from src.model.architecture import build_model

app = FastAPI(title="Corpus API", description="Human body part identification from images.")

with open("configs/model.yaml") as f:
    cfg = yaml.safe_load(f)

# Load class descriptions
with open("src/model/classes.json") as f:
    CLASS_INFO = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=cfg["model"]["num_classes"], pretrained=False)
model.load_state_dict(torch.load(os.getenv("MODEL_WEIGHTS_PATH", "src/model/weights/best.pt"), map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class IdentifyRequest(BaseModel):
    image: str  # base64-encoded image


class IdentifyResponse(BaseModel):
    part: str
    confidence: float
    description: str


@app.post("/identify", response_model=IdentifyResponse)
def identify(request: IdentifyRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    class_name = list(CLASS_INFO.keys())[predicted.item()]
    description = CLASS_INFO[class_name]

    return IdentifyResponse(
        part=class_name,
        confidence=round(confidence.item(), 4),
        description=description,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
