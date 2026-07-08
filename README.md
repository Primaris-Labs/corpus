# Corpus

A computer vision model that identifies human body parts from images.

Built by [Primaris Labs](https://github.com/Primaris-Labs) as a first open-source ML project.

---

## What it does

Send an image of a human body part or bone. Corpus returns:
- The name of the part
- A confidence score
- A short anatomical description

---

## Stack

- PyTorch + torchvision — model training and inference
- MobileNetV2 — pre-trained backbone (transfer learning)
- FastAPI — REST API
- Gradio — browser interface
- pytest + GitHub Actions — tests and CI/CD

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Primaris-Labs/corpus.git
cd corpus
```

### 2. Open in VS Code Dev Container

Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, then:

```
Ctrl+Shift+P > Dev Containers: Reopen in Container
```

Dependencies install automatically.

### 3. Set up environment

```bash
cp .env.example .env
# Fill in your Kaggle credentials
```

### 4. Download the dataset

```bash
python data/download.py
```

### 5. Train the model

```bash
python src/model/train.py
```

Or open `notebooks/explore.ipynb` on Google Colab for GPU training.

### 6. Run the API and UI

```bash
docker-compose up
```

- API: http://localhost:8000
- UI: http://localhost:7860

---

## API

### POST /identify

**Request**
```json
{ "image": "base64_encoded_image_string" }
```

**Response**
```json
{
  "part": "Femur",
  "confidence": 0.94,
  "description": "Longest bone in the human body, located in the thigh."
}
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

MIT
