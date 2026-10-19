"""
Gradio interface for Corpus.
Upload an image and get the predicted body part with confidence.
"""
import base64
import io
import requests
import gradio as gr
from PIL import Image


API_URL = "http://localhost:8000/identify"


def predict(image: Image.Image):
    if image is None:
        return "No image provided.", "", ""
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = requests.post(API_URL, json={"image": encoded})
    if response.status_code != 200:
        return f"Error: {response.status_code}", "", ""

    data = response.json()
    return data["part"], f"{data['confidence']*100:.1f}%", data["description"]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Text(label="Body Part"),
        gr.Text(label="Confidence"),
        gr.Text(label="Description"),
    ],
    title="Corpus",
    description="Upload an image of a human body part to identify it.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
