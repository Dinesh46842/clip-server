from fastapi import FastAPI, UploadFile, File
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

app = FastAPI()

# Load CLIP model and processor from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Preprocess and get features
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        
    return {"embedding": features[0].tolist()}
