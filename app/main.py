from fastapi import FastAPI, UploadFile, File
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import logging

app = FastAPI()

# Suppress warnings if running in limited memory environment
logging.getLogger("transformers").setLevel(logging.ERROR)

# Force CPU + float16 to save memory
device = torch.device("cpu")
model = CLIPModel.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    torch_dtype=torch.float16
).to(device)

processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)

        # Convert to list of floats for JSON
        return {"embedding": features[0].cpu().tolist()}

    except Exception as e:
        return {"error": str(e)}
