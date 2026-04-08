from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
import requests

app = FastAPI()

# =========================
# CORS (Allow frontend)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL DOWNLOAD (ROBUST)
# =========================
MODEL_PATH = "model/model.h5"

def download_model():
    print("⬇️ Downloading model...")

    url = "https://drive.google.com/uc?export=download&id=1wUz7ZZS-b0m7bqFYuMkENedUYJ1BiufI"

    session = requests.Session()
    response = session.get(url, stream=True)

    # Handle large file warning from Google
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url_confirm = url + "&confirm=" + value
            response = session.get(url_confirm, stream=True)

    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded!")

# Download if not exists
if not os.path.exists(MODEL_PATH):
    download_model()

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "Skin Cancer API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        return {
            "result": "Malignant",
            "confidence": float(prediction)
        }
    else:
        return {
            "result": "Benign",
            "confidence": float(1 - prediction)
        }
