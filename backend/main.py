from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/model.h5")

# =========================
# PREPROCESS
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# ROUTE
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