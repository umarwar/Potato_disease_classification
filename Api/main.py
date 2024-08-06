from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the saved model
MODEL = tf.saved_model.load("U:/potato-disease-classification/Saved_models/1")

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# Define the class names
Class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def bytes_to_nparray(data) -> np.ndarray:
    array = np.array(Image.open(BytesIO(data)))
    return array

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    image = bytes_to_nparray(bytes)
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.signatures["serving_default"](tf.constant(img_batch, dtype=tf.float32))['dense_1']
    predicted_class = Class_name[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)