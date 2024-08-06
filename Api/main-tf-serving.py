from fastapi import FastAPI, File, UploadFile
import requests
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

app = FastAPI()

endpoint = "http://localhost:8502/v1/models/Potatoes_model:predict"

# Define the class names
Class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def bytes_to_nparray(data) -> np.ndarray:
    array = np.array(Image.open(BytesIO(data)))
    return array

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image = bytes_to_nparray(bytes)
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    
    # Debugging information
    if response.status_code != 200:
        return {"error": "Failed to get a response from TensorFlow Serving", "status_code": response.status_code, "content": response.content}
    
    response_data = response.json()
    if "predictions" not in response_data:
        return {"error": "'predictions' key not found in response", "response_data": response_data}
    
    prediction = np.array(response_data["predictions"][0])
    predicted_class = Class_name[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


# Start TensorFlow Serving with Your Model using this command
# docker run -t --rm -p 8502:8502 -v U:/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8502 --model_config_file=/potato-disease-classification/models.config