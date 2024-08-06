from fastapi import FastAPI, File, UploadFile
import requests
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

# Create a FastAPI app instance
app = FastAPI()

# Define the TensorFlow Serving endpoint URL
endpoint = "http://localhost:8502/v1/models/Potatoes_model:predict"

# Define the class names
Class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Function to convert byte data to a NumPy array
def bytes_to_nparray(data) -> np.ndarray:
    array = np.array(Image.open(BytesIO(data)))
    return array

# Define the '/predict' endpoint with a POST method
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    bytes = await file.read()
    # Convert the byte data to a NumPy array
    image = bytes_to_nparray(bytes)
    # Expand dimensions to create a batch of one image
    img_batch = np.expand_dims(image, 0)

    # Prepare the JSON payload for the TensorFlow Serving request
    json_data = {
        "instances": img_batch.tolist()
    }

    # Send the request to the TensorFlow Serving endpoint
    response = requests.post(endpoint, json=json_data)
    
    # Debugging information
    if response.status_code != 200:
        return {"error": "Failed to get a response from TensorFlow Serving", "status_code": response.status_code, "content": response.content}
    
    response_data = response.json()
    if "predictions" not in response_data:
        return {"error": "'predictions' key not found in response", "response_data": response_data}
    
    # Extract the predictions from the response data
    prediction = np.array(response_data["predictions"][0])

    # Determine the predicted class by finding the index of the maximum value in the prediction array
    predicted_class = Class_name[np.argmax(prediction)]
    
    # Get the confidence score of the prediction
    confidence = np.max(prediction)

    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


# Start TensorFlow Serving with Your Model using this command
# docker run -t --rm -p 8502:8502 -v U:/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8502 --model_config_file=/potato-disease-classification/models.config
