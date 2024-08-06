# Potato Disease Classification
This project aims to classify potato plant diseases by leverages deep learning techniques using Convolutional Neural Networks (CNNs) and TensorFlow to achieve accurate classification. It leverages TensorFlow for model training and FastAPI for serving the model via a RESTful API. The model is trained to classify images of potato leaves into three categories: Early Blight, Late Blight, and Healthy.

# Dataset
The dataset is sourced from the PlantVillage dataset and includes images of potato leaves classified into three categories:
<ul>
  <li>Potato___Early_blight</li>
  <li>Potato___healthy</li>
  <li>Potato___Late_blight</li>
</ul>

# Training
The model is trained using the following steps:
<ol>
  <li>Load the dataset and split it into training, validation, and test sets.</li>
  <li>Apply data augmentation to the training data to improve generalization.</li>
  <li>Compile the model with Adam optimizer and Sparse Categorical Crossentropy loss function.</li>
  <li>Train the model for 30 epochs, monitoring training and validation accuracy and loss.</li>
</ol>

# Inference
For inference, a FastAPI application is created to handle image uploads and interact with the TensorFlow Serving endpoint. The application receives an image, processes it, and sends a request to the TensorFlow Serving model for prediction.

# Dependencies
<ul>
  <li>TensorFlow</li>
  <li>FastAPI</li>
  <li>Uvicorn</li>
  <li>Requests</li>
  <li>PIL (Pillow)</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
</ul>
Ensure all dependencies are installed using pip:

```
pip install tensorflow fastapi uvicorn requests pillow numpy matplotlib
              OR
// navigate to the Api directory
cd Api
//install required dependencies through txt file
pip install -r requirements.txt 
```

# Installation and Setup
<ul>
  <li>Install Docker: TensorFlow Serving is run using Docker. Ensure Docker is installed on your system.</li>
  <li>Start TensorFlow Serving: </li>
</ul>

```
docker run -t --rm -p 8502:8502 -v U:/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8502 --model_config_file=/potato-d      disease-classification/models.config
```
<ul>
  <li>Create a `models.config` file to configure TensorFlow Serving. This file tells TensorFlow Serving to load a model named Potatoes_model from the specified base_path </li>
</ul>

# FastAPI Usage
<ul>
  <li>Run TensorFlow Serving: Ensure TensorFlow Serving is running using the Docker command provided.</li>
  <li>Run FastAPI Application: </li>
</ul>

```
uvicorn main-tf-serving:app --reload
```
<ul>
  <li>Make Predictions: Use an API client (e.g., Postman) or a web interface to upload images and receive predictions.</li>
</ul>







