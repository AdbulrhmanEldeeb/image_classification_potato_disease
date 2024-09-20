# Potato Disease Recognition App
This project is a Potato Disease Recognition App built using deep learning. The app leverages a custom Convolutional Neural Network (CNN) and a pre-trained ResNet model to classify potato leaf diseases. The models have been trained on a dataset consisting of images of potato leaves and achieved high accuracy, with an **F1 score of 99%.** The app allows users to upload images of potato leaves and predicts whether the leaf is healthy or affected by a disease, such as Early Blight or Late Blight

## Features
  - Image Upload: Users can upload an image of a potato leaf, and the app will classify it into one of three categories: Healthy, Early Blight, or Late Blight.
  - Deep Learning Models: The app uses both a custom CNN and a pre-trained ResNet model fine-tuned on potato leaf images.
  - High Accuracy: The models have been trained and tested on a subset of the PlantVillage dataset, achieving an F1 score of 99%.
  - Real-time Prediction: The app provides instant predictions and displays the predicted disease along with the uploaded image.

## Installation
  -Clone the repository:
    ```bash
    git clone https://github.com/yourusername/potato-disease-recognition.git
    cd potato-disease-recognition
  -Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
  -Install the required dependencies:
    ```bash 
    pip install -r requirements.txt
  -Download the trained models: Make sure to download the pre-trained CNN and ResNet models and place them in the models/ directory.
  -Run the Streamlit app:
    ```bash 
    streamlit run app.py
## How to Use
  - Open the app: After running the Streamlit command, a local server will start, and you can access the app through your browser.
  
  - Upload an image: Click on the "Upload" button and select an image of a potato leaf. The app will automatically preprocess the image and predict the disease.
  
  - View the result: The predicted disease (Healthy, Early Blight, or Late Blight) will be displayed below the uploaded image.







