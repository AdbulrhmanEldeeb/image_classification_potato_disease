# Potato Disease Recognition App

This project is a **Potato Disease Recognition App** built using **deep learning**. The app leverages a **custom Convolutional Neural Network (CNN)** and a **pre-trained ResNet model** to classify potato leaf diseases. The models have been trained on a dataset consisting of images of potato leaves and achieved high accuracy, with an **F1 score of 99%**. The app allows users to upload images of potato leaves and predicts whether the leaf is healthy or affected by a disease, such as Early Blight or Late Blight.

You can find the related notebook on Kaggle:
[Kaggle Notebook](https://www.kaggle.com/code/abdulrhmaneldeeb/potato-disease-classification)

## Features
- **Image Upload**: Users can upload an image of a potato leaf, and the app will classify it into one of three categories: Healthy, Early Blight, or Late Blight.
- **Deep Learning Models**: The app uses both a custom CNN and a pre-trained ResNet model fine-tuned on potato leaf images.
- **High Accuracy**: The models have been trained and tested on a subset of the PlantVillage dataset, achieving an F1 score of 99%.
- **Real-time Prediction**: The app provides instant predictions and displays the predicted disease along with the uploaded image.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/potato-disease-recognition.git
    cd potato-disease-recognition
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the trained models: Make sure to download the pre-trained CNN and ResNet models and place them in the `models/` directory.

5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## How to Use
- **Open the app**: After running the Streamlit command, a local server will start, and you can access the app through your browser.
  
- **Upload an image**: Click on the "Upload" button and select an image of a potato leaf. The app will automatically preprocess the image and predict the disease.
  
- **View the result**: The predicted disease (Healthy, Early Blight, or Late Blight) will be displayed below the uploaded image.

## Project Structure
    ├── app.py                     # Streamlit app code
    ├── models
    │   ├── cnn_model.keras         # Pre-trained CNN model
    │   ├── resnet_model.h5         # Pre-trained ResNet model
    ├── data
    │   └── potato_disease.jfif     # Potato disease image for the sidebar
    ├── requirements.txt            # Dependencies
    └── README.md                   # Project README
## Models
The app uses two deep learning models:
- **Custom CNN**: A Convolutional Neural Network trained on 20% of the PlantVillage dataset, specifically for potato diseases. The CNN was designed for lightweight performance while maintaining high accuracy.
- **Pre-trained ResNet**: A ResNet model fine-tuned for image classification tasks. This model provides a higher degree of generalization and performs well on the dataset even with a limited amount of training data.

## Dataset
The dataset used in this project is from the **PlantVillage** dataset, which includes thousands of images of healthy and diseased potato leaves. For this project, we focused on three classes:
- **Healthy**: No disease
- **Early Blight**: A common disease caused by the fungus *Alternaria solani*.
- **Late Blight**: A more severe disease caused by the *Phytophthora infestans* pathogen.

[Kaggle Dataset Link](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

## Evaluation
The models were evaluated using several metrics, including **accuracy**, **precision**, **recall**, and the **F1 score**. The custom CNN and pre-trained ResNet models both achieved an F1 score of 99%, demonstrating their effectiveness for the task of potato disease recognition.

## Contact 
For any questions or feedback, feel free to reach out at:
- **Email**: [abdodebo3@gmail.com](mailto:abdodebo3@gmail.com)
- **Phone**: +201026821545

