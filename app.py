import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
logo_path='data\my_logo_2.png'

st.set_page_config('Potato Disease Recognation',logo_path)
# Load the model and classes
cnn_model_path = 'models/cnn_model.keras'
cnn_model = load_model(cnn_model_path)
classes = ['Early blight', 'Late blight', 'healthy']

# Load the image for display in the sidebar
potato_disease_image = Image.open('data/potato_disease.jfif')

# Streamlit app configuration
st.sidebar.header('Potato Disease Recognition Using Deep Learning')
st.sidebar.image(potato_disease_image)
st.sidebar.markdown('This app is developed for classification of potato disease using CNN with 99% macro average F1 score')
# Upload image through Streamlit file uploader
uploaded_file = st.file_uploader("Upload a Potato image to predict its disease")

# Check if an image has been uploaded
if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image = image.resize((128, 128))
        image_arr = img_to_array(image)
        image_arr = image_arr / 255.0

        # Predict the class
        scores = cnn_model.predict(np.expand_dims(image_arr, axis=0), verbose=0)
        pred_label = np.argmax(scores, axis=1)
        pred_class = classes[pred_label[0]]

        # Display the prediction
        st.markdown(f'### Prediction:')
        st.write(f'**{pred_class}**')

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an image to get a prediction.")
