from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the model and define classes
cnn_model_path = 'models/cnn_model.keras'
cnn_model = load_model(cnn_model_path)
classes = ['Early blight', 'Late blight', 'healthy']

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image has been uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process and predict the uploaded image
            image = Image.open(filepath)
            image = image.resize((128, 128))
            image_arr = img_to_array(image)
            image_arr = image_arr / 255.0

            # Predict the class
            scores = cnn_model.predict(np.expand_dims(image_arr, axis=0), verbose=0)
            pred_label = np.argmax(scores, axis=1)
            pred_class = classes[pred_label[0]]

            # Pass the correct path to the image
            image_url = f'uploads/{file.filename}'

            return render_template('result.html', image_url=image_url, prediction=pred_class)
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
