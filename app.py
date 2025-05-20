from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2 
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained Keras model
model = load_model('smart_waste_classifier.keras')

# Category folders in the dataset (ensure these match the class labels used during model training)
categories = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
              'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Mapping of detailed categories to 3 main classes
category_mapping = {
    'battery': 'Toxic',
    'biological': 'Organic',
    'brown-glass': 'Reusable',
    'cardboard': 'Reusable',
    'clothes': 'Reusable',
    'green-glass': 'Reusable',
    'metal': 'Reusable',
    'paper': 'Reusable',
    'plastic': 'Reusable',
    'shoes': 'Reusable',
    'trash': 'Toxic',
    'white-glass': 'Reusable'
}

def predict_class(img_path):
    # Load the image and resize it to the input size expected by the model
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure this matches the size used in training
    img_array = image.img_to_array(img) / 255.0  # Normalize the image to match the model's training preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)  # Get the index of the highest predicted class
    predicted_label = categories[predicted_index]  # Get the predicted label (category)
    
    # Map the detailed class to one of the main categories (Toxic, Organic, Reusable)
    mapped_label = category_mapping[predicted_label]
    
    return mapped_label, predicted_label

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Handle Image Upload & Prediction
@app.route('/predict', methods=['POST'])
def upload_and_classify():
    if 'file' not in request.files:
        return "No file part in the request"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Predict the class and get the mapped label
        final_category, original_class = predict_class(filepath)

        # Return the results page
        return render_template('results.html', user_image=filepath,
                               final_category=final_category,
                               original_class=original_class)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
