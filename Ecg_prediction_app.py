from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


app = Flask(__name__)

# Load the pre-trained model (using classifier1 as it is tuned)
with open('classifier1.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function (same as original)
def preprocess_image(image_path, target_size=(128, 128)):
    # Convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image
    img = cv2.resize(img, target_size)
    # Normalize pixel values to [0,1]
    img = img.astype('float32') / 255.0
    # Add channel dimension for CNN
    img = np.expand_dims(img, axis=-1)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Prediction function
def predict_ecg(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(img)
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Map class index to label
    class_labels = {
        0: "Normal",
        1: "Myocardial Infarction",
        2: "History of Myocardial Infarction",
        3: "Abnormal Heartbeat"
    }
    return class_labels[predicted_class], prediction[0][predicted_class]

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Make prediction
        label, confidence = predict_ecg(filepath)
        # Pass results to template
        return render_template('result.html', label=label, confidence=confidence*100, image_path=filepath)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)