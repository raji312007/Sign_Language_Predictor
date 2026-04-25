import os
import cv2
import numpy as np
import pickle
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained linear model
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Warning: sign_model.pkl not found! Please run train.py first.")
    model = None

IMG_SIZE = 64

def process_and_predict(img):
    if model is None:
        return "Model Offline"
    
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    # 2. Gaussian Blur & Resize
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (IMG_SIZE, IMG_SIZE))
    
    # 3. Flatten for the Linear Model
    flattened = resized.flatten().reshape(1, -1)
    
    prediction = model.predict(flattened)[0]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    
    img = cv2.imread(filepath)
    prediction = process_and_predict(img)
    
    return jsonify({'prediction': prediction})

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data'})
    
    # Decode the base64 image from the frontend
    img_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    prediction = process_and_predict(img)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)