import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

DATASET_DIR = 'dataset'
IMG_SIZE = 64  # Resize all images to 64x64 to keep the model fast and consistent

X = []
y = []

print("Loading and preprocessing images...")
for label_folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, label_folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Extracts the letter 'A' from 'A-Sample'
    label = label_folder.split('-')[0] if '-' in label_folder else label_folder
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # 1. Convert to Grayscale (Black & White)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 2. Apply Gaussian Blur to smooth out edges and noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # 3. Resize image
            resized = cv2.resize(blurred, (IMG_SIZE, IMG_SIZE))
            # 4. Flatten the 2D image matrix into a 1D array for the linear model
            flattened = resized.flatten()
            
            X.append(flattened)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Data loaded successfully! Found {len(X)} images.")

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Linear Model (Logistic Regression)... This might take a minute.")
model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Validation Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("Saving model to sign_model.pkl...")
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Training complete! You are ready to start the Flask app.")