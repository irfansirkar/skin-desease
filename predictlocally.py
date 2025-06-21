import cv2
import numpy as np
from keras.models import load_model
import os

# Set the size of the input images
img_size = (224, 224)

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

def predict_from_file(image_path):
    """Predict from a single image file"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    img_resized = cv2.resize(img, img_size)
    img_processed = np.expand_dims(img_resized, axis=0)
    img_processed = img_processed / 255.0
    
    # Make prediction
    prediction = model.predict(img_processed)
    probability = prediction[0][0]
    label = 'Cancer' if probability > 0.5 else 'Non_Cancer'
    
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Probability: {probability:.4f}")
    print("-" * 40)

# Example usage
if __name__ == "__main__":
    # Test with a single image
    image_path = input("Enter path to image file (or 'quit' to exit): ")
    
    while image_path.lower() != 'quit':
        predict_from_file(image_path)
        image_path = input("Enter path to image file (or 'quit' to exit): ")
