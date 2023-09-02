import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
import tempfile

# Load the pre-trained model
model_path = "fmodel_resnet50.h5"
loaded_model_imageNet = load_model(model_path)

# Define the class labels
class_labels = ['benign', 'Malignant']

# Create a Streamlit app
def main():
    st.title("Skin Cancer Classification App")
    st.sidebar.title("Upload Image")

    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict the image
        if st.sidebar.button("Predict"):
            image = preprocess_uploaded_image(uploaded_image)
            if image is not None:
                prediction = predict_skin_cancer(image)
                st.write(f"Prediction: {class_labels[prediction]}")

# Preprocess the uploaded image
def preprocess_uploaded_image(uploaded_image):
    try:
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, "uploaded_image.jpg")

        # Save the uploaded image to the temporary directory
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_image.read())

        # Read the uploaded image using the absolute file path
        img = cv2.imread(temp_file_path)

        if img is None:
            st.error("Error: Unable to read the uploaded image. Please make sure it's a valid image file.")
            return None

        # Resize the image to (100, 100)
        img = cv2.resize(img, (100, 100))

        # Expand dimensions to match the model input shape
        img = np.expand_dims(img, axis=0)

        # Preprocess the image
        img = preprocess_input(img)

        return img
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Predict the skin cancer type
def predict_skin_cancer(img):
    result = loaded_model_imageNet.predict(img)
    return np.argmax(result)

if __name__ == "__main__":
    main()
