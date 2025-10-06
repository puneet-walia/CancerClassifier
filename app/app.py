import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
import tempfile

# Load the pre-trained model
MODEL_PATH = "models/saved_models/fmodel_resnet50.h5"
loaded_model_imageNet = load_model(MODEL_PATH)

# Define the class labels
CLASS_LABELS = ['Benign', 'Malignant']

def main():
    st.title("🩺 Skin Cancer Classification App")
    st.sidebar.title("Upload an Image")

    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.sidebar.button("Predict"):
            image = preprocess_uploaded_image(uploaded_image)
            if image is not None:
                prediction = predict_skin_cancer(image)
                st.success(f"Prediction: **{CLASS_LABELS[prediction]}**")

# Preprocess the uploaded image
def preprocess_uploaded_image(uploaded_image):
    try:
        # Open the uploaded image using PIL
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((100, 100))  # Resize to match model input
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Predict the skin cancer type
def predict_skin_cancer(img):
    result = loaded_model_imageNet.predict(img)
    return np.argmax(result)

if __name__ == "__main__":
    main()
