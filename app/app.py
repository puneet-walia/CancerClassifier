import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_cancer_model():
    model_path = "models/saved_models/fmodel_resnet50.h5"
    return load_model(model_path)

model = load_cancer_model()

CLASS_LABELS = ['Benign', 'Malignant']

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("🩺 Skin Cancer Classification App")
    st.markdown("""
    Welcome to the **Skin Cancer Detection App**.  
    Upload a **clear, close-up image of a skin lesion** below to predict whether it is *benign* or *malignant*.
    
    ⚠️ **Disclaimer:**  
    This tool is for **educational and research purposes only**.  
    It is **not a substitute for professional medical diagnosis**.  
    Always consult a dermatologist for medical concerns.
    """)

    st.divider()

    uploaded_image = st.file_uploader(
        "📸 Upload a skin lesion image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the affected skin area."
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        st.info("✅ Image uploaded successfully. Click **Predict** to analyze.")

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image... Please wait ⏳"):
                img = preprocess_uploaded_image(uploaded_image)
                if img is not None:
                    prediction = predict_skin_cancer(img)
                    display_result(prediction)

    st.divider()
    st.caption("Developed for research and awareness purposes by Puneet Walia © 2025")

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_uploaded_image(uploaded_image):
    try:
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize((100, 100))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")
        return None

# -----------------------------
# Prediction
# -----------------------------
def predict_skin_cancer(img):
    result = model.predict(img)
    return np.argmax(result)

# -----------------------------
# Display Result
# -----------------------------
def display_result(prediction):
    label = CLASS_LABELS[prediction]
    if label == 'Malignant':
        st.error("🧬 Prediction: **Malignant (Possible Skin Cancer)**")
        st.write("Please consult a **qualified dermatologist** for further diagnosis.")
    else:
        st.success("🩹 Prediction: **Benign (Likely Non-Cancerous)**")
        st.write("This lesion appears **non-cancerous**, but always seek professional advice if in doubt.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
