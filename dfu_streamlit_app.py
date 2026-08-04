import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np

st.set_page_config(page_title="DFU Detection AI", page_icon="🩺", layout="centered")

st.title("🩺 Smart AI-Driven Plantar Pressure & DFU Analysis")
st.write("Upload an image of the foot/wound to detect potential Diabetic Foot Ulcer (DFU) risks.")

# Google Drive-la irunthu model-ai automatic-ah download panrathu
@st.cache_resource
def load_model():
    # Direct download link for Google Drive file
    file_id = "1A99tLHNZppJsSkey77aqlkhzgiPoXewy"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "dfu_model.h5"
    
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive... Please wait."):
            gdown.download(url, output, quiet=False)
            
    model = tf.keras.models.load_model(output)
    return model

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# File uploader for user image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            image = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            st.write(f"Raw Score: {score}")
            
            # Display result based on threshold
            if score > 0.5:
                st.error(f"⚠️ High Risk / Ulcer Detected (Confidence: {score:.2f})")
            else:
                st.success(f"✅ Low Risk / Normal (Confidence: {1 - score:.2f})")
