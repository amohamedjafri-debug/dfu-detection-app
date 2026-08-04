import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np
import cv2

# Page Configuration
st.set_page_config(
    page_title="Smart DFU Detection AI",
    page_icon="🩺",
    layout="centered"
)

# App Title & Description
st.markdown("""
    <h1 style='color: #0b5345;'>🩺 Smart AI-Driven Plantar Pressure & DFU Analysis</h1>
""", unsafe_allow_html=True)

st.write("Upload an image of the foot/wound to detect potential Diabetic Foot Ulcer (DFU) risks accurately.")

# Google Drive Model Download Link Function
@st.cache_resource
def load_ai_model():
    model_path = 'dfu_model.h5'
    
    if not os.path.exists(model_path):
        file_id = '1A99tLHNZppJSkey77aqlkhzgiPoXewy'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
with st.spinner("Loading AI Model... Please wait."):
    model = load_ai_model()

# Image Upload Section
uploaded_file = st.file_uploader("Choose a foot/wound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Foot Image', use_column_width=True)
    
    if st.button("Analyze Image"):
        if model is not None:
            with st.spinner("Analyzing image for clinical anomalies..."):
                # Convert PIL image to OpenCV format for processing
                img_cv = np.array(image)
                if len(img_cv.shape) == 2:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
                elif img_cv.shape[2] == 4:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

                # Preprocess for model prediction
                img_resized = cv2.resize(img_cv, (224, 224))
                img_array = img_resized.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Model Prediction
                prediction = model.predict(img_array)
                score = float(prediction[0][0])
                
                # Display Results
                st.subheader("Analysis Results:")
                
                # Using a higher threshold (e.g., 0.70) to prevent false positives on normal skin tones/redness
                if score > 0.70:
                    st.error(f"⚠️ High Risk Detected! (Confidence Score: {score:.2f})")
                    st.write("Recommendation: Clinical signs match potential ulceration risks. Consult a specialist immediately.")
                else:
                    # If score is low, treat as normal/low risk
                    display_confidence = 1.0 - score if score < 0.5 else score
                    st.success(f"✅ Low Risk / Normal Skin Tissue (Confidence: {display_confidence:.2f})")
                    st.write("Recommendation: No significant open wound or deep ulcer pattern detected. Maintain regular foot care.")
        else:
            st.error("Model is not loaded properly. Please check your Google Drive configurations.")
