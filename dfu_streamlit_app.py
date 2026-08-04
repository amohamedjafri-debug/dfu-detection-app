import streamlit as st
import os
import gdown
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="DFU Detection AI",
    page_icon="🩺",
    layout="centered"
)

# App Title & Description
st.markdown("""
    <h1 style='color: #0b5345;'>🩺 Smart AI-Driven Plantar Pressure & DFU Analysis</h1>
""", unsafe_allow_html=True)

st.write("Upload an image of the foot/wound to detect potential Diabetic Foot Ulcer (DFU) risks.")

# Google Drive Model Download Link Function
@st.cache_resource
def load_ai_model():
    model_path = 'dfu_model.h5'
    
    # Check if model file already exists, if not download using gdown
    if not os.path.exists(model_path):
        # Replace this URL with your actual Google Drive shareable link ID
        file_id = '1A99tLHNZppJSkey77aqlkhzgiPoXewy'
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            return None

    # Load model using TensorFlow
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
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Foot Image', use_column_width=True)
    
    if st.button("Analyze Image"):
        if model is not None:
            with st.spinner("Analyzing for DFU risks..."):
                # Preprocess image for model (adjust size if required by your training, e.g., 224x224)
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                
                # Check if image has 4 channels (RGBA) and convert to RGB
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]
                    
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediction
                prediction = model.predict(img_array)
                score = prediction[0][0]
                
                # Display Results based on your model's output logic
                st.subheader("Analysis Results:")
                if score > 0.5:
                    st.error(f"⚠️ High Risk Detected! (Confidence Score: {score:.2f})")
                    st.write("Recommendation: Consult a clinical specialist immediately for detailed wound assessment.")
                else:
                    st.success(f"✅ Low Risk / Normal (Confidence Score: {1 - score:.2f})")
                    st.write("Recommendation: Routine maintenance and regular foot monitoring advised.")
        else:
            st.error("Model is not loaded properly. Please check your Google Drive permissions.")
