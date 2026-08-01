import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DFU Detection AI",
    page_icon="🩺",
    layout="centered"
)

# ==========================================
# 2. ANALYSIS & MEASUREMENT FUNCTIONS
# ==========================================
def segment_and_measure_ulcer(img_rgb, pixels_per_cm=100):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([0, 50, 20])
    upper_bound = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb, mask, 0.0, "None"
        
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest) / (pixels_per_cm ** 2)
    
    img_with_contours = img_rgb.copy()
    cv2.drawContours(img_with_contours, [largest], -1, (0, 255, 0), 3)
    
    severity = "Mild" if area < 2.0 else "Moderate" if area < 5.0 else "Severe"
    return img_with_contours, mask, area, severity

# ==========================================
# 3. UI & APP LOGIC
# ==========================================
st.title("🩺 Diabetic Foot Ulcer Detection AI")
st.write("Upload or capture a clinical foot image for automated lesion tracking and boundary segmentation.")

# Camera and Upload Options
app_mode = st.radio("Choose Image Source:", ["📸 Use Mobile Camera", "📂 Upload from Gallery"])

uploaded_file = None
if app_mode == "📸 Use Mobile Camera":
    uploaded_file = st.camera_input("Take a clear picture of the foot")
else:
    uploaded_file = st.file_uploader("Choose a clinical foot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_rgb_original = np.array(image)
    
    if app_mode == "📂 Upload from Gallery":
        st.image(image, caption='Uploaded Clinical Image', use_container_width=True)
    
    if st.button("Run Clinical AI Analysis", type="primary"):
        with st.spinner("Processing image modules & calculating wound area..."):
            
            img_c, mask, area, severity = segment_and_measure_ulcer(img_rgb_original)
            
            st.divider()
            if area > 0.05:
                st.error("### ⚠️ CLINICAL FOOT ULCER DETECTED")
                
                col1, col2 = st.columns(2)
                col1.metric("Wound Surface Area", f"{area:.2f} cm²")
                col2.metric("Estimated Severity", severity)
                
                st.subheader("Visual Analysis Modules")
                t1, t2 = st.tabs(["Wound Boundary Tracking", "Binary Wound Mask"])
                
                with t1:
                    st.image(img_c, use_container_width=True, caption="Phase 2: Boundary Contour Mapping")
                with t2:
                    st.image(mask, use_container_width=True, caption="Phase 3: Pixel-level Mask Segmentation")
            else:
                st.success("### ✅ NORMAL / NO ULCER DETECTED")
                st.info("The selected region shows no prominent clinical lesion patterns based on HSV color thresholding.")
