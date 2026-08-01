import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
# 3. PDF REPORT GENERATOR FUNCTION
# ==========================================
def generate_pdf_report(patient_name, area, severity, risk_score, recommendations):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title & Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "DFU-Vision AI: Clinical Assessment Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Patient Name / ID: {patient_name}")
    c.drawString(50, height - 100, f"Assessment Date: Live AI Analysis")
    
    # Divider line
    c.line(50, height - 115, width - 50, height - 115)
    
    # Metrics Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 145, "Wound Measurements & Classification:")
    
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 170, f"- Calculated Wound Surface Area: {area:.2f} cm²")
    c.drawString(70, height - 195, f"- Estimated Clinical Severity: {severity}")
    c.drawString(70, height - 220, f"- Combined Clinical Risk Score: {risk_score} / 10")
    
    # Recommendations Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 260, "Clinical Recommendations & Next Steps:")
    
    c.setFont("Helvetica", 11)
    y_pos = height - 285
    for rec in recommendations:
        c.drawString(70, y_pos, f"• {rec}")
        y_pos -= 22
        
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Generated via DFU-Vision AI: Automated Multi-Module Morphometric Analysis Suite")
    
    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================
st.title("🩺 DFU-Vision AI")
st.markdown("**Automated Multi-Module Diabetic Foot Ulcer Detection & Morphometric Analysis Suite**")
st.write("Capture or upload clinical foot images, compute ulcer measurements, evaluate updated clinical history, and generate formal PDF reports.")

# Sidebar for Patient Metadata & History
st.sidebar.header("📋 Patient Clinical Metadata")
patient_name = st.sidebar.text_input("Patient Name / ID", "Patient_001")
diabetes_duration = st.sidebar.slider("Diabetes Duration (Years)", 0, 30, 5)

# New history inputs added here
has_previous_ulcer = st.sidebar.checkbox("History of Previous Foot Ulcer / Surgery")
hba1c_high = st.sidebar.checkbox("High Blood Sugar / HbA1c > 8.0%")

has_neuropathy = st.sidebar.checkbox("Symptoms of Neuropathy / Numbness")
is_smoker = st.sidebar.checkbox("History of Smoking")
previous_area = st.sidebar.number_input("Previous Wound Area (cm² - Optional)", min_value=0.0, value=0.0)

# Calculate Risk Score including the 2 new parameters
risk_score = min(10, int(diabetes_duration / 3) + (3 if has_previous_ulcer else 0) + (2 if hba1c_high else 0) + (2 if has_neuropathy else 0) + (1 if is_smoker else 0))

# Input Mode: Live Camera or Gallery
app_mode = st.radio("Choose Input Mode:", ["📷 Live Phone Camera", "📁 Upload from Gallery"])
uploaded_file = st.camera_input("Take a picture of the foot ulcer") if app_mode == "📷 Live Phone Camera" else st.file_uploader("Choose a clinical foot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_rgb_original = np.array(image)
    
    st.image(image, caption='Selected Clinical Image', use_container_width=True)
    
    with st.spinner("Executing multi-module clinical analysis pipeline..."):
        img_c, mask, area, severity = segment_and_measure_ulcer(img_rgb_original)
        
        st.divider()
        if area > 0.05:
            st.error("### ⚠️ CLINICAL FOOT ULCER DETECTED")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Wound Surface Area", f"{area:.2f} cm²")
            col2.metric("Estimated Severity", severity)
            col3.metric("Clinical Risk Score", f"{risk_score}/10")
            
            # Healing progress tracking if previous area exists
            if previous_area > 0:
                diff = previous_area - area
                pct = (diff / previous_area) * 100
                if diff > 0:
                    st.success(f"📈 Healing Progress: Wound area decreased by {pct:.1f}% ({diff:.2f} cm² reduction) compared to previous visit.")
                else:
                    st.warning(f"📉 Healing Progress: Wound area increased by {abs(pct):.1f}% compared to previous visit.")
            
            st.subheader("Visual Analysis Modules")
            t1, t2 = st.tabs(["Wound Boundary Tracking", "Binary Wound Mask"])
            with t1:
                st.image(img_c, use_container_width=True, caption="Phase 2: Boundary Contour Mapping")
            with t2:
                st.image(mask, use_container_width=True, caption="Phase 3: Pixel-level Mask Segmentation")
                
            # Recommendations logic
            recs = [
                "Offload pressure immediately using specialized therapeutic footwear.",
                "Maintain strict glycemic control and monitor daily blood glucose levels.",
                "Clean wound area regularly with sterile saline and apply prescribed dressings.",
                "Consult a vascular specialist or podiatrist for clinical evaluation."
            ]
            if severity == "Severe" or risk_score >= 7:
                recs.insert(0, "URGENT: High risk profile detected based on clinical history and lesion size. Immediate intervention required.")
                
            st.divider()
            st.subheader("📄 Download Official Clinical Report")
            pdf_buffer = generate_pdf_report(patient_name, area, severity, risk_score, recs)
            
            st.download_button(
                label="📥 Download PDF Assessment Report",
                data=pdf_buffer,
                file_name=f"{patient_name}_DFU_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.success("### ✅ NORMAL / NO ULCER DETECTED")
            st.info("The selected region shows no prominent clinical lesion patterns based on HSV color thresholding.")
