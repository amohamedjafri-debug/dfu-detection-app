import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime
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
# 2. ANALYSIS & MEASUREMENT FUNCTION (ULTRA-STRICT FALSE POSITIVE GUARD)
# ==========================================
def segment_and_measure_ulcer(img_rgb, pixels_per_cm=100):
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Ultra-strict red ranges for active open wounds / inflammation
        lower_red1 = np.array([0, 110, 60])
        upper_red1 = np.array([10, 255, 230])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        lower_red2 = np.array([170, 110, 60])
        upper_red2 = np.array([180, 255, 230])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Dark / Necrotic wound tissue range
        lower_necrotic = np.array([0, 0, 0])
        upper_necrotic = np.array([180, 45, 50])
        mask_necrotic = cv2.inRange(hsv, lower_necrotic, upper_necrotic)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        final_mask = cv2.bitwise_or(red_mask, mask_necrotic)
        
        # Check total non-zero pixel percentage in the entire image
        total_pixels = final_mask.shape[0] * final_mask.shape[1]
        wound_pixel_count = cv2.countNonZero(final_mask)
        redness_percentage = (wound_pixel_count / total_pixels) * 100
        
        # Strict Guard: If wound/redness content is less than 2.0% of the image, treat as normal skin!
        if redness_percentage < 2.0:
            return img_rgb, np.zeros_like(final_mask), 0.0, "None"
        
        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_rgb, np.zeros_like(final_mask), 0.0, "None"
            
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        
        # High area threshold to filter out skin creases, scratches, or minor texture spots
        if contour_area < 1500:
            return img_rgb, np.zeros_like(final_mask), 0.0, "None"
            
        area = contour_area / (pixels_per_cm ** 2)
        
        if area < 0.6: # Minimum physical area threshold for clinical ulcers
            return img_rgb, np.zeros_like(final_mask), 0.0, "None"
            
        img_with_contours = img_rgb.copy()
        cv2.drawContours(img_with_contours, [largest], -1, (0, 255, 0), 3)
        
        severity = "Mild" if area < 2.0 else "Moderate" if area < 5.0 else "Severe"
        return img_with_contours, final_mask, area, severity
        
    except Exception as e:
        return img_rgb, np.zeros((100, 100), dtype=np.uint8), 0.0, "None"

# ==========================================
# 3. PDF REPORT GENERATOR FUNCTION
# ==========================================
def generate_pdf_report(patient_name, blood_glucose, upload_timestamp, area, severity, risk_score, recommendations):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "DFU Detection AI: Clinical Assessment Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Patient Name / ID: {patient_name}")
    c.drawString(50, height - 100, f"Blood Glucose Level: {blood_glucose} mg/dL")
    c.drawString(50, height - 120, f"Assessment Timestamp: {upload_timestamp}")
    
    c.line(50, height - 135, width - 50, height - 135)
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 165, "Wound Measurements & Classification:")
    
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 190, f"- Calculated Wound Surface Area: {area:.2f} cm²")
    c.drawString(70, height - 215, f"- Estimated Clinical Severity: {severity}")
    c.drawString(70, height - 240, f"- Combined Clinical Risk Score: {risk_score} / 10")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 280, "Clinical Recommendations & Next Steps:")
    
    c.setFont("Helvetica", 11)
    y_pos = height - 305
    for rec in recommendations:
        c.drawString(70, y_pos, f"• {rec}")
        y_pos -= 22
        
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Generated via DFU Detection AI: Automated Multi-Module Morphometric Analysis Suite")
    
    c.save()
    buffer.seek(0)
    return buffer

# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================
st.title("🩺 DFU Detection AI")
st.markdown("**Automated Multi-Module Diabetic Foot Ulcer Detection & Morphometric Analysis Suite**")
st.markdown("*A core visual diagnostic module for the Smart AI driven plantar pressure analysis pipeline.*")
st.write("Capture or upload clinical foot images, compute ulcer measurements, evaluate clinical history, and generate formal PDF reports.")

st.sidebar.header("📋 Patient Clinical Metadata")
patient_name = st.sidebar.text_input("Patient Name / ID", "Patient_001")
diabetes_duration = st.sidebar.slider("Diabetes Duration (Years)", 0, 30, 5)
blood_glucose = st.sidebar.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=600, value=140, step=5)

has_previous_ulcer = st.sidebar.checkbox("History of Previous Foot Ulcer / Surgery")
has_neuropathy = st.sidebar.checkbox("Symptoms of Neuropathy / Numbness")
is_smoker = st.sidebar.checkbox("History of Smoking")
previous_area = st.sidebar.number_input("Previous Wound Area (cm² - Optional)", min_value=0.0, value=0.0)

glucose_risk = 3 if blood_glucose > 200 else (1 if blood_glucose > 140 else 0)
risk_score = min(10, int(diabetes_duration / 3) + glucose_risk + (3 if has_previous_ulcer else 0) + (2 if has_neuropathy else 0) + (1 if is_smoker else 0))

app_mode = st.radio("Choose Input Mode:", ["📷 Live Phone Camera", "📁 Upload from Gallery"])
uploaded_file = st.camera_input("Take a picture of the foot ulcer") if app_mode == "📷 Live Phone Camera" else st.file_uploader("Choose a clinical foot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    image = Image.open(uploaded_file).convert('RGB')
    img_rgb_original = np.array(image)
    
    st.image(image, caption=f'Selected Clinical Image (Uploaded at: {upload_timestamp})', use_container_width=True)
    
    with st.spinner("Executing multi-module clinical analysis pipeline..."):
        img_c, mask, area, severity = segment_and_measure_ulcer(img_rgb_original)
        
        st.divider()
        if area > 0.0:
            st.error("### ⚠️ CLINICAL FOOT ULCER DETECTED")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Wound Surface Area", f"{area:.2f} cm²")
            col2.metric("Estimated Severity", severity)
            col3.metric("Clinical Risk Score", f"{risk_score}/10")
            
            st.info(f"🩸 Recorded Blood Glucose: **{blood_glucose} mg/dL** | ⏱️ Analysis Time: **{upload_timestamp}**")
            
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
            pdf_buffer = generate_pdf_report(patient_name, blood_glucose, upload_timestamp, area, severity, risk_score, recs)
            
            st.download_button(
                label="📥 Download PDF Assessment Report",
                data=pdf_buffer,
                file_name=f"{patient_name}_DFU_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.success("### ✅ NORMAL / NO ULCER DETECTED")
            st.info("The selected region shows no prominent clinical lesion patterns based on morphometric analysis.")
