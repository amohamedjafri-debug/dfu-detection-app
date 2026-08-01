import streamlit as st
import numpy as np
import cv2
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION (Video Style)
# ==========================================
st.set_page_config(
    page_title="DFU Detection AI",
    page_icon="🩺",
    layout="centered" # Center-la azhaga vara
)

# ==========================================
# 2. MODEL BUILDING (To avoid JSON Error)
# ==========================================
def build_model():
    from tensorflow.keras import layers, applications
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    base_model = applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="dense")(x)

    return tf.keras.models.Model(inputs, outputs)

@st.cache_resource
def load_trained_model():
    model = build_model()
    # Weights mattum load panrom - No JSON Error!
    model.load_weights("dfu_efficientnetb0_model.keras") 
    return model

# ==========================================
# 3. ANALYSIS FUNCTIONS
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

def make_gradcam_heatmap(img_array, model):
    try:
        base_model = model.get_layer('efficientnetb0')
        last_conv_layer = base_model.get_layer('top_activation')
        grad_model = tf.keras.models.Model([base_model.inputs], [last_conv_layer.output, base_model.output])
        
        with tf.GradientTape() as tape:
            aug_in = model.get_layer('data_augmentation')(img_array, training=False)
            conv_out, base_out = grad_model(aug_in)
            tape.watch(conv_out)
            preds = model.get_layer('dense')(model.get_layer('dropout')(model.get_layer('global_average_pooling2d')(base_out), training=False))
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

# ==========================================
# 4. UI & LOGIC
# ==========================================
st.title("🩺 Diabetic Foot Ulcer Detection AI")
st.write("Upload a foot image to classify it, measure severity, and view AI focus.")

try:
    model = load_trained_model()
    st.success("✅ AI Model Ready!")
except Exception as e:
    st.error(f"Model File Error: Please ensure 'dfu_efficientnetb0_model.keras' is in the folder.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Memory-la direct-a load panrom (No PermissionError!)
    image = Image.open(uploaded_file).convert('RGB')
    img_rgb_original = np.array(image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Analyzing..."):
            # CNN Prep
            img_resized = image.resize((224, 224))
            img_array_batch = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img_resized), axis=0)
            
            # Predict
            prob = model.predict(img_array_batch)[0][0]
            
            st.divider()
            if prob > 0.5:
                st.error(f"### ⚠️ ULCER DETECTED (AI Confidence: {prob*100:.2f}%)")
                
                # Analysis
                img_c, mask, area, severity = segment_and_measure_ulcer(img_rgb_original)
                
                # Metrics (Video style)
                col1, col2 = st.columns(2)
                col1.metric("Estimated Area", f"{area:.2f} cm²")
                col2.metric("Severity Level", severity)
                
                # Grad-CAM Overlay
                heatmap = make_gradcam_heatmap(img_array_batch, model)
                
                # Tabs (Video style)
                st.subheader("Visual Analysis")
                t1, t2, t3 = st.tabs(["AI Focus (Grad-CAM)", "Wound Boundary", "Wound Mask"])
                
                with t1:
                    if heatmap is not None:
                        # Overlay
                        h_resized = cv2.resize(heatmap, (img_resized.size[0], img_resized.size[1]))
                        h_colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                        overlay = cv2.addWeighted(np.array(img_resized), 0.6, h_colored, 0.4, 0)
                        st.image(overlay, use_container_width=True)
                    else:
                        st.warning("Heatmap error.")
                
                with t2:
                    st.image(img_c, use_container_width=True, caption="Boundary Detection")
                with t3:
                    st.image(mask, use_container_width=True, caption="Wound Mask")

            else:
                st.success(f"### ✅ NORMAL (Confidence: {(1-prob)*100:.2f}%)")
