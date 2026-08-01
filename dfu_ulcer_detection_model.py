import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    DATA_DIR = "dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    SEED = 42
    MODEL_SAVE_PATH = "dfu_efficientnetb0_model.keras"

tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

# ==========================================
# 2. DATA PREPARATION & SPLIT
# ==========================================
def load_and_split_data(data_dir):
    filepaths = []
    labels = []
    classes = ['Normal', 'Ulcer']
    
    for class_idx, class_name in enumerate(classes):
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Directory {class_dir} not found. Please check paths.")
            continue
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        for ext in extensions:
            for img_path in class_dir.glob(ext):
                filepaths.append(str(img_path))
                labels.append(class_idx)
                
    if len(filepaths) == 0:
        raise ValueError(f"\n[ERROR] Images kedaikala! {data_dir} folder check pannunga.\n")
            
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=0.30, stratify=labels, random_state=Config.SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=Config.SEED
    )
    
    print(f"Total images: {len(filepaths)}")
    print(f"Training (70%): {len(X_train)} | Validation (15%): {len(X_val)} | Test (15%): {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_tf_dataset(filepaths, labels, shuffle=True):
    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, Config.IMG_SIZE)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(filepaths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# ==========================================
# 3. MODEL BUILDING
# ==========================================
def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=Config.IMG_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=Config.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="dense")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 4. CALLBACKS
# ==========================================
def get_callbacks():
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True, verbose=1
    )

    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=Config.MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  # <--- MAGIC LINE ITHUTHAAN! Ithu JSON error-ai block pannidum.
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]
    
# ==========================================
# 5. EVALUATION
# ==========================================
def evaluate_and_plot(model, test_dataset, y_true):
    y_pred_probs = model.predict(test_dataset)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# ==========================================
# 6. ADVANCED FEATURES
# ==========================================
def segment_and_measure_ulcer(image_path, pixels_per_cm=100):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0.0, "Error loading image"
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_bound = np.array([0, 50, 20])
    upper_bound = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_rgb, mask, 0.0, "None"
        
    largest_contour = max(contours, key=cv2.contourArea)
    pixel_area = cv2.contourArea(largest_contour)
    real_area_cm2 = pixel_area / (pixels_per_cm ** 2)
    
    cv2.drawContours(img_rgb, [largest_contour], -1, (0, 255, 0), 3)
    
    if real_area_cm2 < 2.0:
        severity = "Mild"
    elif 2.0 <= real_area_cm2 < 5.0:
        severity = "Moderate"
    else:
        severity = "Severe"
        
    return img_rgb, mask, real_area_cm2, severity

def generate_grad_cam(model, image_path, img_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    try:
        base_model = model.get_layer('efficientnetb0')
    except ValueError:
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break
                
    last_conv_layer_name = 'top_activation'
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except:
        last_conv_layer = base_model.layers[-1]
    
    grad_model = tf.keras.models.Model(
        [base_model.inputs], 
        [last_conv_layer.output, base_model.output]
    )
    
    with tf.GradientTape() as tape:
        augmented_inputs = model.get_layer('data_augmentation')(img_array, training=False)
        conv_outputs, base_outputs = grad_model(augmented_inputs)
        tape.watch(conv_outputs)
        
        x = model.get_layer('global_average_pooling2d')(base_outputs)
        x = model.get_layer('dropout')(x, training=False)
        preds = model.get_layer('dense')(x)
        
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    img_rgb = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, img_size)
    
    heatmap_resized = cv2.resize(heatmap, img_size)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    return img_rgb, heatmap_colored, superimposed_img

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train_data, val_data, test_data = load_and_split_data(Config.DATA_DIR)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    train_ds = create_tf_dataset(X_train, y_train)
    val_ds = create_tf_dataset(X_val, y_val, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, shuffle=False)

    model = build_model()

    # NOTE: Json Error vara koodathu nu class_weight enga irunthu thookiyachu
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=get_callbacks()
    )

    evaluate_and_plot(model, test_ds, y_test)
    print("Training Complete!")

    if len(X_test) > 0:
        print("\n--- Testing Advanced Features ---")
        sample_img_path = None
        for path, label in zip(X_test, y_test):
            if label == 1:
                sample_img_path = path
                break
        
        if sample_img_path is None:
            sample_img_path = X_test[0]
            
        print(f"Analyzing: {sample_img_path}")
        
        img_contours, mask, area, severity = segment_and_measure_ulcer(sample_img_path)
        
        try:
            if img_contours is not None:
                orig, heatmap, overlay = generate_grad_cam(model, sample_img_path, img_size=Config.IMG_SIZE)
                
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(img_contours)
                ax[0].set_title(f"Area: {area:.2f}cm² | {severity}")
                ax[0].axis('off')

                ax[1].imshow(mask, cmap='gray')
                ax[1].set_title("Wound Mask")
                ax[1].axis('off')

                ax[2].imshow(heatmap)
                ax[2].set_title("Grad-CAM Heatmap")
                ax[2].axis('off')

                ax[3].imshow(overlay)
                ax[3].set_title("AI Focus Overlay")
                ax[3].axis('off')

                plt.tight_layout()
                plt.show()
            else:
                print("Image load aagala, pathai check pannunga.")
        except Exception as e:
            print(f"Could not generate Grad-CAM: {e}")