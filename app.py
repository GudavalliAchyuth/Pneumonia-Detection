# app.py
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.cm as cm
import cv2

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="wide")

# --- XAI ALGORITHM: GRAD-CAM ---
def make_gradcam_heatmap(img_array, model):
    # 1. Find the last convolutional layer automatically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if layer.__class__.__name__ == 'Conv2D': 
            last_conv_layer = layer
            break

    if not last_conv_layer:
        return None

    # 2. Split the model into two distinct parts to force gradient tracking
    # Part A: Input to the last conv layer
    last_conv_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    # Part B: From the last conv layer to the final prediction
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    layer_idx = model.layers.index(last_conv_layer)
    for layer in model.layers[layer_idx + 1:]:
        x = layer(x)
        
    classifier_model = tf.keras.models.Model(classifier_input, x)

    # 3. Compute gradients using the explicit split models
    with tf.GradientTape() as tape:
        # Generate the intermediate activations
        last_conv_layer_output = last_conv_model(img_array)
        
        # CRITICAL: Explicitly tell the tape to watch these activations
        tape.watch(last_conv_layer_output)
        
        # Generate the final predictions
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, 0]

    # 4. Multiply feature map by pooled gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Failsafe: if gradients are still lost, return None to avoid a crash
    if grads is None:
        return None
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
def overlay_heatmap(img, heatmap, alpha=0.5):
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.width, img.height))
    
    # Convert heatmap to RGB using a colormap (Jet)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Superimpose the heatmap on the original image
    img_array = np.array(img) / 255.0
    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    return superimposed_img

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Clinical Settings")
st.sidebar.write("Adjust the AI diagnostic parameters.")
threshold = st.sidebar.slider("Diagnostic Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

st.sidebar.divider()
st.sidebar.header("📋 Patient Details")
patient_name = st.sidebar.text_input("Patient ID / Name")
patient_age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)

# --- MAIN PAGE ---
st.title("🫁 Pneumonia Detection AI")
st.write("Upload a pediatric chest X-ray to get a real-time clinical prediction.")

@st.cache_resource
def load_ai_model():
    # 1. Get the absolute path of the directory app.py is sitting in
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Safely route directly into the models folder
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_detector.h5')
    
    return load_model(MODEL_PATH)
try:
    model = load_ai_model()
except Exception as e:
    st.error("Model not found! Please ensure 'pneumonia_detector.h5' is in the 'models/' folder.")
    st.stop()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Original Chest X-Ray', use_container_width=True)
    
    with col2:
        with st.spinner("🔄 Analyzing scan layers..."):
            img_resized = img.resize((150, 150))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            prediction = model.predict(img_array)[0][0]
            
        st.subheader("Diagnostic Results")
        if patient_name:
            st.write(f"**Patient:** {patient_name} (Age: {patient_age})")
            
        if prediction >= threshold:
            st.error(f"### 🚨 PNEUMONIA DETECTED")
            st.write(f"**AI Confidence:** {prediction * 100:.2f}%")
            
            # --- SHOW HEATMAP EXPLANATION ---
            st.divider()
            st.write("#### 🔍 Explainable AI Analysis")
            st.write("The heatmap below highlights the areas of the lung with fluid or opacity that triggered the positive diagnosis.")
            with st.spinner("Generating Grad-CAM Heatmap..."):
                heatmap = make_gradcam_heatmap(img_array, model)
                if heatmap is not None:
                    overlay = overlay_heatmap(img, heatmap)
                    st.image(overlay, caption="Red/Yellow areas indicate high AI focus.", use_container_width=True)
                else:
                    st.warning("Heatmap unavailable: Could not locate Convolutional layers.")
                    
        else:
            st.success(f"### ✅ NORMAL (Clear Lungs)")
            st.write(f"**AI Confidence:** {(1 - prediction) * 100:.2f}%")
            st.write("The lungs appear clear. No significant opacities detected.")