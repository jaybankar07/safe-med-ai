import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from PIL import Image

# -----------------------------------------------------------
# Rebuild same model architecture and load weights
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.load_weights("safe_med_ai_model.h5")
    return model

model = load_model()

# -----------------------------------------------------------
# Streamlit App UI
# -----------------------------------------------------------
st.set_page_config(page_title="Safe-Med AI", page_icon="💊", layout="centered")
st.title("💊 Safe-Med AI")
st.markdown("### A CNN-based system for detecting **medicine seal integrity** using mobile imagery.")

uploaded_file = st.file_uploader("📤 Upload an image of a medicine package", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")  # ensure 3 channels
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array, axis=0))

    # Predict
    with st.spinner("Analyzing..."):
        pred_prob = model.predict(img_array)[0][0]

    pred_class = "No Defect ✅" if pred_prob > 0.5 else "Defect ⚠️"

    # Display result
    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: **{pred_prob:.2f}**")

    if pred_class.startswith("No Defect"):
        st.success("✅ This medicine package appears properly sealed.")
    else:
        st.error("⚠️ Warning: Possible broken or tampered seal detected!")
