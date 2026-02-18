import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from soil_model import load_soil_model, preprocess_soil_image, SOIL_CLASSES
from veg_model import load_veg_model, predict_vegetation

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="AI Soil & Vegetation Analyzer",
    layout="wide"
)

st.title("🌱 AI Soil & Vegetation Intelligence System")
st.markdown(
    """
    Deep Learning powered **Soil Classification** and **Vegetation Segmentation**
    deployed using PyTorch and Streamlit.
    """
)

# -------------------------------------------------
# Sidebar Info (Recruiter Visible)
# -------------------------------------------------
st.sidebar.header("🔍 Model Information")

st.sidebar.markdown("### Soil Classifier")
st.sidebar.write("Architecture: CNN")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Accuracy: 90%")
st.sidebar.write("Output: Softmax probabilities")

st.sidebar.markdown("### Vegetation Model")
st.sidebar.write("Architecture: U-Net (Lite)")
st.sidebar.write("Task: Binary Segmentation")
st.sidebar.write("Metric: Vegetation Coverage %")

st.sidebar.markdown("---")
st.sidebar.write("Deployment: Streamlit Cloud")
st.sidebar.write("Backend Model: PyTorch (.pth)")

# -------------------------------------------------
# Model Caching
# -------------------------------------------------
@st.cache_resource
def get_soil_model():
    return load_soil_model()

@st.cache_resource
def get_veg_model():
    return load_veg_model()

# -------------------------------------------------
# Task Selection
# -------------------------------------------------
option = st.selectbox(
    "Select AI Task",
    ["Soil Classification", "Vegetation Detection"]
)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ==========================================================
    # SOIL CLASSIFICATION
    # ==========================================================
    if option == "Soil Classification":

        model = get_soil_model()
        model.eval()

        input_tensor = preprocess_soil_image(image)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        predicted_label = SOIL_CLASSES[predicted_class.item()]
        confidence_score = confidence.item() * 100

        st.success(f"🌍 Predicted Soil Type: **{predicted_label}**")

        # Confidence Bar
        st.subheader("Model Confidence")
        st.progress(int(confidence_score))
        st.write(f"Confidence Score: **{confidence_score:.2f}%**")

        # Probability Distribution
        st.subheader("Class Probability Distribution")
        prob_array = probabilities.cpu().numpy()[0]

        prob_dict = {
            SOIL_CLASSES[i]: float(prob_array[i] * 100)
            for i in range(len(SOIL_CLASSES))
        }

        st.bar_chart(prob_dict)

    # ==========================================================
    # VEGETATION SEGMENTATION
    # ==========================================================
    elif option == "Vegetation Detection":

        model = get_veg_model()
        mask_image, veg_percent = predict_vegetation(model, image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image)

        with col2:
            st.subheader("Predicted Vegetation Mask")
            st.image(mask_image)

        st.subheader("Vegetation Coverage Estimation")
        st.progress(int(veg_percent))
        st.write(f"Vegetation Area: **{veg_percent:.2f}%** of total image")
