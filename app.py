import streamlit as st
import cv2
import numpy as np
import os
from signatureExtractor import extract_signature

st.set_page_config(page_title="Signature Extractor", layout="centered")

st.title("🖋️ Signature Extraction Tool")
st.markdown("Upload a document or signature page to automatically detect and extract the signature.")

uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image to a temporary location
    input_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Show uploaded image
    st.image(input_path, caption="Uploaded Image", use_container_width=True)

    # Process the image
    if st.button("🔍 Extract Signature"):
        with st.spinner("Processing..."):
            result = extract_signature(input_path)
        
        # Show result
        if result:
            output_folder = "extracted_signatures"
            extracted_images = [f for f in os.listdir(output_folder) if uploaded_file.name in f]
            if extracted_images:
                st.success(f"Found {len(extracted_images)} signature(s).")
                for img_file in extracted_images:
                    st.image(os.path.join(output_folder, img_file), caption=img_file)
            else:
                st.warning("Signature detected but no output saved.")
        else:
            st.error("No signature found in the uploaded image.")
