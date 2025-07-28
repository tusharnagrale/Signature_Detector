import streamlit as st
import cv2
import numpy as np
import os
import shutil
from signatureExtractor import extract_signature

st.set_page_config(page_title="Signature Extractor", layout="centered")

st.title("🖋️ Signature Extraction Tool")
st.markdown("Upload a document or signature page to automatically detect and extract the signature.")

# Initialize session state for uploaded file
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Function to clear the uploaded file
def clear_upload():
    st.session_state.uploaded_file = None
    if os.path.exists("temp_uploads"):
        shutil.rmtree("temp_uploads")
    if os.path.exists("extracted_signatures"):
        shutil.rmtree("extracted_signatures")

# File uploader
uploaded_file = st.file_uploader("Upload an Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    # Save uploaded image to a temporary location
    input_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Show uploaded image
    st.image(input_path, caption="Uploaded Image", use_container_width=True)

    # Clear button
    st.button("❌ Clear Upload", on_click=clear_upload)

    # Process the image
    if st.button("🔍 Extract Signature"):
        with st.spinner("Processing..."):
            result = extract_signature(input_path)
        
        # Show result
        if result:
            output_folder = "extracted_signatures"
            os.makedirs(output_folder, exist_ok=True)
            extracted_images = [f for f in os.listdir(output_folder) if uploaded_file.name in f]
            
            if extracted_images:
                st.success(f"Found {len(extracted_images)} signature(s).")
                for img_file in extracted_images:
                    img_path = os.path.join(output_folder, img_file)
                    st.image(img_path, caption=img_file)
                    
                    # Add download button for each extracted signature
                    with open(img_path, "rb") as file:
                        btn = st.download_button(
                            label=f"Download {img_file}",
                            data=file,
                            file_name=img_file,
                            mime="image/png"
                        )
            else:
                st.warning("Signature detected but no output saved.")
        else:
            st.error("No signature found in the uploaded image.")
elif st.session_state.uploaded_file:
    # This handles the case when the clear button is pressed
    st.session_state.uploaded_file = None