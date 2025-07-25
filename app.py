import streamlit as st
import os
import fitz  # PyMuPDF
from signatureExtractor import extract_signature
from PIL import Image
import io

st.set_page_config(page_title="Signature Extractor", layout="centered")
st.title("🖋️ Signature Extraction Tool")
st.markdown("Upload an image or PDF to detect and extract signature(s).")

uploaded_file = st.file_uploader("Upload Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("extracted_signatures", exist_ok=True)

if uploaded_file:
    file_type = uploaded_file.type
    st.write(f"Detected file type: `{file_type}`")

    if file_type == "application/pdf":
        # Read PDF using PyMuPDF
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_signatures = 0

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)  # Adjust DPI for clarity
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            temp_img_path = f"temp_uploads/page_{i+1}.png"
            img.save(temp_img_path)

            st.image(temp_img_path, caption=f"Page {i+1}", use_container_width=True)
            st.write("🔍 Extracting signature...")

            extracted, msg = extract_signature(temp_img_path)
            st.info(msg)

            matches = [f for f in os.listdir("extracted_signatures") if f"page_{i+1}" in f]
            for sig_file in matches:
                st.image(os.path.join("extracted_signatures", sig_file), caption=sig_file, use_container_width=True)
                total_signatures += 1

        if total_signatures == 0:
            st.warning("No signatures found in the PDF.")
        else:
            st.success(f"Extracted {total_signatures} signature(s) from PDF.")

    else:
        # Handle image uploads
        img_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        st.image(img_path, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Extract Signature"):
            with st.spinner("Processing..."):
                extracted, msg = extract_signature(img_path)
                st.info(msg)

                matches = [f for f in os.listdir("extracted_signatures") if uploaded_file.name in f]
                if matches:
                    for sig_file in matches:
                        st.image(os.path.join("extracted_signatures", sig_file), caption=sig_file, use_container_width=True)
                    st.success(f"Extracted {len(matches)} signature(s).")
                else:
                    st.warning("No signature found in the image.")
