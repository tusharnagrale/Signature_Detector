# Signature Extraction Tool 🖋️

Automatically detect and extract signatures from documents and images.

## Features ✨
- Extract signatures from documents
- Works with complex documents containing text
- Handles various background colors
- Simple web interface

## Prerequisites 📋
- Docker installed on your system

## Installation 🛠️

### Using Docker (Recommended)
1. Clone this repository:
   ```bash
   git clone https://github.com/tusharnagrale/Signature_Detector.git
   cd Signature_Detector

### Project Structure 📂

Signature_Detector/
├── app.py               # Main application file
├── signatureExtractor.py # Core signature extraction logic
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── temp_uploads/        # Temporary upload storage
└── extracted_signatures/ # Extracted signatures storage

### Build the Docker image:

*bash
docker build -t signature-detector .
Run the container:

*bash
docker run -p 8501:8501 signature-detector   
Open your browser at http://0.0.0.0:8501


*Without Docker (For Development)
Install Tesseract OCR on your system:

Windows: Download installer from UB Mannheim

MacOS: brew install tesseract

Linux: sudo apt install tesseract-ocr

Install Python dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py


### Troubleshooting 🔧

If you get OCR errors, ensure Tesseract is properly installed in your system

For Docker permission issues, try:

bash
docker run -p 8501:8501 -v $(pwd)/temp_uploads:/app/temp_uploads -v $(pwd)/extracted_signatures:/app/extracted_signatures signature-extractor
