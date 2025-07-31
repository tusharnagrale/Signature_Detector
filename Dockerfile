# Use official Python image with 3.13
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y curl

# Install system dependencies including Tesseract
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip uninstall -y opencv-python && pip install opencv-python-headless

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p temp_uploads extracted_signatures

# Set permissions for the directories
RUN chmod -R a+rwx temp_uploads extracted_signatures

# Expose port (Streamlit default)
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]