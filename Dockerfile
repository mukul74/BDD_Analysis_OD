# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the selected Python files
COPY DataAnalysis.py utilities.py ./

# Copy only the necessary folders
COPY bdd100k_images_100k/ bdd100k_images_100k/
COPY bdd100k_labels_release/ bdd100k_labels_release/

# Create output folder (optional, if your scripts write outputs)
# RUN mkdir -p outputs

# Default command
CMD ["python", "DataAnalysis.py"]
