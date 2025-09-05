FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Marker directly from PyPI and API dependencies
RUN pip install --no-cache-dir marker-pdf uvicorn fastapi python-multipart

# Create necessary directories
RUN mkdir -p /app/temp /app/input

# Expose port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy our custom API server
COPY marker_api_server.py /app/

# Start the API server
CMD ["python", "marker_api_server.py"]
