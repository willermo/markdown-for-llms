# Local Marker API Setup Guide

## Prerequisites

1. **Docker and Docker Compose installed**

   ```bash
   # Check if Docker is installed
   docker --version
   docker compose version
   ```

2. **System Requirements:**
   - At least 4GB RAM (8GB recommended)
   - 5GB free disk space
   - NVIDIA GPU (optional, for better performance)

## Quick Start

⚠️ **Note**: The Docker build takes 5-10 minutes due to large ML dependencies.

### Option 1: Docker Build (Full Setup)

```bash
# Build and start the Marker API service (takes 5-10 minutes)
docker compose up -d --build

# Monitor build progress
docker compose logs -f marker-api

# Once built, test the API
curl http://localhost:8000/health
```

### Option 2: Direct Installation (Faster)

If Docker is too slow, install directly:

```bash
# Install Marker locally
pip install marker-pdf

# Start the API server
python marker_api_server.py

# Test in another terminal
curl http://localhost:8000/health
```

### Option 3: Cloud Marker API (Instant)

Skip local setup entirely:

```bash
# Get API key from https://www.datalab.to/
echo "MARKER_API_KEY=your_key_here" >> .env

# Update pipeline_config.json:
# "pdf_converter": "cloud_marker"

# Run pipeline immediately
python master_workflow.py
```

## Verification

1. **Check service status:**

   ```bash
   docker compose ps
   ```

2. **Test API health:**

   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy"}
   ```

3. **Test with your pipeline:**
   ```bash
   python -c "from unified_converter import UnifiedDocumentConverter; from config import get_config_manager; config = get_config_manager().load_config(); converter = UnifiedDocumentConverter(config.conversion_settings); print('✓ Local Marker API configured')"
   ```

## Configuration

### GPU Support (Optional)

To enable GPU acceleration, edit `docker-compose.yml`:

```yaml
services:
  marker-api:
    environment:
      - TORCH_DEVICE=cuda # Change from 'cpu' to 'cuda'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Memory Limits

Adjust memory limits in `docker-compose.yml` based on your system:

```yaml
deploy:
  resources:
    limits:
      memory: 8G # Increase for better performance
    reservations:
      memory: 4G # Minimum required
```

## Pipeline Integration

### For Docker Setup

Once Docker service is running:

1. **Verify service is healthy:**

   ```bash
   curl http://localhost:8000/health
   # Should return: {"status": "healthy"}
   ```

2. **Configure pipeline (`pipeline_config.json`):**

   ```json
   {
     "conversion_settings": {
       "pdf_converter": "local_marker",
       "marker_local_base_url": "http://localhost:8000"
     }
   }
   ```

3. **Run the pipeline:**
   ```bash
   python master_workflow.py
   ```

### For Direct Installation

If using direct Python installation:

1. **Start API server:**

   ```bash
   python marker_api_server.py &
   ```

2. **Same pipeline configuration as above**

### For Cloud API

If using cloud service:

1. **Configure for cloud (`pipeline_config.json`):**

   ```json
   {
     "conversion_settings": {
       "pdf_converter": "cloud_marker"
     }
   }
   ```

2. **Ensure `.env` has API key:**
   ```bash
   grep MARKER_API_KEY .env
   ```

## Troubleshooting

### Docker Build Issues

**Build taking too long (>10 minutes):**

```bash
# Cancel current build
docker compose down

# Use cloud API instead (instant setup)
echo "MARKER_API_KEY=your_key" >> .env
# Edit pipeline_config.json: "pdf_converter": "cloud_marker"
python master_workflow.py
```

**Build failed:**

```bash
# Clean Docker cache and retry
docker system prune -f
docker compose build --no-cache
```

### Service Issues

```bash
# Check Docker logs
docker compose logs marker-api

# Restart services
docker compose restart

# Stop and remove containers
docker compose down

# Rebuild if needed
docker compose up -d --build
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check GPU usage (if enabled)
nvidia-smi

# Check available memory
free -h
```

### API Connection Issues

```bash
# Test direct connection
curl -v http://localhost:8000/health

# Check if port is accessible
netstat -tlnp | grep 8000

# Test from Python
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

### Common Solutions

1. **Port already in use:**

   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Out of memory:**

   - Reduce concurrent workers in pipeline config
   - Increase Docker memory limits
   - Process smaller batches

3. **Service won't start:**

   ```bash
   # Clean up and restart
   docker compose down -v
   docker system prune -f
   docker compose up -d --build
   ```

4. **Docker build too slow:**

   ```bash
   # Switch to cloud API (fastest option)
   echo "MARKER_API_KEY=get_from_datalab.to" >> .env
   # Update config: "pdf_converter": "cloud_marker"
   python master_workflow.py
   ```

5. **Want to test without Docker:**
   ```bash
   # Install locally and run API server
   pip install marker-pdf uvicorn fastapi python-multipart
   python marker_api_server.py
   ```

## Alternative: Cloud Marker API

If local setup is problematic, use the cloud API:

1. **Get API key from:** https://www.datalab.to/
2. **Add to `.env` file:**
   ```bash
   echo "MARKER_API_KEY=your_api_key_here" >> .env
   ```
3. **Update `pipeline_config.json`:**
   ```json
   {
     "conversion_settings": {
       "pdf_converter": "cloud_marker"
     }
   }
   ```

For more details, see: https://documentation.datalab.to/docs/on-prem/self-serve/api
