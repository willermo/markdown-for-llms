# Document Intelligence Pipeline for LLMs

A comprehensive, production-ready Python pipeline for converting various document formats into clean, validated, and optimally chunked Markdown files ready for Large Language Model (LLM) consumption and NotebookLM notebooks.

## Table of Contents

- [🎯 Overview](#overview)
- [✨ Key Features](#key-features)
- [🚀 Quick Start](#quick-start)
- [📁 Project Structure](#project-structure)
- [🛠️ Installation Guide](#installation-guide)
- [⚙️ Configuration](#configuration)
- [📖 Usage Scenarios](#usage-scenarios)
- [🔄 Pipeline Steps](#pipeline-steps)
- [📊 Quality Metrics](#quality-metrics)
- [🧪 Testing](#testing)
- [🐳 Docker Setup & Testing](#docker-setup--testing)
- [🎯 Output Usage Guide](#output-usage-guide)
- [🚨 Troubleshooting](#troubleshooting)
- [🙏 Acknowledgments](#acknowledgments)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

## Overview

This pipeline transforms documents through a **4-step process** with enterprise-grade configuration, logging, and error handling:

**Workflow**: `Source Documents` → `Raw Markdown` → `Cleaned Markdown` → `Validated Markdown` → `LLM-Optimized Chunks`

### Pipeline Steps

1. **Conversion**: PDF/EPUB/MOBI/AZW/HTML/DOCX/RTF → Raw Markdown
2. **Cleaning**: Remove artifacts, normalize formatting, fix OCR errors
3. **Validation**: Quality assessment, issue detection, grading (A-F)
4. **Chunking**: Intelligent splitting optimized for specific LLMs

## Key Features

- **Unified Conversion System**: Smart routing between Pandoc, Local Marker API, and Cloud Marker API
- **Flexible Source Management**: Separate directories for PDFs and other documents
- **Full JSON Configuration**: All settings controlled via `pipeline_config.json`
- **Conversion Method Choice**: Real choice between local (free) and cloud (paid) PDF conversion
- **Advanced Logging**: Colored console output, file rotation, progress tracking
- **Comprehensive Testing**: Unit tests, integration tests, mocking support
- **Error Handling**: Custom exceptions with context and recovery strategies
- **Production Ready**: Proper dependency management, CLI tools, performance optimization
- **LLM Optimization**: Presets for GPT-4, Claude-3, Llama-2, and custom models

## Quick Start

### 🚀 Prerequisites

- **Python 3.9+** (3.12 recommended for Docker)
- **Pandoc** (for ebook/document conversion)
- **Optional**: Marker API key for high-quality PDF conversion

### 📦 Installation

```bash
# Clone the repository
git clone git@github.com:willermo/markdown-for-llms.git
cd markdown-for-llms

# Create virtual environment with meaningful name
python3 -m venv markdown-pipeline-env
source markdown-pipeline-env/bin/activate

# Install with pip (recommended)
pip install -e .

# Verify installation
pandoc --version
python -c "import tiktoken; print('✓ Dependencies installed')"
```

### ⚡ Basic Usage

```bash
# 1. Initialize configuration
python -c "from config import get_config_manager; get_config_manager().save_config()"

# 2. Place your documents in appropriate directories
mkdir source_pdfs source_documents
cp *.pdf source_pdfs/
cp *.epub *.mobi *.azw3 *.docx source_documents/

# 3. Configure conversion method in pipeline_config.json
# Edit "pdf_converter": "local_marker" or "cloud_marker"

# 4. Run the complete pipeline
python master_workflow.py

# 5. Find processed chunks in chunked_markdown/
ls chunked_markdown/
```

## Project Structure

```
markdown-for-llms/
├── source_pdfs/                # Input: PDF files (routed to Marker API)
├── source_documents/           # Input: Other documents (routed to Pandoc)
├── converted_markdown/         # Step 1: Raw markdown from unified converter
├── cleaned_markdown/           # Step 2: Cleaned markdown
├── validated_markdown/         # Step 3: Quality-validated markdown
├── chunked_markdown/           # Step 4: LLM-optimized chunks
├── pipeline_logs/              # Execution logs and reports
├── config.py                   # Configuration management
├── unified_converter.py        # NEW: Unified document converter
├── exceptions.py               # Custom exception classes
├── logging_config.py           # Advanced logging system
├── clean_markdown.py           # Cleaning module
├── validate_markdown.py        # Validation module
├── chunk_markdown.py           # Chunking module
├── master_workflow.py          # Pipeline orchestrator (refactored)
├── pdf_converter.py            # Cloud Marker API converter
├── onprem_pdf_converter.py     # Local Marker API converter
├── requirements.txt            # Dependencies
├── setup.py                    # Package installation
├── pipeline_config.json        # Runtime configuration (enhanced)
├── .env                        # Environment variables (create from .env.example)
└── tests/                      # Test suite
    ├── unit/                   # Unit tests
    └── integration/            # Integration tests
```

## Installation Guide

### 🐍 Method 1: Virtual Environment (Recommended)

```bash
# Create virtual environment with descriptive name
python3 -m venv markdown-pipeline-env

# Activate environment
source markdown-pipeline-env/bin/activate  # Linux/macOS
# OR
markdown-pipeline-env\Scripts\activate     # Windows

# Install project
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### 💻 Method 2: System Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Pandoc
# Ubuntu/Debian:
sudo apt install pandoc

# macOS:
brew install pandoc

# Windows: Download from https://pandoc.org/installing.html
```

### 🎭 Method 3: Poetry (Modern Approach)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run in Poetry environment
poetry run python -m master_workflow
```

## Configuration

The pipeline uses **two configuration approaches**:

### ⚙️ 1. Pipeline Configuration (JSON)

Create and edit `pipeline_config.json`:

```bash
# Generate default configuration
python -c "from config import get_config_manager; get_config_manager().save_config()"
```

```json
{
  "directories": {
    "source_pdfs": "source_pdfs",
    "source_documents": "source_documents",
    "converted": "converted_markdown",
    "cleaned": "cleaned_markdown",
    "validated": "validated_markdown",
    "chunked": "chunked_markdown",
    "logs": "pipeline_logs"
  },
  "conversion_settings": {
    "pdf_converter": "local_marker",
    "document_converter": "pandoc",
    "marker_cloud_api_key_env": "MARKER_API_KEY",
    "marker_local_base_url": "http://localhost:8000",
    "marker_cloud_base_url": "https://www.datalab.to/api/v1",
    "pandoc_options": "--wrap=none --strip-comments --markdown-headings=atx",
    "supported_pdf_formats": ["pdf"],
    "supported_document_formats": [
      "epub",
      "mobi",
      "azw",
      "azw3",
      "html",
      "htm",
      "docx",
      "rtf"
    ],
    "conversion_timeout": 3600,
    "max_retries": 3,
    "retry_delay": 10
  },
  "llm_presets": {
    "gpt-3.5-turbo": { "chunk_size": 3000, "overlap": 150 },
    "gpt-4": { "chunk_size": 6000, "overlap": 300 },
    "claude-3": { "chunk_size": 8000, "overlap": 400 },
    "llama-2": { "chunk_size": 3500, "overlap": 175 },
    "gemini-pro": { "chunk_size": 7000, "overlap": 350 },
    "custom": { "chunk_size": 4000, "overlap": 200 }
  },
  "validation_thresholds": {
    "min_content_length": 100,
    "max_content_length": 10000000,
    "min_words": 50,
    "max_artifact_ratio": 0.05,
    "min_readability_score": 20,
    "min_structure_score": 50
  },
  "cleaning_settings": {
    "aggressive_cleaning": true,
    "preserve_tables": true,
    "min_line_length": 3,
    "remove_html_tags": true,
    "normalize_whitespace": true
  },
  "chunking_settings": {
    "target_llm": "custom",
    "chunk_size": 4000,
    "overlap": 200,
    "min_chunk_size": 1000,
    "max_chunk_size": 8000,
    "strategy": "semantic"
  },
  "pipeline_settings": {
    "skip_existing": true,
    "max_workers": 3,
    "log_level": "INFO"
  }
}
```

### 🔧 Conversion Method Configuration

The `conversion_settings` section controls how different document types are processed:

**PDF Conversion Options:**

- `"pdf_converter": "local_marker"` - Use local Marker API (free, requires Docker setup)
- `"pdf_converter": "cloud_marker"` - Use cloud Marker API (paid, requires API key)

**Document Conversion:**

- `"document_converter": "pandoc"` - Use Pandoc for epub, mobi, azw3, docx, html, etc.

**Setup Examples:**

**Option 1: Local Processing (Recommended)**

```json
"conversion_settings": {
  "pdf_converter": "local_marker",
  "document_converter": "pandoc"
}
```

_Requires: Pandoc + Local Marker API running on localhost:8000_

**Option 2: Cloud PDF Processing**

```json
"conversion_settings": {
  "pdf_converter": "cloud_marker",
  "document_converter": "pandoc"
}
```

_Requires: Pandoc + MARKER_API_KEY in .env file_

### 🔐 2. Environment Variables (Sensitive Data)

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` for API keys and sensitive settings:

```bash
# Marker API configuration (for high-quality PDF conversion)
MARKER_API_KEY=your_api_key_here

# Optional: Override configuration settings
CHUNK_SIZE=4000
OVERLAP_SIZE=200
TARGET_LLM=claude-3
MAX_WORKERS=4
LOG_LEVEL=INFO
```

**Environment variables override JSON settings** for deployment flexibility.

## Usage Scenarios

### 🔄 Scenario 1: Complete Pipeline (Recommended)

```bash
# Process all documents with default settings
python master_workflow.py

# Target specific LLM
python master_workflow.py --llm gpt-4

# Custom chunk size
python master_workflow.py --chunk-size 5000 --overlap 250

# Force reprocess all files
python master_workflow.py --force
```

### 🎯 Scenario 2: Individual Steps

```bash
# Run only conversion
python master_workflow.py --step conversion

# Run only cleaning
python master_workflow.py --step cleaning

# Run only validation
python master_workflow.py --step validation

# Run only chunking
python master_workflow.py --step chunking
```

### 📚 Scenario 3: Unified Document Processing

**Setup Source Directories:**

```bash
# Create source directories
mkdir source_pdfs source_documents

# Place PDFs in dedicated directory
cp *.pdf source_pdfs/

# Place other documents in documents directory
cp *.epub *.mobi *.azw3 *.docx *.html source_documents/
```

**Configure Conversion Method:**

**Option A: Local Processing (Free)**

```bash
# 1. Start local Marker API
docker compose up -d --build

# 2. Wait for service to be ready (takes 5-10 minutes first time)
docker compose logs -f marker-api

# 3. Test API health
curl http://localhost:8000/health

# 4. Ensure pipeline_config.json has:
# "pdf_converter": "local_marker"

# 5. Run pipeline
python master_workflow.py
```

**Option B: Cloud Processing (Paid)**

```bash
# 1. Set up API key
echo "MARKER_API_KEY=your_key_here" >> .env

# 2. Configure for cloud in pipeline_config.json:
# "pdf_converter": "cloud_marker"

# 3. Run pipeline
python master_workflow.py
```

**Direct Converter Usage:**

```bash
# Test unified converter directly
python unified_converter.py --source-dir source_pdfs --output-dir test_output
python unified_converter.py --source-dir source_documents --output-dir test_output
```

### 🧪 Scenario 4: Testing and Development

```bash
# Run test suite
pytest tests/ -v

# Run with debug logging
LOG_LEVEL=DEBUG python master_workflow.py

# Dry run (see what would be processed)
python master_workflow.py --dry-run

# Force reprocess existing files
python master_workflow.py --force

# Test configuration
python -c "from config import get_config_manager; print('Config OK')"

# Test unified converter
python -c "from unified_converter import UnifiedDocumentConverter; print('Converter OK')"
```

## Pipeline Steps

### 📄 Step 1: Unified Document Conversion

**Smart Format Routing:**

The unified converter automatically routes files based on extension and configuration:

**PDF Files** (`source_pdfs/`) → **Marker API**

- **Local Marker** (free): Requires Docker setup, processes locally
- **Cloud Marker** (paid): API-based, requires MARKER_API_KEY
- High-quality ML-based conversion with table/image preservation

**Other Documents** (`source_documents/`) → **Pandoc**

- **EPUB** (.epub) - Standard ebook format
- **MOBI** (.mobi) - Amazon Kindle format
- **AZW/AZW3** (.azw, .azw3) - Amazon formats
- **HTML** (.html, .htm) - Web pages
- **DOCX** (.docx) - Word documents
- **RTF** (.rtf) - Rich text format

**Conversion Features:**

- Batch processing with threading
- Automatic retry logic with exponential backoff
- Progress tracking and detailed logging
- Error handling with graceful fallbacks
- Configurable timeouts and retry limits

### 🧹 Step 2: Markdown Cleaning

**Artifacts Removed:**

- **OCR Artifacts**: `D[AN] S. K[ENNEDY]` → `DAN S. KENNEDY`
- **HTML Remnants**: `<div>`, `<span>` tags, entities (`&nbsp;`, `&amp;`)
- **Format Artifacts**: Pandoc anchors `[]{...}`, broken emphasis
- **Structure Issues**: Orphaned punctuation, excessive whitespace

### ✅ Step 3: Quality Validation

**Quality Metrics:**

- **Content Length**: Minimum/maximum thresholds
- **Artifact Ratio**: Percentage of remaining artifacts
- **Readability Score**: Automated readability assessment
- **Structure Score**: Heading/paragraph balance
- **Encoding Quality**: UTF-8 compliance

**Grading System (A-F):**

- **A (90-100%)**: Excellent quality, ready for production
- **B (80-89%)**: Good quality, minor issues
- **C (70-79%)**: Acceptable quality, some cleanup needed
- **D (60-69%)**: Poor quality, significant issues
- **F (<60%)**: Failed validation, major problems

### 🔪 Step 4: Intelligent Chunking

**Chunking Strategies:**

1. **Semantic**: Split on natural boundaries (headings, paragraphs)
2. **Sliding Window**: Overlapping fixed-size windows
3. **Fixed Size**: Simple token-based splitting
4. **Hierarchical**: Preserve document structure

**LLM Presets:**

| LLM             | Chunk Size | Overlap  | Notes                        |
| --------------- | ---------- | -------- | ---------------------------- |
| `gpt-3.5-turbo` | 3,000      | 150      | Conservative for older model |
| `gpt-4`         | 6,000      | 300      | Balanced performance         |
| `claude-3`      | 8,000      | 400      | Large context window         |
| `llama-2`       | 3,500      | 175      | Open-source optimized        |
| `gemini-pro`    | 7,000      | 350      | Google's model               |
| `custom`        | Variable   | Variable | Your custom settings         |

## Quality Metrics

### 📊 Validation Report Example

```json
{
  "summary": {
    "total_files": 25,
    "valid_files": 23,
    "validation_rate": 92.0,
    "grade_distribution": { "A": 15, "B": 6, "C": 2, "D": 1, "F": 1 },
    "average_readability": 78.5,
    "average_structure_score": 85.2
  },
  "issues_found": {
    "html_artifacts": 12,
    "encoding_issues": 3,
    "formatting_problems": 8
  }
}
```

### 📈 Chunking Report Example

```json
{
  "chunking_summary": {
    "total_source_files": 23,
    "total_chunks": 342,
    "total_tokens": 1456789,
    "average_chunk_size": 4260,
    "chunk_distribution": {
      "heading_section": 156,
      "paragraph": 142,
      "list": 28,
      "table": 12,
      "code_block": 4
    }
  }
}
```

### 📋 Output Format

Each chunk includes comprehensive metadata:

```markdown
---
chunk_id: book_name_chunk_001
source_file: book_name.md
chunk_index: 0
total_chunks: 45
token_count: 4127
word_count: 2856
heading_context: ["Chapter 1", "Introduction", "Overview"]
content_type: paragraph
chunking_strategy: semantic
target_llm: claude-3
quality_grade: A
---

# Chapter 1: Introduction

## Overview

Lorem ipsum dolor sit amet, consectetur adipiscing elit...
```

## Testing

### 🧪 Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v          # Unit tests only
pytest tests/integration/ -v   # Integration tests only

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_cleaning.py -v
```

### 📁 Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/
│   ├── test_cleaning.py     # Cleaning functionality tests
│   ├── test_validation.py   # Validation tests
│   └── test_chunking.py     # Chunking tests
└── integration/
    └── test_pipeline.py     # End-to-end pipeline tests
```

### ✍️ Writing Tests

The test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Fixtures**: Sample data and mock objects
- **Mocking**: External API and file system mocking

## Docker Setup & Testing

### 🐳 Prerequisites

**System Requirements:**

- Docker and Docker Compose installed
- At least 4GB RAM (8GB recommended for better performance)
- 5GB free disk space for Docker images
- Optional: NVIDIA GPU for accelerated processing

```bash
# Verify Docker installation
docker --version
docker compose version
```

### 🚀 Quick Docker Setup

**⚠️ Important**: First-time setup downloads ~2GB of ML models and takes 5-10 minutes.

```bash
# 1. Build and start Marker API service
docker compose up -d --build

# 2. Monitor build progress (first time only)
docker compose logs -f marker-api

# 3. Wait for "Application startup complete" message
# 4. IMPORTANT: Wait for model downloads to complete before first conversion
#    Look for: "✓ Model dict created" in logs

# 5. Test API health
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

**⚡ Model Download Behavior:**
- **First startup**: Downloads 5 ML models (~2GB total)
- **Subsequent startups**: Uses cached models (fast startup)
- **Model persistence**: Enabled via Docker volumes
- **Timeout prevention**: Models cached after first download

### 🧪 Docker Container Testing

#### **1. Health Check Testing**

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with verbose output
curl -v http://localhost:8000/health

# Test from Python
python -c "import requests; print(requests.get('http://localhost:8000/health').json())"
```

#### **2. API Functionality Testing**

```bash
# Test API documentation endpoint
curl http://localhost:8000/docs
# Should return HTML page

# Test with sample PDF (create test file first)
echo "Test PDF content" > test.pdf
curl -X POST "http://localhost:8000/convert" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test.pdf"
```

#### **3. Container Management & Monitoring**

```bash
# Check container status
docker compose ps

# View real-time logs
docker compose logs -f marker-api

# Monitor resource usage
docker stats marker-api

# Check container health
docker inspect marker-api --format='{{.State.Health.Status}}'
```

#### **4. Performance Testing**

```bash
# Test with actual PDF file
cp /path/to/sample.pdf test_conversion.pdf

# Time the conversion
time curl -X POST "http://localhost:8000/convert" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_conversion.pdf" \
     -o converted_output.json

# Check conversion quality
cat converted_output.json | jq '.markdown' | head -20
```

#### **5. Integration Testing with Pipeline**

```bash
# Test unified converter integration
python -c "
from unified_converter import UnifiedDocumentConverter
from config import get_config_manager
config = get_config_manager().load_config()
converter = UnifiedDocumentConverter(config.conversion_settings)
print('✓ Local Marker API integration ready')
"

# Test end-to-end with sample PDF
mkdir -p source_pdfs test_output
cp test_conversion.pdf source_pdfs/
python unified_converter.py --source-dir source_pdfs --output-dir test_output

# Verify output
ls test_output/
cat test_output/*.md | head -10
```

### 🔧 Container Configuration

#### **GPU Support (Optional)**

Edit `docker-compose.yml` for GPU acceleration:

```yaml
services:
  marker-api:
    environment:
      - TORCH_DEVICE=cuda # Change from 'cpu'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### **Memory Optimization**

Adjust memory limits based on your system:

```yaml
deploy:
  resources:
    limits:
      memory: 8G # Maximum memory usage
    reservations:
      memory: 4G # Minimum required memory
```

### 🚨 Docker Troubleshooting

#### **Container Won't Start**

```bash
# Check Docker daemon
sudo systemctl status docker

# View detailed logs
docker compose logs marker-api

# Restart with clean state
docker compose down -v
docker system prune -f
docker compose up -d --build
```

#### **Build Failures**

```bash
# Clean build cache
docker builder prune -f

# Build with no cache
docker compose build --no-cache

# Check disk space
df -h
```

#### **API Connection Issues**

```bash
# Check if port is accessible
netstat -tlnp | grep 8000

# Test with different port (edit docker-compose.yml)
# ports: ["8001:8000"]

# Check firewall settings
sudo ufw status
```

#### **Performance Issues**

```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check available system resources
free -h
top -p $(docker inspect -f '{{.State.Pid}}' marker-api)
```

#### **Memory Issues**

```bash
# Increase Docker memory limit
# Edit ~/.docker/daemon.json:
{
  "default-runtime": "runc",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-shm-size": "2g"
}

# Restart Docker daemon
sudo systemctl restart docker
```

### 🔄 Container Lifecycle Management

```bash
# Start services
docker compose up -d

# Stop services (preserves data)
docker compose stop

# Restart services
docker compose restart

# Stop and remove containers
docker compose down

# Stop, remove containers, and clean volumes
docker compose down -v

# Rebuild and restart
docker compose up -d --build

# View service status
docker compose ps -a
```

### ⚡ Alternative Setup Options

#### **Option 1: Direct Python Installation (Faster)**

```bash
# Install Marker locally (faster than Docker)
pip install marker-pdf uvicorn fastapi python-multipart

# Start API server directly
python marker_api_server.py

# Test in another terminal
curl http://localhost:8000/health
```

#### **Option 2: Cloud Marker API (Instant)**

```bash
# Get API key from https://www.datalab.to/
echo "MARKER_API_KEY=your_key_here" >> .env

# Update pipeline_config.json
# "pdf_converter": "cloud_marker"

# Run pipeline immediately (no Docker needed)
python master_workflow.py
```

### 📊 Docker Testing Checklist

- [ ] Docker and Docker Compose installed
- [ ] Container builds successfully (5-10 minutes)
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] API docs accessible at `http://localhost:8000/docs`
- [ ] Sample PDF conversion works
- [ ] Pipeline integration test passes
- [ ] Resource usage within acceptable limits
- [ ] Logs show no errors during operation
- [ ] Container restarts cleanly
- [ ] Performance meets expectations

## Output Usage Guide

### 🤔 Choosing Between Validated Files vs. Chunks

The pipeline produces two main outputs, each optimized for different use cases:

#### **Use Validated Files When:**

**📚 Knowledge Base & Research**

- **NotebookLM**: Upload entire validated files for document Q&A and research
- **Document Analysis**: Analyzing writing style, themes, or document organization
- **Reference Systems**: When you need to cite specific sections with full context
- **Conversational AI**: Chat interfaces that need to understand document structure

**Example Scenarios:**

```bash
# Research scenario: Upload to NotebookLM
cp validated_markdown/*.md ~/NotebookLM_uploads/

# Academic analysis: Full document context needed
python analyze_documents.py --input validated_markdown/ --mode full_context
```

#### **Use Chunks When:**

**🤖 Machine Learning & Training**

- **Fine-tuning**: Each chunk becomes a focused training example
- **Vector Databases**: Better semantic search granularity
- **Embedding Systems**: Optimal size for embedding models
- **Batch Processing**: Parallel processing of document sections

**Example Scenarios:**

```bash
# Fine-tuning preparation
python prepare_training_data.py --input chunked_markdown/ --format jsonl

# Vector database ingestion
python embed_chunks.py --chunks chunked_markdown/ --output embeddings.db

# RAG system setup
for chunk in chunked_markdown/*.md; do
    python add_to_vector_store.py "$chunk"
done
```

#### **Hybrid Approach (Recommended)**

Use both outputs strategically:

1. **Immediate Research**: NotebookLM with validated files
2. **Production RAG**: Vector search with chunks
3. **Model Training**: Fine-tuning with chunks
4. **Content Management**: Full documents for editing and review

**Real-World Example:**

```bash
# Step 1: Research with NotebookLM (validated files)
cp validated_markdown/technical_books/*.md ~/research_notebook/

# Step 2: Build RAG system (chunks)
python build_rag_system.py \
    --chunks chunked_markdown/technical_books/ \
    --index technical_knowledge_base

# Step 3: Fine-tune domain model (chunks)
python fine_tune_model.py \
    --training_data chunked_markdown/technical_books/ \
    --model_name technical_assistant
```

### 📊 Output Quality Indicators

**Validated Files:**

- Grade A-B: Ready for production use
- Grade C: Good for research, may need manual review
- Grade D-F: Requires re-processing or manual cleanup

**Chunks:**

- Token count consistency: ±10% of target size
- Semantic boundaries: Chunks start/end at logical points
- Metadata completeness: All heading context preserved

### 🔗 Integration Examples

**NotebookLM Integration:**

```bash
# Upload high-quality validated files
find validated_markdown/ -name "*.md" -exec grep -l "Grade: [AB]" {} \; | \
xargs cp -t ~/NotebookLM_sources/
```

**Vector Database Integration:**

```python
# Process chunks for embedding
import os
from pathlib import Path

for chunk_file in Path('chunked_markdown').glob('*.md'):
    with open(chunk_file) as f:
        content = f.read()
        # Extract metadata and content
        # Add to vector store with metadata
```

**Training Data Preparation:**

```bash
# Convert chunks to training format
python -c "
from pathlib import Path
import json

training_data = []
for chunk in Path('chunked_markdown').glob('*.md'):
    with open(chunk) as f:
        content = f.read()
        training_data.append({'text': content})

with open('training_data.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')
"
```

## 🎉 Success!

Your documents are now converted into LLM-ready markdown chunks optimized for your target AI model. Use these chunks for:

- **RAG Systems**: Vector database ingestion
- **NotebookLM**: Google's notebook AI
- **Fine-tuning**: Training data preparation
- **Analysis**: Document processing pipelines
- **Research**: Academic document processing

Happy chunking! 🚀

---

## Troubleshooting

### 🚨 Common Issues

**No files converted:**

```bash
# Check source directory
ls source_ebooks/

# Verify Pandoc installation
pandoc --version

# Check conversion logs
tail -f pipeline_logs/conversion_*.log
```

**Poor quality scores:**

```bash
# Review validation report
cat validated_markdown/validation_report.json

# Adjust thresholds in config
python -m config --show
```

**Chunks too large/small:**

```bash
# Adjust chunk size
python -m master_workflow --chunk-size 5000

# Try different strategy
python -m master_workflow --chunking-strategy semantic
```

**API Key Issues:**

```bash
# Check environment variables
echo $MARKER_API_KEY

# Verify .env file
cat .env
```

### 🐛 Debug Mode

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python -m master_workflow

# Monitor logs in real-time
tail -f pipeline_logs/pipeline_*.log

# Test individual components
python -c "from clean_markdown import MarkdownCleaner; print('✓ Cleaning module works')"
```

### 📝 Log Files

- `conversion_*.log` - Document conversion details
- `cleaning_*.log` - Artifact removal progress
- `validation_*.log` - Quality assessment details
- `chunking_*.log` - Tokenization and chunking info
- `orchestrator_*.log` - Pipeline orchestration
- `pipeline_*.log` - Complete execution logs

## Acknowledgments

### 🙏 Core Dependencies

This project builds upon excellent open-source tools:

- **[Pandoc](https://pandoc.org/)** - Universal document converter by John MacFarlane
- **[datalab/marker](https://github.com/VikParuchuri/marker)** - High-quality PDF to markdown conversion using ML
- **[tiktoken](https://github.com/openai/tiktoken)** - Fast BPE tokenizer for accurate token counting
- **[pytest](https://pytest.org/)** - Testing framework for reliable code quality

### ✨ Special Thanks

- **Marker API** by [datalab/marker](https://github.com/VikParuchuri/marker) for providing state-of-the-art PDF conversion that preserves document structure and formatting
- **OpenAI tiktoken** for accurate token counting across different LLM tokenizers
- **Pandoc community** for maintaining the most comprehensive document conversion tool

### 💡 Inspiration

This pipeline was inspired by the need for high-quality document processing in the LLM era, where document structure and token optimization are crucial for effective AI applications.

## Contributing

### 🛠️ Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/markdown-for-llms.git
cd markdown-for-llms

# Create development environment
python3 -m venv markdown-pipeline-dev
source markdown-pipeline-dev/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests before making changes
pytest tests/ -v
```

### 🎨 Code Style

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy .
```

### ✨ Adding Features

The pipeline is designed to be modular and extensible:

- **New Input Formats**: Extend conversion scripts
- **New LLMs**: Add presets to configuration
- **Custom Cleaning**: Extend `MarkdownCleaner` class
- **Validation Rules**: Modify validation thresholds
- **Chunking Strategies**: Implement new chunking methods

### 📤 Submitting Changes

1. Create feature branch: `git checkout -b feature-name`
2. Make changes and add tests
3. Run test suite: `pytest tests/ -v`
4. Submit pull request with description

## License

MIT License - See LICENSE file for details.
