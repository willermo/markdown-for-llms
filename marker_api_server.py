#!/usr/bin/env python3
"""
Simple FastAPI server for Marker conversion (multi-format)
Compatible with the unified converter pipeline

Supports PDFs and additional document types via Marker providers
(PDF, EPUB, DOCX, PPTX, XLSX, HTML, images), depending on the installed
Marker version and its dependencies.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Marker Converter API", version="1.1.0")

# Global model cache to avoid reloading on each request
_model_dict = None
_converter = None

def get_converter():
    """Get cached converter instance"""
    global _model_dict, _converter
    if _converter is None:
        # Full Marker stack with provider registry
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        
        logger.info("Loading ML models (one-time setup)...")
        _model_dict = create_model_dict()
        _converter = PdfConverter(
            artifact_dict=_model_dict,
            processor_list=None,
            renderer=None
        )
        logger.info("✓ Models loaded and converter ready")
    return _converter

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Initializing Marker API server...")
    # Pre-load models to avoid delays on first request
    try:
        get_converter()
        logger.info("✓ Server ready to accept requests")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        # Continue startup anyway - models will load on first request

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "marker-api"}

@app.post("/convert")
@app.post("/marker")
async def convert_document(
    file: UploadFile = File(...),
    output_format: str = "markdown",
    use_llm: bool = False,
    force_ocr: bool = False,
    paginate: bool = False,
    strip_existing_ocr: bool = True,  # Default to True for text PDFs
    disable_image_extraction: bool = True,  # Default to True for LLM use
    max_pages: Optional[int] = None
):
    """Convert a document to markdown using Marker (multi-format)."""
    
    try:
        # Preserve original file suffix to help provider detection
        suffix = ''.join(Path(file.filename).suffixes) or '.bin'
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Get cached converter (loads models only once)
            converter = get_converter()
            
            # Convert document
            logger.info(f"Converting: {file.filename}")
            document = converter(temp_file_path)
            
            # Handle different document output types
            if hasattr(document, 'render'):
                full_text = document.render()
            elif hasattr(document, 'markdown'):
                full_text = document.markdown
            elif hasattr(document, 'text'):
                full_text = document.text
            else:
                # Fallback: convert to string
                full_text = str(document)
            
            page_count = getattr(converter, 'page_count', 0) or 0
            logger.info(f"✓ Conversion completed for {file.filename} (pages: {page_count})")
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Prepare response
            result = {
                "success": True,
                "markdown": full_text,
                "metadata": {
                    "conversion_method": "marker",
                    "filename": file.filename,
                    "output_format": output_format,
                    "use_llm": use_llm,
                    "force_ocr": force_ocr,
                    "paginate": paginate,
                    "strip_existing_ocr": strip_existing_ocr,
                    "disable_image_extraction": disable_image_extraction,
                },
                "images": {},  # Images handling would need additional implementation
                "page_count": page_count,
            }
            
            return JSONResponse(content=result)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
            
    except ImportError as e:
        logger.error(f"Marker import failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Marker library not properly installed"
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversion failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Marker PDF Converter API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "convert": "/convert (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Marker API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
