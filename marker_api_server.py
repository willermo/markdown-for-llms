#!/usr/bin/env python3
"""
Simple FastAPI server for Marker PDF conversion
Compatible with the unified converter pipeline
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

app = FastAPI(title="Marker PDF Converter API", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "marker-api"}

@app.post("/convert")
async def convert_pdf(
    file: UploadFile = File(...),
    output_format: str = "markdown",
    use_llm: bool = False,
    force_ocr: bool = False,
    paginate: bool = False,
    strip_existing_ocr: bool = False,
    disable_image_extraction: bool = False,
    max_pages: Optional[int] = None
):
    """Convert PDF to markdown using Marker"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Import marker here to catch import errors
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
        
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create configuration
            config = ConfigParser({
                'output_format': output_format,
                'languages': ['English'],
                'force_ocr': force_ocr,
                'paginate': paginate,
                'max_pages': max_pages
            })
            
            # Load models (this is expensive, should be cached in production)
            model_dict = create_model_dict()
            
            # Create converter
            converter = PdfConverter(
                artifact_dict=model_dict,
                processor_list=None,
                renderer=None
            )
            
            # Convert PDF
            document = converter(temp_file_path)
            full_text = document.render()
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Prepare response
            result = {
                "success": True,
                "markdown": full_text,
                "metadata": {"conversion_method": "marker-pdf", "filename": file.filename},
                "images": {},  # Images handling would need additional implementation
                "page_count": 0  # Page count would need additional implementation
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
