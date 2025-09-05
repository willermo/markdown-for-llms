#!/usr/bin/env python3
"""
Custom exception classes for the Document Intelligence Pipeline
"""

class DocumentProcessingError(Exception):
    """Base exception for all document processing errors"""
    
    def __init__(self, message: str, file_path: str = None, stage: str = None):
        self.message = message
        self.file_path = file_path
        self.stage = stage
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with context"""
        parts = []
        if self.stage:
            parts.append(f"[{self.stage}]")
        if self.file_path:
            parts.append(f"File: {self.file_path}")
        parts.append(self.message)
        return " - ".join(parts)

class FileProcessingError(DocumentProcessingError):
    """Raised when file operations fail"""
    pass

class FileReadError(FileProcessingError):
    """Raised when file cannot be read"""
    pass

class FileWriteError(FileProcessingError):
    """Raised when file cannot be written"""
    pass

class ConversionError(DocumentProcessingError):
    """Raised when document conversion fails"""
    pass

class PandocError(ConversionError):
    """Raised when Pandoc conversion fails"""
    pass

class MarkerAPIError(ConversionError):
    """Raised when Marker API calls fail"""
    
    def __init__(self, message: str, status_code: int = None, api_response: str = None, **kwargs):
        self.status_code = status_code
        self.api_response = api_response
        super().__init__(message, **kwargs)

class CleaningError(DocumentProcessingError):
    """Raised when markdown cleaning fails"""
    pass

class ValidationError(DocumentProcessingError):
    """Raised when document validation fails"""
    pass

class ChunkingError(DocumentProcessingError):
    """Raised when document chunking fails"""
    pass

class TokenizationError(ChunkingError):
    """Raised when token counting or splitting fails"""
    pass

class ConfigurationError(DocumentProcessingError):
    """Raised when configuration is invalid"""
    pass

class PipelineError(DocumentProcessingError):
    """Raised when pipeline orchestration fails"""
    
    def __init__(self, message: str, failed_step: str = None, **kwargs):
        self.failed_step = failed_step
        super().__init__(message, **kwargs)

class DependencyError(DocumentProcessingError):
    """Raised when required dependencies are missing"""
    pass

# Error context manager for better error handling
from contextlib import contextmanager
from typing import Type, Optional
import logging

@contextmanager
def error_context(stage: str, file_path: Optional[str] = None, 
                 exception_type: Type[DocumentProcessingError] = DocumentProcessingError):
    """Context manager for consistent error handling"""
    try:
        yield
    except DocumentProcessingError:
        # Re-raise our custom exceptions as-is
        raise
    except Exception as e:
        # Wrap other exceptions in our custom exception
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error in {stage}: {str(e)}")
        raise exception_type(
            message=f"Unexpected error: {str(e)}",
            file_path=file_path,
            stage=stage
        ) from e
