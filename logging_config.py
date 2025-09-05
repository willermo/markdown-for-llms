#!/usr/bin/env python3
"""
Standardized logging configuration for the Document Intelligence Pipeline
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

from config import get_config

class PipelineFormatter(logging.Formatter):
    """Custom formatter with color support and structured output"""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_context: bool = True):
        self.use_colors = use_colors
        self.include_context = include_context
        
        if include_context:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        else:
            fmt = '%(asctime)s - %(levelname)s - %(message)s'
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            # Add color to level name
            level_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            
            # Store original levelname
            original_levelname = record.levelname
            record.levelname = f"{level_color}{record.levelname}{reset_color}"
            
            # Format the message
            formatted = super().format(record)
            
            # Restore original levelname
            record.levelname = original_levelname
            
            return formatted
        else:
            return super().format(record)

class PipelineLogger:
    """Centralized logger setup for pipeline components"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.name = name
        self.log_dir = log_dir or get_config().directories.logs
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file and console handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Set log level from config
        config = get_config()
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # File handler with rotation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.log_dir) / f"{self.name}_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB files, keep 5
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_formatter = PipelineFormatter(use_colors=False, include_context=True)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = PipelineFormatter(use_colors=True, include_context=False)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def get_logger(self) -> logging.Logger:
        """Get configured logger instance"""
        return self.logger

# Convenience functions for common loggers
def get_pipeline_logger(component_name: str) -> logging.Logger:
    """Get a standardized logger for a pipeline component"""
    pipeline_logger = PipelineLogger(component_name)
    return pipeline_logger.get_logger()

def get_conversion_logger() -> logging.Logger:
    """Get logger for conversion operations"""
    return get_pipeline_logger("conversion")

def get_cleaning_logger() -> logging.Logger:
    """Get logger for cleaning operations"""
    return get_pipeline_logger("cleaning")

def get_validation_logger() -> logging.Logger:
    """Get logger for validation operations"""
    return get_pipeline_logger("validation")

def get_chunking_logger() -> logging.Logger:
    """Get logger for chunking operations"""
    return get_pipeline_logger("chunking")

def get_orchestrator_logger() -> logging.Logger:
    """Get logger for pipeline orchestration"""
    return get_pipeline_logger("orchestrator")

# Context manager for operation logging
from contextlib import contextmanager
from typing import Any
import time

@contextmanager
def log_operation(logger: logging.Logger, operation: str, **context):
    """Context manager for logging operations with timing"""
    start_time = time.time()
    
    # Log operation start
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.info(f"Starting {operation}" + (f" ({context_str})" if context_str else ""))
    
    try:
        yield
        # Log successful completion
        duration = time.time() - start_time
        logger.info(f"✓ Completed {operation} in {duration:.2f}s")
    except Exception as e:
        # Log failure
        duration = time.time() - start_time
        logger.error(f"✗ Failed {operation} after {duration:.2f}s: {str(e)}")
        raise

# Performance logging decorator
def log_performance(logger: logging.Logger, operation_name: str = None):
    """Decorator for logging function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            with log_operation(logger, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Progress logging helper
class ProgressLogger:
    """Helper for logging progress of batch operations"""
    
    def __init__(self, logger: logging.Logger, total_items: int, operation: str):
        self.logger = logger
        self.total_items = total_items
        self.operation = operation
        self.processed = 0
        self.start_time = time.time()
        
        self.logger.info(f"Starting {operation}: {total_items} items to process")
    
    def update(self, increment: int = 1, item_name: str = None):
        """Update progress counter"""
        self.processed += increment
        
        if self.processed % max(1, self.total_items // 10) == 0 or self.processed == self.total_items:
            # Log every 10% or at completion
            percentage = (self.processed / self.total_items) * 100
            elapsed = time.time() - self.start_time
            
            if self.processed < self.total_items and elapsed > 0:
                # Estimate remaining time
                rate = self.processed / elapsed
                remaining = (self.total_items - self.processed) / rate
                eta_str = f", ETA: {remaining:.0f}s"
            else:
                eta_str = ""
            
            item_str = f" - {item_name}" if item_name else ""
            self.logger.info(f"{self.operation} progress: {self.processed}/{self.total_items} ({percentage:.1f}%){eta_str}{item_str}")
    
    def complete(self):
        """Mark operation as complete"""
        total_time = time.time() - self.start_time
        rate = self.processed / total_time if total_time > 0 else 0
        self.logger.info(f"✓ {self.operation} completed: {self.processed} items in {total_time:.2f}s ({rate:.1f} items/s)")

if __name__ == "__main__":
    # Test the logging system
    logger = get_pipeline_logger("test")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test operation logging
    with log_operation(logger, "test operation", file_count=5):
        time.sleep(0.1)  # Simulate work
    
    # Test progress logging
    progress = ProgressLogger(logger, 5, "test batch operation")
    for i in range(5):
        time.sleep(0.05)  # Simulate work
        progress.update(item_name=f"item_{i}")
    progress.complete()
