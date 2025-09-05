#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for the Document Intelligence Pipeline tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace with standard directory structure"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create standard directory structure
    directories = [
        "source_ebooks",
        "converted_markdown", 
        "cleaned_markdown",
        "validated_markdown",
        "chunked_markdown",
        "pipeline_logs"
    ]
    
    for directory in directories:
        (temp_dir / directory).mkdir(parents=True, exist_ok=True)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_markdown_content():
    """Sample markdown content with various artifacts for testing"""
    return """
# Chapter [T]itle with OCR artifacts

This is a paragraph with D[AN] S. K[ENNEDY] name artifacts and some content.

<span class="artifact">HTML content that should be removed</span>

Some text with &nbsp; HTML entities &amp; more entities like &#8217; quotes.

[]{#pandoc-anchor-artifact}

![Images artifact content that should be cleaned

** broken emphasis formatting **

Multiple    spaces   everywhere    that need normalization.



Excessive newlines above that should be reduced.

## Valid Section

This is good content that should be preserved during cleaning.

- Valid list item 1
- Valid list item 2

### Subsection

More good content here with proper formatting.

ab
cd

Valid paragraph after short lines that should be removed.
"""

@pytest.fixture
def high_quality_content():
    """High-quality markdown content for validation testing"""
    return """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on 
algorithms that can learn from and make predictions on data. This field 
has grown tremendously in recent years due to advances in computing power 
and the availability of large datasets.

## Types of Machine Learning

There are three main types of machine learning approaches:

### Supervised Learning

Supervised learning uses labeled training data to learn a mapping function
from input variables to output variables. Common applications include:

- Image classification and computer vision
- Natural language processing and sentiment analysis
- Predictive analytics and forecasting
- Medical diagnosis and healthcare applications

The quality of supervised learning models depends heavily on the quality 
and quantity of training data available. More diverse and representative 
datasets typically lead to better model performance.

### Unsupervised Learning

Unsupervised learning finds patterns in data without labeled examples.
Key techniques include clustering, dimensionality reduction, and 
association rule learning.

### Reinforcement Learning

Reinforcement learning involves agents learning through interaction
with an environment, receiving rewards or penalties for actions taken.
This approach has shown remarkable success in game playing, robotics,
and autonomous systems.

## Conclusion

Machine learning continues to evolve rapidly, with new architectures
and techniques emerging regularly. Understanding these fundamental
concepts provides a solid foundation for exploring more advanced topics.
"""

@pytest.fixture
def low_quality_content():
    """Low-quality content with many issues for testing"""
    return """
# Bad [D]oc

<div class="bad">HTML content</div>

Short.

&nbsp;&amp;

** bad format **

ab
x
"""

@pytest.fixture
def sample_epub_file(temp_workspace):
    """Create a sample EPUB file for testing"""
    epub_file = temp_workspace / "source_ebooks" / "test_book.epub"
    epub_file.write_text("Sample EPUB content for testing")
    return epub_file

@pytest.fixture
def sample_pdf_file(temp_workspace):
    """Create a sample PDF file for testing"""
    pdf_file = temp_workspace / "source_ebooks" / "test_document.pdf"
    pdf_file.write_text("Sample PDF content for testing")
    return pdf_file

@pytest.fixture
def mock_pandoc_success(monkeypatch):
    """Mock successful Pandoc conversion"""
    def mock_run(*args, **kwargs):
        # Extract output file from args
        if len(args) > 0 and isinstance(args[0], list):
            cmd_args = args[0]
            if '-o' in cmd_args:
                output_idx = cmd_args.index('-o') + 1
                if output_idx < len(cmd_args):
                    output_file = Path(cmd_args[output_idx])
                    output_file.write_text("# Converted Content\n\nSample converted markdown.")
        
        # Return mock successful result
        return type('MockResult', (), {
            'returncode': 0,
            'stdout': 'Conversion successful',
            'stderr': ''
        })()
    
    monkeypatch.setattr('subprocess.run', mock_run)

@pytest.fixture
def mock_pandoc_failure(monkeypatch):
    """Mock failed Pandoc conversion"""
    def mock_run(*args, **kwargs):
        return type('MockResult', (), {
            'returncode': 1,
            'stdout': '',
            'stderr': 'Pandoc conversion failed'
        })()
    
    monkeypatch.setattr('subprocess.run', mock_run)

@pytest.fixture
def test_config():
    """Test configuration for pipeline components"""
    from config import PipelineConfig, DirectoryConfig, ValidationThresholds, CleaningConfig, ChunkingConfig, LLMModel, ConversionConfig
    
    return PipelineConfig(
        directories=DirectoryConfig(),
        llm_presets={},
        validation=ValidationThresholds(
            min_content_length=50,
            min_words=10,
            max_artifact_ratio=0.1
        ),
        cleaning=CleaningConfig(
            aggressive_cleaning=True,
            min_line_length=2
        ),
        chunking=ChunkingConfig(
            target_llm=LLMModel.CUSTOM,
            chunk_size=1000,
            overlap=100
        ),
        conversion=ConversionConfig()
    )

# Test data constants
SAMPLE_CHUNK_METADATA = {
    "chunk_id": "test_chunk_001",
    "source_file": "test.md",
    "chunk_index": 0,
    "total_chunks": 3,
    "token_count": 150,
    "word_count": 100,
    "heading_context": ["Chapter 1", "Introduction"],
    "content_type": "paragraph"
}

SAMPLE_VALIDATION_ISSUES = {
    "html_tags": ["<div>", "<span>"],
    "encoding_issues": ["Smart quote issue"],
    "formatting_issues": ["Broken emphasis"],
    "suspicious_patterns": ["ocr_artifacts: 3 instances"]
}
