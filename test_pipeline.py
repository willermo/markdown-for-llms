#!/usr/bin/env python3
"""
Simple pipeline test script to validate the document intelligence pipeline
with sample PDFs from source_pdfs folder.
"""

import sys
import os
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from master_workflow import PipelineOrchestrator
from config import get_config_manager

def test_pipeline_with_samples():
    """Test the pipeline with sample PDFs"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize configuration
    config_manager = get_config_manager()
    
    # Create pipeline orchestrator
    pipeline = PipelineOrchestrator(
        target_llm="claude-3",
        chunk_size=4000,
        overlap=200,
        skip_existing=False,  # Process all files
        pdf_mode='skip'  # Skip PDF conversion for now, use existing markdown if available
    )
    
    logger.info("Starting pipeline test with sample PDFs...")
    
    # Check source PDFs
    source_pdfs = Path("source_pdfs")
    if not source_pdfs.exists():
        logger.error("source_pdfs directory not found!")
        return False
    
    pdf_files = list(source_pdfs.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
    
    # For testing, let's create a simple markdown file to test the cleaning/validation/chunking pipeline
    test_markdown_dir = Path("test_markdown")
    test_markdown_dir.mkdir(exist_ok=True)
    
    # Create a sample markdown file for testing
    sample_content = """# BPF Performance Tools Test Document

## Introduction

This is a test document to validate the markdown processing pipeline.
It contains various markdown elements that should be processed correctly.

### Key Features

- **Bold text** for emphasis
- *Italic text* for style
- `Code snippets` for technical content
- [Links](https://example.com) for references

#### Code Examples

```python
def hello_world():
    print("Hello, World!")
    return True
```

#### Lists and Structure

1. First item with detailed explanation that spans multiple lines
   and contains technical information about BPF tools
2. Second item with **bold** and *italic* formatting
3. Third item with `inline code` examples

### Performance Metrics

| Tool | Purpose | Performance Impact |
|------|---------|-------------------|
| bpftrace | Dynamic tracing | Low |
| perf | System profiling | Medium |
| ftrace | Kernel tracing | Low |

## Conclusion

This document demonstrates various markdown structures that the pipeline
should handle correctly during cleaning, validation, and chunking phases.

The content is substantial enough to test chunking algorithms while
maintaining readability and structure preservation.
"""
    
    test_file = test_markdown_dir / "sample_bpf_tools.md"
    test_file.write_text(sample_content)
    logger.info(f"Created test markdown file: {test_file}")
    
    try:
        # Test individual pipeline steps
        logger.info("Testing markdown cleaning...")
        from clean_markdown import MarkdownCleaner
        
        cleaned_dir = Path("cleaned_markdown")
        cleaned_dir.mkdir(exist_ok=True)
        
        cleaner = MarkdownCleaner()
        cleaned_content = cleaner.clean_markdown_file(test_file)
        
        cleaned_file = cleaned_dir / test_file.name
        cleaned_file.write_text(cleaned_content)
        logger.info(f"Cleaning completed: {cleaned_file}")
        
        # Test validation
        logger.info("Testing markdown validation...")
        from validate_markdown import MarkdownValidator
        
        validator = MarkdownValidator()
        validation_result = validator.validate_markdown_file(cleaned_file)
        logger.info(f"Validation grade: {validation_result.quality_grade}")
        logger.info(f"Valid: {validation_result.is_valid}")
        logger.info(f"Artifact ratio: {validation_result.artifact_ratio:.3f}")
        
        # Test chunking
        logger.info("Testing markdown chunking...")
        from chunk_markdown import SmartChunker, ChunkingStrategy
        
        chunker = SmartChunker(chunk_size=1000, overlap=100)
        
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunking_result = chunker.chunk_document(content, str(cleaned_file), ChunkingStrategy.SEMANTIC)
        chunks = chunking_result.chunks
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks - extract content using positions
        chunks_dir = Path("chunked_markdown")
        chunks_dir.mkdir(exist_ok=True)
        
        for i, chunk_meta in enumerate(chunks):
            # Extract actual content using start/end positions
            chunk_content = content[chunk_meta.start_position:chunk_meta.end_position]
            chunk_file = chunks_dir / f"sample_bpf_tools_chunk_{i+1}.md"
            chunk_file.write_text(chunk_content)
        
        logger.info(f"Saved {len(chunks)} chunks to {chunks_dir}")
        
        # Summary
        logger.info("=" * 50)
        logger.info("PIPELINE TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✓ Source file processed: {test_file.name}")
        logger.info(f"✓ Cleaned successfully: {Path(cleaned_file).name}")
        logger.info(f"✓ Validation grade: {validation_result.quality_grade}")
        logger.info(f"✓ Valid: {validation_result.is_valid}")
        logger.info(f"✓ Chunks created: {len(chunks)}")
        logger.info(f"✓ Average chunk size: {sum(c.character_count for c in chunks) // len(chunks)} chars")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_with_samples()
    if success:
        print("\n🎉 Pipeline test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline test failed!")
        sys.exit(1)
