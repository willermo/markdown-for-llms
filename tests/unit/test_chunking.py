#!/usr/bin/env python3
"""
Unit tests for markdown chunking functionality
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from chunk_markdown import MarkdownChunker, ChunkMetadata
from exceptions import ChunkingError, TokenizationError

class TestMarkdownChunker:
    
    @pytest.fixture
    def chunker(self, test_config):
        """Create a MarkdownChunker instance for testing"""
        return MarkdownChunker(
            chunk_size=test_config.chunking.chunk_size,
            overlap=test_config.chunking.overlap,
            target_llm=test_config.chunking.target_llm.value
        )
    
    def test_count_tokens_basic(self, chunker):
        """Test basic token counting"""
        text = "This is a simple test sentence."
        token_count = chunker.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 20  # Should be reasonable for short text
    
    def test_count_tokens_empty(self, chunker):
        """Test token counting with empty text"""
        assert chunker.count_tokens("") == 0
        assert chunker.count_tokens("   ") == 0
    
    def test_split_by_sentences(self, chunker):
        """Test sentence splitting functionality"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = chunker.split_by_sentences(text)
        
        assert len(sentences) == 4
        assert "First sentence." in sentences
        assert "Second sentence!" in sentences
        assert "Third sentence?" in sentences
        assert "Fourth sentence." in sentences
    
    def test_extract_headings(self, chunker, high_quality_content):
        """Test heading extraction from markdown"""
        headings = chunker.extract_headings(high_quality_content)
        
        assert len(headings) > 0
        assert any("Introduction to Machine Learning" in h for h in headings)
        assert any("Types of Machine Learning" in h for h in headings)
        assert any("Supervised Learning" in h for h in headings)
    
    def test_semantic_chunking(self, chunker, high_quality_content):
        """Test semantic chunking strategy"""
        chunks = chunker.chunk_by_semantic_sections(high_quality_content)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Each chunk should be within size limits
        for chunk in chunks:
            token_count = chunker.count_tokens(chunk)
            assert token_count <= chunker.chunk_size + chunker.overlap
    
    def test_fixed_size_chunking(self, chunker, high_quality_content):
        """Test fixed-size chunking strategy"""
        chunks = chunker.chunk_by_fixed_size(high_quality_content)
        
        assert len(chunks) > 0
        
        # All chunks except possibly the last should be close to target size
        for i, chunk in enumerate(chunks[:-1]):
            token_count = chunker.count_tokens(chunk)
            assert token_count >= chunker.chunk_size * 0.8  # Allow some variance
    
    def test_sliding_window_chunking(self, chunker, high_quality_content):
        """Test sliding window chunking strategy"""
        chunks = chunker.chunk_by_sliding_window(high_quality_content)
        
        assert len(chunks) > 0
        
        # Should have overlap between consecutive chunks
        if len(chunks) > 1:
            # Check for some content overlap (simplified check)
            first_chunk_end = chunks[0][-100:]  # Last 100 chars
            second_chunk_start = chunks[1][:200]  # First 200 chars
            
            # There should be some common words due to overlap
            first_words = set(first_chunk_end.split())
            second_words = set(second_chunk_start.split())
            common_words = first_words.intersection(second_words)
            assert len(common_words) > 0
    
    def test_create_chunk_metadata(self, chunker):
        """Test chunk metadata creation"""
        chunk_content = "# Test Heading\n\nThis is test content for metadata."
        source_file = Path("test.md")
        chunk_index = 0
        total_chunks = 3
        
        metadata = chunker.create_chunk_metadata(
            chunk_content, source_file, chunk_index, total_chunks
        )
        
        assert isinstance(metadata, ChunkMetadata)
        assert metadata.source_file == "test.md"
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 3
        assert metadata.token_count > 0
        assert metadata.word_count > 0
        assert len(metadata.heading_context) > 0
        assert "Test Heading" in metadata.heading_context
    
    def test_save_chunk_with_metadata(self, chunker, tmp_path):
        """Test saving chunks with metadata headers"""
        chunk_content = "# Test Content\n\nThis is a test chunk."
        metadata = ChunkMetadata(
            chunk_id="test_001",
            source_file="test.md",
            chunk_index=0,
            total_chunks=1,
            token_count=10,
            word_count=8,
            heading_context=["Test Content"],
            content_type="section"
        )
        
        output_file = tmp_path / "chunk_001.md"
        chunker.save_chunk_with_metadata(chunk_content, metadata, output_file)
        
        # Verify file was created and contains metadata
        assert output_file.exists()
        content = output_file.read_text()
        
        assert "---" in content  # YAML frontmatter
        assert "chunk_id: test_001" in content
        assert "source_file: test.md" in content
        assert "# Test Content" in content
    
    def test_chunk_markdown_file_integration(self, chunker, high_quality_content, tmp_path):
        """Integration test for complete file chunking"""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text(high_quality_content)
        
        # Create output directory
        output_dir = tmp_path / "chunks"
        output_dir.mkdir()
        
        # Chunk the file
        chunk_files = chunker.chunk_markdown_file(test_file, output_dir)
        
        assert len(chunk_files) > 0
        assert all(f.exists() for f in chunk_files)
        
        # Verify chunk files contain proper content and metadata
        for chunk_file in chunk_files:
            content = chunk_file.read_text()
            assert "---" in content  # Should have metadata
            assert len(content.strip()) > 0
    
    def test_batch_chunking(self, chunker, high_quality_content, tmp_path):
        """Test batch processing of multiple files"""
        # Create multiple test files
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        for i in range(3):
            test_file = input_dir / f"test_{i}.md"
            test_file.write_text(high_quality_content)
        
        # Process batch
        results = chunker.process_directory(input_dir, output_dir)
        
        assert len(results) == 3
        assert all(len(chunks) > 0 for chunks in results.values())
        
        # Verify output files exist
        for source_file, chunk_files in results.items():
            assert all(f.exists() for f in chunk_files)
    
    def test_error_handling_invalid_file(self, chunker, tmp_path):
        """Test error handling for invalid files"""
        non_existent = tmp_path / "missing.md"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should handle missing file gracefully
        result = chunker.chunk_markdown_file(non_existent, output_dir)
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_error_handling_empty_content(self, chunker, tmp_path):
        """Test handling of empty content"""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = chunker.chunk_markdown_file(empty_file, output_dir)
        assert isinstance(result, list)
        # May return empty list or single minimal chunk
        assert len(result) <= 1
    
    @pytest.mark.parametrize("strategy", ["semantic", "fixed_size", "sliding_window"])
    def test_chunking_strategies(self, chunker, high_quality_content, strategy):
        """Test different chunking strategies"""
        if strategy == "semantic":
            chunks = chunker.chunk_by_semantic_sections(high_quality_content)
        elif strategy == "fixed_size":
            chunks = chunker.chunk_by_fixed_size(high_quality_content)
        elif strategy == "sliding_window":
            chunks = chunker.chunk_by_sliding_window(high_quality_content)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)
    
    def test_chunk_size_limits(self, chunker):
        """Test chunk size enforcement"""
        # Create content that's longer than chunk size
        long_content = "This is a sentence. " * 1000
        
        chunks = chunker.chunk_by_fixed_size(long_content)
        
        # All chunks should respect size limits
        for chunk in chunks:
            token_count = chunker.count_tokens(chunk)
            assert token_count <= chunker.chunk_size + chunker.overlap
    
    def test_preserve_markdown_structure(self, chunker):
        """Test that markdown structure is preserved in chunks"""
        content_with_structure = """
        # Main Heading
        
        ## Subheading 1
        
        Content under subheading 1 with **bold** and *italic* text.
        
        - List item 1
        - List item 2
        
        ## Subheading 2
        
        More content here with [links](http://example.com) and `code`.
        
        ```python
        def example():
            return "code block"
        ```
        """
        
        chunks = chunker.chunk_by_semantic_sections(content_with_structure)
        
        # Should preserve markdown formatting
        combined = "\n".join(chunks)
        assert "# Main Heading" in combined
        assert "## Subheading" in combined
        assert "**bold**" in combined
        assert "- List item" in combined
        assert "```python" in combined
    
    def test_unicode_handling(self, chunker, tmp_path):
        """Test proper Unicode handling in chunking"""
        unicode_content = """
        # Título con Émojis 🚀
        
        Contenido con caracteres especiales: café, naïve, résumé.
        
        ## Sección con Más Contenido
        
        Este párrafo contiene suficiente texto para ser dividido en chunks
        mientras preserva los caracteres Unicode correctamente.
        """
        
        test_file = tmp_path / "unicode.md"
        test_file.write_text(unicode_content, encoding='utf-8')
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        chunk_files = chunker.chunk_markdown_file(test_file, output_dir)
        
        # Verify Unicode is preserved
        for chunk_file in chunk_files:
            content = chunk_file.read_text(encoding='utf-8')
            # Should contain Unicode characters
            assert any(char in content for char in ['é', '🚀', 'ñ'])
    
    def test_statistics_tracking(self, chunker, high_quality_content, tmp_path):
        """Test chunking statistics tracking"""
        test_file = tmp_path / "test.md"
        test_file.write_text(high_quality_content)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Track initial stats
        initial_stats = chunker.get_statistics()
        
        # Process file
        chunker.chunk_markdown_file(test_file, output_dir)
        
        # Check updated stats
        final_stats = chunker.get_statistics()
        
        assert final_stats['files_processed'] > initial_stats['files_processed']
        assert final_stats['total_chunks'] >= initial_stats['total_chunks']
    
    def test_chunk_metadata_completeness(self, chunker):
        """Test that chunk metadata contains all required fields"""
        chunk_content = "# Test\n\nContent here."
        metadata = chunker.create_chunk_metadata(
            chunk_content, Path("test.md"), 0, 1
        )
        
        # Verify all required fields are present
        assert hasattr(metadata, 'chunk_id')
        assert hasattr(metadata, 'source_file')
        assert hasattr(metadata, 'chunk_index')
        assert hasattr(metadata, 'total_chunks')
        assert hasattr(metadata, 'token_count')
        assert hasattr(metadata, 'word_count')
        assert hasattr(metadata, 'heading_context')
        assert hasattr(metadata, 'content_type')
        
        # Verify values are reasonable
        assert metadata.token_count > 0
        assert metadata.word_count > 0
        assert len(metadata.heading_context) > 0
