#!/usr/bin/env python3
"""
Unit tests for markdown cleaning functionality
"""

import pytest
from pathlib import Path
import tempfile
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from clean_markdown import MarkdownCleaner
from exceptions import CleaningError, FileReadError

class TestMarkdownCleaner:
    
    @pytest.fixture
    def cleaner(self):
        """Create a MarkdownCleaner instance for testing"""
        return MarkdownCleaner(aggressive=True)
    
    def test_clean_ocr_artifacts(self, cleaner, sample_markdown_content):
        """Test OCR artifact removal"""
        result = cleaner.clean_ocr_artifacts(sample_markdown_content)
        
        # Should fix bracketed letters
        assert "[T]itle" not in result
        assert "Title" in result
        
        # Should fix bracketed names
        assert "D[AN] S. K[ENNEDY]" not in result
        assert "DAN S. KENNEDY" in result
    
    def test_clean_html_artifacts(self, cleaner):
        """Test HTML tag and entity removal"""
        content = '<span class="test">Text</span> with &nbsp; entities &amp; more'
        result = cleaner.clean_html_artifacts(content)
        
        # Should remove HTML tags
        assert "<span" not in result
        assert "</span>" not in result
        assert "Text" in result
        
        # Should replace HTML entities
        assert "&nbsp;" not in result
        assert "&amp;" not in result
        assert " " in result  # nbsp replacement
        assert "&" in result  # amp replacement
    
    def test_clean_pandoc_artifacts(self, cleaner):
        """Test Pandoc artifact removal"""
        content = "Text []{#anchor} with {.class} attributes and ::: divs"
        result = cleaner.clean_pandoc_artifacts(content)
        
        # Should remove Pandoc artifacts
        assert "[]{#anchor}" not in result
        assert "{.class}" not in result
        assert ":::" not in result
        assert "Text" in result
        assert "with" in result
        assert "attributes" in result
    
    def test_normalize_whitespace(self, cleaner):
        """Test whitespace normalization"""
        content = "Text  with   multiple    spaces\n\n\n\nand excessive newlines"
        result = cleaner.normalize_whitespace(content)
        
        # Should normalize multiple spaces
        assert "  " not in result
        
        # Should normalize excessive newlines
        assert "\n\n\n" not in result
        assert "\n\n" in result  # Should keep paragraph breaks
    
    def test_remove_short_lines(self, cleaner):
        """Test removal of meaningless short lines"""
        lines = [
            "# Valid Heading",
            "Valid paragraph content here",
            "ab",  # Too short
            "- Valid list item",
            "x",   # Too short
            "",    # Empty line (should be kept)
            "Another valid paragraph"
        ]
        
        result = cleaner.remove_short_lines(lines, min_length=3)
        
        # Should keep valid content
        assert "# Valid Heading" in result
        assert "Valid paragraph content here" in result
        assert "- Valid list item" in result
        assert "Another valid paragraph" in result
        
        # Should remove short lines
        assert "ab" not in result
        assert "x" not in result
        
        # Should keep empty lines for structure
        assert "" in result
    
    def test_clean_formatting_artifacts(self, cleaner):
        """Test formatting artifact cleanup"""
        content = "** broken emphasis ** and _ broken italic _"
        result = cleaner.clean_formatting_artifacts(content)
        
        # Should fix broken formatting
        assert "**broken emphasis**" in result
        assert "_broken italic_" in result
    
    def test_clean_image_artifacts(self, cleaner):
        """Test image artifact removal"""
        content = "Text !Images artifact and ![](empty.jpg) reference"
        result = cleaner.clean_image_artifacts(content)
        
        # Should remove image artifacts
        assert "!Images" not in result
        assert "![](empty.jpg)" not in result
        assert "Text" in result
        assert "reference" in result
    
    def test_clean_markdown_file_integration(self, cleaner, sample_markdown_content, tmp_path):
        """Integration test for complete file cleaning"""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text(sample_markdown_content)
        
        # Clean the file
        result = cleaner.clean_markdown_file(test_file)
        
        # Verify cleaning results
        assert "[T]itle" not in result
        assert "Title" in result
        assert "DAN S. KENNEDY" in result
        assert "<span" not in result
        assert "&nbsp;" not in result
        assert "**broken emphasis formatting**" in result  # Should fix spacing
        # Verify the standalone short line 'ab' was removed (avoid false positives like 'above')
        assert "ab" not in [ln.strip() for ln in result.splitlines()]
        assert "Valid Section" in result
        assert "This is good content" in result
    
    def test_statistics_tracking(self, cleaner, sample_markdown_content, tmp_path):
        """Test that cleaning statistics are properly tracked"""
        test_file = tmp_path / "test.md"
        test_file.write_text(sample_markdown_content)
        
        original_stats = cleaner.stats.copy()
        
        # Process content
        result = cleaner.clean_markdown_file(test_file)
        
        # Verify stats were updated
        assert cleaner.stats['files_processed'] > original_stats['files_processed']
        assert cleaner.stats['characters_removed'] >= 0
        assert cleaner.stats['lines_removed'] >= 0
    
    def test_error_handling_invalid_file(self, cleaner, tmp_path):
        """Test error handling for invalid files"""
        non_existent_file = tmp_path / "nonexistent.md"
        
        # Should raise a FileReadError for missing file
        with pytest.raises(FileReadError):
            cleaner.clean_markdown_file(non_existent_file)
    
    def test_error_handling_permission_denied(self, cleaner, tmp_path, monkeypatch):
        """Test error handling for permission issues"""
        test_file = tmp_path / "protected.md"
        test_file.write_text("test content")
        
        # Mock permission error
        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")
        
        monkeypatch.setattr('builtins.open', mock_open)
        
        # Should raise FileReadError on permission issues
        with pytest.raises(FileReadError):
            cleaner.clean_markdown_file(test_file)
    
    @pytest.mark.parametrize("aggressive,min_length,expected_removals", [
        (True, 3, True),   # Aggressive mode should remove short lines
        (False, 3, False), # Non-aggressive should be more conservative
        (True, 1, False),  # Very low threshold should keep most lines
    ])
    def test_aggressive_mode_behavior(self, aggressive, min_length, expected_removals):
        """Test different cleaning modes"""
        cleaner = MarkdownCleaner(aggressive=aggressive)
        lines = ["# Heading", "ab", "cd", "Valid content here"]
        
        result = cleaner.remove_short_lines(lines, min_length=min_length)
        
        if expected_removals and min_length > 2:
            # Should remove short lines in aggressive mode with reasonable threshold
            assert "ab" not in result
            assert "cd" not in result
        
        # Should always keep valid content
        assert "# Heading" in result
        assert "Valid content here" in result
    
    def test_get_statistics(self, cleaner):
        """Test statistics reporting"""
        # Process some content to generate stats
        cleaner.stats['files_processed'] = 5
        cleaner.stats['characters_removed'] = 1000
        cleaner.stats['lines_removed'] = 50
        
        stats_report = cleaner.get_statistics()
        
        assert "Files processed: 5" in stats_report
        assert "characters removed: 1,000" in stats_report
        assert "Lines removed: 50" in stats_report
        assert "Average reduction: 200" in stats_report
    
    def test_edge_cases_empty_content(self, cleaner, tmp_path):
        """Test handling of empty files"""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        result = cleaner.clean_markdown_file(empty_file)
        assert result == ""
    
    def test_edge_cases_whitespace_only(self, cleaner, tmp_path):
        """Test handling of whitespace-only files"""
        whitespace_file = tmp_path / "whitespace.md"
        whitespace_file.write_text("   \n\n   \t\t  \n")
        
        result = cleaner.clean_markdown_file(whitespace_file)
        # Should be empty or minimal after cleaning
        assert len(result.strip()) == 0
    
    def test_preserve_valid_markdown_structure(self, cleaner, high_quality_content, tmp_path):
        """Test that valid markdown structure is preserved"""
        test_file = tmp_path / "quality.md"
        test_file.write_text(high_quality_content)
        
        result = cleaner.clean_markdown_file(test_file)
        
        # Should preserve headings
        assert "# Introduction to Machine Learning" in result
        assert "## Types of Machine Learning" in result
        assert "### Supervised Learning" in result
        
        # Should preserve lists
        assert "- Image classification" in result
        
        # Should preserve paragraph structure
        assert "Machine learning is a subset" in result
    
    def test_unicode_handling(self, cleaner, tmp_path):
        """Test proper Unicode character handling"""
        unicode_content = "Text with émojis 🚀 and accénted characters"
        test_file = tmp_path / "unicode.md"
        test_file.write_text(unicode_content, encoding='utf-8')
        
        result = cleaner.clean_markdown_file(test_file)
        
        # Should preserve Unicode characters
        assert "émojis" in result
        assert "🚀" in result
        assert "accénted" in result
    
    def test_large_file_handling(self, cleaner, tmp_path):
        """Test handling of large files"""
        # Create a large content string
        large_content = "# Large Document\n\n" + ("This is a paragraph. " * 1000)
        test_file = tmp_path / "large.md"
        test_file.write_text(large_content)
        
        result = cleaner.clean_markdown_file(test_file)
        
        # Should handle large files without issues
        assert "# Large Document" in result
        assert len(result) > 0
        assert "This is a paragraph." in result
