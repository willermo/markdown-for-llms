#!/usr/bin/env python3
"""
Unit tests for markdown validation functionality
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validate_markdown import MarkdownValidator, ValidationResult
from exceptions import ValidationError

class TestMarkdownValidator:
    
    @pytest.fixture
    def validator(self, test_config):
        """Create a MarkdownValidator instance for testing"""
        return MarkdownValidator(config=test_config.validation)
    
    def test_detect_html_artifacts(self, validator):
        """Test HTML artifact detection"""
        content_with_html = "Text with <div>HTML</div> and <span class='test'>more</span>"
        content_clean = "Clean text without HTML artifacts"
        
        html_tags = validator.detect_html_artifacts(content_with_html)
        clean_tags = validator.detect_html_artifacts(content_clean)
        
        assert len(html_tags) == 2
        assert "<div>" in html_tags
        assert "<span class='test'>" in html_tags
        assert len(clean_tags) == 0
    
    def test_detect_encoding_issues(self, validator):
        """Test encoding issue detection"""
        content_with_issues = "Text with â€™ smart quotes and Ã© accents"
        content_clean = "Text with proper 'quotes' and é accents"
        
        issues = validator.detect_encoding_issues(content_with_issues)
        clean_issues = validator.detect_encoding_issues(content_clean)
        
        assert len(issues) > 0
        assert any("â€™" in issue for issue in issues)
        assert len(clean_issues) == 0
    
    def test_detect_formatting_issues(self, validator):
        """Test formatting issue detection"""
        content_with_issues = "Text with ** broken bold ** and _ broken italic _"
        content_clean = "Text with **proper bold** and _proper italic_"
        
        issues = validator.detect_formatting_issues(content_with_issues)
        clean_issues = validator.detect_formatting_issues(content_clean)
        
        assert len(issues) > 0
        assert any("broken bold" in issue for issue in issues)
        assert len(clean_issues) == 0
    
    def test_calculate_readability_score(self, validator, high_quality_content, low_quality_content):
        """Test readability score calculation"""
        high_score = validator.calculate_readability_score(high_quality_content)
        low_score = validator.calculate_readability_score(low_quality_content)
        
        # High quality content should score better
        assert high_score > low_score
        assert high_score >= 50  # Should be reasonably readable
        assert low_score < 50    # Should score poorly
    
    def test_calculate_structure_score(self, validator, high_quality_content, low_quality_content):
        """Test structure score calculation"""
        high_score = validator.calculate_structure_score(high_quality_content)
        low_score = validator.calculate_structure_score(low_quality_content)
        
        # High quality content should have better structure
        assert high_score > low_score
        assert high_score >= 70  # Should have good structure
    
    def test_detect_suspicious_patterns(self, validator):
        """Test suspicious pattern detection"""
        content_with_patterns = """
        Text with D[AN] S. K[ENNEDY] OCR artifacts and
        multiple    spaces   everywhere and
        !Images artifact content and
        []{#pandoc-anchor} artifacts
        """
        
        patterns = validator.detect_suspicious_patterns(content_with_patterns)
        
        assert len(patterns) > 0
        assert any("ocr_artifacts" in pattern for pattern in patterns)
        assert any("excessive_whitespace" in pattern for pattern in patterns)
    
    def test_assign_quality_grade(self, validator):
        """Test quality grade assignment"""
        # Test different score combinations
        assert validator.assign_quality_grade(95, 90, 0.01) == 'A'
        assert validator.assign_quality_grade(85, 80, 0.02) == 'B'
        assert validator.assign_quality_grade(75, 70, 0.03) == 'C'
        assert validator.assign_quality_grade(65, 60, 0.04) == 'D'
        assert validator.assign_quality_grade(45, 40, 0.08) == 'F'
    
    def test_generate_recommendations(self, validator):
        """Test recommendation generation"""
        issues = {
            'html_tags': ['<div>', '<span>'],
            'encoding_issues': ['Smart quote issue'],
            'formatting_issues': ['Broken emphasis'],
            'suspicious_patterns': ['ocr_artifacts: 3 instances']
        }
        
        recommendations = validator.generate_recommendations(issues)
        
        assert len(recommendations) > 0
        assert any("HTML tags" in rec for rec in recommendations)
        assert any("encoding" in rec for rec in recommendations)
        assert any("OCR artifacts" in rec for rec in recommendations)
    
    def test_validate_markdown_file_high_quality(self, validator, high_quality_content, tmp_path):
        """Test validation of high-quality content"""
        test_file = tmp_path / "high_quality.md"
        test_file.write_text(high_quality_content)
        
        result = validator.validate_markdown_file(test_file)
        
        assert isinstance(result, ValidationResult)
        assert result.file_path == test_file
        assert result.quality_grade in ['A', 'B']  # Should be high quality
        assert result.readability_score >= 50
        assert result.structure_score >= 70
        assert len(result.issues['html_tags']) == 0
    
    def test_validate_markdown_file_low_quality(self, validator, low_quality_content, tmp_path):
        """Test validation of low-quality content"""
        test_file = tmp_path / "low_quality.md"
        test_file.write_text(low_quality_content)
        
        result = validator.validate_markdown_file(test_file)
        
        assert isinstance(result, ValidationResult)
        assert result.quality_grade in ['D', 'F']  # Should be low quality
        assert len(result.issues['html_tags']) > 0
        assert len(result.recommendations) > 0
    
    def test_validate_nonexistent_file(self, validator, tmp_path):
        """Test validation of non-existent file"""
        non_existent = tmp_path / "missing.md"
        
        result = validator.validate_markdown_file(non_existent)
        
        # Should handle gracefully
        assert result is None or result.quality_grade == 'F'
    
    def test_batch_validation(self, validator, high_quality_content, low_quality_content, tmp_path):
        """Test batch validation of multiple files"""
        # Create test files
        files = []
        for i, content in enumerate([high_quality_content, low_quality_content]):
            test_file = tmp_path / f"test_{i}.md"
            test_file.write_text(content)
            files.append(test_file)
        
        results = validator.validate_directory(tmp_path)
        
        assert len(results) == 2
        assert all(isinstance(r, ValidationResult) for r in results)
        
        # Should have different quality grades
        grades = [r.quality_grade for r in results]
        assert len(set(grades)) > 1  # Should have variety in grades
    
    def test_validation_statistics(self, validator, tmp_path):
        """Test validation statistics tracking"""
        # Create test files with different quality levels
        contents = [
            "# Good Content\n\nThis is well-formatted content with proper structure.",
            "<div>Bad HTML content</div> with ** broken ** formatting",
            "# Another Good One\n\nMore quality content here."
        ]
        
        for i, content in enumerate(contents):
            test_file = tmp_path / f"test_{i}.md"
            test_file.write_text(content)
        
        results = validator.validate_directory(tmp_path)
        stats = validator.get_validation_statistics(results)
        
        assert 'total_files' in stats
        assert 'grade_distribution' in stats
        assert 'average_scores' in stats
        assert stats['total_files'] == 3
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass functionality"""
        result = ValidationResult(
            file_path=Path("test.md"),
            content_length=1000,
            word_count=150,
            readability_score=75,
            structure_score=80,
            artifact_ratio=0.02,
            quality_grade='B',
            issues={'html_tags': [], 'encoding_issues': []},
            recommendations=['Improve readability']
        )
        
        assert result.file_path == Path("test.md")
        assert result.quality_grade == 'B'
        assert len(result.recommendations) == 1
    
    @pytest.mark.parametrize("content_length,word_count,expected_valid", [
        (1000, 150, True),   # Good length
        (50, 8, False),      # Too short
        (20000000, 3000000, False),  # Too long
        (500, 75, True),     # Acceptable
    ])
    def test_content_length_validation(self, validator, content_length, word_count, expected_valid):
        """Test content length validation with different inputs"""
        # Create content of specified length
        content = "word " * word_count
        content = content[:content_length] if len(content) > content_length else content
        
        # Mock the file reading to return our test content
        class MockPath:
            def read_text(self, encoding=None):
                return content
            def __str__(self):
                return "test.md"
        
        result = validator.validate_markdown_file(MockPath())
        
        if expected_valid:
            assert result.quality_grade != 'F'
        else:
            # Very short or very long content should get poor grades
            assert result.quality_grade in ['D', 'F']
    
    def test_edge_cases_empty_file(self, validator, tmp_path):
        """Test validation of empty files"""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        result = validator.validate_markdown_file(empty_file)
        
        assert result.quality_grade == 'F'
        assert result.content_length == 0
        assert result.word_count == 0
    
    def test_edge_cases_unicode_content(self, validator, tmp_path):
        """Test validation of Unicode content"""
        unicode_content = """
        # Título con Acentos
        
        Contenido con émojis 🚀 y caracteres especiales: café, naïve, résumé.
        
        ## Sección con Más Contenido
        
        Este es un párrafo con suficiente contenido para ser considerado
        de buena calidad desde el punto de vista de longitud y estructura.
        """
        
        test_file = tmp_path / "unicode.md"
        test_file.write_text(unicode_content, encoding='utf-8')
        
        result = validator.validate_markdown_file(test_file)
        
        # Should handle Unicode properly
        assert result is not None
        assert result.content_length > 0
        assert result.word_count > 0
    
    def test_performance_large_file(self, validator, tmp_path):
        """Test validation performance on large files"""
        # Create a large file
        large_content = "# Large Document\n\n" + ("This is a sentence. " * 5000)
        test_file = tmp_path / "large.md"
        test_file.write_text(large_content)
        
        import time
        start_time = time.time()
        result = validator.validate_markdown_file(test_file)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0
        assert result is not None
        assert result.content_length > 50000
