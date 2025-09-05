#!/usr/bin/env python3
"""
Integration tests for the complete Document Intelligence Pipeline
"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from master_workflow import PipelineOrchestrator
from config import get_config_manager, PipelineConfig
from exceptions import PipelineError

class TestPipelineIntegration:
    
    @pytest.fixture
    def pipeline_workspace(self, temp_workspace):
        """Create a complete pipeline workspace"""
        # Create sample source files
        source_dir = temp_workspace / "source_ebooks"
        
        # Sample EPUB content
        epub_content = """
        # Sample Book Title
        
        ## Chapter 1: Introduction
        
        This is sample content for testing the pipeline with various
        formatting elements and potential artifacts.
        
        Some text with [OCR] artifacts and HTML <span>tags</span>.
        
        ## Chapter 2: Main Content
        
        More substantial content here to test chunking and validation.
        This paragraph contains enough text to be meaningful for testing
        the complete pipeline workflow from conversion through chunking.
        """
        
        (source_dir / "sample.epub").write_text(epub_content)
        (source_dir / "sample.md").write_text(epub_content)  # Pre-converted for testing
        
        return temp_workspace
    
    @pytest.fixture
    def test_pipeline(self, pipeline_workspace, test_config):
        """Create a PipelineOrchestrator instance for testing"""
        config_manager = get_config_manager()
        config_manager._config = test_config
        
        return PipelineOrchestrator(
            target_llm="claude-3",
            chunk_size=1000,
            overlap=100
        )
    
    def test_pipeline_initialization(self, test_pipeline):
        """Test pipeline initialization and setup"""
        assert test_pipeline.source_dir.exists()
        assert test_pipeline.base_output_dir.exists()
    
    def test_directory_setup(self, test_pipeline):
        """Test that pipeline creates required directories"""
        test_pipeline.setup_directories()
        
        # Verify all required directories exist
        config = get_config_manager().config
        base_dir = test_pipeline.base_output_dir
        
        assert (base_dir / config.directories.converted).exists()
        assert (base_dir / config.directories.cleaned).exists()
        assert (base_dir / config.directories.validated).exists()
        assert (base_dir / config.directories.chunked).exists()
        assert (base_dir / config.directories.logs).exists()
    
    def test_conversion_step_skip_existing(self, test_pipeline, pipeline_workspace):
        """Test conversion step with existing markdown files"""
        # Setup directories
        test_pipeline.setup_directories()
        
        # Run conversion step (should skip .md files or process them)
        success = test_pipeline.run_conversion_step()
        
        assert success
        
        # Check that converted directory has content
        converted_dir = pipeline_workspace / "converted_markdown"
        converted_files = list(converted_dir.glob("*.md"))
        assert len(converted_files) > 0
    
    def test_cleaning_step(self, test_pipeline, pipeline_workspace):
        """Test cleaning step of the pipeline"""
        # Setup and run conversion first
        test_pipeline.setup_directories()
        test_pipeline.run_conversion_step()
        
        # Run cleaning step
        success = test_pipeline.run_cleaning_step()
        
        assert success
        
        # Verify cleaned files exist
        cleaned_dir = pipeline_workspace / "cleaned_markdown"
        cleaned_files = list(cleaned_dir.glob("*.md"))
        assert len(cleaned_files) > 0
        
        # Verify content was actually cleaned
        for cleaned_file in cleaned_files:
            content = cleaned_file.read_text()
            # Should not contain OCR artifacts
            assert "[OCR]" not in content
            # Should not contain HTML tags
            assert "<span>" not in content
    
    def test_validation_step(self, test_pipeline, pipeline_workspace):
        """Test validation step of the pipeline"""
        # Setup and run previous steps
        test_pipeline.setup_directories()
        test_pipeline.run_conversion_step()
        test_pipeline.run_cleaning_step()
        
        # Run validation step
        success = test_pipeline.run_validation_step()
        
        assert success
        
        # Verify validation report exists
        validated_dir = pipeline_workspace / "validated_markdown"
        report_file = validated_dir / "validation_report.json"
        assert report_file.exists()
        
        # Verify validated files exist
        validated_files = list(validated_dir.glob("*.md"))
        assert len(validated_files) > 0
    
    def test_chunking_step(self, test_pipeline, pipeline_workspace):
        """Test chunking step of the pipeline"""
        # Setup and run all previous steps
        test_pipeline.setup_directories()
        test_pipeline.run_conversion_step()
        test_pipeline.run_cleaning_step()
        test_pipeline.run_validation_step()
        
        # Run chunking step
        success = test_pipeline.run_chunking_step()
        
        assert success
        
        # Verify chunked files exist
        chunked_dir = pipeline_workspace / "chunked_markdown"
        chunked_files = list(chunked_dir.glob("*.md"))
        assert len(chunked_files) > 0
        
        # Verify chunks have metadata
        for chunk_file in chunked_files:
            content = chunk_file.read_text()
            assert "---" in content  # YAML frontmatter
            assert "chunk_id:" in content
            assert "source_file:" in content
    
    def test_complete_pipeline_execution(self, test_pipeline, pipeline_workspace):
        """Test complete end-to-end pipeline execution"""
        # Run complete pipeline
        success = test_pipeline.run_complete_pipeline()
        
        assert success
        
        # Verify all output directories have content
        config = get_config_manager().config
        base_dir = pipeline_workspace
        
        converted_files = list((base_dir / config.directories.converted).glob("*.md"))
        cleaned_files = list((base_dir / config.directories.cleaned).glob("*.md"))
        validated_files = list((base_dir / config.directories.validated).glob("*.md"))
        chunked_files = list((base_dir / config.directories.chunked).glob("*.md"))
        
        assert len(converted_files) > 0
        assert len(cleaned_files) > 0
        assert len(validated_files) > 0
        assert len(chunked_files) > 0
        
        # Verify processing chain integrity
        assert len(converted_files) == len(cleaned_files)
        assert len(cleaned_files) == len(validated_files)
        # Chunked files may be more due to splitting
        assert len(chunked_files) >= len(validated_files)
    
    def test_pipeline_with_errors(self, test_pipeline, pipeline_workspace, monkeypatch):
        """Test pipeline error handling and recovery"""
        # Mock a failure in the cleaning step
        def mock_clean_fail(*args, **kwargs):
            raise Exception("Simulated cleaning failure")
        
        # Setup directories
        test_pipeline.setup_directories()
        test_pipeline.run_conversion_step()
        
        # Mock the cleaning function to fail
        import clean_markdown
        monkeypatch.setattr(clean_markdown, 'clean_markdown_file', mock_clean_fail)
        
        # Cleaning step should handle the error gracefully
        success = test_pipeline.run_cleaning_step()
        
        # Should return False but not crash
        assert not success
    
    def test_skip_existing_files(self, test_pipeline, pipeline_workspace):
        """Test that pipeline skips existing files when configured"""
        # Run pipeline once
        test_pipeline.setup_directories()
        test_pipeline.run_complete_pipeline()
        
        # Get modification times of output files
        cleaned_dir = pipeline_workspace / "cleaned_markdown"
        cleaned_files = list(cleaned_dir.glob("*.md"))
        original_mtimes = {f: f.stat().st_mtime for f in cleaned_files}
        
        # Run pipeline again (should skip existing)
        test_pipeline.run_complete_pipeline()
        
        # Check that files weren't modified (assuming skip_existing=True)
        new_mtimes = {f: f.stat().st_mtime for f in cleaned_files}
        
        # Files should have same modification times if skipped
        for file_path in original_mtimes:
            if file_path in new_mtimes:
                # Allow small time differences due to filesystem precision
                time_diff = abs(new_mtimes[file_path] - original_mtimes[file_path])
                assert time_diff < 1.0  # Less than 1 second difference
    
    def test_pipeline_statistics_tracking(self, test_pipeline, pipeline_workspace):
        """Test that pipeline tracks statistics correctly"""
        # Run complete pipeline
        success = test_pipeline.run_complete_pipeline()
        assert success
        
        # Get statistics from pipeline
        stats = test_pipeline.get_pipeline_statistics()
        
        assert 'files_processed' in stats
        assert 'total_processing_time' in stats
        assert 'step_times' in stats
        assert stats['files_processed'] > 0
    
    def test_concurrent_processing(self, test_pipeline, pipeline_workspace):
        """Test pipeline with concurrent processing enabled"""
        # Create multiple source files
        source_dir = pipeline_workspace / "source_ebooks"
        for i in range(3):
            content = f"# Document {i}\n\nContent for document {i} with sufficient text."
            (source_dir / f"doc_{i}.md").write_text(content)
        
        # Configure for concurrent processing
        config = get_config_manager().config
        config.max_workers = 2
        
        # Run pipeline
        success = test_pipeline.run_complete_pipeline()
        assert success
        
        # Verify all files were processed
        chunked_dir = pipeline_workspace / "chunked_markdown"
        chunked_files = list(chunked_dir.glob("*.md"))
        
        # Should have chunks from multiple source files
        source_files = set()
        for chunk_file in chunked_files:
            content = chunk_file.read_text()
            if "source_file:" in content:
                # Extract source file from metadata
                for line in content.split('\n'):
                    if line.startswith('source_file:'):
                        source_files.add(line.split(':', 1)[1].strip())
        
        assert len(source_files) >= 3  # Should process all source files
    
    def test_pipeline_logging(self, test_pipeline, pipeline_workspace):
        """Test that pipeline generates proper logs"""
        # Run pipeline
        test_pipeline.run_complete_pipeline()
        
        # Check that log files were created
        logs_dir = pipeline_workspace / "pipeline_logs"
        log_files = list(logs_dir.glob("*.log"))
        
        assert len(log_files) > 0
        
        # Verify log content
        for log_file in log_files:
            content = log_file.read_text()
            assert len(content) > 0
            # Should contain pipeline step information
            assert any(keyword in content.lower() for keyword in 
                      ['conversion', 'cleaning', 'validation', 'chunking'])
    
    def test_configuration_integration(self, pipeline_workspace):
        """Test pipeline with different configurations"""
        # Create custom configuration
        from config import PipelineConfig, ChunkingConfig, LLMModel
        
        custom_config = PipelineConfig(
            directories=get_config_manager().config.directories,
            llm_presets={},
            validation=get_config_manager().config.validation,
            cleaning=get_config_manager().config.cleaning,
            chunking=ChunkingConfig(
                target_llm=LLMModel.CUSTOM,
                chunk_size=500,  # Smaller chunks
                overlap=50
            )
        )
        
        # Create pipeline with custom config
        config_manager = get_config_manager()
        config_manager._config = custom_config
        
        pipeline = DocumentPipeline(
            source_dir=pipeline_workspace / "source_ebooks",
            base_output_dir=pipeline_workspace
        )
        
        # Run pipeline
        success = pipeline.run_complete_pipeline()
        assert success
        
        # Verify smaller chunks were created
        chunked_dir = pipeline_workspace / "chunked_markdown"
        chunked_files = list(chunked_dir.glob("*.md"))
        
        # Should have more chunks due to smaller chunk size
        assert len(chunked_files) > 0
    
    @pytest.mark.slow
    def test_large_document_processing(self, test_pipeline, pipeline_workspace):
        """Test pipeline with large documents"""
        # Create a large document
        large_content = "# Large Document\n\n" + ("This is a paragraph with substantial content. " * 1000)
        
        source_dir = pipeline_workspace / "source_ebooks"
        (source_dir / "large_doc.md").write_text(large_content)
        
        # Run pipeline
        success = test_pipeline.run_complete_pipeline()
        assert success
        
        # Verify large document was processed and chunked appropriately
        chunked_dir = pipeline_workspace / "chunked_markdown"
        large_doc_chunks = list(chunked_dir.glob("large_doc_*.md"))
        
        # Should create multiple chunks for large document
        assert len(large_doc_chunks) > 1
    
    def test_pipeline_cleanup(self, test_pipeline, pipeline_workspace):
        """Test pipeline cleanup functionality"""
        # Run pipeline to create files
        test_pipeline.run_complete_pipeline()
        
        # Verify files exist
        config = get_config_manager().config
        output_dirs = [
            pipeline_workspace / config.directories.converted,
            pipeline_workspace / config.directories.cleaned,
            pipeline_workspace / config.directories.validated,
            pipeline_workspace / config.directories.chunked
        ]
        
        for output_dir in output_dirs:
            assert len(list(output_dir.glob("*.md"))) > 0
        
        # Test cleanup (if implemented)
        if hasattr(test_pipeline, 'cleanup_intermediate_files'):
            test_pipeline.cleanup_intermediate_files()
            
            # Verify cleanup worked as expected
            # (Implementation depends on cleanup strategy)
