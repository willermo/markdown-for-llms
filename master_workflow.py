#!/usr/bin/env python3
"""
Document Intelligence Pipeline Orchestrator

This module orchestrates the complete document processing pipeline:
1. Document conversion (PDF/EPUB -> Markdown)
2. Markdown cleaning and normalization
3. Quality validation and assessment
4. Intelligent chunking for LLM consumption
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import new infrastructure
from config import get_config_manager, get_config, apply_env_overrides
from logging_config import get_orchestrator_logger, log_operation, ProgressLogger
from exceptions import PipelineError, ConversionError, CleaningError, ValidationError, ChunkingError, error_context
from unified_converter import UnifiedDocumentConverter

# --- Configuration ---
BASE_DIR = Path.cwd()

# Default values for command line arguments
DEFAULT_TARGET_LLM = 'custom'
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_OVERLAP = 200

SCRIPTS = {
    'clean': 'clean_markdown.py',
    'validate': 'validate_markdown.py',
    'chunk': 'chunk_markdown.py'
}

class PipelineOrchestrator:
    def __init__(self, config_manager=None):
        # Initialize configuration
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.config
        
        # Apply environment overrides
        self.config = apply_env_overrides(self.config)
        
        # Setup logging
        self.logger = get_orchestrator_logger()
        
        # Initialize unified converter
        self.converter = UnifiedDocumentConverter(self.config_manager)
        
        # Initialize directories
        self.setup_directories()
        
        # Pipeline state
        self.pipeline_state = {
            'start_time': time.time(),
            'steps_completed': [],
            'steps_failed': [],
            'file_counts': {},
            'total_processing_time': 0,
            'settings': {
                'target_llm': self.config.chunking.target_llm.value,
                'chunk_size': self.config.chunking.chunk_size,
                'overlap': self.config.chunking.overlap,
                'skip_existing': self.config.skip_existing
            }
        }
    
    def setup_directories(self):
        """Create all necessary directories from configuration"""
        self.config_manager.create_directories()
        self.logger.info("Pipeline directories created/verified from configuration")
    
    def check_prerequisites(self) -> bool:
        """Check if all required scripts and tools are available"""
        self.logger.info("Checking prerequisites...")
        
        # Check for required scripts
        missing_scripts = []
        for script_name, script_file in SCRIPTS.items():
            script_path = BASE_DIR / script_file
            if not script_path.exists():
                missing_scripts.append(script_file)
        
        if missing_scripts:
            self.logger.error(f"Missing required scripts: {missing_scripts}")
            return False
        
        # Check for pandoc
        try:
            result = subprocess.run(['pandoc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.error("Pandoc not found or not working")
                return False
            self.logger.info(f"Pandoc available: {result.stdout.split()[1]}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.error("Pandoc not found in PATH")
            return False
        
        # Check for required Python packages
        required_packages = ['tiktoken', 're', 'pathlib']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing Python packages: {missing_packages}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        self.logger.info("✓ All prerequisites satisfied")
        return True
    
    def count_files(self, directory_type: str, pattern: str = "*") -> int:
        """Count files in a directory"""
        dir_path = self.config_manager.get_directory_path(directory_type)
        if not dir_path.exists():
            return 0
        return len(list(dir_path.glob(pattern)))
    
    def run_conversion(self) -> bool:
        """Step 1: Convert documents to markdown using unified converter"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Converting documents to markdown")
        self.logger.info("=" * 60)
        
        # Get source directories from config
        source_dirs = []
        try:
            pdf_dir = self.config_manager.get_directory_path('source_pdfs')
            if pdf_dir.exists():
                source_dirs.append(('source_pdfs', pdf_dir))
        except ValueError:
            pass
        
        try:
            doc_dir = self.config_manager.get_directory_path('source_documents')
            if doc_dir.exists():
                source_dirs.append(('source_documents', doc_dir))
        except ValueError:
            pass
        
        # Fallback to legacy source directory if configured
        try:
            legacy_source = self.config_manager.get_directory_path('source')
            if legacy_source.exists():
                source_dirs.append(('source', legacy_source))
        except ValueError:
            pass
        
        if not source_dirs:
            self.logger.warning("No source directories found or configured")
            return False
        
        # Count total source files
        total_source_files = 0
        for dir_name, dir_path in source_dirs:
            supported_formats = (
                self.converter.conversion_config.get("supported_pdf_formats", []) +
                self.converter.conversion_config.get("supported_document_formats", [])
            )
            for ext in supported_formats:
                total_source_files += len(list(dir_path.glob(f"*.{ext}")))
        
        if total_source_files == 0:
            self.logger.warning("No supported files found in source directories")
            return False
        
        self.logger.info(f"Found {total_source_files} source files across {len(source_dirs)} directories")
        
        # Check if we should skip existing
        converted_count = self.count_files('converted', '*.md')
        if self.config.skip_existing and converted_count > 0:
            self.logger.info(f"Found {converted_count} existing converted files, skipping conversion")
            self.pipeline_state['file_counts']['converted'] = converted_count
            return True
        
        # Get output directory
        output_dir = self.config_manager.get_directory_path('converted')
        
        # Convert files from all source directories
        all_results = []
        for dir_name, source_dir in source_dirs:
            self.logger.info(f"Processing {dir_name}: {source_dir}")
            
            results = self.converter.convert_directory(
                str(source_dir),
                str(output_dir),
                max_workers=self.config.max_workers,
                skip_existing=self.config.skip_existing
            )
            all_results.extend(results)
        
        # Count successful conversions
        successful_count = sum(1 for r in all_results if r.success)
        failed_count = len(all_results) - successful_count
        
        self.logger.info(f"✓ Conversion completed: {successful_count} successful, {failed_count} failed")
        self.pipeline_state['file_counts']['converted'] = successful_count
        
        return successful_count > 0
    
    def run_cleaning(self) -> bool:
        """Step 2: Clean converted markdown"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Cleaning markdown files")
        self.logger.info("=" * 60)
        
        converted_count = self.count_files('converted', '*.md')
        if converted_count == 0:
            self.logger.error("No converted files found for cleaning")
            return False
        
        cleaned_count = self.count_files('cleaned', '*.md')
        if self.config.skip_existing and cleaned_count > 0:
            self.logger.info(f"Found {cleaned_count} existing cleaned files, skipping cleaning")
            self.pipeline_state['file_counts']['cleaned'] = cleaned_count
            return True
        
        # Run cleaning script
        script_path = BASE_DIR / SCRIPTS['clean']
        
        try:
            result = subprocess.run([sys.executable, str(script_path)], 
                                  cwd=BASE_DIR, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=1800)  # 30 minutes timeout
            
            if result.returncode == 0:
                cleaned_count = self.count_files('cleaned', '*.md')
                self.logger.info(f"✓ Cleaning completed: {cleaned_count} files")
                self.pipeline_state['file_counts']['cleaned'] = cleaned_count
                return True
            else:
                self.logger.error(f"Cleaning failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Cleaning timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Cleaning error: {str(e)}")
            return False
    
    def run_validation(self) -> bool:
        """Step 3: Validate cleaned markdown"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Validating markdown files")
        self.logger.info("=" * 60)
        
        cleaned_count = self.count_files('cleaned', '*.md')
        if cleaned_count == 0:
            self.logger.error("No cleaned files found for validation")
            return False
        
        validated_count = self.count_files('validated', '*.md')
        if self.config.skip_existing and validated_count > 0:
            self.logger.info(f"Found {validated_count} existing validated files, skipping validation")
            self.pipeline_state['file_counts']['validated'] = validated_count
            return True
        
        # Run validation script
        script_path = BASE_DIR / SCRIPTS['validate']
        
        try:
            result = subprocess.run([sys.executable, str(script_path)], 
                                  cwd=BASE_DIR, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=900)  # 15 minutes timeout
            
            if result.returncode == 0:
                validated_count = self.count_files('validated', '*.md')
                self.logger.info(f"✓ Validation completed: {validated_count} valid files")
                self.pipeline_state['file_counts']['validated'] = validated_count
                
                # Load and report validation summary
                try:
                    with open(BASE_DIR / 'validation_report.json', 'r') as f:
                        report = json.load(f)
                        summary = report.get('summary', {})
                        self.logger.info(f"Validation rate: {summary.get('validation_rate', 0)}%")
                        self.logger.info(f"Grade distribution: {summary.get('grade_distribution', {})}")
                except Exception as e:
                    self.logger.warning(f"Could not load validation report: {e}")
                
                return True
            else:
                self.logger.error(f"Validation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Validation timed out after 15 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
    
    def run_chunking(self) -> bool:
        """Step 4: Chunk validated markdown for LLM consumption"""
        self.logger.info("=" * 60)
        self.logger.info(f"STEP 4: Chunking markdown for {self.config.chunking.target_llm.value}")
        self.logger.info("=" * 60)
        
        validated_count = self.count_files('validated', '*.md')
        if validated_count == 0:
            self.logger.error("No validated files found for chunking")
            return False
        
        chunked_count = self.count_files('chunked', '*.md')
        if self.config.skip_existing and chunked_count > 0:
            self.logger.info(f"Found {chunked_count} existing chunked files, skipping chunking")
            self.pipeline_state['file_counts']['chunked'] = chunked_count
            return True
        
        # Update chunking script configuration
        chunk_script_path = BASE_DIR / SCRIPTS['chunk']
        
        # Temporarily modify the script's TARGET_LLM setting
        # This is a bit hacky but ensures the right LLM settings are used
        try:
            with open(chunk_script_path, 'r') as f:
                script_content = f.read()
            
            # Replace TARGET_LLM setting
            modified_content = script_content.replace(
                "TARGET_LLM = 'custom'",
                f"TARGET_LLM = '{self.config.chunking.target_llm.value}'"
            )
            
            # Write temporary script
            temp_script_path = BASE_DIR / 'temp_chunk_markdown.py'
            with open(temp_script_path, 'w') as f:
                f.write(modified_content)
            
            # Run chunking script
            result = subprocess.run([sys.executable, str(temp_script_path)], 
                                  cwd=BASE_DIR, 
                                  capture_output=True, 
                                  text=True,
                                  timeout=1800)  # 30 minutes timeout
            
            # Clean up temporary script
            temp_script_path.unlink()
            
            if result.returncode == 0:
                chunked_count = self.count_files('chunked', '*.md')
                self.logger.info(f"✓ Chunking completed: {chunked_count} chunks")
                self.pipeline_state['file_counts']['chunked'] = chunked_count
                
                # Load and report chunking summary
                try:
                    metadata_path = BASE_DIR / DIRECTORIES['chunked'] / 'chunks_index.json'
                    with open(metadata_path, 'r') as f:
                        index = json.load(f)
                        summary = index.get('chunking_summary', {})
                        self.logger.info(f"Total tokens: {summary.get('total_tokens', 0):,}")
                        self.logger.info(f"Average chunk size: {summary.get('average_chunk_size', 0):.0f} tokens")
                except Exception as e:
                    self.logger.warning(f"Could not load chunking summary: {e}")
                
                return True
            else:
                self.logger.error(f"Chunking failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Chunking timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Chunking error: {str(e)}")
            return False
    
    def save_pipeline_state(self):
        """Save final pipeline state"""
        self.pipeline_state['end_time'] = time.time()
        self.pipeline_state['total_processing_time'] = (
            self.pipeline_state['end_time'] - 
            self.pipeline_state['start_time']
        )
        
        state_file = self.config_manager.get_directory_path('logs') / 'pipeline_state.json'
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline state saved to: {state_file}")
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline"""
        start_time = time.time()
        
        self.logger.info("🚀 Starting Full LLM-Ready Markdown Pipeline")
        self.logger.info(f"Target LLM: {self.config.chunking.target_llm.value}")
        self.logger.info(f"Chunk size: {self.config.chunking.chunk_size} tokens")
        self.logger.info(f"Overlap: {self.config.chunking.overlap} tokens")
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("❌ Prerequisites not met, aborting pipeline")
            return False
        
        # Run pipeline steps
        steps = [
            ('conversion', self.run_conversion),
            ('cleaning', self.run_cleaning), 
            ('validation', self.run_validation),
            ('chunking', self.run_chunking)
        ]
        
        for step_name, step_func in steps:
            try:
                step_start = time.time()
                self.logger.info(f"Starting {step_name}...")
                
                if step_func():
                    step_time = time.time() - step_start
                    self.pipeline_state['steps_completed'].append({
                        'step': step_name,
                        'duration': step_time,
                        'timestamp': time.time()
                    })
                    self.logger.info(f"✅ {step_name.capitalize()} completed in {step_time:.1f}s")
                else:
                    self.pipeline_state['steps_failed'].append({
                        'step': step_name,
                        'timestamp': time.time()
                    })
                    self.logger.error(f"❌ {step_name.capitalize()} failed")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ {step_name.capitalize()} crashed: {str(e)}")
                self.pipeline_state['steps_failed'].append({
                    'step': step_name,
                    'error': str(e),
                    'timestamp': time.time()
                })
                return False
        
        # Pipeline completed successfully
        total_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("=" * 60)
        self.logger.info(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        # Final file count summary
        for stage, count in self.pipeline_state['file_counts'].items():
            self.logger.info(f"{stage.capitalize()} files: {count}")
        
        # Calculate efficiency metrics
        if 'chunked' in self.pipeline_state['file_counts']:
            chunks = self.pipeline_state['file_counts']['chunked']
            source = self.pipeline_state['file_counts'].get('converted', 1)
            self.logger.info(f"Chunks per source file: {chunks/source:.1f}")
        
        self.save_pipeline_state()
        
        self.logger.info("\n📁 Output directories:")
        for dir_type in ['converted', 'cleaned', 'validated', 'chunked']:
            try:
                dir_path = self.config_manager.get_directory_path(dir_type)
                file_count = self.count_files(dir_type, '*.md' if dir_type != 'chunked' else '*')
                self.logger.info(f"  {dir_path}: {file_count} files")
            except ValueError:
                pass
        
        self.logger.info("\n🔍 Next steps:")
        self.logger.info("  1. Review validation report for quality metrics")
        self.logger.info("  2. Check chunks_index.json for chunking details")
        chunked_dir = self.config_manager.get_directory_path('chunked')
        self.logger.info(f"  3. Your LLM-ready files are in: {chunked_dir}")
        
        return True

def create_config_file():
    """Create a configuration file for the pipeline"""
    config_manager = get_config_manager()
    config_manager.save_config()
    print(f"Configuration file created: {config_manager.config_path}")
    print("Edit this file to customize pipeline settings")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='LLM-Ready Markdown Pipeline: Convert ebooks to optimized markdown chunks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py                          # Run with defaults
  python run_full_pipeline.py --llm gpt-4             # Target GPT-4
  python run_full_pipeline.py --chunk-size 5000       # Custom chunk size
  python run_full_pipeline.py --force                 # Reprocess all files
  python run_full_pipeline.py --config-only           # Just create config file
  python run_full_pipeline.py --step validation       # Run single step
        """
    )
    
    parser.add_argument('--llm', '--target-llm', 
                       choices=['gpt-3.5-turbo', 'gpt-4', 'claude-3', 'llama-2', 'gemini-pro', 'custom'],
                       default=DEFAULT_TARGET_LLM,
                       help='Target LLM for chunking optimization')
    
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                       help='Target chunk size in tokens')
    
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                       help='Overlap between chunks in tokens')
    
    parser.add_argument('--force', action='store_true',
                       help='Reprocess all files, ignore existing outputs')
    
    parser.add_argument('--step', choices=['conversion', 'cleaning', 'validation', 'chunking'],
                       help='Run only a specific pipeline step')
    
    parser.add_argument('--config-only', action='store_true',
                       help='Create configuration file and exit')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without running')
    
    args = parser.parse_args()
    
    # Handle config-only mode
    if args.config_only:
        create_config_file()
        return 0
    
    # Initialize orchestrator with configuration
    config_manager = get_config_manager()
    
    # Override config with command line arguments
    if args.llm != 'claude-3':  # Only override if different from default
        config_manager.update_llm_settings(args.llm)
    
    if args.chunk_size != 8000:  # Only override if different from default
        config_manager.config.chunking.chunk_size = args.chunk_size
    
    if args.overlap != 400:  # Only override if different from default
        config_manager.config.chunking.overlap = args.overlap
    
    if args.force:
        config_manager.config.skip_existing = False
    
    orchestrator = PipelineOrchestrator(config_manager)
    
    # Handle dry-run mode
    if args.dry_run:
        print("DRY RUN - Would process:")
        # Show source directories
        for source_type in ['source_pdfs', 'source_documents', 'source']:
            try:
                dir_path = orchestrator.config_manager.get_directory_path(source_type)
                if dir_path.exists():
                    count = orchestrator.count_files(source_type, '*')
                    print(f"  {dir_path}: {count} source files")
            except ValueError:
                pass
        
        # Show output directories
        for dir_type in ['converted', 'cleaned', 'validated', 'chunked']:
            try:
                dir_path = orchestrator.config_manager.get_directory_path(dir_type)
                count = orchestrator.count_files(dir_type, '*.md')
                print(f"  {dir_path}: {count} existing files")
            except ValueError:
                pass
        print(f"\nTarget LLM: {args.llm}")
        print(f"Chunk size: {args.chunk_size} tokens")
        print(f"Overlap: {args.overlap} tokens")
        print(f"Force reprocess: {args.force}")
        return 0
    
    # Handle single step mode
    if args.step:
        step_functions = {
            'conversion': orchestrator.run_conversion,
            'cleaning': orchestrator.run_cleaning,
            'validation': orchestrator.run_validation,
            'chunking': orchestrator.run_chunking
        }
        
        if orchestrator.check_prerequisites():
            success = step_functions[args.step]()
            orchestrator.save_pipeline_state()
            return 0 if success else 1
        else:
            return 1
    
    # Run full pipeline
    success = orchestrator.run_full_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())