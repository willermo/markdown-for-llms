#!/usr/bin/env python3
"""
Unified Document Converter for the Document Intelligence Pipeline

This module provides a unified interface for converting various document formats
to markdown using different conversion backends:
- Pandoc for non-PDF formats (epub, mobi, azw3, docx, html, etc.)
- Local Marker API for PDFs (on-premises)
- Cloud Marker API for PDFs (with API key)

The converter automatically routes files to the appropriate backend based on
format and configuration settings.
"""

import os
import sys
import time
import json
import base64
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import requests
from dotenv import load_dotenv

from config import get_config_manager
from logging_config import get_orchestrator_logger
from exceptions import ConversionError, error_context

# Load environment variables
load_dotenv()

class ConverterType(Enum):
    PANDOC = "pandoc"
    LOCAL_MARKER = "local_marker"
    CLOUD_MARKER = "cloud_marker"

@dataclass
class ConversionResult:
    """Result of a document conversion operation."""
    file_path: str
    success: bool
    markdown_content: Optional[str] = None
    images: Optional[Dict[str, str]] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    page_count: Optional[int] = None
    processing_time: Optional[float] = None
    file_size: Optional[int] = None
    retry_count: int = 0
    converter_used: Optional[str] = None

class PandocConverter:
    """Converter for non-PDF documents using Pandoc."""
    
    def __init__(self, config):
        self.config = config
        self.logger = get_orchestrator_logger()
        self.pandoc_options = config.get("pandoc_options", "--wrap=none --strip-comments --markdown-headings=atx")
    
    def is_available(self) -> bool:
        """Check if Pandoc is available."""
        try:
            result = subprocess.run(['pandoc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert(self, file_path: str) -> ConversionResult:
        """Convert document using Pandoc."""
        start_time = time.time()
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        
        if not self.is_available():
            return ConversionResult(
                file_path=file_path,
                success=False,
                error="Pandoc not available",
                file_size=file_size,
                converter_used="pandoc"
            )
        
        try:
            # Create temporary output file
            temp_output = Path(file_path).with_suffix('.temp.md')
            
            # Build pandoc command
            cmd = ['pandoc', file_path] + self.pandoc_options.split() + ['-o', str(temp_output)]
            
            self.logger.debug(f"Running pandoc: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get("conversion_timeout", 300)
            )
            
            if result.returncode == 0 and temp_output.exists():
                # Read converted content
                with open(temp_output, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # Clean up temp file
                temp_output.unlink()
                
                if len(markdown_content.strip()) < 10:
                    self.logger.warning(f"Suspicious result for {file_path}: very short content")
                
                return ConversionResult(
                    file_path=file_path,
                    success=True,
                    markdown_content=markdown_content,
                    processing_time=time.time() - start_time,
                    file_size=file_size,
                    converter_used="pandoc"
                )
            else:
                error_msg = result.stderr or "Pandoc conversion failed"
                return ConversionResult(
                    file_path=file_path,
                    success=False,
                    error=error_msg,
                    file_size=file_size,
                    converter_used="pandoc"
                )
                
        except subprocess.TimeoutExpired:
            return ConversionResult(
                file_path=file_path,
                success=False,
                error="Pandoc conversion timeout",
                file_size=file_size,
                converter_used="pandoc"
            )
        except Exception as e:
            return ConversionResult(
                file_path=file_path,
                success=False,
                error=f"Pandoc error: {str(e)}",
                file_size=file_size,
                converter_used="pandoc"
            )

class MarkerConverter:
    """Base class for Marker API converters."""
    
    def __init__(self, config, is_local=False):
        self.config = config
        self.is_local = is_local
        self.logger = get_orchestrator_logger()
        self.session = requests.Session()
        # Polling configuration
        self.max_polls = int(self.config.get("max_polls", 300))
        self.poll_interval = int(self.config.get("poll_interval", 2))
        
        if not is_local:
            # Cloud API requires API key
            api_key = os.getenv(config.get("marker_cloud_api_key_env", "MARKER_API_KEY"))
            if api_key:
                self.session.headers.update({"X-Api-Key": api_key})
            else:
                self.logger.warning("No Marker API key found in environment")
        
        self.base_url = (config.get("marker_local_base_url", "http://localhost:8000") 
                        if is_local 
                        else config.get("marker_cloud_base_url", "https://www.datalab.to/api/v1"))
    
    def is_available(self) -> bool:
        """Check if Marker API is available."""
        try:
            if self.is_local:
                health_url = f"{self.base_url.rstrip('/')}/health"
                resp = self.session.get(health_url, timeout=10)
                return resp.status_code == 200
            else:
                # Cloud: endpoint may not support GET; treat 200/401/403/405 as reachable
                marker_url = f"{self.base_url.rstrip('/')}/marker"
                resp = self.session.get(marker_url, timeout=10)
                return resp.status_code in (200, 401, 403, 405)
        except requests.exceptions.RequestException:
            return False
    
    def convert(self, file_path: str) -> ConversionResult:
        """Convert PDF using Marker API."""
        start_time = time.time()
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        
        if not self.is_available():
            service_type = "local" if self.is_local else "cloud"
            return ConversionResult(
                file_path=file_path,
                success=False,
                error=f"Marker {service_type} API not available",
                file_size=file_size,
                converter_used=f"marker_{service_type}"
            )
        
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay", 10)
        
        for attempt in range(max_retries + 1):
            try:
                if self.is_local:
                    result_data = self._convert_local(file_path)
                else:
                    result_data = self._convert_cloud(file_path)
                
                if result_data:
                    markdown_content = result_data.get("markdown", "")
                    if len(markdown_content.strip()) < 10:
                        self.logger.warning(f"Suspicious result for {file_path}: very short content")
                    
                    service_type = "local" if self.is_local else "cloud"
                    return ConversionResult(
                        file_path=file_path,
                        success=True,
                        markdown_content=markdown_content,
                        images=result_data.get("images", {}),
                        metadata=result_data.get("metadata", {}),
                        page_count=result_data.get("page_count", 0),
                        processing_time=time.time() - start_time,
                        file_size=file_size,
                        retry_count=attempt,
                        converter_used=f"marker_{service_type}"
                    )
                
                if attempt < max_retries:
                    self.logger.warning(f"Conversion failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Error converting {file_path}: {str(e)}, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    service_type = "local" if self.is_local else "cloud"
                    return ConversionResult(
                        file_path=file_path,
                        success=False,
                        error=f"Marker {service_type} error: {str(e)}",
                        file_size=file_size,
                        retry_count=attempt,
                        converter_used=f"marker_{service_type}"
                    )
        
        service_type = "local" if self.is_local else "cloud"
        return ConversionResult(
            file_path=file_path,
            success=False,
            error=f"Marker {service_type} conversion failed after retries",
            file_size=file_size,
            retry_count=max_retries,
            converter_used=f"marker_{service_type}"
        )
    
    def _convert_local(self, file_path: str) -> Optional[Dict]:
        """Convert using local Marker API."""
        url = f"{self.base_url.rstrip('/')}/convert"
        
        with open(file_path, "rb") as fh:
            form_data = {
                "file": (os.path.basename(file_path), fh, "application/pdf"),
                "output_format": (None, "markdown"),
                "use_llm": (None, "true" if self.config.get("use_llm", False) else "false"),
                "force_ocr": (None, "true" if self.config.get("force_ocr", False) else "false"),
                "strip_existing_ocr": (None, "true" if self.config.get("strip_existing_ocr", True) else "false"),
                "disable_image_extraction": (None, "true" if self.config.get("disable_image_extraction", True) else "false"),
                "paginate": (None, "true" if self.config.get("paginate", False) else "false"),
            }
            
            timeout = self.config.get("conversion_timeout", 3600)
            resp = self.session.post(url, files=form_data, timeout=timeout)
            resp.raise_for_status()
            
            result = resp.json()
            return result if result.get("success") else None
    
    def _convert_cloud(self, file_path: str) -> Optional[Dict]:
        """Convert using cloud Marker API with polling."""
        # Submit for conversion
        request_id, check_url = self._submit_conversion(file_path)
        if not request_id or not check_url:
            return None
        
        # Poll for completion
        return self._poll_for_completion(check_url)
    
    def _submit_conversion(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Submit PDF for cloud conversion."""
        url = f"{self.base_url.rstrip('/')}/marker"
        
        with open(file_path, "rb") as fh:
            form_data = {
                "file": (os.path.basename(file_path), fh, "application/pdf"),
                "output_format": (None, "markdown"),
                "use_llm": (None, "true" if self.config.get("use_llm", False) else "false"),
                "force_ocr": (None, "true" if self.config.get("force_ocr", False) else "false"),
                "strip_existing_ocr": (None, "true" if self.config.get("strip_existing_ocr", True) else "false"),
                "disable_image_extraction": (None, "true" if self.config.get("disable_image_extraction", True) else "false"),
                "paginate": (None, "true" if self.config.get("paginate", False) else "false"),
            }
            
            timeout = self.config.get("conversion_timeout", 30)
            resp = self.session.post(url, files=form_data, timeout=timeout)
            resp.raise_for_status()
            
            js = resp.json()
            if js.get("success"):
                return js.get("request_id"), js.get("request_check_url") or js.get("check_url")
            
            return None, None
    
    def _poll_for_completion(self, check_url: str) -> Optional[Dict]:
        """Poll for cloud conversion completion."""
        max_polls = int(self.config.get("max_polls", self.max_polls))
        poll_interval = int(self.config.get("poll_interval", self.poll_interval))
        
        for poll_count in range(max_polls):
            try:
                resp = self.session.get(check_url, timeout=30)
                resp.raise_for_status()
                
                data = resp.json()
                status = (data.get("status") or "").lower()
                
                if status in ("complete", "finished") or (data.get("success") and "markdown" in data):
                    return data if data.get("success") else None
                
                if status in ("queued", "processing", "running", "in_progress"):
                    if poll_count % max(1, int(60 / max(1, poll_interval))) == 0:  # roughly every minute
                        self.logger.info(f"Still processing... (poll {poll_count + 1})")
                    time.sleep(poll_interval)
                    continue
                
                if status in ("error", "failed"):
                    self.logger.error(f"Processing failed: {data.get('error', 'Unknown error')}")
                    return None
                
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Polling error: {str(e)}")
                time.sleep(poll_interval)
                continue
        
        self.logger.error("Polling timeout reached")
        return None

class UnifiedDocumentConverter:
    """Unified converter that routes documents to appropriate backends."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.config = self.config_manager.config
        self.logger = get_orchestrator_logger()
        
        # Get conversion settings
        self.conversion_config = self._get_conversion_config()
        
        # Initialize converters
        self.pandoc_converter = PandocConverter(self.conversion_config)
        self.local_marker_converter = MarkerConverter(self.conversion_config, is_local=True)
        self.cloud_marker_converter = MarkerConverter(self.conversion_config, is_local=False)
        
        self.logger.info(f"Unified converter initialized:")
        self.logger.info(f"  PDF converter: {self.conversion_config.get('pdf_converter', 'local_marker')}")
        self.logger.info(f"  Document converter: {self.conversion_config.get('document_converter', 'pandoc')}")
    
    def _get_conversion_config(self) -> Dict:
        """Get conversion configuration from config manager."""
        # Try to get from pipeline_config.json first
        config_dict = self.config_manager._config_to_dict(self.config)
        conversion_settings = config_dict.get("conversion_settings", {})
        
        # Set defaults if not present
        defaults = {
            "pdf_converter": "local_marker",
            "document_converter": "pandoc",
            "marker_cloud_api_key_env": "MARKER_API_KEY",
            "marker_local_base_url": "http://localhost:8000",
            "marker_cloud_base_url": "https://www.datalab.to/api/v1",
            "pandoc_options": "--wrap=none --strip-comments --markdown-headings=atx",
            "supported_pdf_formats": ["pdf"],
            "supported_document_formats": ["epub", "mobi", "azw", "azw3", "html", "htm", "docx", "rtf"],
            "conversion_timeout": 3600,
            "max_retries": 3,
            "retry_delay": 10
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in conversion_settings:
                conversion_settings[key] = value
        
        return conversion_settings
    
    def get_converter_for_file(self, file_path: str) -> Tuple[Optional[object], str]:
        """Determine which converter to use for a file."""
        file_ext = Path(file_path).suffix.lower().lstrip('.')
        
        if file_ext in self.conversion_config.get("supported_pdf_formats", ["pdf"]):
            # PDF file - use configured PDF converter
            pdf_converter = self.conversion_config.get("pdf_converter", "local_marker")
            
            if pdf_converter == "cloud_marker":
                return self.cloud_marker_converter, "cloud_marker"
            elif pdf_converter == "local_marker":
                return self.local_marker_converter, "local_marker"
            else:
                return None, f"unknown_pdf_converter_{pdf_converter}"
        
        elif file_ext in self.conversion_config.get("supported_document_formats", []):
            # Non-PDF document - use configured document converter
            doc_converter = self.conversion_config.get("document_converter", "pandoc")
            
            if doc_converter == "pandoc":
                return self.pandoc_converter, "pandoc"
            else:
                return None, f"unknown_document_converter_{doc_converter}"
        
        else:
            return None, f"unsupported_format_{file_ext}"
    
    def convert_file(self, file_path: str) -> ConversionResult:
        """Convert a single file using the appropriate converter."""
        converter, converter_type = self.get_converter_for_file(file_path)
        
        if converter is None:
            return ConversionResult(
                file_path=file_path,
                success=False,
                error=f"No converter available: {converter_type}",
                converter_used=converter_type
            )
        
        self.logger.info(f"Converting {Path(file_path).name} using {converter_type}")
        
        try:
            result = converter.convert(file_path)
            result.converter_used = converter_type
            return result
        except Exception as e:
            return ConversionResult(
                file_path=file_path,
                success=False,
                error=f"Conversion error: {str(e)}",
                converter_used=converter_type
            )
    
    def convert_directory(self, source_dir: str, output_dir: str, max_workers: int = 3, 
                         skip_existing: bool = True) -> List[ConversionResult]:
        """Convert all supported files in a directory."""
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not source_path.exists():
            self.logger.error(f"Source directory not found: {source_dir}")
            return []
        
        # Find all supported files
        supported_formats = (
            self.conversion_config.get("supported_pdf_formats", []) +
            self.conversion_config.get("supported_document_formats", [])
        )
        
        all_files = []
        for ext in supported_formats:
            all_files.extend(source_path.glob(f"*.{ext}"))
        
        if not all_files:
            self.logger.warning(f"No supported files found in {source_dir}")
            return []
        
        # Filter out existing files if skip_existing is True
        if skip_existing:
            existing_outputs = set(f.stem for f in output_path.glob("*.md"))
            files_to_process = [f for f in all_files if f.stem not in existing_outputs]
            
            if len(files_to_process) < len(all_files):
                skipped = len(all_files) - len(files_to_process)
                self.logger.info(f"Skipping {skipped} existing files, processing {len(files_to_process)} files")
        else:
            files_to_process = all_files
        
        if not files_to_process:
            self.logger.info("All files already processed")
            return []
        
        # Group files by converter type for better resource management
        files_by_converter = {}
        for file_path in files_to_process:
            _, converter_type = self.get_converter_for_file(str(file_path))
            if converter_type not in files_by_converter:
                files_by_converter[converter_type] = []
            files_by_converter[converter_type].append(file_path)
        
        self.logger.info(f"Files to process by converter:")
        for conv_type, files in files_by_converter.items():
            self.logger.info(f"  {conv_type}: {len(files)} files")
        
        # Convert files by converter group to control concurrency for heavy backends
        results = []
        total_files = len(files_to_process)
        completed = 0

        def handle_result(file_path: Path, result: ConversionResult):
            nonlocal completed
            results.append(result)
            completed += 1
            if result.success:
                # Save the converted markdown
                output_file = output_path / f"{file_path.stem}.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.markdown_content)
                # Save images if present
                if result.images:
                    img_dir = output_path / "images" / file_path.stem
                    img_dir.mkdir(parents=True, exist_ok=True)
                    for img_name, img_b64 in result.images.items():
                        try:
                            img_data = base64.b64decode(img_b64)
                            with open(img_dir / img_name, "wb") as f:
                                f.write(img_data)
                        except Exception as e:
                            self.logger.warning(f"Failed to save image {img_name}: {e}")
                self.logger.info(f"✓ [{completed}/{total_files}] {file_path.name} -> {result.converter_used}")
            else:
                self.logger.error(f"✗ [{completed}/{total_files}] {file_path.name}: {result.error}")

        for conv_type, files in files_by_converter.items():
            # Limit local_marker to sequential processing to avoid OOM / restarts
            group_workers = 1 if conv_type == 'local_marker' else max_workers
            if group_workers <= 1:
                for file_path in files:
                    try:
                        res = self.convert_file(str(file_path))
                        handle_result(file_path, res)
                    except Exception as e:
                        self.logger.error(f"✗ [{completed+1}/{total_files}] {file_path.name}: {str(e)}")
                        results.append(ConversionResult(
                            file_path=str(file_path),
                            success=False,
                            error=str(e)
                        ))
                        completed += 1
            else:
                with ThreadPoolExecutor(max_workers=group_workers) as executor:
                    future_to_file = {executor.submit(self.convert_file, str(fp)): fp for fp in files}
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            res = future.result()
                            handle_result(file_path, res)
                        except Exception as e:
                            self.logger.error(f"✗ [{completed+1}/{total_files}] {file_path.name}: {str(e)}")
                            results.append(ConversionResult(
                                file_path=str(file_path),
                                success=False,
                                error=str(e)
                            ))
                            completed += 1
        
        # Write summary
        self._write_conversion_summary(results, output_path)
        
        return results
    
    def _write_conversion_summary(self, results: List[ConversionResult], output_path: Path):
        """Write conversion summary to file."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Group by converter
        by_converter = {}
        for result in results:
            conv = result.converter_used or "unknown"
            if conv not in by_converter:
                by_converter[conv] = {"total": 0, "successful": 0, "failed": 0}
            by_converter[conv]["total"] += 1
            if result.success:
                by_converter[conv]["successful"] += 1
            else:
                by_converter[conv]["failed"] += 1
        
        summary = {
            "conversion_summary": {
                "total_files": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) * 100 if results else 0,
                "by_converter": by_converter
            },
            "failed_files": [
                {"file": r.file_path, "error": r.error, "converter": r.converter_used}
                for r in failed
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path / "unified_conversion_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nConversion Summary:")
        self.logger.info(f"  Total files: {len(results)}")
        self.logger.info(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        self.logger.info(f"  Failed: {len(failed)}")
        self.logger.info(f"  By converter: {by_converter}")

def main():
    """CLI interface for unified converter."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Document Converter")
    parser.add_argument("--source-dir", required=True, help="Source directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-workers", type=int, default=3, help="Max workers")
    parser.add_argument("--force", action="store_true", help="Reprocess existing files")
    
    args = parser.parse_args()
    
    converter = UnifiedDocumentConverter()
    results = converter.convert_directory(
        args.source_dir,
        args.output_dir,
        max_workers=args.max_workers,
        skip_existing=not args.force
    )
    
    success_rate = sum(1 for r in results if r.success) / len(results) * 100 if results else 0
    print(f"Conversion completed: {success_rate:.1f}% success rate")
    
    return 0 if success_rate > 50 else 1

if __name__ == "__main__":
    sys.exit(main())
