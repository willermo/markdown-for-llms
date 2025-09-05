#!/usr/bin/env python3
"""
Datalab Marker On-Premises Batch PDF to Markdown Converter
Uses local self-serve API for free personal research use
Requires: docker-compose setup from https://github.com/datalab-to/marker
"""

import os
import sys
import time
import json
import base64
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


# --- dataclasses ---------------------------------------------------------

@dataclass
class OnPremConfig:
    base_url: str = "http://localhost:8000"
    output_format: str = "markdown"
    use_llm: bool = False
    force_ocr: bool = False
    paginate: bool = False
    strip_existing_ocr: bool = False
    disable_image_extraction: bool = False
    max_pages: Optional[int] = None
    timeout: int = 300  # Longer timeout for local processing
    max_retries: int = 3
    retry_delay: int = 10
    health_check_timeout: int = 30
    docker_compose_file: Optional[str] = None
    auto_start_docker: bool = False


@dataclass
class ConversionResult:
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


# --- docker management --------------------------------------------------

class DockerManager:
    """Manage Docker Compose for Marker on-premises setup."""
    
    def __init__(self, compose_file: Optional[str] = None):
        self.compose_file = compose_file or "docker-compose.yml"
        self.logger = logging.getLogger(__name__)
    
    def is_docker_running(self) -> bool:
        """Check if Docker is running."""
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def is_compose_file_present(self) -> bool:
        """Check if docker-compose file exists."""
        return Path(self.compose_file).exists()
    
    def start_services(self) -> bool:
        """Start Docker Compose services."""
        if not self.is_docker_running():
            self.logger.error("Docker is not running")
            return False
        
        if not self.is_compose_file_present():
            self.logger.error(f"Docker Compose file not found: {self.compose_file}")
            return False
        
        try:
            self.logger.info("Starting Marker services...")
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file, "up", "-d"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.logger.info("Marker services started successfully")
                return True
            else:
                self.logger.error(f"Failed to start services: {result.stderr}")
                return False
                
        except subprocess.SubprocessError as e:
            self.logger.error(f"Error starting Docker services: {str(e)}")
            return False
    
    def stop_services(self) -> bool:
        """Stop Docker Compose services."""
        try:
            self.logger.info("Stopping Marker services...")
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file, "down"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.logger.info("Marker services stopped successfully")
                return True
            else:
                self.logger.warning(f"Error stopping services: {result.stderr}")
                return False
                
        except subprocess.SubprocessError as e:
            self.logger.error(f"Error stopping Docker services: {str(e)}")
            return False
    
    def get_service_status(self) -> Dict[str, str]:
        """Get status of all services."""
        try:
            result = subprocess.run([
                "docker-compose", "-f", self.compose_file, "ps", "--format", "json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                services = {}
                for line in result.stdout.strip().split('\n'):
                    if line:
                        service_data = json.loads(line)
                        services[service_data.get('Service', 'unknown')] = service_data.get('State', 'unknown')
                return services
            else:
                return {}
                
        except (subprocess.SubprocessError, json.JSONDecodeError):
            return {}


# --- client --------------------------------------------------------------

class OnPremMarkerClient:
    """Client for on-premises Marker API."""

    def __init__(self, config: OnPremConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BatchPDFConverter-OnPrem/1.0"
        })
        
        self.docker_manager = DockerManager(config.docker_compose_file) if config.auto_start_docker else None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("conversion_onprem.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_service_health(self) -> bool:
        """Check if the on-premises Marker service is healthy."""
        try:
            health_url = f"{self.config.base_url.rstrip('/')}/health"
            resp = self.session.get(health_url, timeout=self.config.health_check_timeout)
            
            if resp.status_code == 200:
                health_data = resp.json()
                status = health_data.get("status", "unknown")
                if status == "healthy":
                    self.logger.info("Marker service is healthy")
                    return True
                else:
                    self.logger.warning(f"Marker service status: {status}")
                    return False
            else:
                self.logger.error(f"Health check failed: HTTP {resp.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    def ensure_service_running(self) -> bool:
        """Ensure the Marker service is running, start if needed."""
        if self.check_service_health():
            return True
        
        if self.docker_manager:
            self.logger.info("Service not healthy, attempting to start...")
            if self.docker_manager.start_services():
                # Wait for service to be ready
                for attempt in range(30):  # Wait up to 5 minutes
                    time.sleep(10)
                    if self.check_service_health():
                        return True
                    self.logger.info(f"Waiting for service to be ready... ({attempt + 1}/30)")
                
                self.logger.error("Service failed to become healthy after startup")
                return False
            else:
                self.logger.error("Failed to start Docker services")
                return False
        else:
            self.logger.error("Service not available and auto-start disabled")
            return False

    def convert_pdf(self, pdf_path: str) -> ConversionResult:
        """Convert PDF using on-premises API."""
        start_time = time.time()
        file_size = Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0
        
        # Ensure service is running
        if not self.ensure_service_running():
            return ConversionResult(
                file_path=pdf_path,
                success=False,
                error="Marker service not available",
                file_size=file_size
            )
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Converting {Path(pdf_path).name} (attempt {attempt + 1})")
                
                # For on-premises, we can use direct conversion endpoint
                result_data = self._convert_direct(pdf_path)
                
                if not result_data:
                    if attempt < self.config.max_retries:
                        self.logger.warning(f"Conversion failed, retrying in {self.config.retry_delay}s...")
                        time.sleep(self.config.retry_delay)
                        continue
                    return ConversionResult(
                        file_path=pdf_path,
                        success=False,
                        error="Conversion failed after retries",
                        file_size=file_size,
                        retry_count=attempt
                    )

                # Validate result
                markdown_content = result_data.get("markdown", "")
                if not markdown_content or len(markdown_content.strip()) < 10:
                    self.logger.warning(f"Suspicious result for {pdf_path}: very short markdown content")

                return ConversionResult(
                    file_path=pdf_path,
                    success=True,
                    markdown_content=markdown_content,
                    images=result_data.get("images", {}) or {},
                    metadata=result_data.get("metadata", {}) or {},
                    page_count=result_data.get("page_count") or result_data.get("pages") or 0,
                    processing_time=time.time() - start_time,
                    file_size=file_size,
                    retry_count=attempt
                )
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Request error for {pdf_path}: {str(e)}, retrying...")
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    self.logger.error(f"Request failed for {pdf_path} after retries: {str(e)}")
                    return ConversionResult(
                        file_path=pdf_path,
                        success=False,
                        error=f"Request error: {str(e)}",
                        file_size=file_size,
                        retry_count=attempt
                    )
            except Exception as e:
                self.logger.error(f"Unexpected error converting {pdf_path}: {str(e)}")
                return ConversionResult(
                    file_path=pdf_path,
                    success=False,
                    error=f"Unexpected error: {str(e)}",
                    file_size=file_size,
                    retry_count=attempt
                )

        return ConversionResult(
            file_path=pdf_path,
            success=False,
            error="Max retries exceeded",
            file_size=file_size,
            retry_count=self.config.max_retries
        )

    def _convert_direct(self, pdf_path: str) -> Optional[Dict]:
        """Direct conversion for on-premises API."""
        url = f"{self.config.base_url.rstrip('/')}/convert"
        
        # Validate file before submission
        if not Path(pdf_path).exists():
            self.logger.error(f"File not found: {pdf_path}")
            return None
        
        file_size = Path(pdf_path).stat().st_size
        if file_size == 0:
            self.logger.error(f"Empty file: {pdf_path}")
            return None
        
        if file_size > 500 * 1024 * 1024:  # 500MB limit for local processing
            self.logger.warning(f"Very large file ({file_size / 1024 / 1024:.1f}MB): {pdf_path}")

        with open(pdf_path, "rb") as fh:
            # Build form data for on-premises API
            form_data = {
                "file": (os.path.basename(pdf_path), fh, "application/pdf"),
                "output_format": (None, self.config.output_format),
            }
            
            # Add optional parameters
            if self.config.use_llm:
                form_data["use_llm"] = (None, "true")
            if self.config.force_ocr:
                form_data["force_ocr"] = (None, "true")
            if self.config.paginate:
                form_data["paginate"] = (None, "true")
            if self.config.strip_existing_ocr:
                form_data["strip_existing_ocr"] = (None, "true")
            if self.config.disable_image_extraction:
                form_data["disable_image_extraction"] = (None, "true")
            if self.config.max_pages:
                form_data["max_pages"] = (None, str(self.config.max_pages))

            try:
                self.logger.debug(f"Submitting {pdf_path} for direct conversion")
                resp = self.session.post(url, files=form_data, timeout=self.config.timeout)
                resp.raise_for_status()
                
                result = resp.json()
                if result.get("success"):
                    self.logger.debug(f"Direct conversion successful for {pdf_path}")
                    return result
                else:
                    error_msg = result.get("error", "Unknown error")
                    self.logger.error(f"Conversion error for {pdf_path}: {error_msg}")
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"Timeout converting {pdf_path}")
                return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for {pdf_path}: {str(e)}")
                return None

        return None


# --- batch converter -----------------------------------------------------

class OnPremBatchPDFConverter:
    """Batch converter for on-premises Marker API."""
    
    def __init__(self, config: OnPremConfig):
        self.config = config
        self.client = OnPremMarkerClient(config)
        self.logger = logging.getLogger(__name__)

    def convert_directory(self, input_dir: str, output_dir: str, max_workers: int = 2, resume: bool = False) -> List[ConversionResult]:
        """Convert directory with on-premises optimizations."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files in {input_dir}")
            return []

        total_size = sum(f.stat().st_size for f in pdf_files) / 1024 / 1024  # MB
        self.logger.info(f"Found {len(pdf_files)} PDF files ({total_size:.1f} MB total)")

        if resume:
            existing_outputs = set(f.stem for f in output_path.glob("*.md"))
            pdf_files = [f for f in pdf_files if f.stem not in existing_outputs]
            self.logger.info(f"Resume mode: {len(pdf_files)} files remaining")

        if not pdf_files:
            self.logger.info("All files already processed")
            return []

        # Check service health before starting
        if not self.client.check_service_health():
            self.logger.error("Marker service not available")
            return []

        results: List[ConversionResult] = []
        start_time = time.time()
        
        # Use fewer workers for on-premises to avoid overwhelming local resources
        max_workers = min(max_workers, 2)
        
        # For very large files, use single worker
        large_files = [f for f in pdf_files if f.stat().st_size > 50 * 1024 * 1024]  # >50MB
        if large_files:
            max_workers = 1
            self.logger.info(f"Using single worker due to {len(large_files)} large files")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.client.convert_pdf, str(f)): f for f in pdf_files}
            
            with tqdm(total=len(pdf_files), desc="Converting PDFs (On-Prem)", unit="file") as pbar:
                for future in as_completed(future_map):
                    pdf_file = future_map[future]
                    try:
                        res = future.result()
                        results.append(res)
                        
                        if res.success:
                            self._save_result(res, output_path)
                            pages_info = f"{res.page_count} pages" if res.page_count else "? pages"
                            size_info = f"{res.file_size / 1024:.0f}KB" if res.file_size else ""
                            time_info = f"{res.processing_time:.1f}s" if res.processing_time else ""
                            retry_info = f" (retries: {res.retry_count})" if res.retry_count > 0 else ""
                            self.logger.info(f"✓ {pdf_file.name} - {pages_info}, {size_info}, {time_info}{retry_info}")
                        else:
                            error_short = res.error[:50] + "..." if res.error and len(res.error) > 50 else res.error
                            self.logger.error(f"✗ {pdf_file.name}: {error_short}")
                            
                    except Exception as e:
                        self.logger.error(f"✗ {pdf_file.name} failed: {str(e)}")
                        results.append(ConversionResult(file_path=str(pdf_file), success=False, error=str(e)))
                    
                    pbar.update(1)

        total_time = time.time() - start_time
        self._write_summary(results, output_path, total_time)
        return results

    def _save_result(self, res: ConversionResult, out: Path):
        """Save conversion result with validation."""
        stem = Path(res.file_path).stem
        
        # Save markdown
        if res.markdown_content:
            md_path = out / f"{stem}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(res.markdown_content)
            
            # Validate saved file
            if md_path.stat().st_size == 0:
                self.logger.error(f"Empty markdown file saved: {md_path}")
        
        # Save images with validation
        if res.images:
            img_dir = out / "images" / stem
            img_dir.mkdir(parents=True, exist_ok=True)
            saved_images = 0
            
            for fn, b64 in res.images.items():
                try:
                    img_data = base64.b64decode(b64)
                    if len(img_data) > 0:  # Validate image data
                        with open(img_dir / fn, "wb") as f:
                            f.write(img_data)
                        saved_images += 1
                    else:
                        self.logger.warning(f"Empty image data: {fn}")
                except Exception as e:
                    self.logger.error(f"Image save failed {fn}: {str(e)}")
            
            if saved_images > 0:
                self.logger.debug(f"Saved {saved_images} images for {stem}")
        
        # Save metadata
        if res.metadata:
            with open(out / f"{stem}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(res.metadata, f, indent=2)

    def _write_summary(self, results: List[ConversionResult], out: Path, total_time: float):
        """Write enhanced conversion summary."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Calculate statistics
        total_pages = sum(r.page_count or 0 for r in successful_results)
        total_size = sum(r.file_size or 0 for r in results)
        avg_processing_time = sum(r.processing_time or 0 for r in successful_results) / len(successful_results) if successful_results else 0
        total_retries = sum(r.retry_count for r in results)
        
        # Error analysis
        error_types = {}
        for r in failed_results:
            if r.error:
                error_key = r.error.split(':')[0]  # Get error type
                error_types[error_key] = error_types.get(error_key, 0) + 1

        summary = {
            "conversion_stats": {
                "total_files": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) * 100 if results else 0,
                "total_pages": total_pages,
                "total_size_mb": total_size / 1024 / 1024,
                "total_processing_time": total_time,
                "avg_processing_time_per_file": avg_processing_time,
                "total_retries": total_retries,
            },
            "performance_metrics": {
                "pages_per_minute": total_pages / (total_time / 60) if total_time > 0 else 0,
                "files_per_minute": len(results) / (total_time / 60) if total_time > 0 else 0,
                "mb_per_minute": (total_size / 1024 / 1024) / (total_time / 60) if total_time > 0 else 0,
            },
            "error_analysis": error_types,
            "failed_files": [{"file": r.file_path, "error": r.error} for r in failed_results],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "mode": "on-premises",
                "base_url": self.config.base_url,
                "use_llm": self.config.use_llm,
                "force_ocr": self.config.force_ocr,
                "max_pages": self.config.max_pages,
            }
        }
        
        with open(out / "conversion_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        print("\n=== ON-PREMISES CONVERSION SUMMARY ===")
        print(f"Files processed: {summary['conversion_stats']['total_files']}")
        print(f"Successful: {summary['conversion_stats']['successful']} ({summary['conversion_stats']['success_rate']:.1f}%)")
        print(f"Failed: {summary['conversion_stats']['failed']}")
        print(f"Total pages: {summary['conversion_stats']['total_pages']}")
        print(f"Total size: {summary['conversion_stats']['total_size_mb']:.1f} MB")
        print(f"Processing time: {summary['conversion_stats']['total_processing_time']:.1f}s")
        print(f"Performance: {summary['performance_metrics']['pages_per_minute']:.1f} pages/min")
        if total_retries > 0:
            print(f"Total retries: {total_retries}")
        if error_types:
            print(f"Error types: {error_types}")


# --- setup helpers ------------------------------------------------------

def create_setup_guide():
    """Create setup guide for on-premises Marker."""
    guide = """
# Datalab Marker On-Premises Setup Guide

## Prerequisites
1. Docker and Docker Compose installed
2. NVIDIA GPU (recommended for better performance)
3. At least 8GB RAM
4. 10GB free disk space

## Setup Steps

1. **Clone the Marker repository:**
   ```bash
   git clone https://github.com/datalab-to/marker.git
   cd marker
   ```

2. **Setup environment:**
   ```bash
   cp .env.example .env
   # Edit .env file as needed
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running:**
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

5. **Run this script:**
   ```bash
   python batch-pdf-converter-onprem.py --input-dir ./pdfs --output-dir ./output
   ```

## Troubleshooting
- Check Docker logs: `docker-compose logs`
- Restart services: `docker-compose restart`
- Check GPU usage: `nvidia-smi` (if using GPU)
- Monitor resources: `docker stats`

For more details, see: https://documentation.datalab.to/docs/on-prem/self-serve/api
"""
    
    with open("ONPREM_SETUP.md", "w") as f:
        f.write(guide)
    
    print("Setup guide created: ONPREM_SETUP.md")


# --- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="On-Premises Batch PDF to Markdown converter using Datalab Marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (assumes service running on localhost:8000)
  python batch-pdf-converter-onprem.py --input-dir ./pdfs --output-dir ./output

  # With LLM enhancement
  python batch-pdf-converter-onprem.py --input-dir ./pdfs --output-dir ./output --use-llm

  # Auto-start Docker services
  python batch-pdf-converter-onprem.py --input-dir ./pdfs --output-dir ./output --auto-start-docker

  # Custom API endpoint
  python batch-pdf-converter-onprem.py --input-dir ./pdfs --output-dir ./output --base-url http://192.168.1.100:8000

  # Create setup guide
  python batch-pdf-converter-onprem.py --setup-guide
        """
    )

    parser.add_argument("--input-dir", help="Directory containing PDF files to convert")
    parser.add_argument("--output-dir", help="Directory to save converted Markdown files")

    # Service configuration
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL for on-premises Marker API")
    parser.add_argument("--auto-start-docker", action="store_true",
                        help="Automatically start Docker services if needed")
    parser.add_argument("--docker-compose-file", default="docker-compose.yml",
                        help="Path to docker-compose.yml file")

    # Processing options
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for enhanced accuracy")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on every page")
    parser.add_argument("--paginate", action="store_true", help="Add page delimiters")
    parser.add_argument("--strip-existing-ocr", action="store_true", help="Remove existing OCR text and redo OCR")
    parser.add_argument("--disable-image-extraction", action="store_true", help="Disable image extraction")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process per PDF")

    # Performance options
    parser.add_argument("--max-workers", type=int, default=2, help="Concurrent conversions (max 2 for on-prem)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts per file")
    parser.add_argument("--retry-delay", type=int, default=10, help="Delay between retries (seconds)")

    # Timeout options
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--health-check-timeout", type=int, default=30, help="Health check timeout")

    # Utility options
    parser.add_argument("--setup-guide", action="store_true", help="Create setup guide and exit")
    parser.add_argument("--check-health", action="store_true", help="Check service health and exit")

    args = parser.parse_args()

    # Handle utility commands
    if args.setup_guide:
        create_setup_guide()
        return 0

    if args.check_health:
        config = OnPremConfig(base_url=args.base_url)
        client = OnPremMarkerClient(config)
        if client.check_service_health():
            print("✓ Marker service is healthy")
            return 0
        else:
            print("✗ Marker service is not available")
            return 1

    # Validate required arguments
    if not args.input_dir or not args.output_dir:
        print("Error: --input-dir and --output-dir are required")
        return 2

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return 1

    # Limit workers for on-premises
    max_workers = min(args.max_workers, 2)
    if args.max_workers > 2:
        print("Warning: On-premises processing limited to 2 workers maximum")

    config = OnPremConfig(
        base_url=args.base_url,
        use_llm=args.use_llm,
        force_ocr=args.force_ocr,
        paginate=args.paginate,
        strip_existing_ocr=args.strip_existing_ocr,
        disable_image_extraction=args.disable_image_extraction,
        max_pages=args.max_pages,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        health_check_timeout=args.health_check_timeout,
        docker_compose_file=args.docker_compose_file,
        auto_start_docker=args.auto_start_docker,
    )

    converter = OnPremBatchPDFConverter(config)
    results = converter.convert_directory(args.input_dir, args.output_dir, max_workers, args.resume)

    failed = sum(1 for r in results if not r.success)
    success_rate = (len(results) - failed) / len(results) * 100 if results else 0
    
    print(f"\nOn-premises conversion completed: {success_rate:.1f}% success rate")
    return 1 if failed > len(results) * 0.5 else 0  # Exit with error if >50% failed


if __name__ == "__main__":
    sys.exit(main())