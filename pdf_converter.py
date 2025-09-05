#!/usr/bin/env python3
"""
Enhanced Datalab Marker API Batch PDF to Markdown Converter
Improvements: Better error handling, retry logic, rate limiting, validation
"""

import os
import sys
import time
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from dotenv import load_dotenv


# --- helpers -------------------------------------------------------------

def _bool_str(v: bool) -> str:
    """Convert Python bool to lowercase string for form posts."""
    return "true" if bool(v) else "false"


# --- dataclasses ---------------------------------------------------------

@dataclass
class ConversionConfig:
    api_key: str
    base_url: str = "https://www.datalab.to/api/v1"
    output_format: str = "markdown"
    use_llm: bool = False
    force_ocr: bool = False
    paginate: bool = False
    strip_existing_ocr: bool = False
    disable_image_extraction: bool = False
    max_pages: Optional[int] = None
    max_polls: int = 300
    poll_interval: int = 2
    timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit_delay: int = 1


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


# --- client --------------------------------------------------------------

class DataLabMarkerClient:
    """Enhanced client for interacting with Datalab's Marker API."""

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Key": config.api_key,
            "User-Agent": "BatchPDFConverter/1.2"
        })

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("conversion.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def convert_pdf(self, pdf_path: str) -> ConversionResult:
        """Convert PDF with retry logic and enhanced error handling."""
        start_time = time.time()
        file_size = Path(pdf_path).stat().st_size if Path(pdf_path).exists() else 0
        
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Converting {Path(pdf_path).name} (attempt {attempt + 1})")
                
                request_id, check_url = self._submit_conversion(pdf_path)
                if not request_id or not check_url:
                    if attempt < self.config.max_retries:
                        self.logger.warning(f"Submission failed, retrying in {self.config.retry_delay}s...")
                        time.sleep(self.config.retry_delay)
                        continue
                    return ConversionResult(
                        file_path=pdf_path, 
                        success=False, 
                        error="Failed to submit request after retries",
                        file_size=file_size,
                        retry_count=attempt
                    )

                result_data = self._poll_for_completion(check_url)
                if not result_data:
                    if attempt < self.config.max_retries:
                        self.logger.warning(f"Polling failed, retrying in {self.config.retry_delay}s...")
                        time.sleep(self.config.retry_delay)
                        continue
                    return ConversionResult(
                        file_path=pdf_path, 
                        success=False, 
                        error="No result returned after retries",
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
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif attempt < self.config.max_retries:
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

    def _submit_conversion(self, pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Submit a PDF for conversion with enhanced error handling."""
        url = f"{self.config.base_url.rstrip('/')}/marker"

        # Validate file before submission
        if not Path(pdf_path).exists():
            self.logger.error(f"File not found: {pdf_path}")
            return None, None
        
        file_size = Path(pdf_path).stat().st_size
        if file_size == 0:
            self.logger.error(f"Empty file: {pdf_path}")
            return None, None
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            self.logger.warning(f"Large file ({file_size / 1024 / 1024:.1f}MB): {pdf_path}")

        with open(pdf_path, "rb") as fh:
            # Build multipart form according to Datalab docs
            form_data = {
                "file": (os.path.basename(pdf_path), fh, "application/pdf"),
                "output_format": (None, self.config.output_format),
                "use_llm": (None, str(self.config.use_llm).lower()),
                "force_ocr": (None, str(self.config.force_ocr).lower()),
                "paginate": (None, str(self.config.paginate).lower()),
                "strip_existing_ocr": (None, str(self.config.strip_existing_ocr).lower()),
                "disable_image_extraction": (None, str(self.config.disable_image_extraction).lower()),
            }
            if self.config.max_pages:
                form_data["max_pages"] = (None, str(self.config.max_pages))

            try:
                # Add rate limiting delay
                time.sleep(self.config.rate_limit_delay)
                
                resp = self.session.post(url, files=form_data, timeout=self.config.timeout)
                
                if resp.status_code == 429:
                    self.logger.warning(f"Rate limited for {pdf_path}")
                    return None, None
                
                resp.raise_for_status()
                js = resp.json()

                if js.get("success"):
                    request_id = js.get("request_id")
                    check_url = js.get("request_check_url") or js.get("check_url")
                    if request_id and check_url:
                        self.logger.debug(f"Submitted {pdf_path}, request_id: {request_id}")
                        return request_id, check_url

                error_msg = js.get("error", "Unknown error")
                self.logger.error(f"API error for {pdf_path}: {error_msg}")
                return None, None
                
            except requests.exceptions.Timeout:
                self.logger.error(f"Timeout submitting {pdf_path}")
                return None, None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed for {pdf_path}: {str(e)}")
                return None, None

        return None, None

    def _poll_for_completion(self, check_url: str) -> Optional[Dict]:
        """Poll for completion with better error handling and logging."""
        poll_count = 0
        
        for poll_count in range(self.config.max_polls):
            try:
                resp = self.session.get(check_url, timeout=self.config.timeout)
                
                if resp.status_code == 429:
                    self.logger.warning("Rate limited during polling, increasing delay")
                    time.sleep(self.config.poll_interval * 2)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                status = (data.get("status") or "").lower()

                if status in ("complete", "finished") or (data.get("success") and "markdown" in data):
                    self.logger.debug(f"Conversion completed after {poll_count + 1} polls")
                    return data if data.get("success") else None
                    
                if status in ("queued", "processing", "running", "in_progress"):
                    if poll_count % 30 == 0:  # Log every 30 polls (1 minute)
                        self.logger.info(f"Still processing... (poll {poll_count + 1})")
                    time.sleep(self.config.poll_interval)
                    continue
                
                if status == "error" or status == "failed":
                    error_msg = data.get("error", "Processing failed")
                    self.logger.error(f"Processing failed with status '{status}': {error_msg}")
                    return None

                self.logger.warning(f"Unexpected status: {status}")
                time.sleep(self.config.poll_interval)
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Polling timeout (attempt {poll_count + 1})")
                time.sleep(self.config.poll_interval * 2)
                continue
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Polling failed (attempt {poll_count + 1}): {str(e)}")
                time.sleep(self.config.poll_interval)
                continue
        
        self.logger.error(f"Polling timeout reached after {self.config.max_polls} attempts")
        return None


# --- batch converter -----------------------------------------------------

class BatchPDFConverter:
    """Enhanced batch converter with better statistics and validation."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.client = DataLabMarkerClient(config)
        self.logger = logging.getLogger(__name__)

    def convert_directory(self, input_dir: str, output_dir: str, max_workers: int = 3, resume: bool = False) -> List[ConversionResult]:
        """Convert directory with enhanced progress tracking."""
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

        results: List[ConversionResult] = []
        start_time = time.time()
        
        # Use single thread for small batches to avoid rate limiting
        if len(pdf_files) <= 5:
            max_workers = 1
            self.logger.info("Using single thread for small batch")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.client.convert_pdf, str(f)): f for f in pdf_files}
            
            with tqdm(total=len(pdf_files), desc="Converting PDFs", unit="file") as pbar:
                for future in as_completed(future_map):
                    pdf_file = future_map[future]
                    try:
                        res = future.result()
                        results.append(res)
                        
                        if res.success:
                            self._save_result(res, output_path)
                            pages_info = f"{res.page_count} pages" if res.page_count else "? pages"
                            size_info = f"{res.file_size / 1024:.0f}KB" if res.file_size else ""
                            retry_info = f" (retries: {res.retry_count})" if res.retry_count > 0 else ""
                            self.logger.info(f"✓ {pdf_file.name} - {pages_info}, {size_info}{retry_info}")
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
                "use_llm": self.config.use_llm,
                "force_ocr": self.config.force_ocr,
                "max_pages": self.config.max_pages,
                "max_workers": "auto"  # This would need to be passed from caller
            }
        }
        
        with open(out / "conversion_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        print("\n=== CONVERSION SUMMARY ===")
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


# --- main ---------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Enhanced Batch PDF to Markdown converter using Datalab Marker API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (API key in .env)
  python batch-pdf-converter.py --input-dir ./pdfs --output-dir ./output

  # With LLM enhancement and quality settings
  python batch-pdf-converter.py --input-dir ./pdfs --output-dir ./output --use-llm --max-retries 5

  # Force OCR with custom timeouts
  python batch-pdf-converter.py --input-dir ./pdfs --output-dir ./output --force-ocr --timeout 60

  # Resume interrupted batch with verbose logging
  python batch-pdf-converter.py --input-dir ./pdfs --output-dir ./output --resume --max-workers 1
        """
    )

    parser.add_argument("--api-key", default=os.getenv("MARKER_API_KEY"),
                        help="Datalab API key (or set in .env as MARKER_API_KEY)")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files to convert")
    parser.add_argument("--output-dir", required=True, help="Directory to save converted Markdown files")

    # Quality and processing options
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for enhanced accuracy")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on every page")
    parser.add_argument("--paginate", action="store_true", help="Add page delimiters")
    parser.add_argument("--strip-existing-ocr", action="store_true", help="Remove existing OCR text and redo OCR")
    parser.add_argument("--disable-image-extraction", action="store_true", help="Disable image extraction")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to process per PDF")

    # Performance and reliability options
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent conversions")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts per file")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries (seconds)")
    parser.add_argument("--rate-limit-delay", type=int, default=1, help="Delay between requests (seconds)")

    # Timeout and polling options
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--max-polls", type=int, default=300, help="Maximum polling attempts")
    parser.add_argument("--poll-interval", type=int, default=2, help="Polling interval in seconds")
    
    parser.add_argument("--base-url", default=os.getenv("MARKER_BASE_URL", "https://www.datalab.to/api/v1"),
                        help="Override API base URL (or set MARKER_BASE_URL in .env)")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key not provided. Set MARKER_API_KEY in .env or use --api-key.")
        sys.exit(2)
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        sys.exit(1)

    config = ConversionConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        use_llm=args.use_llm,
        force_ocr=args.force_ocr,
        paginate=args.paginate,
        strip_existing_ocr=args.strip_existing_ocr,
        disable_image_extraction=args.disable_image_extraction,
        max_pages=args.max_pages,
        timeout=args.timeout,
        max_polls=args.max_polls,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        rate_limit_delay=args.rate_limit_delay,
    )

    converter = BatchPDFConverter(config)
    results = converter.convert_directory(args.input_dir, args.output_dir, args.max_workers, args.resume)

    failed = sum(1 for r in results if not r.success)
    success_rate = (len(results) - failed) / len(results) * 100 if results else 0
    
    print(f"\nConversion completed: {success_rate:.1f}% success rate")
    sys.exit(1 if failed > len(results) * 0.5 else 0)  # Exit with error if >50% failed


if __name__ == "__main__":
    main()