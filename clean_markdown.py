#!/usr/bin/env python3
"""
Markdown Cleaning Pipeline

This module provides comprehensive cleaning functionality for markdown files,
removing various artifacts and normalizing content for LLM consumption.
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Import new infrastructure
from config import get_config
from logging_config import get_cleaning_logger, log_operation, ProgressLogger
from exceptions import CleaningError, FileReadError, FileWriteError, error_context

# --- Configuration ---
INPUT_DIR = "converted_markdown"
OUTPUT_DIR = "cleaned_markdown"
LOG_FILE = "cleaning.log"

# Advanced cleaning options
AGGRESSIVE_CLEANING = True  # More thorough cleaning for LLM optimization
PRESERVE_TABLES = True      # Keep table formatting
MIN_LINE_LENGTH = 3         # Remove lines shorter than this (likely artifacts)
# ---------------------

class MarkdownCleaner:
    """Comprehensive markdown cleaning with configurable options"""
    
    def __init__(self, aggressive: bool = None, config=None):
        # Use config if provided, otherwise get global config
        self.config = config or get_config().cleaning
        self.aggressive = aggressive if aggressive is not None else self.config.aggressive_cleaning
        
        self.stats = {
            'files_processed': 0,
            'characters_removed': 0,
            'lines_removed': 0,
            'artifacts_cleaned': 0
        }
        
        # Setup logging using new system
        self.logger = get_cleaning_logger()
    
    def clean_ocr_artifacts(self, content: str) -> str:
        """Remove OCR artifacts like [A]BC -> ABC, [T]here -> There"""
        # Pattern for single bracketed letters at start of words
        content = re.sub(r'\[([A-Za-z])\]([a-zA-Z])', r'\1\2', content)
        
        # Pattern for names like D[AN] S. K[ENNEDY] -> DAN S. KENNEDY
        content = re.sub(r'([A-Z])\[([A-Z]+)\]', r'\1\2', content)
        
        # Pattern for mid-word brackets like wor[d] -> word
        content = re.sub(r'([a-zA-Z])\[([a-zA-Z]+)\]', r'\1\2', content)
        
        return content
    
    def clean_html_artifacts(self, content: str) -> str:
        """Remove HTML tags and artifacts"""
        # Remove all HTML tags (more comprehensive than original)
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#8217;': "'",
            '&#8220;': '"',
            '&#8221;': '"',
            '&#8211;': '–',
            '&#8212;': '—'
        }
        for entity, replacement in html_entities.items():
            content = content.replace(entity, replacement)
        
        return content
    
    def clean_pandoc_artifacts(self, content: str) -> str:
        """Remove Pandoc-specific artifacts"""
        # Multi-line Pandoc index term anchors
        content = re.sub(r'\[\]\{.*?\}', '', content, flags=re.DOTALL)
        
        # Simplify Markdown links to text only
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove Pandoc attributes
        content = re.sub(r'\{[^\}]+\}', '', content)
        
        # Remove Pandoc fenced divs
        content = re.sub(r'^\s*:::.*$', '', content, flags=re.MULTILINE)
        
        return content
    
    def clean_image_artifacts(self, content: str) -> str:
        """Remove image-related artifacts"""
        # Remove !Images references
        content = re.sub(r'!Images?', '', content, flags=re.IGNORECASE)
        
        # Remove empty image references
        content = re.sub(r'!\[\]\([^\)]*\)', '', content)
        
        # Remove standalone image references that don't add value
        content = re.sub(r'^\s*!\[.*?\]\(.*?\)\s*$', '', content, flags=re.MULTILINE)
        
        return content
    
    def normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace for better LLM consumption"""
        # Replace multiple spaces with single space
        content = re.sub(r' {2,}', ' ', content)
        
        # Replace multiple newlines with double newline (paragraph break)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove trailing spaces from lines
        content = re.sub(r' +$', '', content, flags=re.MULTILINE)
        
        return content
    
    def clean_formatting_artifacts(self, content: str) -> str:
        """Clean up formatting artifacts"""
        # Remove orphaned formatting characters
        content = re.sub(r'(?:^|\s)[*_]{1,2}(?:\s|$)', ' ', content)
        
        # Fix broken emphasis (e.g., "** text **" -> "**text**")
        content = re.sub(r'\*\* ([^*]+) \*\*', r'**\1**', content)
        content = re.sub(r'_ ([^_]+) _', r'_\1_', content)
        
        # Remove standalone punctuation lines
        content = re.sub(r'^\s*[.,;:!?]+\s*$', '', content, flags=re.MULTILINE)
        
        return content
    
    def remove_short_lines(self, lines: List[str], min_length: int = None) -> List[str]:
        """Remove meaningless short lines while preserving structure"""
        if min_length is None:
            min_length = self.config.min_line_length
            
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep markdown headers, lists, and meaningful short lines
            if (len(stripped) >= min_length or 
                stripped.startswith('#') or 
                stripped.startswith('-') or 
                stripped.startswith('*') or 
                stripped.startswith('1.') or
                stripped == ''):
                cleaned_lines.append(line)
            else:
                self.stats['lines_removed'] += 1
        return cleaned_lines
    
    def clean_markdown_file(self, file_path: Path) -> str:
        """Clean a single markdown file and return the cleaned content"""
        with error_context("cleaning", str(file_path), CleaningError):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except FileNotFoundError:
                self.logger.error(f"File not found: {file_path}")
                raise FileReadError(f"File not found: {file_path}", str(file_path), "cleaning")
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {str(e)}")
                raise FileReadError(f"Error reading file: {str(e)}", str(file_path), "cleaning")
            
            original_length = len(content)
            original_lines = content.count('\n')
            
            # Apply cleaning steps
            content = self.clean_ocr_artifacts(content)
            content = self.clean_html_artifacts(content)
            content = self.clean_pandoc_artifacts(content)
            content = self.clean_image_artifacts(content)
            content = self.clean_formatting_artifacts(content)
            content = self.normalize_whitespace(content)
            
            # Process line by line for final cleanup
            lines = content.splitlines()
            
            # Remove short/meaningless lines
            if self.aggressive:
                lines = self.remove_short_lines(lines, self.config.min_line_length)
            
            # Final line-level cleaning
            cleaned_lines = []
            for line in lines:
                # Skip empty image tags
                if re.match(r'^!\[\]\([^\)]+\)$', line.strip()):
                    continue
                # Skip Pandoc fenced divs
                if re.match(r'^\s*:::', line):
                    continue
                # Skip lines with only special characters
                if re.match(r'^[\s\-_=*#]+$', line.strip()) and len(line.strip()) < 10:
                    continue
                
                cleaned_lines.append(line)
            
            final_content = "\n".join(cleaned_lines)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['characters_removed'] += original_length - len(final_content)
            self.stats['lines_removed'] += original_lines - final_content.count('\n')
            
            self.logger.info(f"Cleaned {file_path.name}: {original_length} -> {len(final_content)} chars")
            
            return final_content
    
    def get_statistics(self) -> str:
        """Return cleaning statistics"""
        return f"""
Cleaning Statistics:
- Files processed: {self.stats['files_processed']}
- Total characters removed: {self.stats['characters_removed']:,}
- Lines removed: {self.stats['lines_removed']}
- Average reduction: {self.stats['characters_removed']/max(1, self.stats['files_processed']):.0f} chars/file
        """.strip()

def main():
    """Main function to process all markdown files"""
    # Use configuration system
    config = get_config()
    input_path = Path(config.directories.converted)
    output_path = Path(config.directories.cleaned)
    
    # Use new logging system
    logger = get_cleaning_logger()
    
    if not input_path.exists():
        logger.error(f"Input directory '{input_path}' not found.")
        return
    
    output_path.mkdir(exist_ok=True)
    logger.info(f"Created/verified output directory: '{output_path}'")
    
    cleaner = MarkdownCleaner()
    
    markdown_files = list(input_path.glob("*.md"))
    
    if not markdown_files:
        logger.warning("No .md files found in input directory")
        return
    
    # Use progress logger for batch processing
    progress = ProgressLogger(logger, len(markdown_files), "markdown cleaning")
    
    with log_operation(logger, "batch markdown cleaning", file_count=len(markdown_files)):
        for md_file in markdown_files:
            try:
                with error_context("cleaning", str(md_file), CleaningError):
                    cleaned_content = cleaner.clean_markdown_file(md_file)
                    
                    output_file = output_path / md_file.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    progress.update(item_name=md_file.name)
                    
            except Exception as e:
                logger.error(f"Failed to process {md_file.name}: {str(e)}")
                progress.update()  # Still count as processed for progress
        
        progress.complete()
        logger.info(cleaner.get_statistics())

if __name__ == "__main__":
    main()