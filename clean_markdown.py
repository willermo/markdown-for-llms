import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

# --- Configuration ---
INPUT_DIR = "converted_markdown"
OUTPUT_DIR = "cleaned_markdown"
LOG_FILE = "cleaning.log"

# Advanced cleaning options
AGGRESSIVE_CLEANING = True  # More thorough cleaning for LLM optimization
PRESERVE_TABLES = True      # Keep table formatting
MIN_LINE_LENGTH = 3         # Remove lines shorter than this (likely artifacts)
# ---------------------

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class MarkdownCleaner:
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.stats = {
            'files_processed': 0,
            'total_removals': 0,
            'lines_removed': 0,
            'characters_removed': 0
        }
    
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
    
    def remove_short_lines(self, lines: List[str], min_length: int = 3) -> List[str]:
        """Remove lines that are too short to be meaningful"""
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
        """Main cleaning function"""
        logging.info(f"Processing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
        
        original_length = len(original_content)
        content = original_content
        
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
            lines = self.remove_short_lines(lines, MIN_LINE_LENGTH)
        
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
        
        # Calculate statistics
        chars_removed = original_length - len(final_content)
        self.stats['characters_removed'] += chars_removed
        self.stats['total_removals'] += 1
        
        logging.info(f"  Reduced by {chars_removed} characters ({chars_removed/original_length*100:.1f}%)")
        
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
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        logging.error(f"Input directory '{INPUT_DIR}' not found.")
        return
    
    output_path.mkdir(exist_ok=True)
    logging.info(f"Created/verified output directory: '{OUTPUT_DIR}'")
    
    cleaner = MarkdownCleaner(aggressive=AGGRESSIVE_CLEANING)
    
    logging.info("Starting comprehensive Markdown cleaning...")
    logging.info(f"Input folder: {INPUT_DIR}")
    logging.info(f"Output folder: {OUTPUT_DIR}")
    logging.info("=" * 50)
    
    markdown_files = list(input_path.glob("*.md"))
    
    if not markdown_files:
        logging.warning("No .md files found in input directory")
        return
    
    for md_file in markdown_files:
        try:
            cleaned_content = cleaner.clean_markdown_file(md_file)
            
            output_file = output_path / md_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            cleaner.stats['files_processed'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process {md_file.name}: {str(e)}")
    
    logging.info("=" * 50)
    logging.info("Cleaning completed!")
    logging.info(cleaner.get_statistics())

if __name__ == "__main__":
    main()