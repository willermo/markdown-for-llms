#!/usr/bin/env python3
"""
Markdown Validation Pipeline

This module provides comprehensive validation for cleaned markdown files,
assessing quality and providing recommendations for improvement.
"""

import re
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import new infrastructure
from config import get_config
from logging_config import get_validation_logger, log_operation, ProgressLogger
from exceptions import ValidationError, FileReadError, error_context

# --- Configuration ---
INPUT_DIR = "cleaned_markdown"
OUTPUT_DIR = "validated_markdown"
VALIDATION_REPORT = "validation_report.json"
LOG_FILE = "validation.log"

# Validation thresholds
MIN_CONTENT_LENGTH = 100        # Minimum characters for valid content
MAX_CONTENT_LENGTH = 10_000_000 # Maximum characters (10MB text files)
MIN_WORDS = 50                  # Minimum word count for valid content
MAX_ARTIFACT_RATIO = 0.05       # Maximum ratio of artifacts to content
MIN_READABILITY_SCORE = 20      # Minimum readability score (0-100)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

@dataclass
class ValidationResult:
    filename: str
    file_size: int
    character_count: int
    word_count: int
    paragraph_count: int
    heading_count: int
    line_count: int
    
    # Quality metrics
    artifact_ratio: float
    readability_score: float
    structure_score: float
    
    # Issues found
    html_tags: List[str]
    suspicious_patterns: List[str]
    encoding_issues: List[str]
    formatting_issues: List[str]
    
    # Overall assessment
    is_valid: bool
    quality_grade: str  # A, B, C, D, F
    recommendations: List[str]

class MarkdownValidator:
    """Comprehensive markdown validation with quality scoring"""
    
    def __init__(self, config=None):
        # Use config if provided, otherwise get global config
        self.config = config or get_config().validation
        
        # Setup logging using new system
        self.logger = get_validation_logger()
        
        self.html_pattern = re.compile(r'<[^>]+>')
        self.suspicious_patterns = {
            'ocr_artifacts': re.compile(r'\[[A-Za-z]\]'),
            'multiple_spaces': re.compile(r'  +'),
            'broken_formatting': re.compile(r'\*\* +|\*\*$|^ +\*\*'),
            'orphaned_punctuation': re.compile(r'^[.,;:!?]+$'),
            'image_artifacts': re.compile(r'!Images?|!\[\]'),
            'pandoc_artifacts': re.compile(r'\{[^}]*\}|:::'),
            'encoding_issues': re.compile(r'[^\x00-\x7F\u00A0-\u017F\u0180-\u024F\u1E00-\u1EFF]'),
            'excessive_newlines': re.compile(r'\n{4,}'),
            'malformed_links': re.compile(r'\]\([^)]*$|\[[^\]]*$'),
        }
    
    def count_artifacts(self, content: str) -> Tuple[int, List[str]]:
        """Count various artifacts and return suspicious patterns found"""
        total_artifacts = 0
        found_patterns = []
        
        for pattern_name, pattern in self.suspicious_patterns.items():
            matches = pattern.findall(content)
            if matches:
                total_artifacts += len(matches)
                found_patterns.append(f"{pattern_name}: {len(matches)} instances")
        
        return total_artifacts, found_patterns
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score (0-100)"""
        words = content.split()
        if not words:
            return 0
        
        # Basic metrics
        content_length = len(content)
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content)) or 1
        
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / word_count
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Simple readability score (inverse of complexity)
        # Lower complexity = higher readability
        complexity = (avg_word_length * 0.5) + (avg_sentence_length * 0.1)
        readability = max(0, min(100, 100 - complexity))
        
        return round(readability, 1)
    
    def calculate_structure_score(self, content: str) -> float:
        """Calculate structure quality score based on markdown elements"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines == 0:
            return 0
        
        # Count structural elements
        headings = len([line for line in lines if line.strip().startswith('#')])
        paragraphs = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        empty_lines = len([line for line in lines if not line.strip()])
        
        # Structure ratios
        heading_ratio = headings / total_lines
        paragraph_ratio = paragraphs / total_lines
        whitespace_ratio = empty_lines / total_lines
        
        # Good structure has reasonable ratios
        structure_score = 0
        
        # Heading ratio (should be 5-15% for good structure)
        if 0.05 <= heading_ratio <= 0.15:
            structure_score += 30
        elif 0.02 <= heading_ratio <= 0.25:
            structure_score += 20
        
        # Paragraph ratio (should be majority of content)
        if paragraph_ratio >= 0.6:
            structure_score += 40
        elif paragraph_ratio >= 0.4:
            structure_score += 25
        
        # Whitespace ratio (should be reasonable for readability)
        if 0.1 <= whitespace_ratio <= 0.3:
            structure_score += 30
        elif whitespace_ratio <= 0.4:
            structure_score += 20
        
        return min(100, structure_score)
    
    def detect_html_tags(self, content: str) -> List[str]:
        """Find remaining HTML tags"""
        html_matches = self.html_pattern.findall(content)
        return list(set(html_matches))  # Remove duplicates
    
    def detect_encoding_issues(self, content: str) -> List[str]:
        """Detect potential encoding problems"""
        issues = []
        
        # Common encoding artifacts
        encoding_artifacts = {
            'â€™': "Smart quote encoding issue",
            'â€œ': "Smart quote encoding issue", 
            'â€': "Smart quote encoding issue",
            'Ã¡': "Accented character encoding issue",
            'Ã©': "Accented character encoding issue",
            'â€¦': "Ellipsis encoding issue"
        }
        
        for artifact, description in encoding_artifacts.items():
            if artifact in content:
                issues.append(f"{description}: found '{artifact}'")
        
        # Check for unusual Unicode characters
        unusual_chars = re.findall(r'[^\x00-\x7F\u00A0-\u017F\u0180-\u024F\u1E00-\u1EFF]', content)
        if unusual_chars:
            unique_chars = list(set(unusual_chars))[:10]  # Limit to first 10 unique
            issues.append(f"Unusual Unicode characters: {', '.join(unique_chars)}")
        
        return issues
    
    def detect_formatting_issues(self, content: str) -> List[str]:
        """Detect formatting problems"""
        issues = []
        
        # Check for broken emphasis
        if re.search(r'\*\* +[^*]+ +\*\*', content):
            issues.append("Broken bold formatting with extra spaces")
        
        if re.search(r'_ +[^_]+ +_', content):
            issues.append("Broken italic formatting with extra spaces")
        
        # Check for malformed headings
        malformed_headings = re.findall(r'^#{7,}|^# *$', content, re.MULTILINE)
        if malformed_headings:
            issues.append(f"Malformed headings: {len(malformed_headings)} instances")
        
        # Check for excessive punctuation
        excessive_punct = re.findall(r'[.!?]{3,}', content)
        if excessive_punct:
            issues.append(f"Excessive punctuation: {len(excessive_punct)} instances")
        
        return issues
    
    def validate_markdown_file(self, file_path: Path) -> ValidationResult:
        """Validate a single markdown file"""
        with error_context("validation", str(file_path), ValidationError):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {str(e)}")
                raise FileReadError(f"Error reading file: {str(e)}", str(file_path), "validation")
            
            # Basic metrics
            file_size = file_path.stat().st_size
            character_count = len(content)
            words = content.split()
            word_count = len(words)
            lines = content.split('\n')
            line_count = len(lines)
            
            # Count paragraphs and headings
            non_empty_lines = [line for line in lines if line.strip()]
            paragraph_count = len([line for line in non_empty_lines if not line.strip().startswith('#')])
            heading_count = len([line for line in lines if line.strip().startswith('#')])
            
            # Calculate artifact ratio
            total_artifacts = len(self.detect_html_tags(content)) + len(self.detect_encoding_issues(content)) + len(self.detect_formatting_issues(content))
            artifact_ratio = total_artifacts / max(1, word_count)
            
            # Calculate scores
            readability_score = self.calculate_readability_score(content)
            structure_score = self.calculate_structure_score(content)
            
            # Quality checks
            html_tags = self.detect_html_tags(content)
            encoding_issues = self.detect_encoding_issues(content)
            formatting_issues = self.detect_formatting_issues(content)
            suspicious_patterns = self.count_artifacts(content)[1]
            
            # Create result
            result = ValidationResult(
                filename=file_path.name,
                file_size=file_size,
                character_count=character_count,
                word_count=word_count,
                paragraph_count=paragraph_count,
                heading_count=heading_count,
                line_count=line_count,
                artifact_ratio=artifact_ratio,
                readability_score=readability_score,
                structure_score=structure_score,
                html_tags=html_tags,
                suspicious_patterns=suspicious_patterns,
                encoding_issues=encoding_issues,
                formatting_issues=formatting_issues,
                is_valid=False,  # Will be set below
                quality_grade='',  # Will be set below
                recommendations=[]  # Will be set below
            )
            
            # Generate recommendations and grade
            issues = {
                'html_tags': html_tags,
                'encoding_issues': encoding_issues,
                'formatting_issues': formatting_issues,
                'suspicious_patterns': suspicious_patterns
            }
            result.recommendations = self.generate_recommendations(issues)
            result.quality_grade = self.assign_quality_grade(readability_score, structure_score, artifact_ratio)
            result.is_valid = (
                result.character_count >= MIN_CONTENT_LENGTH and
                result.word_count >= MIN_WORDS and
                result.artifact_ratio <= MAX_ARTIFACT_RATIO and
                len(result.html_tags) <= 5 and
                len(result.encoding_issues) <= 3
            )
            
            self.logger.info(f"  Grade: {result.quality_grade}, Valid: {result.is_valid}")
            
            return result
    
    def generate_recommendations(self, issues: Dict[str, List[str]]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if issues['html_tags']:
            recommendations.append("Remove remaining HTML tags")
        
        if issues['encoding_issues']:
            recommendations.append("Fix encoding issues")
        
        if issues['formatting_issues']:
            recommendations.append("Fix formatting issues")
        
        if issues['suspicious_patterns']:
            recommendations.append("Investigate suspicious patterns")
        
        return recommendations
    
    def assign_quality_grade(self, readability_score: float, structure_score: float, artifact_ratio: float) -> str:
        """Assign quality grade A-F based on metrics"""
        score = 0
        
        # Content length (20 points)
        if readability_score >= MIN_READABILITY_SCORE:
            score += 20
        
        # Low artifact ratio (20 points)
        if artifact_ratio <= MAX_ARTIFACT_RATIO:
            score += 20
        
        # Readability (20 points)
        score += min(20, readability_score / 5)
        
        # Structure (20 points)
        score += min(20, structure_score / 5)
        
        # No major issues (20 points) - simplified without undefined issues variable
        if artifact_ratio <= 0.1:  # Very low artifact ratio
            score += 20
        elif artifact_ratio <= 0.2:  # Moderate artifact ratio
            score += 10
        
        # Grade assignment
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
        
        logging.info(f"  Grade: {result.quality_grade}, Valid: {result.is_valid}")
        
        return result

def main():
    """Main validation function"""
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        logging.error(f"Input directory '{INPUT_DIR}' not found.")
        return
    
    output_path.mkdir(exist_ok=True)
    
    validator = MarkdownValidator()
    results = []
    
    logging.info("Starting markdown validation...")
    logging.info(f"Input folder: {INPUT_DIR}")
    logging.info(f"Output folder: {OUTPUT_DIR}")
    logging.info("=" * 60)
    
    markdown_files = list(input_path.glob("*.md"))
    
    if not markdown_files:
        logging.warning("No .md files found in input directory")
        return
    
    # Validate all files
    for md_file in markdown_files:
        result = validator.validate_file(md_file)
        results.append(result)
        
        # Copy valid files to output directory
        if result.is_valid:
            import shutil
            shutil.copy2(md_file, output_path / md_file.name)
    
    # Generate summary statistics
    total_files = len(results)
    valid_files = len([r for r in results if r.is_valid])
    grade_distribution = Counter(r.quality_grade for r in results)
    avg_readability = sum(r.readability_score for r in results) / total_files if total_files > 0 else 0
    avg_structure = sum(r.structure_score for r in results) / total_files if total_files > 0 else 0
    
    # Create detailed report
    report = {
        'summary': {
            'total_files': total_files,
            'valid_files': valid_files,
            'validation_rate': round(valid_files / total_files * 100, 1) if total_files > 0 else 0,
            'grade_distribution': dict(grade_distribution),
            'average_readability': round(avg_readability, 1),
            'average_structure_score': round(avg_structure, 1)
        },
        'detailed_results': [asdict(result) for result in results]
    }
    
    # Save report
    with open(VALIDATION_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Log summary
    logging.info("=" * 60)
    logging.info("VALIDATION SUMMARY:")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Valid files: {valid_files} ({valid_files/total_files*100:.1f}%)")
    logging.info(f"Grade distribution: {dict(grade_distribution)}")
    logging.info(f"Average readability: {avg_readability:.1f}")
    logging.info(f"Average structure score: {avg_structure:.1f}")
    logging.info(f"Detailed report saved to: {VALIDATION_REPORT}")
    
    if valid_files < total_files:
        logging.warning(f"{total_files - valid_files} files failed validation - check individual results")

if __name__ == "__main__":
    main()