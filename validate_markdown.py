import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter

# --- Configuration ---
INPUT_DIR = "cleaned_markdown"
OUTPUT_DIR = "validated_markdown"
VALIDATION_REPORT = "validation_report.json"
LOG_FILE = "validation.log"

# Validation thresholds
MIN_CONTENT_LENGTH = 100        # Minimum characters for valid content
MAX_CONTENT_LENGTH = 10_000_000 # Maximum characters (10MB text files)
MIN_WORDS = 50                  # Minimum word count
MAX_ARTIFACT_RATIO = 0.05       # Maximum ratio of artifacts to content
MIN_READABILITY_SCORE = 20      # Minimum readability score (0-100)
# ---------------------

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
    def __init__(self):
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
        word_count = len(words)
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
    
    def generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if result.character_count < MIN_CONTENT_LENGTH:
            recommendations.append("File too short - may not contain meaningful content")
        
        if result.artifact_ratio > MAX_ARTIFACT_RATIO:
            recommendations.append("High artifact ratio - run additional cleaning")
        
        if result.readability_score < MIN_READABILITY_SCORE:
            recommendations.append("Low readability score - check for formatting issues")
        
        if result.html_tags:
            recommendations.append("Remove remaining HTML tags")
        
        if result.encoding_issues:
            recommendations.append("Fix encoding issues")
        
        if result.structure_score < 50:
            recommendations.append("Improve document structure (headings, paragraphs)")
        
        if result.word_count < MIN_WORDS:
            recommendations.append("Content may be too brief for meaningful analysis")
        
        return recommendations
    
    def calculate_quality_grade(self, result: ValidationResult) -> str:
        """Assign quality grade A-F based on metrics"""
        score = 0
        
        # Content length (20 points)
        if result.character_count >= MIN_CONTENT_LENGTH:
            score += 20
        
        # Low artifact ratio (20 points)
        if result.artifact_ratio <= MAX_ARTIFACT_RATIO:
            score += 20
        
        # Readability (20 points)
        score += min(20, result.readability_score / 5)
        
        # Structure (20 points)
        score += min(20, result.structure_score / 5)
        
        # No major issues (20 points)
        if not result.html_tags and not result.encoding_issues:
            score += 20
        elif len(result.html_tags) + len(result.encoding_issues) <= 2:
            score += 10
        
        # Grade assignment
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate a single markdown file"""
        logging.info(f"Validating: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Failed to read {file_path.name}: {str(e)}")
            # Return minimal failed result
            return ValidationResult(
                filename=file_path.name,
                file_size=0,
                character_count=0,
                word_count=0,
                paragraph_count=0,
                heading_count=0,
                line_count=0,
                artifact_ratio=1.0,
                readability_score=0,
                structure_score=0,
                html_tags=[],
                suspicious_patterns=[f"File read error: {str(e)}"],
                encoding_issues=[],
                formatting_issues=[],
                is_valid=False,
                quality_grade='F',
                recommendations=["Fix file reading issues"]
            )
        
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
        
        # Quality analysis
        artifact_count, suspicious_patterns = self.count_artifacts(content)
        artifact_ratio = artifact_count / max(1, character_count)
        
        readability_score = self.calculate_readability_score(content)
        structure_score = self.calculate_structure_score(content)
        
        # Issue detection
        html_tags = self.detect_html_tags(content)
        encoding_issues = self.detect_encoding_issues(content)
        formatting_issues = self.detect_formatting_issues(content)
        
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
        result.recommendations = self.generate_recommendations(result)
        result.quality_grade = self.calculate_quality_grade(result)
        result.is_valid = (
            result.character_count >= MIN_CONTENT_LENGTH and
            result.word_count >= MIN_WORDS and
            result.artifact_ratio <= MAX_ARTIFACT_RATIO and
            len(result.html_tags) <= 5 and
            len(result.encoding_issues) <= 3
        )
        
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