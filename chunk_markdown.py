import os
import re
import json
import logging
import tiktoken
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

# --- Configuration ---
INPUT_DIR = "validated_markdown"
OUTPUT_DIR = "chunked_markdown"
METADATA_FILE = "chunking_metadata.json"
LOG_FILE = "chunking.log"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 4000      # Target tokens per chunk
DEFAULT_OVERLAP = 200          # Token overlap between chunks
MIN_CHUNK_SIZE = 1000          # Minimum viable chunk size
MAX_CHUNK_SIZE = 8000          # Maximum chunk size

# LLM-specific presets
LLM_PRESETS = {
    'gpt-3.5-turbo': {'chunk_size': 3000, 'overlap': 150},
    'gpt-4': {'chunk_size': 6000, 'overlap': 300},
    'claude-3': {'chunk_size': 8000, 'overlap': 400},
    'llama-2': {'chunk_size': 3500, 'overlap': 175},
    'gemini-pro': {'chunk_size': 7000, 'overlap': 350},
    'custom': {'chunk_size': DEFAULT_CHUNK_SIZE, 'overlap': DEFAULT_OVERLAP}
}

TARGET_LLM = 'custom'  # Change this to use specific LLM presets
# ---------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class ChunkingStrategy(Enum):
    SEMANTIC = "semantic"          # Split on natural boundaries (headings, paragraphs)
    FIXED_SIZE = "fixed_size"      # Split on fixed token counts
    SLIDING_WINDOW = "sliding_window"  # Overlapping sliding windows
    HIERARCHICAL = "hierarchical"  # Preserve document hierarchy

@dataclass
class ChunkMetadata:
    chunk_id: str
    source_file: str
    chunk_index: int
    total_chunks: int
    token_count: int
    character_count: int
    word_count: int
    start_position: int
    end_position: int
    heading_context: List[str]  # Hierarchical headings leading to this chunk
    content_type: str           # paragraph, heading, list, etc.
    overlap_with_previous: int
    overlap_with_next: int

@dataclass
class ChunkingResult:
    source_file: str
    total_chunks: int
    total_tokens: int
    chunking_strategy: str
    chunk_size_target: int
    overlap_size: int
    chunks: List[ChunkMetadata]

class TokenCounter:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize token counter with specific model encoding"""
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logging.warning(f"Unknown model {model}, using cl100k_base encoding")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text by token count"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

class SmartChunker:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.token_counter = TokenCounter()
        self.heading_hierarchy = []
        
    def extract_headings(self, content: str) -> List[Tuple[int, str, str]]:
        """Extract headings with their levels and positions"""
        headings = []
        lines = content.split('\n')
        position = 0
        
        for line in lines:
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.strip('#').strip()
                headings.append((position, level, heading_text))
            position += len(line) + 1  # +1 for newline
        
        return headings
    
    def get_heading_context(self, position: int, headings: List[Tuple[int, str, str]]) -> List[str]:
        """Get hierarchical heading context for a given position"""
        context = []
        current_levels = {}
        
        for heading_pos, level, text in headings:
            if heading_pos > position:
                break
            
            # Update hierarchy
            current_levels[level] = text
            # Remove deeper levels when we encounter a shallower heading
            current_levels = {k: v for k, v in current_levels.items() if k <= level}
        
        # Build context from current hierarchy
        for level in sorted(current_levels.keys()):
            context.append(current_levels[level])
        
        return context
    
    def split_on_sentences(self, text: str, max_tokens: int) -> List[str]:
        """Split text on sentence boundaries while respecting token limits"""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.token_counter.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence too long, force split
                    chunks.extend(self.token_counter.split_by_tokens(sentence, max_tokens))
                    current_chunk = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_semantic(self, content: str) -> List[Tuple[str, List[str]]]:
        """Chunk content based on semantic boundaries (headings, paragraphs)"""
        chunks = []
        headings = self.extract_headings(content)
        
        # Split content by major headings
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            section_tokens = self.token_counter.count_tokens(section)
            
            if section_tokens <= self.chunk_size:
                # Section fits in one chunk
                position = content.find(section)
                heading_context = self.get_heading_context(position, headings)
                chunks.append((section.strip(), heading_context))
            else:
                # Section too large, split by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for paragraph in paragraphs:
                    test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
                    
                    if self.token_counter.count_tokens(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            position = content.find(current_chunk)
                            heading_context = self.get_heading_context(position, headings)
                            chunks.append((current_chunk.strip(), heading_context))
                        
                        # Handle large paragraphs
                        if self.token_counter.count_tokens(paragraph) > self.chunk_size:
                            # Split large paragraph by sentences
                            sentence_chunks = self.split_on_sentences(paragraph, self.chunk_size)
                            for chunk in sentence_chunks:
                                position = content.find(chunk)
                                heading_context = self.get_heading_context(position, headings)
                                chunks.append((chunk, heading_context))
                            current_chunk = ""
                        else:
                            current_chunk = paragraph
                
                if current_chunk:
                    position = content.find(current_chunk)
                    heading_context = self.get_heading_context(position, headings)
                    chunks.append((current_chunk.strip(), heading_context))
        
        return chunks
    
    def chunk_sliding_window(self, content: str) -> List[Tuple[str, List[str]]]:
        """Create overlapping chunks using sliding window"""
        chunks = []
        headings = self.extract_headings(content)
        
        # Split into sentences for better boundary detection
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_position = 0
        
        while current_position < len(content):
            # Find chunk end position
            chunk_tokens = 0
            chunk_end = current_position
            chunk_text = ""
            
            # Build chunk until we reach target size
            temp_pos = current_position
            while chunk_tokens < self.chunk_size and temp_pos < len(content):
                # Find next sentence boundary
                next_sentence_end = content.find('.', temp_pos)
                if next_sentence_end == -1:
                    next_sentence_end = len(content)
                else:
                    next_sentence_end += 1
                
                test_text = content[current_position:next_sentence_end]
                test_tokens = self.token_counter.count_tokens(test_text)
                
                if test_tokens <= self.chunk_size:
                    chunk_text = test_text
                    chunk_tokens = test_tokens
                    chunk_end = next_sentence_end
                    temp_pos = next_sentence_end
                else:
                    break
            
            if not chunk_text:
                # Force at least some content
                chunk_text = content[current_position:current_position + 1000]
                chunk_end = current_position + 1000
            
            heading_context = self.get_heading_context(current_position, headings)
            chunks.append((chunk_text.strip(), heading_context))
            
            # Move position with overlap
            overlap_chars = int(len(chunk_text) * self.overlap / chunk_tokens) if chunk_tokens > 0 else 0
            current_position = max(current_position + 1, chunk_end - overlap_chars)
            
            if current_position >= len(content):
                break
        
        return chunks
    
    def add_overlap(self, chunks: List[str], content: str) -> List[Tuple[str, int, int]]:
        """Add overlap between chunks and return (chunk, overlap_prev, overlap_next)"""
        if not chunks or self.overlap <= 0:
            return [(chunk, 0, 0) for chunk in chunks]
        
        result = []
        
        for i, chunk in enumerate(chunks):
            overlap_prev = 0
            overlap_next = 0
            
            # Find chunk position in original content
            chunk_start = content.find(chunk)
            chunk_end = chunk_start + len(chunk)
            
            # Add overlap with previous chunk
            if i > 0:
                overlap_start = max(0, chunk_start - self.overlap)
                overlap_text = content[overlap_start:chunk_start]
                overlap_tokens = self.token_counter.count_tokens(overlap_text)
                chunk = overlap_text + chunk
                overlap_prev = overlap_tokens
            
            # Add overlap with next chunk
            if i < len(chunks) - 1:
                overlap_end = min(len(content), chunk_end + self.overlap)
                overlap_text = content[chunk_end:overlap_end]
                overlap_tokens = self.token_counter.count_tokens(overlap_text)
                chunk = chunk + overlap_text
                overlap_next = overlap_tokens
            
            result.append((chunk, overlap_prev, overlap_next))
        
        return result
    
    def chunk_document(self, content: str, filename: str, strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC) -> ChunkingResult:
        """Main chunking function"""
        logging.info(f"Chunking {filename} using {strategy.value} strategy")
        
        # Choose chunking strategy
        if strategy == ChunkingStrategy.SEMANTIC:
            raw_chunks = self.chunk_semantic(content)
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            raw_chunks = self.chunk_sliding_window(content)
        else:  # FIXED_SIZE fallback
            token_chunks = self.token_counter.split_by_tokens(content, self.chunk_size)
            headings = self.extract_headings(content)
            raw_chunks = []
            for chunk in token_chunks:
                position = content.find(chunk)
                heading_context = self.get_heading_context(position, headings)
                raw_chunks.append((chunk, heading_context))
        
        # Process chunks with metadata
        chunks_metadata = []
        total_tokens = 0
        
        for i, (chunk_text, heading_context) in enumerate(raw_chunks):
            # Calculate overlap
            overlap_prev = 0
            overlap_next = 0
            
            if i > 0 and self.overlap > 0:
                # Find overlap with previous chunk
                prev_chunk = raw_chunks[i-1][0]
                overlap_text = self.find_overlap(prev_chunk, chunk_text)
                overlap_prev = self.token_counter.count_tokens(overlap_text)
            
            if i < len(raw_chunks) - 1 and self.overlap > 0:
                # Find overlap with next chunk
                next_chunk = raw_chunks[i+1][0]
                overlap_text = self.find_overlap(chunk_text, next_chunk)
                overlap_next = self.token_counter.count_tokens(overlap_text)
            
            # Create metadata
            token_count = self.token_counter.count_tokens(chunk_text)
            char_count = len(chunk_text)
            word_count = len(chunk_text.split())
            start_pos = content.find(chunk_text)
            end_pos = start_pos + char_count
            
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{filename}_chunk_{i+1:03d}",
                source_file=filename,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                token_count=token_count,
                character_count=char_count,
                word_count=word_count,
                start_position=start_pos,
                end_position=end_pos,
                heading_context=heading_context,
                content_type=self.classify_content(chunk_text),
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next
            )
            
            chunks_metadata.append(chunk_metadata)
            total_tokens += token_count
        
        return ChunkingResult(
            source_file=filename,
            total_chunks=len(chunks_metadata),
            total_tokens=total_tokens,
            chunking_strategy=strategy.value,
            chunk_size_target=self.chunk_size,
            overlap_size=self.overlap,
            chunks=chunks_metadata
        )
    
    def find_overlap(self, chunk1: str, chunk2: str) -> str:
        """Find overlapping text between two chunks"""
        # Simple overlap detection - find common suffix/prefix
        max_overlap = min(len(chunk1), len(chunk2), self.overlap * 10)  # char approximation
        
        for i in range(max_overlap, 0, -1):
            if chunk1[-i:] == chunk2[:i]:
                return chunk1[-i:]
        
        return ""
    
    def classify_content(self, chunk_text: str) -> str:
        """Classify the type of content in a chunk"""
        lines = chunk_text.strip().split('\n')
        
        if not lines:
            return "empty"
        
        first_line = lines[0].strip()
        
        if first_line.startswith('#'):
            return "heading_section"
        elif re.match(r'^\d+\.|\*|-', first_line):
            return "list"
        elif len([l for l in lines if l.strip().startswith('|')]) > 2:
            return "table"
        elif first_line.startswith('>'):
            return "quote"
        elif first_line.startswith('```'):
            return "code_block"
        else:
            return "paragraph"

class MarkdownChunker:
    def __init__(self, target_llm: str = TARGET_LLM):
        self.target_llm = target_llm
        self.chunk_size = LLM_PRESETS[target_llm]['chunk_size']
        self.overlap = LLM_PRESETS[target_llm]['overlap']
        self.chunker = SmartChunker(self.chunk_size, self.overlap)
        
    def process_file(self, file_path: Path, strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC) -> Tuple[ChunkingResult, List[str]]:
        """Process a single markdown file and return chunks with metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Failed to read {file_path.name}: {str(e)}")
            return None, []
        
        if not content.strip():
            logging.warning(f"Empty file: {file_path.name}")
            return None, []
        
        # Chunk the document
        result = self.chunker.chunk_document(content, file_path.name, strategy)
        
        # Extract chunk texts for saving
        chunks = []
        for i, (chunk_text, _) in enumerate(self.chunker.chunk_semantic(content) if strategy == ChunkingStrategy.SEMANTIC else 
                                           self.chunker.chunk_sliding_window(content) if strategy == ChunkingStrategy.SLIDING_WINDOW else
                                           [(chunk, []) for chunk in self.chunker.token_counter.split_by_tokens(content, self.chunk_size)]):
            chunks.append(chunk_text)
        
        logging.info(f"  Created {len(chunks)} chunks, {result.total_tokens} total tokens")
        
        return result, chunks
    
    def save_chunks(self, filename: str, chunks: List[str], metadata: ChunkingResult, output_path: Path):
        """Save chunks to individual files with metadata"""
        base_name = Path(filename).stem
        
        for i, chunk_text in enumerate(chunks):
            chunk_filename = f"{base_name}_chunk_{i+1:03d}.md"
            chunk_path = output_path / chunk_filename
            
            # Create chunk file with metadata header
            chunk_metadata = metadata.chunks[i] if i < len(metadata.chunks) else None
            
            with open(chunk_path, 'w', encoding='utf-8') as f:
                # Write metadata header
                f.write("---\n")
                f.write(f"chunk_id: {chunk_metadata.chunk_id if chunk_metadata else f'{base_name}_chunk_{i+1:03d}'}\n")
                f.write(f"source_file: {filename}\n")
                f.write(f"chunk_index: {i}\n")
                f.write(f"total_chunks: {len(chunks)}\n")
                if chunk_metadata:
                    f.write(f"token_count: {chunk_metadata.token_count}\n")
                    f.write(f"word_count: {chunk_metadata.word_count}\n")
                    if chunk_metadata.heading_context:
                        f.write(f"heading_context: {' > '.join(chunk_metadata.heading_context)}\n")
                    f.write(f"content_type: {chunk_metadata.content_type}\n")
                f.write(f"chunking_strategy: {metadata.chunking_strategy}\n")
                f.write(f"target_llm: {self.target_llm}\n")
                f.write("---\n\n")
                
                # Write chunk content
                f.write(chunk_text)

def create_index_file(all_results: List[ChunkingResult], output_path: Path):
    """Create an index file listing all chunks"""
    index_data = {
        'chunking_summary': {
            'total_source_files': len(all_results),
            'total_chunks': sum(r.total_chunks for r in all_results),
            'total_tokens': sum(r.total_tokens for r in all_results),
            'average_chunk_size': sum(r.total_tokens for r in all_results) / sum(r.total_chunks for r in all_results) if all_results else 0,
            'target_llm': TARGET_LLM,
            'chunk_size_target': LLM_PRESETS[TARGET_LLM]['chunk_size'],
            'overlap_size': LLM_PRESETS[TARGET_LLM]['overlap']
        },
        'files': []
    }
    
    for result in all_results:
        file_info = {
            'source_file': result.source_file,
            'chunk_count': result.total_chunks,
            'total_tokens': result.total_tokens,
            'chunking_strategy': result.chunking_strategy,
            'chunks': []
        }
        
        for chunk in result.chunks:
            chunk_info = {
                'chunk_id': chunk.chunk_id,
                'filename': f"{Path(result.source_file).stem}_chunk_{chunk.chunk_index+1:03d}.md",
                'token_count': chunk.token_count,
                'content_type': chunk.content_type,
                'heading_context': chunk.heading_context
            }
            file_info['chunks'].append(chunk_info)
        
        index_data['files'].append(file_info)
    
    # Save index
    with open(output_path / 'chunks_index.json', 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

def main():
    """Main chunking function"""
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    if not input_path.exists():
        logging.error(f"Input directory '{INPUT_DIR}' not found.")
        return
    
    output_path.mkdir(exist_ok=True)
    
    chunker = MarkdownChunker(TARGET_LLM)
    all_results = []
    
    logging.info(f"Starting markdown chunking for {TARGET_LLM}...")
    logging.info(f"Target chunk size: {chunker.chunk_size} tokens")
    logging.info(f"Overlap size: {chunker.overlap} tokens")
    logging.info(f"Input folder: {INPUT_DIR}")
    logging.info(f"Output folder: {OUTPUT_DIR}")
    logging.info("=" * 60)
    
    markdown_files = list(input_path.glob("*.md"))
    
    if not markdown_files:
        logging.warning("No .md files found in input directory")
        return
    
    # Process each file
    for md_file in markdown_files:
        try:
            # Try semantic chunking first
            result, chunks = chunker.process_file(md_file, ChunkingStrategy.SEMANTIC)
            
            if result and chunks:
                # Check if chunks are too large, switch to sliding window
                oversized_chunks = [c for c in result.chunks if c.token_count > chunker.chunk_size * 1.5]
                
                if len(oversized_chunks) > len(result.chunks) * 0.3:  # More than 30% oversized
                    logging.info(f"  Switching to sliding window for {md_file.name}")
                    result, chunks = chunker.process_file(md_file, ChunkingStrategy.SLIDING_WINDOW)
                
                if result and chunks:
                    chunker.save_chunks(md_file.name, chunks, result, output_path)
                    all_results.append(result)
                    
                    logging.info(f"  ✓ Processed: {result.total_chunks} chunks, avg {result.total_tokens//result.total_chunks} tokens/chunk")
                else:
                    logging.error(f"  ✗ Failed to chunk: {md_file.name}")
            else:
                logging.error(f"  ✗ Failed to process: {md_file.name}")
                
        except Exception as e:
            logging.error(f"Error processing {md_file.name}: {str(e)}")
    
    # Create comprehensive metadata
    if all_results:
        # Save detailed metadata
        metadata = {
            'chunking_parameters': {
                'target_llm': TARGET_LLM,
                'chunk_size': chunker.chunk_size,
                'overlap': chunker.overlap,
                'min_chunk_size': MIN_CHUNK_SIZE,
                'max_chunk_size': MAX_CHUNK_SIZE
            },
            'results': [asdict(result) for result in all_results]
        }
        
        with open(output_path / METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create index file
        create_index_file(all_results, output_path)
        
        # Summary statistics
        total_files = len(all_results)
        total_chunks = sum(r.total_chunks for r in all_results)
        total_tokens = sum(r.total_tokens for r in all_results)
        avg_chunk_size = total_tokens / total_chunks if total_chunks > 0 else 0
        
        logging.info("=" * 60)
        logging.info("CHUNKING SUMMARY:")
        logging.info(f"Files processed: {total_files}")
        logging.info(f"Total chunks created: {total_chunks}")
        logging.info(f"Total tokens: {total_tokens:,}")
        logging.info(f"Average chunk size: {avg_chunk_size:.0f} tokens")
        logging.info(f"Target chunk size: {chunker.chunk_size} tokens")
        logging.info(f"Metadata saved to: {METADATA_FILE}")
        logging.info(f"Index saved to: chunks_index.json")
        
        # Quality metrics
        oversized_chunks = []
        undersized_chunks = []
        for result in all_results:
            for chunk in result.chunks:
                if chunk.token_count > chunker.chunk_size * 1.2:
                    oversized_chunks.append(chunk)
                elif chunk.token_count < chunker.chunk_size * 0.3:
                    undersized_chunks.append(chunk)
        
        if oversized_chunks:
            logging.warning(f"  {len(oversized_chunks)} chunks exceed target size by >20%")
        if undersized_chunks:
            logging.warning(f"  {len(undersized_chunks)} chunks are <30% of target size")
        
        logging.info("🎉 Chunking completed successfully!")
    else:
        logging.error("No files were successfully processed")

if __name__ == "__main__":
    main()
        