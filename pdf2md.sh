#!/bin/bash
# Batch converts PDFs in a source folder to Markdown using Marker.
# --- Configuration ---
INPUT_DIR="source_pdfs"
OUTPUT_DIR="./converted_markdown"
MARKER_FLAGS="--workers 0"
# ---------------------

set -e
mkdir -p "$OUTPUT_DIR"

echo "Starting Marker batch conversion..."
echo "Input folder: $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "---------------------------"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist!"
    exit 1
fi

# Check if there are any PDF files
pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.pdf" | wc -l)
if [ "$pdf_count" -eq 0 ]; then
    echo "No PDF files found in '$INPUT_DIR'"
    exit 1
fi

echo "Found $pdf_count PDF file(s) to convert"

# Run marker on the entire input directory
echo "Running: marker $MARKER_FLAGS --output_dir \"$OUTPUT_DIR\" \"$INPUT_DIR\""
marker $MARKER_FLAGS --output_dir "$OUTPUT_DIR" "$INPUT_DIR"

echo "---------------------------"
echo "All PDF conversions complete!"
echo "Converted files are in: $OUTPUT_DIR"
