#!/bin/bash
# Batch converts PDFs in a source folder to Markdown using Marker.
# --- Configuration ---
INPUT_DIR="source_pdfs"
OUTPUT_DIR="converted_markdown"
MARKER_FLAGS="--workers 0 --batch_multiplier 2"
# ---------------------
set -e
mkdir -p "$OUTPUT_DIR"
echo "Starting Marker batch conversion..."
echo "Input folder: $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "---------------------------"
while IFS= read -r pdf_file; do
base_name=$(basename "$pdf_file" .pdf)
echo "Converting: $base_name.pdf"
marker "$pdf_file" "$OUTPUT_DIR/$base_name.md" $MARKER_FLAGS
echo "Finished: $base_name.md"
echo "---------------------------"
done < <(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.pdf")
echo "All PDF conversions complete!"
