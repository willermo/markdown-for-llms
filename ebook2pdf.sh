#!/bin/bash
# Batch converts various ebook and document formats to Markdown using Pandoc.

# --- Configuration ---
INPUT_DIR="source_ebooks"
OUTPUT_DIR="converted_markdown"
# ---------------------

set -e
mkdir -p "$OUTPUT_DIR"

echo "Starting Pandoc batch conversion..."
echo "Input folder: $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "---------------------------"

# Find files with common ebook/document extensions and loop through them
while IFS= read -r input_file; do
    # Get the filename and the extension
    base_name=$(basename "$input_file")
    extension="${base_name##*.}"
    filename="${base_name%.*}"

    echo "Converting: $base_name"

    # Run the pandoc command
    pandoc "$input_file" -o "$OUTPUT_DIR/$filename.md"

    echo "Finished: $filename.md"
    echo "---------------------------"

done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name "*.epub" -o -name "*.mobi" -o -name "*.azw" -o -name "*.html" \))

echo "All ebook conversions complete!"
