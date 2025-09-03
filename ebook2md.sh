#!/bin/bash
# Enhanced batch converter for ebooks to Markdown optimized for LLM consumption

# --- Configuration ---
INPUT_DIR="source_ebooks"
OUTPUT_DIR="converted_markdown"
LOG_FILE="conversion.log"

# Pandoc options for better LLM-friendly output
PANDOC_OPTS="--wrap=none --strip-comments --markdown-headings=atx"
# ---------------------

set -e
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "Conversion started: $(date)" > "$LOG_FILE"

echo "Starting enhanced Pandoc batch conversion..."
echo "Input folder: $INPUT_DIR"
echo "Output folder: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "---------------------------"

# Counter for statistics
total_files=0
successful_conversions=0
failed_conversions=0

# Find files with common ebook/document extensions
while IFS= read -r input_file; do
    base_name=$(basename "$input_file")
    extension="${base_name##*.}"
    filename="${base_name%.*}"
    output_file="$OUTPUT_DIR/$filename.md"
    
    total_files=$((total_files + 1))
    
    echo "Converting: $base_name"
    echo "Processing: $input_file" >> "$LOG_FILE"
    
    # Attempt conversion with error handling
    if pandoc "$input_file" $PANDOC_OPTS -o "$output_file" 2>>"$LOG_FILE"; then
        echo "✓ Success: $filename.md"
        echo "✓ Success: $filename.md" >> "$LOG_FILE"
        successful_conversions=$((successful_conversions + 1))
        
        # Get file size for stats
        file_size=$(wc -c < "$output_file" 2>/dev/null || echo "0")
        echo "  Size: $file_size bytes"
    else
        echo "✗ Failed: $base_name (check log for details)"
        echo "✗ Failed: $base_name" >> "$LOG_FILE"
        failed_conversions=$((failed_conversions + 1))
    fi
    
    echo "---------------------------"

done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( \
    -name "*.epub" -o -name "*.mobi" -o -name "*.azw" -o \
    -name "*.azw3" -o -name "*.html" -o -name "*.htm" -o \
    -name "*.docx" -o -name "*.pdf" -o -name "*.rtf" \))

# Print statistics
echo "CONVERSION SUMMARY:"
echo "Total files processed: $total_files"
echo "Successful conversions: $successful_conversions"
echo "Failed conversions: $failed_conversions"
echo "Conversion completed: $(date)" >> "$LOG_FILE"

if [ $failed_conversions -gt 0 ]; then
    echo "⚠️  Some conversions failed. Check $LOG_FILE for details."
    exit 1
else
    echo "🎉 All conversions successful!"
fi