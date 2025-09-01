# Markdown for LLMs

This guide is a walkthrough to derive a clean markdown collection of resources from other common source formats.

It uses `Marker` to convert from `.pdf` files and `pandoc` to convert other common formats (`epub`, `mobi`, `azw3`).

## The two step procedure

To obtain a clean collection of `md` files for LLM usage we will follow a two step procedure consisting in:

1. conversion from original source
2. cleaning the residual noise derived from conversion

 



## Marker: PDF Conversion

Marker is a specialized tool that uses machine learning to convert **PDF files** into high-quality Markdown.

### Marker Installation on Ubuntu Linux



The recommended method is to use a Python virtual environment to avoid conflicts with system packages.

1. **Install Prerequisites:**

   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```
   
2. **Create and Activate Virtual Environment:**

   ```bash
   # Create a project folder
   mkdir marker_project && cd marker_project
   
   # Create the environment
   python3 -m venv marker-env
   
   # Activate it
   source marker-env/bin/activate
   ```
   
3. **Install Marker:**

   ```bash
   # This can take several minutes and download a few GB of data
   pip install marker-pdf
   ```



### Marker Usage and Code Examples



The basic command is `marker <input.pdf> <output.md>`.

**Scenario 1: Standard Technical Book** Leverage a powerful machine for faster conversion.

```bash
marker --workers 0 --batch_multiplier 2 "Advanced Python Programming.pdf" "Advanced Python Programming.md"
```

**Scenario 2: Very Large Book (1000+ pages)** First, test a small portion of the book.

```bash
# Test the first 20 pages
marker --max_pages 20 "The Complete Reference.pdf" "The Complete Reference_test.md"

# If satisfied, run the full conversion
marker --workers 0 --batch_multiplier 2 "The Complete Reference.pdf" "The Complete Reference.md"
```

**Scenario 3: Scanned Book (OCR Required)** Engage the OCR engine to extract text from images. This is much slower.

```bash
marker --workers 0 --ocr_langs english "Scanned Legacy Manual.pdf" "Scanned Legacy Manual.md"
```



### Marker Batch Conversion Script



Save this code as `pdf2md.sh` to process all PDFs in a specified folder.

```bash
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
```

**To Use:**

1. Place PDFs in a folder named `source_pdfs`.
2. Run `chmod +x pdf2md.sh`.
3. Execute with `./pdf2md.sh` (remember to activate the `marker-env` first).

------



## Pandoc: Ebook and Other Formats



For structured formats like **EPUB, MOBI, AZW, and HTML**, Marker is not the right tool. The best command-line utility for this is **Pandoc**.



### Pandoc Installation on Ubuntu Linux



Pandoc is available directly from the `apt` repositories.

```bash
sudo apt update
sudo apt install pandoc
```



### Pandoc Usage and Code Examples



Pandoc's syntax is straightforward.

```bash
# Convert an EPUB file to Markdown
pandoc "My Book.epub" -o "My Book.md"

# Convert an HTML file to Markdown
pandoc "My Document.html" -o "My Document.md"

# Convert a MOBI file to Markdown
pandoc "Another Book.mobi" -o "Another Book.md"
```



### Pandoc Batch Conversion Script



Save this code as `ebook2pdf.sh` to process various ebook/document formats.

```bash
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
```

**To Use:**

1. Place your EPUB, MOBI, HTML, etc., files in a folder named `source_ebooks`.
2. Run `chmod +x ebook2pdf.sh`.
3. Execute with `./ebook2pdf.sh`.



### Pandoc Batch Cleaning Script

This Python script reads the entire content of each file, allowing the regular expressions to match patterns across multiple lines. It incorporates all the cleaning rules for both **Marker** and **Pandoc** noise.



```python
import os
import re

# --- Configuration ---
# Directory containing your Markdown files to be cleaned.
INPUT_DIR = "converted_markdown"

# Directory where the cleaned Markdown files will be saved.
OUTPUT_DIR = "cleaned_markdown"
# ---------------------

def clean_markdown_file(file_path):
    """Reads, cleans, and returns the content of a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Rule 1: Remove multi-line Pandoc index term anchors like []{...}
    # The re.DOTALL flag is crucial as it allows '.' to match newlines.
    content = re.sub(r'\[\]\{.*?\}', '', content, flags=re.DOTALL)

    # Rule 2: Simplify all Markdown links [Text](URL) to just Text
    content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

    # Rule 3: Remove remaining single-line Pandoc attributes like {.label}
    content = re.sub(r'\{[^\}]+\}', '', content)
    
    # Rule 4: Remove empty HTML span tags from Marker
    content = re.sub(r'<span[^>]*><\/span>', '', content)

    # Process remaining rules line by line
    cleaned_lines = []
    for line in content.splitlines():
        # Rule 5: Remove empty image tags from Marker
        if re.match(r'^!\[\]\([^\)]+\)$', line.strip()):
            continue
        # Rule 6: Remove Pandoc fenced divs
        if re.match(r'^\s*:::', line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def main():
    """Main function to process all markdown files in the directory."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: '{OUTPUT_DIR}'")

    print("Starting final cleanup of Markdown files...")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("---------------------------")

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".md"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            print(f"Cleaning: {filename}")
            
            cleaned_content = clean_markdown_file(input_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

    print("---------------------------")
    print("All files cleaned!")

if __name__ == "__main__":
    main()
```

