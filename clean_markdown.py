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
