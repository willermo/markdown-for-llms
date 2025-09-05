#!/usr/bin/env python3
"""
Test script to monitor PDF conversion progress
"""

import requests
import time
import os
import sys
from pathlib import Path

def test_small_conversion():
    """Test with a small dummy PDF to verify API works"""
    print("Testing API with small dummy file...")
    files = {'file': ('test.pdf', b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n9\n%%EOF', 'application/pdf')}
    
    try:
        response = requests.post('http://localhost:8000/convert', files=files, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ API is working! Response length: {len(result.get('markdown', ''))}")
            return True
        else:
            print(f"✗ API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def monitor_conversion(pdf_path, check_interval=30):
    """Monitor a PDF conversion with progress tracking"""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False
    
    print(f"Starting conversion of: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB")
    print("This may take 15-30 minutes for large PDFs...")
    print("\nTo monitor progress, run in another terminal:")
    print("docker compose logs marker-api --follow")
    print("\nStarting conversion (no timeout)...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        
        try:
            # Start conversion with very long timeout
            response = requests.post('http://localhost:8000/convert', files=files, timeout=1800)  # 30 minutes
            
            if response.status_code == 200:
                result = response.json()
                markdown = result.get('markdown', '')
                
                # Save result
                output_file = Path(pdf_path).stem + '_converted.md'
                with open(output_file, 'w', encoding='utf-8') as out:
                    out.write(markdown)
                
                print(f"\n✓ Conversion completed successfully!")
                print(f"✓ Output saved to: {output_file}")
                print(f"✓ Markdown length: {len(markdown):,} characters")
                print(f"✓ First 200 chars: {markdown[:200]}...")
                return True
            else:
                print(f"\n✗ Conversion failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"\n⚠ Request timed out after 30 minutes")
            print("The conversion might still be running in the background.")
            print("Check: docker compose logs marker-api --tail 20")
            return False
        except Exception as e:
            print(f"\n✗ Conversion error: {e}")
            return False

def main():
    print("=== Marker API Conversion Test ===\n")
    
    # First test API with small file
    if not test_small_conversion():
        print("API is not responding correctly. Check Docker container.")
        return
    
    print("\n" + "="*50)
    
    # Test with actual PDF if available
    pdf_path = 'source_pdfs/Eric_Ries__Book_reference__The_Lean_Startup.pdf'
    if os.path.exists(pdf_path):
        print(f"\nFound PDF: {pdf_path}")
        choice = input("Start conversion? (y/n): ").lower().strip()
        if choice == 'y':
            monitor_conversion(pdf_path)
    else:
        print(f"\nPDF not found: {pdf_path}")
        print("Place a PDF file in source_pdfs/ directory to test conversion")

if __name__ == "__main__":
    main()
