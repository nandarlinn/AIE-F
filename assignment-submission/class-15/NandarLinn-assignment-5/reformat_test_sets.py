#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reorganizes syllables in test files to have exactly 20 syllables per line.
Developed for assignment5
"""

import os
import sys

def reformat_test_file(filepath):
    if not os.path.exists(filepath):
        print(f"[WARNING] File '{filepath}' does not exist. Skipping.")
        return
        
    print(f"Reformatting '{filepath}'...")
    
    # Read all syllables from the file
    all_syllables = []
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            for line in infile:
                # Syllables are space-separated
                parts = line.strip().split()
                all_syllables.extend(parts)
    except Exception as e:
        print(f"[ERROR] Failed to read '{filepath}': {e}")
        return
        
    # Group into chunks of 20 syllables
    chunk_size = 20
    reformatted_lines = []
    
    for i in range(0, len(all_syllables), chunk_size):
        chunk = all_syllables[i:i + chunk_size]
        reformatted_lines.append(" ".join(chunk) + "\n")
        
    # Overwrite the original file with reformatted lines
    try:
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.writelines(reformatted_lines)
        print(f"[SUCCESS] Reformatted '{filepath}'. Total syllables: {len(all_syllables)} | Output lines: {len(reformatted_lines)}")
    except Exception as e:
        print(f"[ERROR] Failed to write '{filepath}': {e}")

def main():
    output_dir = "/home/elio/Downloads/assignment5/cleaned_corpus"
    test_files = [
        "test_medical.txt",
        "test_news.txt",
        "test_literature.txt"
    ]
    
    for filename in test_files:
        filepath = os.path.join(output_dir, filename)
        reformat_test_file(filepath)

if __name__ == "__main__":
    main()
