import os
import re
import argparse

# ==========================================
# Regex Patterns for Garbled Myanmar Text
# ==========================================

# 1. Medial Ya (ျ) after consonants that don't take it in standard Burmese
# Excludes: က, ခ, ဂ, ဃ, ပ, ဖ, ဗ, ဘ, မ, လ
# Note: သျ is allowed ONLY if followed by ှ (e.g., သျှောင်)
INVALID_MEDIAL_YA = re.compile(r"[ငစဆဇဉညဋဌဍဎဏတထဒဓနရဝဟဠအ]ျ|သျ(?!ှ)")

# 2. Medial Ha (ှ) after invalid consonants
# Valid ones are: င, ည, ဏ, န, မ, ယ, ရ, လ, ဝ, ဠ
INVALID_MEDIAL_HA = re.compile(r"[ကခဂဃစဆဇဈဋဌဍဎတထဒဓပဖဗဘသအ]ှ")

# 3. Zha (ဈ) used incorrectly (not followed by common vowels)
# ဈ is rarely used except in words like ဈေး. If it appears as an Asat replacement (e.g. တဈ for တစ်), it's invalid.
INVALID_ZHA = re.compile(r"ဈ(?![ေးာိီုူ])")

def is_garbled(text):
    """
    Returns True if the text contains garbled sequences (non-standard Unicode typing errors).
    """
    if INVALID_MEDIAL_YA.search(text):
        return True
    if INVALID_MEDIAL_HA.search(text):
        return True
    if INVALID_ZHA.search(text):
        return True
    return False

def filter_corpus_directory(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not filepaths:
        print(f"No .txt files found in '{input_dir}'.")
        return

    total_files = len(filepaths)
    total_lines_read = 0
    total_lines_kept = 0
    total_lines_removed = 0

    print(f"Found {total_files} files to process.\n")

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:
             
            for line in infile:
                total_lines_read += 1
                if is_garbled(line):
                    total_lines_removed += 1
                else:
                    outfile.write(line)
                    total_lines_kept += 1

    print("=" * 40)
    print("Filtering Complete!")
    print(f"Total lines read: {total_lines_read}")
    print(f"Total lines kept: {total_lines_kept}")
    print(f"Total lines removed (garbled): {total_lines_removed}")
    print(f"Filtered files saved to: {output_dir}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out garbled (non-standard Unicode) lines from corpus.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing the input .txt files (e.g., segmented_corpus).")
    parser.add_argument("-o", "--output_dir", type=str, default="filtered_corpus", help="Directory to save the filtered files.")
    
    args = parser.parse_args()
    filter_corpus_directory(args.input_dir, args.output_dir)
