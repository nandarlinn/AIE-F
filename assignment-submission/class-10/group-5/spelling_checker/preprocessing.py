import argparse
import os
import re
from myanmartools import ZawgyiDetector
from rabbit import Rabbit as rabbit
from myword import WordTokenizer

words = WordTokenizer()

def apply_myword_segmentation(text):
    """
    Apply Dr. Ye Kyaw Thu's myWord segmenter here.
    """
    try:
        tokens = words.tokenize(text)
        text = [t.strip() for t in tokens]
        return " ".join(text)
    except Exception as e:
        # Catch KeyErrors or other issues from myword
        # If it fails, just return the unsegmented text
        return text

def fix_typing_errors(text):
    """
    Fix standard typing errors and numbers masquerading as characters.
    """
    # Fix standard typing errors using exact string replacement
    text = text.replace("ဥ်", "ဉ်")
    text = text.replace("စဥ်", "စဉ်")
    text = text.replace("ဥာ", "ဉာ")

    # Replace "၇" (Number 7) with "ရ" (Letter Ya) if preceded or followed by a Myanmar consonant (\u1000-\u1021)
    text = re.sub(r'(?<=[\u1000-\u1021])၇|၇(?=[\u1000-\u1021])', 'ရ', text)
    
    # Replace "၀" (Number 0) with "ဝ" (Letter Wa) if preceded or followed by a Myanmar consonant (\u1000-\u1021)
    text = re.sub(r'(?<=[\u1000-\u1021])၀|၀(?=[\u1000-\u1021])', 'ဝ', text)

    return text

def process_myanmar_directory(input_dir, clean_output_dir, segmented_output_dir):
    detector = ZawgyiDetector()

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        return

    # Create output directories if they don't exist
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(segmented_output_dir, exist_ok=True)

    # List all files in the given directory
    filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not filepaths:
        print(f"No files found in {input_dir}.")
        return

    total_lines_processed = 0

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        clean_output_filepath = os.path.join(clean_output_dir, filename)
        segmented_output_filepath = os.path.join(segmented_output_dir, filename)

        print(f"Processing file: {filepath}")
        try:
            # Iterate through the file line-by-line instead of reading all lines into memory
            with open(filepath, 'r', encoding='utf-8') as infile, \
                 open(clean_output_filepath, 'w', encoding='utf-8') as clean_out, \
                 open(segmented_output_filepath, 'w', encoding='utf-8') as seg_out:
                 
                for line in infile:
                    # Step 1: Remove English characters (A-Z, a-z)
                    text_no_english = re.sub(r'[a-zA-Z]', '', line)

                    # Step 2: Keep only Myanmar Unicode blocks and spaces
                    text_only_myanmar = re.sub(r'[^\u1000-\u109F\s]', '', text_no_english)
                    text_cleaned = text_only_myanmar.strip()

                    if not text_cleaned:
                        continue

                    # Step 3: Check Zawgyi probability and convert if necessary
                    zawgyi_prob = detector.get_zawgyi_probability(text_cleaned)
                    if zawgyi_prob > 0.95:
                        unicode_text = rabbit.zg2uni(text_cleaned)
                    else:
                        unicode_text = text_cleaned

                    # Step 3.5: Fix standard typing errors and numbers masquerading as characters
                    unicode_text = fix_typing_errors(unicode_text)

                    # Step 4: Normalize multiple spaces into a single space
                    final_clean_text = re.sub(r'\s+', ' ', unicode_text).strip()

                    if final_clean_text:
                        # Step 5: Save directly to Cleaned output file to save memory
                        clean_out.write(final_clean_text + '\n')
                        
                        # Step 6: Apply Word Segmentation and save directly to Segmented output file
                        segmented_text = apply_myword_segmentation(final_clean_text)
                        seg_out.write(segmented_text + '\n')

                        total_lines_processed += 1

                        # Progress reporting
                        if total_lines_processed % 1000 == 0:
                            print(f"Processed {total_lines_processed} lines...")

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    print(f"Processing complete!")
    print(f"Total lines processed: {total_lines_processed}")
    print(f"1. Cleaned text saved to: {clean_output_dir}")
    print(f"2. Segmented text saved to: {segmented_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Myanmar text files from a directory.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input text files.")
    parser.add_argument("-c", "--clean_output_dir", type=str, default="clean_corpus", help="Directory to save the cleaned output.")
    parser.add_argument("-s", "--segmented_output_dir", type=str, default="segmented_corpus", help="Directory to save the segmented output.")
    
    args = parser.parse_args()
    process_myanmar_directory(args.input_dir, args.clean_output_dir, args.segmented_output_dir)