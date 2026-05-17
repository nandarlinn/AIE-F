#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Myanmar Corpus Preprocessing and Syllable Tokenization Pipeline
Developed for assignment5

This script processes:
- TXT files and extensionless files (medical)
- CSV files (reading ONLY the 'text' column, skipping the 'label' column)

Steps:
1. Strip English characters
2. Convert Zawgyi to Unicode using the integrated Rabbit Converter
3. Apply Myanmar typing errors fixes
4. Strip English & Myanmar numbers/digits
5. Strip Myanmar and ASCII punctuation
6. Preserve only valid Myanmar Unicode alphabetical characters
7. Filter out garbled/corrupted lines
8. Segment text into syllables using Dr. Ye Kyaw Thu's oppaWord pattern
"""

import os
import re
import csv
import sys

# =====================================================================
# 1. Rabbit Zawgyi <-> Unicode Converter
# =====================================================================
from rabbit import Rabbit
from myword import SyllableTokenizer

# Try importing myanmartools.ZawgyiDetector
try:
    from myanmartools import ZawgyiDetector
    detector = ZawgyiDetector()
    has_myanmartools = True
    print("[INFO] myanmartools.ZawgyiDetector found! Using standard probabilistic detection.")
except ImportError:
    detector = None
    has_myanmartools = False
    print("[INFO] myanmartools.ZawgyiDetector not found. Using built-in regex-based Zawgyi fallback heuristic.")

def is_probably_zawgyi(text):
    # --- UNICODE SAFEGUARDS ---
    # 1. If it contains standard Unicode subscript indicator (U+1039) or great sa (U+103F), it is definitely Unicode.
    if "\u1039" in text or "\u103f" in text:
        return False
        
    # 2. Check if the line is already valid, well-formed Unicode by analyzing common patterns.
    # If there is a vowel U+1031 (ေ) that is NOT preceded by a consonant or medial, it is probably Zawgyi.
    has_misplaced_e = False
    if "\u1031" in text:
        if text.startswith("\u1031"):
            has_misplaced_e = True
        else:
            consonants_and_medials = set(
                [chr(c) for c in range(0x1000, 0x1021 + 1)] + 
                [chr(c) for c in range(0x103b, 0x103e + 1)] + 
                ["\u103f"]
            )
            for i in range(1, len(text)):
                if text[i] == "\u1031" and text[i-1] not in consonants_and_medials:
                    has_misplaced_e = True
                    break
                    
    # If there are medials (U+103B-U+103E) not preceded by a consonant or medial/subscript:
    has_misplaced_medial = False
    medials = set([chr(c) for c in range(0x103b, 0x103e + 1)])
    consonants_sub_medials = set(
        [chr(c) for c in range(0x1000, 0x1021 + 1)] + 
        ["\u1039"] + 
        [chr(c) for c in range(0x103b, 0x103e + 1)]
    )
    for m in medials:
        if m in text:
            if text.startswith(m):
                has_misplaced_medial = True
                break
            else:
                for i in range(1, len(text)):
                    if text[i] == m and text[i-1] not in consonants_sub_medials:
                        has_misplaced_medial = True
                        break
                if has_misplaced_medial:
                    break

    # Zawgyi-only subscript or medial rendering glyphs
    has_zawgyi_glyphs = bool(re.search(r'[\u1060-\u108f\u1090-\u1097]', text))

    # If it has misplaced E, misplaced medial, or Zawgyi-only glyphs, it is probably Zawgyi.
    # Otherwise, if it has no misplaced elements or Zawgyi glyphs, it is Unicode. We return False to avoid false positives!
    if not (has_misplaced_e or has_misplaced_medial or has_zawgyi_glyphs):
        return False

    # If the text does trigger one of the Zawgyi heuristics, use the standard ZawgyiDetector if available
    if has_myanmartools and detector is not None:
        try:
            score = detector.get_zawgyi_probability(text)
            return score > 0.95
        except Exception:
            pass

    return True


# =====================================================================
# 2. Garbled Text Detection & Typing Errors Fixes
# =====================================================================
INVALID_MEDIAL_YA = re.compile(r"[ငစဆဇဉညဋဌဍဎဏတထဒဓနရဝဟဠအ]ျ|သျ(?!ှ)")
INVALID_MEDIAL_HA = re.compile(r"[ကခဂဃစဆဇဈဋဌဍဎတထဒဓပဖဗဘသအ]ှ")
INVALID_ZHA = re.compile(r"ဈ(?![ေးာိီုူ])")

def is_garbled(text):
    if INVALID_MEDIAL_YA.search(text): return True
    if INVALID_MEDIAL_HA.search(text): return True
    if INVALID_ZHA.search(text): return True
    return False

def fix_typing_errors(text):
    text = text.replace("ဥ်", "ဉ်").replace("စဥ်", "စဉ်").replace("ဥာ", "ဉာ")
    text = re.sub(r'(?<=[\u1000-\u1021])၇|၇(?=[\u1000-\u1021])', 'ရ', text)
    text = re.sub(r'(?<=[\u1000-\u1021])၀|၀(?=[\u1000-\u1021])', 'ဝ', text)
    return text

# =====================================================================
# 3. myword Syllable Tokenization
# =====================================================================
syltok = SyllableTokenizer()

def myword_syllable_tokenize(text):
    try:
        tokens = syltok.tokenize(text)
        return " ".join([t.strip() for t in tokens if t.strip()])
    except Exception as e:
        return text

# =====================================================================
# 4. Cleaning Pipeline
# =====================================================================
def clean_and_tokenize(line):
    # 1. Remove English characters
    text = re.sub(r'[a-zA-Z]', '', line)
    
    # 2. Zawgyi to Unicode conversion
    if is_probably_zawgyi(text):
        text = Rabbit.zg2uni(text)
            
    # 3. Fix typing errors
    text = fix_typing_errors(text)
    
    # 4. Remove English & Myanmar digits
    text = re.sub(r'[0-9၀-၉]', '', text)
    
    # 5. Remove Burmese punctuation (၊, ။) and standard ASCII punctuation
    text = re.sub(r'[၊။!@#$%^&*()_+\-=\[\]{};\'\':"\\|,.<>\/?~`☃…“”’‘•]', ' ', text)
    
    # 6. Preserve ONLY valid Myanmar Unicode alphabetical characters & spaces
    text = re.sub(r'[^\u1000-\u1021\u1023-\u1027\u1029-\u103E\u103F\u1050-\u108F\s]', '', text)
    
    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return None
        
    # 8. Garbled/Corrupted string filter
    if is_garbled(text):
        return None
        
    # 9. Syllable segmentation using myword
    tokenized = myword_syllable_tokenize(text)
    return tokenized

# =====================================================================
# 5. Main Execution
# =====================================================================
def main():
    input_dir = "/home/elio/Downloads/assignment5/raw_corpus"
    output_dir = "/home/elio/Downloads/assignment5/cleaned_corpus"
    
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    print(f"\nScanning assignment5 directory: '{input_dir}'")
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    
    total_files_processed = 0
    total_raw_lines = 0
    total_clean_lines = 0
    total_garbled_removed = 0
    
    output_filenames = ["clean_syllable_corpus.txt", "test_medical.txt", "test_news.txt", "test_literature.txt", "train_cleaned_corpus.txt"]
    files = [f for f in files if f not in output_filenames and f != "preprocess_corpus.py"]
    
    print(f"Found {len(files)} files to process: {files}\n")
    
    test_medical_lines = []
    test_news_lines = []
    test_literature_lines = []
    train_lines = []
    
    test_medical_syls = 0
    test_news_syls = 0
    test_literature_syls = 0
    
    def add_clean_line(clean_str, fn):
        nonlocal test_medical_syls, test_news_syls, test_literature_syls
        
        words = clean_str.strip().split()
        if not words:
            return
            
        if fn == "medical.txt":
            if test_medical_syls < 300:
                needed = 300 - test_medical_syls
                if len(words) <= needed:
                    test_medical_lines.append(clean_str + "\n")
                    test_medical_syls += len(words)
                else:
                    test_part = " ".join(words[:needed])
                    test_medical_lines.append(test_part + "\n")
                    test_medical_syls = 300
                    train_part = " ".join(words[needed:])
                    if train_part:
                        train_lines.append(train_part + "\n")
            else:
                train_lines.append(clean_str + "\n")
        elif fn == "myanmar-news-test-data-overview.csv":
            if test_news_syls < 300:
                needed = 300 - test_news_syls
                if len(words) <= needed:
                    test_news_lines.append(clean_str + "\n")
                    test_news_syls += len(words)
                else:
                    test_part = " ".join(words[:needed])
                    test_news_lines.append(test_part + "\n")
                    test_news_syls = 300
                    train_part = " ".join(words[needed:])
                    if train_part:
                        train_lines.append(train_part + "\n")
            else:
                train_lines.append(clean_str + "\n")
        elif fn == "corpus-3_2.txt":
            if test_literature_syls < 300:
                needed = 300 - test_literature_syls
                if len(words) <= needed:
                    test_literature_lines.append(clean_str + "\n")
                    test_literature_syls += len(words)
                else:
                    test_part = " ".join(words[:needed])
                    test_literature_lines.append(test_part + "\n")
                    test_literature_syls = 300
                    train_part = " ".join(words[needed:])
                    if train_part:
                        train_lines.append(train_part + "\n")
            else:
                train_lines.append(clean_str + "\n")
        else:
            train_lines.append(clean_str + "\n")
            
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        file_raw_lines = 0
        file_clean_lines = 0
        file_garbled_removed = 0
        
        print(f"-> Processing '{filename}'...")
        
        # Check if the file is a CSV
        if filename.endswith(".csv"):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as csvfile:
                    reader = csv.DictReader(csvfile)
                    fieldnames = reader.fieldnames
                    
                    if fieldnames and 'text' in fieldnames:
                        for row in reader:
                            text_content = row['text']
                            if text_content:
                                file_raw_lines += 1
                                clean_line = clean_and_tokenize(text_content)
                                if clean_line:
                                    add_clean_line(clean_line, filename)
                                    file_clean_lines += 1
                                else:
                                    if text_content.strip() and is_garbled(text_content):
                                        file_garbled_removed += 1
                    else:
                        # Fallback: if 'text' fieldname not explicitly detected, assume index 0 is text
                        csvfile.seek(0)
                        raw_reader = csv.reader(csvfile)
                        # Skip header
                        next(raw_reader, None)
                        for row in raw_reader:
                            if row:
                                text_content = row[0]
                                file_raw_lines += 1
                                clean_line = clean_and_tokenize(text_content)
                                if clean_line:
                                    add_clean_line(clean_line, filename)
                                    file_clean_lines += 1
                                else:
                                    if text_content.strip() and is_garbled(text_content):
                                        file_garbled_removed += 1
            except Exception as e:
                print(f"   [ERROR] Failed to read CSV '{filename}': {e}")
        else:
            # Assume it is a text or extensionless file (like 'medical')
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as txtfile:
                    for line in txtfile:
                        line_str = line.strip()
                        if line_str:
                            file_raw_lines += 1
                            clean_line = clean_and_tokenize(line_str)
                            if clean_line:
                                add_clean_line(clean_line, filename)
                                file_clean_lines += 1
                            else:
                                if is_garbled(line_str):
                                    file_garbled_removed += 1
            except Exception as e:
                print(f"   [ERROR] Failed to read text file '{filename}': {e}")
                
        print(f"   Done. Raw lines: {file_raw_lines} | Clean & Tokenized: {file_clean_lines} | Garbled discarded: {file_garbled_removed}")
        
        total_files_processed += 1
        total_raw_lines += file_raw_lines
        total_clean_lines += file_clean_lines
        total_garbled_removed += file_garbled_removed

    # Write output files
    os.makedirs(output_dir, exist_ok=True)
    
    file_mappings = [
        ("test_medical.txt", test_medical_lines),
        ("test_news.txt", test_news_lines),
        ("test_literature.txt", test_literature_lines),
        ("train_cleaned_corpus.txt", train_lines)
    ]
    
    print("\nWriting output files...")
    for out_name, lines in file_mappings:
        out_path = os.path.join(output_dir, out_name)
        try:
            with open(out_path, 'w', encoding='utf-8') as outfile:
                outfile.writelines(lines)
            print(f"[SUCCESS] Wrote {len(lines)} lines to '{out_path}'")
        except Exception as e:
            print(f"[ERROR] Failed to write to '{out_path}': {e}")
            sys.exit(1)
        
    print("\n==================================================")
    print("               PREPROCESSING STATS                ")
    print("==================================================")
    print(f"Total files processed:          {total_files_processed}")
    print(f"Total raw text/rows read:       {total_raw_lines}")
    print(f"Total clean lines output:       {total_clean_lines}")
    print(f"Total garbled lines removed:    {total_garbled_removed}")
    print("--------------------------------------------------")
    print(f"test_medical.txt:         {len(test_medical_lines)} lines, {test_medical_syls} syllables")
    print(f"test_news.txt:            {len(test_news_lines)} lines, {test_news_syls} syllables")
    print(f"test_literature.txt:      {len(test_literature_lines)} lines, {test_literature_syls} syllables")
    print(f"train_cleaned_corpus.txt: {len(train_lines)} lines")
    print("==================================================\n")

if __name__ == "__main__":
    main()
