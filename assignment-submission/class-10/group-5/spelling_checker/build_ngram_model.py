import os
import json
import argparse
from collections import Counter
import re

def build_ngram_model(input_dir, output_file, min_freq=2):
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return

    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()

    filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
    total_files = len(filepaths)
    
    if total_files == 0:
        print(f"No .txt files found in '{input_dir}'.")
        return

    print(f"Processing {total_files} files from '{input_dir}'...")
    
    lines_processed = 0

    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove extra spaces and punctuation. Segmented corpus should already be space-separated words
                words = [w for w in line.strip().split() if not re.match(r'^[။၊\s\.]+$', w)]
                
                if not words:
                    continue
                
                unigrams.update(words)
                
                # Generate Bigrams
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    bigrams[bigram] += 1
                
                # Generate Trigrams
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    trigrams[trigram] += 1
                
                lines_processed += 1
                if lines_processed % 10000 == 0:
                    print(f"Processed {lines_processed} lines...")

    print("Filtering low frequency N-grams...")
    # Filter by minimum frequency to save memory/storage
    filtered_unigrams = {k: v for k, v in unigrams.items() if v >= min_freq}
    filtered_bigrams = {k: v for k, v in bigrams.items() if v >= min_freq}
    filtered_trigrams = {k: v for k, v in trigrams.items() if v >= min_freq}

    model = {
        "unigrams": filtered_unigrams,
        "bigrams": filtered_bigrams,
        "trigrams": filtered_trigrams
    }

    print(f"Total Unique Unigrams: {len(filtered_unigrams)} (Frequency >= {min_freq})")
    print(f"Total Unique Bigrams: {len(filtered_bigrams)} (Frequency >= {min_freq})")
    print(f"Total Unique Trigrams: {len(filtered_trigrams)} (Frequency >= {min_freq})")

    print(f"Saving model to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # separators=(',', ':') removes whitespace to make file smaller
        json.dump(model, f, ensure_ascii=False, separators=(',', ':'))
        
    print("Model successfully built and saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build N-gram model from text corpus.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory containing text files.")
    parser.add_argument("-o", "--output_file", type=str, default="ngram_model.json", help="Output JSON file for the model.")
    parser.add_argument("-m", "--min_freq", type=int, default=2, help="Minimum frequency threshold to keep an N-gram.")
    
    args = parser.parse_args()
    build_ngram_model(args.input_dir, args.output_file, args.min_freq)
