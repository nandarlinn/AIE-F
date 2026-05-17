#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Burmese Linguistics & NLP Data Engineering Pipeline
Task: Build, compile, and evaluate an n-gram Language Model utilizing KenLM 
for a Myanmar syllable-level tokenization dataset.
"""

import os
import sys
import subprocess
import shutil
import json
import matplotlib.pyplot as plt

# Production Constants & Default Directory Structures
DEFAULT_TRAIN_CORPUS = "cleaned_corpus/train_cleaned_corpus.txt"
DEFAULT_TEST_DIR = "cleaned_corpus"
DEFAULT_OUTPUT_DIR = "cleaned_corpus"

def find_kenlm_binaries():
    """
    Dynamically locate lmplz and build_binary executables in PATH or 
    relative project folder structures to ensure cross-system resilience.
    """
    lmplz_path = shutil.which("lmplz")
    build_binary_path = shutil.which("build_binary")
    
    if lmplz_path and build_binary_path:
        return lmplz_path, build_binary_path
        
    # Checking relative local build output folders
    local_bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kenlm", "build", "bin")
    local_lmplz = os.path.join(local_bin_dir, "lmplz")
    local_build_binary = os.path.join(local_bin_dir, "build_binary")
    
    if os.path.exists(local_lmplz) and os.path.exists(local_build_binary):
        return local_lmplz, local_build_binary
        
    return None, None

def train_language_model(train_corpus, lmplz_path, build_binary_path, arpa_path, bin_path, order=5):
    """
    Task 1: Model Estimation
    Compiles a high-quality n-gram language model utilizing KenLM tools.
    - Estimates n-gram structures using Kneser-Ney smoothing via lmplz.
    - Compiles ARPA output into a high-performance binary file format via build_binary.
    """
    if not os.path.exists(lmplz_path):
        raise FileNotFoundError(f"lmplz binary not found at '{lmplz_path}'")
    if not os.path.exists(build_binary_path):
        raise FileNotFoundError(f"build_binary binary not found at '{build_binary_path}'")
        
    print(f"[INFO] Estimating {order}-gram language model from '{train_corpus}'...")
    
    # Stage 1: Estimate ARPA parameters using subprocess streaming
    try:
        with open(train_corpus, 'r', encoding='utf-8') as infile, \
             open(arpa_path, 'w', encoding='utf-8') as outfile:
             
            process = subprocess.Popen(
                [lmplz_path, "-o", str(order), "--discount_fallback"],
                stdin=infile,
                stdout=outfile,
                stderr=subprocess.PIPE,
                text=True
            )
            _, stderr = process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"lmplz parameter estimation failed (code {process.returncode}): {stderr}")
                
        print(f"[SUCCESS] Estimated {order}-gram ARPA parameters: '{arpa_path}'")
    except Exception as e:
        raise RuntimeError(f"Failed during ARPA estimation phase: {e}")
        
    # Stage 2: Compile ARPA model into binary database format for sub-millisecond querying
    print(f"[INFO] Compiling ARPA parameters to binary trie database...")
    try:
        process = subprocess.run(
            [build_binary_path, arpa_path, bin_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SUCCESS] Compiled language model binary: '{bin_path}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"build_binary compilation failed (code {e.returncode}): {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Failed during model serialization/binarization phase: {e}")

def evaluate_perplexity(model_bin_path, test_files):
    """
    Task 2: Perplexity Evaluation
    Computes exact perplexity (PPL) metrics across the domain files.
    - Normalizes each domain dataset to exactly 300 Myanmar syllable-level tokens.
    - Computes mathematical probability scores using the native kenlm Python wrapper.
    """
    print(f"[INFO] Loading serialized KenLM binary from '{model_bin_path}'...")
    try:
        import kenlm
    except ImportError:
        print("[ERROR] Native 'kenlm' Python wrapper module is not installed in the virtual environment.")
        print("[SUGGESTION] Build and install it via: pip install https://github.com/kpu/kenlm/archive/master.zip")
        sys.exit(1)
        
    try:
        model = kenlm.Model(model_bin_path)
    except Exception as e:
        print(f"[ERROR] Failed to load binary model database: {e}")
        sys.exit(1)
        
    ppl_results = {}
    
    for domain, filepath in test_files.items():
        if not os.path.exists(filepath):
            print(f"[WARNING] Corpus data for domain '{domain}' not found at '{filepath}'. Skipping.")
            continue
            
        print(f"[INFO] Evaluating perplexity metrics for '{domain}' domain...")
        
        # Ingest and clean tokens, truncating to exactly 300 syllables
        tokens = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    words = line.strip().split()
                    tokens.extend(words)
                    if len(tokens) >= 300:
                        break
        except Exception as e:
            print(f"[ERROR] Failed to ingest domain dataset '{filepath}': {e}")
            continue
            
        tokens_300 = tokens[:300]
        if len(tokens_300) < 300:
            print(f"[WARNING] Ingested token count ({len(tokens_300)}) is less than normalized 300 limit for '{domain}'.")
            
        # Join tokens using single-space formatting bounds (strict Myanmar Unicode rules)
        eval_text = " ".join(tokens_300)
        
        try:
            # Perplexity (PPL) = 10^(-score / N), where score = log10 prob of text, N = tokens count
            ppl = model.perplexity(eval_text)
            ppl_results[domain] = {
                "perplexity": float(ppl),
                "tokens_evaluated": len(tokens_300)
            }
            print(f"[RESULT] Domain: {domain:12} | PPL: {ppl:8.4f} | Syllable Count: {len(tokens_300)}")
        except Exception as e:
            print(f"[ERROR] Perplexity calculation failed for '{domain}' domain: {e}")
            
    return ppl_results

def save_results_and_plot(ppl_results, output_dir, order=5):
    """
    Task 3: Visualization & Metadata Logging
    - Plots perplexity levels across the different evaluation sets using a clean, modern aesthetic.
    - Saves execution metadata, computed PPL scores, and hyper-parameters as a structured JSON payload.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Log structured evaluation results
    metadata = {
        "model_parameters": {
            "order": order,
            "smoothing": "Kneser-Ney fallback",
            "format": "KenLM Trie Binary"
        },
        "evaluation_metrics": ppl_results
    }
    
    json_path = os.path.join(output_dir, "evaluation_metadata.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(metadata, jf, indent=4, ensure_ascii=False)
        print(f"[SUCCESS] Execution metadata and PPL parameters logged to '{json_path}'")
    except Exception as e:
        print(f"[ERROR] JSON logging pipeline failed: {e}")
        
    # Prepare data for plotting
    domains = list(ppl_results.keys())
    ppl_values = [metrics["perplexity"] for metrics in ppl_results.values()]
    
    if not domains:
        print("[WARNING] Skipping chart visualization: zero successfully evaluated domains.")
        return
        
    # Styling and visualization configurations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # Modern Glassmorphic Color Palette
    bars_colors = ['#3A86F0', '#00F5D4', '#FFBE0B'] # Sleek Royal Blue, Mint, Bright Yellow
    
    bars = ax.bar(domains, ppl_values, color=bars_colors[:len(domains)], width=0.45, edgecolor='none', zorder=3)
    
    # Title & label formatting
    ax.set_title(f"Domain Hardness Mapping (Myanmar {order}-gram Syllable Language Model)", fontsize=13, fontweight='bold', pad=18, color='#1A252C')
    ax.set_ylabel("Perplexity (PPL) - Log Scale (Lower = More Predictable)", fontsize=10.5, labelpad=12, color='#2C3E50')
    ax.set_xlabel("Evaluation Datasets (Normalized 300 Syllables)", fontsize=10.5, labelpad=12, color='#2C3E50')
    
    # Annotate computed PPL labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'PPL: {height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1A252C')
                    
    # Refine grid design parameters
    ax.grid(axis='y', linestyle='--', alpha=0.65, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"ppl_comparison_{order}gram.png")
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"[SUCCESS] Compiled Perplexity domain hardness comparison saved to '{plot_path}'")
    except Exception as e:
        print(f"[ERROR] Visualization pipeline failed: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Myanmar Syllable-level KenLM Pipeline")
    parser.add_argument("--train-corpus", default=DEFAULT_TRAIN_CORPUS, help="Path to clean training corpus file")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Path to directory containing test_*.txt files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to output results directory")
    parser.add_argument("--lmplz", help="Absolute path to KenLM lmplz binary")
    parser.add_argument("--build-binary", help="Absolute path to KenLM build_binary binary")
    parser.add_argument("--order", type=int, default=3, choices=[3, 4, 5], help="N-gram order for the KenLM model (default: 3)")
    
    args = parser.parse_args()
    
    # Locate executables
    lmplz_bin = args.lmplz
    build_bin = args.build_binary
    
    if not lmplz_bin or not build_bin:
        found_lmplz, found_build = find_kenlm_binaries()
        if not lmplz_bin:
            lmplz_bin = found_lmplz
        if not build_bin:
            build_bin = found_build
            
    if not lmplz_bin or not build_bin:
        print("[ERROR] KenLM core binaries ('lmplz' and 'build_binary') could not be automatically located in PATH.")
        print("[SUGGESTION] Specify absolute binary paths, e.g.:")
        print("  python train_evaluate_lm.py --lmplz /path/to/kenlm/build/bin/lmplz --build-binary /path/to/kenlm/build/bin/build_binary")
        sys.exit(1)
        
    arpa_file = os.path.join(args.output_dir, f"myanmar_{args.order}gram.arpa")
    bin_file = os.path.join(args.output_dir, f"myanmar_{args.order}gram.bin")
    
    if not os.path.exists(args.train_corpus):
        print(f"[ERROR] Training corpus file not found at: '{args.train_corpus}'")
        sys.exit(1)
        
    # 1. Model Estimation
    try:
        train_language_model(args.train_corpus, lmplz_bin, build_bin, arpa_file, bin_file, order=args.order)
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        sys.exit(1)
        
    # 2. Perplexity Evaluation
    test_files = {
        "News": os.path.join(args.test_dir, "test_news.txt"),
        "Literature": os.path.join(args.test_dir, "test_literature.txt"),
        "Medical": os.path.join(args.test_dir, "test_medical.txt")
    }
    
    ppl_results = evaluate_perplexity(bin_file, test_files)
    
    # 3. Save logs and render visualization comparison
    save_results_and_plot(ppl_results, args.output_dir, order=args.order)

if __name__ == "__main__":
    main()
