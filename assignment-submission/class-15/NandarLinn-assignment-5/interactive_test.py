#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Myanmar Language Model Real-time Interactive Testing Utility
Uses the compiled KenLM binary model and preprocessor to clean and evaluate text on-the-fly.
"""

import os
import sys

# ANSI Escape Codes for beautiful terminal output styling
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "cleaned_corpus", "myanmar_5gram.bin")
    
    if not os.path.exists(model_path):
        print(f"{RED}[ERROR] Serialized binary model not found at '{model_path}'. Please run train_evaluate_lm.py first.{RESET}")
        sys.exit(1)
        
    print(f"{BOLD}{BLUE}==================================================={RESET}")
    print(f"{BOLD}{GREEN}      Myanmar Language Model Interactive Tester    {RESET}")
    print(f"{BOLD}{BLUE}==================================================={RESET}")
    print(f"[INFO] Loading KenLM model database from '{model_path}'...")
    
    try:
        import kenlm
    except ImportError:
        print(f"{RED}[ERROR] Native 'kenlm' Python module is not installed in the active environment.{RESET}")
        sys.exit(1)
        
    try:
        model = kenlm.Model(model_path)
        print(f"{GREEN}[SUCCESS] Language model loaded successfully.{RESET}")
    except Exception as e:
        print(f"{RED}[ERROR] Failed to load binary model database: {e}{RESET}")
        sys.exit(1)
        
    # Dynamically import preprocessor from local workspace
    try:
        # Append current directory to ensure import resolution
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from preprocess_corpus import clean_and_tokenize
        preprocessor_loaded = True
        print(f"{GREEN}[SUCCESS] Successfully integrated Myanmar preprocessor (clean_and_tokenize).{RESET}")
    except ImportError:
        preprocessor_loaded = False
        print(f"{YELLOW}[WARNING] Preprocessor script (preprocess_corpus.py) not found in path. Inputs must be manually syllable-spaced.{RESET}")

    print(f"\n{BOLD}{CYAN}Type any Myanmar sentence to check its probability and perplexity (PPL) in real-time.{RESET}")
    print(f"Type {BOLD}{RED}'exit'{RESET} or {BOLD}{RED}'quit'{RESET} to stop testing.\n")
    
    while True:
        try:
            # Read input from terminal
            user_input = input(f"{BOLD}Enter sentence >> {RESET}").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{GREEN}Exiting interactive mode. Thank you!{RESET}")
                break
                
            # If preprocessor is available, clean and syllable-tokenize the input on-the-fly
            if preprocessor_loaded:
                processed_text = clean_and_tokenize(user_input)
                if not processed_text:
                    print(f"{RED}[ERROR] Input text discarded (non-Myanmar Unicode or invalid typing sequences).{RESET}\n")
                    continue
            else:
                processed_text = user_input
                
            # Perform log10 score and perplexity calculations
            score = model.score(processed_text, bos=True, eos=True)
            ppl = model.perplexity(processed_text)
            
            # Formulate qualitative domain fit status based on calculated PPL
            if ppl < 20:
                status = f"{GREEN}Excellent (Highly Predictable / In-Domain){RESET}"
            elif ppl < 40:
                status = f"{CYAN}Good (Natural / Common Language){RESET}"
            elif ppl < 70:
                status = f"{YELLOW}Fair (Uncommon / Weak Grammar Structure){RESET}"
            else:
                status = f"{RED}Poor (Highly Unnatural / Out-of-Domain){RESET}"
                
            # Render styled results block
            print(f"\n{BOLD}{BLUE}--- REAL-TIME EVALUATION RESULTS ---{RESET}")
            print(f"  {BOLD}Raw Input:{RESET}         {user_input}")
            if preprocessor_loaded:
                print(f"  {BOLD}Syllable Segmented:{RESET} {processed_text}")
            print(f"  {BOLD}Log10 Prob Score:{RESET}   {score:.4f}")
            print(f"  {BOLD}Perplexity (PPL):{RESET}   {BOLD}{ppl:.4f}{RESET}")
            print(f"  {BOLD}Sentence Status:{RESET}    {status}")
            print(f"{BLUE}------------------------------------{RESET}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{GREEN}Exiting interactive mode. Thank you!{RESET}")
            break
        except Exception as e:
            print(f"{RED}[ERROR] Evaluation pipeline failed: {e}{RESET}\n")

if __name__ == "__main__":
    main()
