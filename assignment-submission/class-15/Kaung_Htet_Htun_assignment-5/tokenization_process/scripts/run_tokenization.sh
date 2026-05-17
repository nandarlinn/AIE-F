#!/usr/bin/env bash
set -euo pipefail

perl tokenization_process/scripts/tokenize_cleaned_csv.pl \
  --mode syllable \
  --output-dir tokenization_process/syllable_based

perl tokenization_process/scripts/tokenize_cleaned_csv.pl \
  --mode word \
  --output-dir tokenization_process/word_based

echo "Tokenization finished."
