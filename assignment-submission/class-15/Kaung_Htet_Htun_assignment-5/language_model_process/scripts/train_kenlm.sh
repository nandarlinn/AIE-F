#!/usr/bin/env bash
set -euo pipefail

ORDER="${ORDER:-5}"
KENLM_BIN="${KENLM_BIN:-/home/phantom/mosesdecoder/bin}"
LMPLZ="${LMPLZ:-$KENLM_BIN/lmplz}"
BUILD_BINARY="${BUILD_BINARY:-$KENLM_BIN/build_binary}"

mkdir -p language_model_process/corpus \
         language_model_process/models/arpa \
         language_model_process/models/binary \
         language_model_process/reports

if [ ! -x "$LMPLZ" ]; then
  echo "lmplz not found or not executable: $LMPLZ" >&2
  exit 1
fi

if [ ! -x "$BUILD_BINARY" ]; then
  echo "build_binary not found or not executable: $BUILD_BINARY" >&2
  exit 1
fi

perl language_model_process/scripts/extract_lm_corpus.pl \
  --input-dir tokenization_process/syllable_based \
  --output language_model_process/corpus/syllable.txt \
  > language_model_process/reports/corpus_syllable_summary.tsv

perl language_model_process/scripts/extract_lm_corpus.pl \
  --input-dir tokenization_process/word_based \
  --output language_model_process/corpus/word.txt \
  > language_model_process/reports/corpus_word_summary.tsv

"$LMPLZ" \
  -o "$ORDER" \
  --text language_model_process/corpus/syllable.txt \
  --arpa language_model_process/models/arpa/syllable_${ORDER}gram.arpa \
  --discount_fallback \
  > language_model_process/reports/train_syllable_${ORDER}gram.log 2>&1

"$BUILD_BINARY" \
  language_model_process/models/arpa/syllable_${ORDER}gram.arpa \
  language_model_process/models/binary/syllable_${ORDER}gram.binary \
  > language_model_process/reports/build_syllable_${ORDER}gram.log 2>&1

"$LMPLZ" \
  -o "$ORDER" \
  --text language_model_process/corpus/word.txt \
  --arpa language_model_process/models/arpa/word_${ORDER}gram.arpa \
  --discount_fallback \
  > language_model_process/reports/train_word_${ORDER}gram.log 2>&1

"$BUILD_BINARY" \
  language_model_process/models/arpa/word_${ORDER}gram.arpa \
  language_model_process/models/binary/word_${ORDER}gram.binary \
  > language_model_process/reports/build_word_${ORDER}gram.log 2>&1

{
  echo -e "model\torder\tcorpus_lines\tarpa\tbinary"
  echo -e "syllable\t$ORDER\t$(wc -l < language_model_process/corpus/syllable.txt)\tlanguage_model_process/models/arpa/syllable_${ORDER}gram.arpa\tlanguage_model_process/models/binary/syllable_${ORDER}gram.binary"
  echo -e "word\t$ORDER\t$(wc -l < language_model_process/corpus/word.txt)\tlanguage_model_process/models/arpa/word_${ORDER}gram.arpa\tlanguage_model_process/models/binary/word_${ORDER}gram.binary"
} > language_model_process/reports/lm_training_summary.tsv

echo "KenLM training finished. Summary: language_model_process/reports/lm_training_summary.tsv"
