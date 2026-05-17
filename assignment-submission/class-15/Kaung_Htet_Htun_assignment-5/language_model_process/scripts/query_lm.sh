#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="${1:-word}"
SENTENCE="${2:-မြန်မာ နိုင်ငံ တွင် မည် သည့် ကုမ္ပဏီ အမျိုးအစား များ အား မှတ်ပုံတင် ရ ပါ သနည်း}"
ORDER="${ORDER:-5}"
QUERY="${QUERY:-/home/phantom/mosesdecoder/bin/query}"
MODEL="language_model_process/models/binary/${MODEL_TYPE}_${ORDER}gram.binary"

if [ ! -x "$QUERY" ]; then
  echo "query not found or not executable: $QUERY" >&2
  exit 1
fi

if [ ! -f "$MODEL" ]; then
  echo "Model not found: $MODEL" >&2
  echo "Run: bash language_model_process/scripts/train_kenlm.sh" >&2
  exit 1
fi

printf '%s\n' "$SENTENCE" | "$QUERY" "$MODEL"
