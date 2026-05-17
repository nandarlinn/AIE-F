#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="${1:-tokenization_process/tools}"
mkdir -p "$TOOLS_DIR"

if [ ! -d "$TOOLS_DIR/sylbreak/.git" ]; then
  git clone https://github.com/ye-kyaw-thu/sylbreak.git "$TOOLS_DIR/sylbreak"
fi

if [ ! -d "$TOOLS_DIR/oppaWord/.git" ]; then
  git clone https://github.com/ye-kyaw-thu/oppaWord.git "$TOOLS_DIR/oppaWord"
fi

echo "Tokenizer tools are under $TOOLS_DIR"
