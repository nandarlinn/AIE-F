"""
train_g2p_transformer_dropout.py
=================================
Experiment 2: Dropout ONLY
  --lr-warmup 0             ← unchanged (baseline)
  --transformer-dropout 0.1 ← CHANGED  (baseline was 0.3)

Kaggle kernel: Grapheme-to-Phoneme NMT using Marian Transformer
Task:          Myanmar grapheme → Phoneme  (my → ph)
Architecture:  Transformer (--type transformer)

Dataset sources (kernel-metadata.json):
  - theinkyawlwin/marian-nmt   → Marian v1.12 binaries
  - theinkyawlwin/g2p-par      → Parallel g2p data (train/dev/test)

GPU:  2x Tesla T4 (15 GB each) — uses both via --devices 0 1 --sync-sgd
"""

import subprocess
import sys
import os
import stat
import shutil
import time as _time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    """Run a shell command, stream stdout/stderr, raise on failure."""
    print(f"\n>>> {cmd}\n{'='*70}")
    subprocess.run(cmd, shell=True, check=True,
                   stdout=sys.stdout, stderr=sys.stderr, **kwargs)


def section(title):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"{'#'*70}\n")


def resolve_path(candidates: list, label: str) -> str:
    """Return the first existing path from candidates, or raise with a clear error."""
    for p in candidates:
        if Path(p).exists():
            print(f"[ok] {label}: {p}")
            return p
    raise FileNotFoundError(
        f"[ERROR] {label} not found. Tried:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )


# ---------------------------------------------------------------------------
# Paths — auto-discover classic vs Kaggle-v2 input layout
# ---------------------------------------------------------------------------

# Marian binaries (dataset: theinkyawlwin/marian-nmt)
BINS_DIR = resolve_path([
    "/kaggle/input/marian-nmt/marian-bins",                              # classic
    "/kaggle/input/datasets/theinkyawlwin/marian-nmt/marian-bins",       # v2
], "marian-bins dir")

MARIAN         = os.path.join(BINS_DIR, "marian")
MARIAN_VOCAB   = os.path.join(BINS_DIR, "marian-vocab")
MARIAN_DECODER = os.path.join(BINS_DIR, "marian-decoder")

# Parallel data (dataset: theinkyawlwin/g2p-par)
DATA_DIR = resolve_path([
    "/kaggle/input/g2p-par/g2p-par",                                     # folder-upload layout
    "/kaggle/input/g2p-par",                                             # flat layout
    "/kaggle/input/datasets/theinkyawlwin/g2p-par/g2p-par",             # v2 + folder
    "/kaggle/input/datasets/theinkyawlwin/g2p-par",                     # v2 flat
], "g2p-par data dir")

# Output (persisted as kernel output)
WORKING_DIR  = "/kaggle/working"
LOCAL_BINS   = os.path.join(WORKING_DIR, "bins")
PREPROC_DIR  = os.path.join(WORKING_DIR, "preprocessing")
VOCAB_DIR    = os.path.join(WORKING_DIR, "vocab")
MODEL_DIR    = os.path.join(WORKING_DIR, "model.transformer.myph.dropout")

# Task: Grapheme → Phoneme
SRC = "my"   # source: Myanmar grapheme
TGT = "ph"   # target: phoneme

# ---------------------------------------------------------------------------
# Experiment hyperparameters
# ---------------------------------------------------------------------------
LR_WARMUP           = 0     # ← unchanged (baseline)
TRANSFORMER_DROPOUT = 0.1   # ← CHANGED from 0.3

print(f"\n[config] Experiment : DROPOUT ONLY")
print(f"[config] lr-warmup            : {LR_WARMUP}  (baseline=0, unchanged)")
print(f"[config] transformer-dropout  : {TRANSFORMER_DROPOUT}  (baseline=0.3)")


# ---------------------------------------------------------------------------
# Step 1: Copy binaries to writable dir and chmod +x
# ---------------------------------------------------------------------------

section("Step 1: Copy Marian binaries to /kaggle/working/bins/ and make executable")

os.makedirs(LOCAL_BINS, exist_ok=True)

for src_bin in ["marian", "marian-vocab", "marian-decoder"]:
    src_path = os.path.join(BINS_DIR, src_bin)
    dst_path = os.path.join(LOCAL_BINS, src_bin)
    shutil.copy2(src_path, dst_path)
    os.chmod(dst_path, 0o755)
    print(f"[ok] copied + chmod +x  {src_bin}  →  {dst_path}")

# Update binary paths to the writable copies
MARIAN         = os.path.join(LOCAL_BINS, "marian")
MARIAN_VOCAB   = os.path.join(LOCAL_BINS, "marian-vocab")
MARIAN_DECODER = os.path.join(LOCAL_BINS, "marian-decoder")

run(f"{MARIAN} --version")


# ---------------------------------------------------------------------------
# Step 2: Build vocabularies
# ---------------------------------------------------------------------------

section("Step 2: Build vocabularies  (marian-vocab)")

os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(VOCAB_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# Concatenate train + dev for richer vocabulary coverage
run(f"cat {DATA_DIR}/train.{SRC} {DATA_DIR}/dev.{SRC} > {PREPROC_DIR}/train-dev.{SRC}")
run(f"cat {DATA_DIR}/train.{TGT} {DATA_DIR}/dev.{TGT} > {PREPROC_DIR}/train-dev.{TGT}")

run(f"wc -l {PREPROC_DIR}/train-dev.{SRC} {PREPROC_DIR}/train-dev.{TGT}")

# Build vocabularies
run(f"{MARIAN_VOCAB} < {PREPROC_DIR}/train-dev.{SRC} > {VOCAB_DIR}/vocab.{SRC}.yml")
run(f"{MARIAN_VOCAB} < {PREPROC_DIR}/train-dev.{TGT} > {VOCAB_DIR}/vocab.{TGT}.yml")

run(f"wc -l {VOCAB_DIR}/vocab.{SRC}.yml {VOCAB_DIR}/vocab.{TGT}.yml")

print(f"\n--- Source vocab (Myanmar grapheme) — first 5 entries ---")
run(f"head -n 5 {VOCAB_DIR}/vocab.{SRC}.yml")
print(f"\n--- Target vocab (phoneme) — first 5 entries ---")
run(f"head -n 5 {VOCAB_DIR}/vocab.{TGT}.yml")


# ---------------------------------------------------------------------------
# Step 3: Dump config then train (Transformer)
# ---------------------------------------------------------------------------

section(f"Step 3: Train Transformer G2P model  (my → ph)  [WARMUP={LR_WARMUP}, DROPOUT={TRANSFORMER_DROPOUT}]")

CONFIG_FILE = os.path.join(MODEL_DIR, f"{SRC}-{TGT}.config.yml")
TRAIN_LOG   = os.path.join(MODEL_DIR, "train.log")
VALID_LOG   = os.path.join(MODEL_DIR, "valid.log")
VALID_OUT   = os.path.join(MODEL_DIR, f"valid.{SRC}-{TGT}.output")

marian_args = " ".join([
    f"--model {MODEL_DIR}/model.npz",
    f"--type transformer",
    f"--train-sets {DATA_DIR}/train.{SRC} {DATA_DIR}/train.{TGT}",
    f"--max-length 200",
    f"--vocabs {VOCAB_DIR}/vocab.{SRC}.yml {VOCAB_DIR}/vocab.{TGT}.yml",
    f"--mini-batch-fit -w 1000 --maxi-batch 100",
    f"--early-stopping 10",
    f"--valid-freq 5000 --save-freq 5000 --disp-freq 500",
    f"--valid-metrics cross-entropy perplexity bleu",
    f"--valid-sets {DATA_DIR}/dev.{SRC} {DATA_DIR}/dev.{TGT}",
    f"--valid-translation-output {VALID_OUT} --quiet-translation",
    f"--valid-mini-batch 64",
    f"--beam-size 6 --normalize 0.6",
    f"--log {TRAIN_LOG} --valid-log {VALID_LOG}",
    f"--enc-depth 2 --dec-depth 2",
    f"--transformer-heads 8",
    f"--transformer-postprocess-emb d",
    f"--transformer-postprocess dan",
    f"--transformer-dropout {TRANSFORMER_DROPOUT} --label-smoothing 0.1",
    f"--learn-rate 0.0003 --lr-warmup {LR_WARMUP} --lr-decay-inv-sqrt 16000 --lr-report",
    f"--clip-norm 5",
    f"--tied-embeddings",
    f"--devices 0 1 --sync-sgd --seed 1111",
    f"--exponential-smoothing",
])

# Dump config YAML (dry-run — no actual training yet)
run(f"{MARIAN} {marian_args} --dump-config > {CONFIG_FILE}")

print("\n--- config.yml ---")
run(f"cat {CONFIG_FILE}")

# Train using config file
print("[info] Starting training...")
_t0 = _time.time()
run(f"{MARIAN} -c {CONFIG_FILE} 2>&1 | tee {MODEL_DIR}/transformer-{SRC}-{TGT}.log")
print(f"[info] Training finished in {(_time.time()-_t0)/60:.1f} minutes")


# ---------------------------------------------------------------------------
# Step 4: Decode test set
# ---------------------------------------------------------------------------

section("Step 4: Decode test set  (marian-decoder)")

hyp_file = os.path.join(WORKING_DIR, f"transformer.{SRC}{TGT}.dropout.hyp.txt")

print("[info] Starting decoding...")
_t1 = _time.time()
run(" ".join([
    MARIAN_DECODER,
    f"-m {MODEL_DIR}/model.npz",
    f"-v {VOCAB_DIR}/vocab.{SRC}.yml {VOCAB_DIR}/vocab.{TGT}.yml",
    f"--devices 0",
    f"< {DATA_DIR}/test.{SRC}",
    f"> {hyp_file}",
]))
print(f"[info] Decoding finished in {(_time.time()-_t1):.1f} seconds")


# ---------------------------------------------------------------------------
# Step 5: Show sample results
# ---------------------------------------------------------------------------

section("Step 5: Sample G2P translations — side-by-side comparison")

print("--- paste: SOURCE | HYPOTHESIS | REFERENCE  (first 30 lines) ---")
run(f"paste {DATA_DIR}/test.{SRC} {hyp_file} {DATA_DIR}/test.{TGT} | head -n 30")

print(f"\n--- Source  (Myanmar grapheme) ---")
run(f"head -n 10 {DATA_DIR}/test.{SRC}")

print(f"\n--- Hypothesis  (model output phoneme) ---")
run(f"head -n 10 {hyp_file}")

print(f"\n--- Reference  (gold phoneme) ---")
run(f"head -n 10 {DATA_DIR}/test.{TGT}")

run(f"wc -l {hyp_file} {DATA_DIR}/test.{TGT}")

print(f"\n[DONE] G2P Transformer training complete — Experiment: DROPOUT ONLY")
print(f"  Model     : {MODEL_DIR}/model.npz")
print(f"  Config    : {CONFIG_FILE}")
print(f"  Train log : {TRAIN_LOG}")
print(f"  Valid log : {VALID_LOG}")
print(f"  Hypothesis: {hyp_file}")
print(f"  lr-warmup           = {LR_WARMUP}")
print(f"  transformer-dropout = {TRANSFORMER_DROPOUT}")
