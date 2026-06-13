"""
train_g2p_s2s.py
================
Kaggle kernel: Grapheme-to-Phoneme NMT using Marian s2s (LSTM seq2seq)
Task:          Myanmar grapheme → Phoneme  (my → ph)
Architecture:  s2s, alternating bidirectional LSTM encoder, LSTM decoder

Dataset sources (kernel-metadata.json):
  - theinkyawlwin/marian-nmt   → Marian v1.12 binaries
  - theinkyawlwin/g2p-par      → Parallel g2p data (train/dev/test)

GPU:  2x Tesla T4 (15 GB each) — uses both via --devices 0 1 --sync-sgd
Reference: Seq2Seq-NMT-marian-ph2gp.ipynb (Ye Kyaw Thu, 24 May 2026)
"""

import subprocess
import sys
import os
import stat
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
# When a folder is uploaded as a Kaggle dataset, Kaggle preserves the folder
# name as a subfolder: /kaggle/input/g2p-par/g2p-par/
DATA_DIR = resolve_path([
    "/kaggle/input/g2p-par/g2p-par",                                     # folder-upload layout
    "/kaggle/input/g2p-par",                                             # flat layout
    "/kaggle/input/datasets/theinkyawlwin/g2p-par/g2p-par",             # v2 + folder
    "/kaggle/input/datasets/theinkyawlwin/g2p-par",                     # v2 flat
], "g2p-par data dir")

# Output (persisted as kernel output)
WORKING_DIR  = "/kaggle/working"
LOCAL_BINS   = os.path.join(WORKING_DIR, "bins")   # writable copy of binaries
PREPROC_DIR  = os.path.join(WORKING_DIR, "preprocessing")
VOCAB_DIR    = os.path.join(WORKING_DIR, "vocab")
MODEL_DIR    = os.path.join(WORKING_DIR, "model.s2s.myph")

# Task: Grapheme → Phoneme
SRC = "my"   # source: Myanmar grapheme
TGT = "ph"   # target: phoneme


# ---------------------------------------------------------------------------
# Step 1: Copy binaries to writable dir and chmod +x
# /kaggle/input/ is read-only — cannot chmod there directly
# ---------------------------------------------------------------------------

section("Step 1: Copy Marian binaries to /kaggle/working/bins/ and make executable")

import shutil
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

print("\n--- Source vocab (Myanmar grapheme) — first 5 entries ---")
run(f"head -n 5 {VOCAB_DIR}/vocab.{SRC}.yml")
print("\n--- Target vocab (phoneme) — first 5 entries ---")
run(f"head -n 5 {VOCAB_DIR}/vocab.{TGT}.yml")


# ---------------------------------------------------------------------------
# Step 3: Dump config then train
# ---------------------------------------------------------------------------

section("Step 3: Train s2s G2P model  (my → ph)  on 2x Tesla T4")

marian_args = " ".join([
    f"--type s2s",
    f"--train-sets {DATA_DIR}/train.{SRC} {DATA_DIR}/train.{TGT}",
    f"--max-length 200",
    f"--valid-sets {DATA_DIR}/dev.{SRC} {DATA_DIR}/dev.{TGT}",
    f"--vocabs {VOCAB_DIR}/vocab.{SRC}.yml {VOCAB_DIR}/vocab.{TGT}.yml",
    f"--model {MODEL_DIR}/model.npz",
    f"--workspace 6000",                # 6 GB per GPU (T4 has 15 GB)
    f"--enc-depth 2 --enc-type alternating --enc-cell lstm --enc-cell-depth 2",
    f"--dec-depth 2 --dec-cell lstm --dec-cell-base-depth 2 --dec-cell-high-depth 2",
    f"--tied-embeddings --layer-normalization --skip",
    f"--mini-batch-fit",                # auto-fit batch size to GPU memory
    f"--valid-mini-batch 32",
    f"--valid-metrics cross-entropy perplexity bleu",
    f"--valid-freq 5000 --save-freq 5000 --disp-freq 500",
    f"--dropout-rnn 0.3 --dropout-src 0.3 --exponential-smoothing",
    f"--early-stopping 10",
    f"--log {MODEL_DIR}/train.log --valid-log {MODEL_DIR}/valid.log",
    f"--devices 0 1 --sync-sgd",        # both T4 GPUs, synchronous
    f"--seed 1111",
])

# Dump config YAML (dry-run, no actual training yet)
run(f"{MARIAN} {marian_args} --dump-config > {MODEL_DIR}/config.yml")

print("\n--- config.yml ---")
run(f"cat {MODEL_DIR}/config.yml")

# Train using config file
# NOTE: 'time' is a bash builtin — not available in /bin/sh used by subprocess.
#       Use Python's time module for elapsed timing instead.
import time as _time
print("[info] Starting training...")
_t0 = _time.time()
run(f"{MARIAN} -c {MODEL_DIR}/config.yml 2>&1 | tee {MODEL_DIR}/train.s2s.myph.log")
print(f"[info] Training finished in {(_time.time()-_t0)/60:.1f} minutes")


# ---------------------------------------------------------------------------
# Step 4: Decode test set
# ---------------------------------------------------------------------------

section("Step 4: Decode test set  (marian-decoder)")

hyp_file = os.path.join(WORKING_DIR, f"s2s.{SRC}-{TGT}.hyp.txt")

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

# Side-by-side: SOURCE (my) | HYPOTHESIS (ph) | REFERENCE (ph)
# mirrors reference notebook:  paste test.ph seq2seq.hyp.txt | head -n 30
print("--- paste: SOURCE | HYPOTHESIS | REFERENCE  (first 30 lines) ---")
run(f"paste {DATA_DIR}/test.{SRC} {hyp_file} {DATA_DIR}/test.{TGT} | head -n 30")

# Individual views for clarity
print("\n--- Source  (Myanmar grapheme) ---")
run(f"head -n 10 {DATA_DIR}/test.{SRC}")

print("\n--- Hypothesis  (model output phoneme) ---")
run(f"head -n 10 {hyp_file}")

print("\n--- Reference  (gold phoneme) ---")
run(f"head -n 10 {DATA_DIR}/test.{TGT}")

run(f"wc -l {hyp_file} {DATA_DIR}/test.{TGT}")

print(f"\n[DONE] G2P s2s training complete.")
print(f"  Model     : {MODEL_DIR}/model.npz")
print(f"  Train log : {MODEL_DIR}/train.log")
print(f"  Valid log : {MODEL_DIR}/valid.log")
print(f"  Hypothesis: {hyp_file}")
