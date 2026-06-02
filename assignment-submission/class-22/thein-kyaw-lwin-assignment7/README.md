# Lab Assignment Report: Myanmar Grapheme-to-Phoneme (G2P) NMT with Marian

**Student:** Thein Kyaw Lwin  
**Assignment:** G2P Neural Machine Translation — Seq2Seq and Transformer  
**Instructor:** Dr. Ye Kyaw Thu, Language Understanding Lab., Myanmar  
**Reference Notebooks:** `Seq2Seq-NMT-marian-ph2gp.ipynb`, `Transformer-NMT-marian-ph2gp.ipynb`  
**Platform:** Kaggle (2× Tesla T4 GPU, Ubuntu 22.04, CUDA 13.0)  

---

## 1. Overview

This report documents the complete pipeline for training Myanmar **Grapheme-to-Phoneme (G2P)** neural machine translation models using the Marian NMT toolkit. The task converts Myanmar script (source) into its phonemic transcription (target). Two architectures were trained and compared:

1. **Seq2Seq (S2S)** — LSTM-based encoder-decoder with attention
2. **Transformer** — Self-attention based architecture

For each architecture, a baseline was first trained following the reference notebook settings. Targeted hyperparameter adjustments were then applied to attempt performance improvement. All training was conducted on Kaggle's free GPU environment.

---

## 2. Environment Setup

### 2.1 Kaggle Platform Specification

| Resource | Details |
|---|---|
| GPU | 2× NVIDIA Tesla T4 (15,360 MiB each) |
| CPU | Intel Xeon (4 cores @ 2.00 GHz) |
| RAM | 31.35 GiB |
| OS | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| CUDA | 13.0 / Driver 580.105.08 |
| NCCL | 2.8.3 |

### 2.2 Marian NMT — Compile-Once, Reuse Strategy

Kaggle does not provide Marian NMT pre-installed. Rather than recompiling from source on every kernel run (which takes ~10–15 minutes), a **compile-once, dataset-reuse** strategy was adopted:

**Step 1 — Compile Marian from source (one-time kernel):**

A dedicated Kaggle kernel (`install_marian.py`) was created to:
1. Install system dependencies (`cmake`, `build-essential`, `libgoogle-perftools-dev`, `ccache`)
2. Clone Marian-NMT from GitHub (`git clone --depth 1`)
3. Configure with CMake (`-DCMAKE_BUILD_TYPE=Release`, `-DUSE_SENTENCEPIECE=off`, `-DCOMPILE_SERVER=off`)
4. Compile with `make -j4` (4 CPU cores)
5. Save the resulting binaries (`marian`, `marian-decoder`, `marian-vocab`) to `/kaggle/working/marian-bins/`

**Step 2 — Publish as a private Kaggle Dataset:**

The compiled binaries were saved as the private Kaggle dataset **`theinkyawlwin/marian-nmt`**, making them reusable across all subsequent training kernels without recompiling.

**Step 3 — Use in training kernels:**

Since `/kaggle/input/` is **read-only** on Kaggle, each training kernel copies the binaries to `/kaggle/working/bins/` and applies `chmod +x` before use:

```python
for src_bin in ["marian", "marian-vocab", "marian-decoder"]:
    shutil.copy2(f"{BINS_DIR}/{src_bin}", f"{LOCAL_BINS}/{src_bin}")
    os.chmod(f"{LOCAL_BINS}/{src_bin}", 0o755)
```

**Marian version used:** `v1.12.0 65bf82f (2023-02-21)`

### 2.3 Dual GPU Training

Both GPUs were utilized via Marian's built-in multi-GPU synchronous training:

```
--devices 0 1 --sync-sgd
```

Marian uses **NCCL** (NVIDIA Collective Communications Library) for GPU-to-GPU communication with global sharding, effectively doubling throughput compared to single-GPU training.

---

## 3. Dataset

The parallel G2P corpus (**`theinkyawlwin/g2p-par`**) consists of Myanmar grapheme-phoneme pairs:

| Split | Sentences | Source | Target |
|---|---|---|---|
| Train | 20,000 | `train.my` | `train.ph` |
| Dev | 2,000 | `dev.my` | `dev.ph` |
| Test | 2,802 | `test.my` | `test.ph` |

**Vocabulary** (built from train + dev combined, 22,000 sentences):

| Vocab | Size |
|---|---|
| Source (Myanmar grapheme) | 2,307 types |
| Target (phoneme) | 1,850 types |

Vocabularies were built using `marian-vocab`, producing character-level token dictionaries. The source vocabulary covers Myanmar Unicode codepoints (syllable components) while the target covers phoneme tokens such as `a-`, `ma-`, `da-`, `te'`, `kya.`, etc.

**Sample pairs (test set):**

| Myanmar Grapheme | Gold Phoneme |
|---|---|
| တက် တက် ပြောင် | `te' te' pjaun` |
| ကပ် ပိ | `ka' pi.` |
| ညှဉ်း ပန်း | `njhin: ban:` |

---

## 4. Model Architectures

### 4.1 Seq2Seq (S2S) — LSTM Encoder-Decoder

The S2S model uses a **bidirectional alternating LSTM** encoder and a **deep LSTM** decoder with attention.

| Parameter | Value |
|---|---|
| Model type | `s2s` |
| Encoder type | `alternating` bidirectional |
| Encoder cell | LSTM, depth 2, cell-depth 2 |
| Decoder cell | LSTM, depth 2, base-depth 2, high-depth 2 |
| Embedding dim | 512 |
| RNN hidden dim | 1,024 |
| Skip connections | ✓ |
| Layer normalization | ✓ |
| Tied embeddings | ✓ |

### 4.2 Transformer

The Transformer model uses multi-head self-attention for both encoder and decoder.

| Parameter | Value |
|---|---|
| Model type | `transformer` |
| Encoder depth | 2 layers |
| Decoder depth | 2 layers |
| Attention heads | 8 |
| Embedding dim | 512 |
| FFN dim | 2,048 |
| FFN activation | Swish |
| Post-processing | `dan` (dropout → add → normalize) |
| Tied embeddings | ✓ |

---

## 5. Baseline Training

### 5.1 S2S Baseline

**Key training parameters:**

| Parameter | Value |
|---|---|
| Learning rate | 0.0001 (Adam default) |
| LR schedule | None (flat) |
| Dropout (RNN) | 0.3 |
| Dropout (source) | 0.3 |
| Label smoothing | 0 (disabled) |
| Clip norm | 1 (default) |
| Beam size | 12 (default) |
| Length normalization | 0 (disabled) |
| Early stopping | 10 stalls |
| Validation frequency | every 5,000 updates |
| GPUs | 2× T4, sync-SGD |

**Validation progression (dev set BLEU):**

| Update | Epoch | Dev BLEU | Note |
|---|---|---|---|
| 5,000 | 265 | **66.9571** | ✅ Best (only best) |
| 10,000 | 530 | 63.7195 | stalled 1× |
| 15,000 | 794 | 60.1361 | stalled 2× |
| 20,000 | 1,057 | 59.4070 | stalled 3× |
| 25,000 | 1,321 | 57.8045 | stalled 4× |
| 30,000 | 1,586 | 58.6133 | stalled 5× |
| 35,000 | 1,850 | 57.1648 | stalled 6× |
| 40,000 | 2,115 | 57.3819 | stalled 7× |
| 45,000 | 2,378 | 57.0389 | stalled 8× |
| 50,000 | 2,642 | 57.5192 | stalled 9× |
| 55,000 | 2,906 | 56.6063 | stalled 10× → **stopped** |

**Training time:** ~5.5 hours

### 5.2 Transformer Baseline

**Key training parameters:**

| Parameter | Value |
|---|---|
| Learning rate | 0.0003 |
| LR schedule | Inverse-sqrt decay from step 16,000 |
| LR warmup | 0 (none) |
| Transformer dropout | 0.3 |
| Label smoothing | 0.1 |
| Clip norm | 5 |
| Beam size | 6 |
| Length normalization | 0.6 |
| Early stopping | 10 stalls |
| Validation frequency | every 5,000 updates |
| GPUs | 2× T4, sync-SGD |

**Validation progression (dev set BLEU):**

| Update | Epoch | Dev BLEU | Note |
|---|---|---|---|
| 5,000 | 145 | 72.6762 | ✅ new best |
| 10,000 | 289 | 71.8873 | stalled 1× |
| 15,000 | 433 | 73.0278 | ✅ new best |
| 20,000 | 577 | 73.3391 | ✅ new best |
| 25,000 | 721 | 73.0544 | stalled 1× |
| **30,000** | **865** | **73.3498** | ✅ **best overall** |
| 35,000 | 1,009 | 72.9386 | stalled 1× |
| 40,000 | 1,152 | 72.6275 | stalled 2× |
| 45,000 | 1,296 | 72.7997 | stalled 3× |
| 50,000 | 1,441 | 72.4101 | stalled 4× |
| 55,000 | 1,585 | 72.7275 | stalled 5× → **stopped** |

**Training time:** ~58 minutes

> The Transformer trained **~5.7× faster** than S2S while achieving **+6.4 BLEU** higher dev score.

---

## 6. Improvement Experiments

### 6.1 S2S — Problem Analysis

The S2S baseline showed a critical failure pattern: BLEU peaked at the very first validation (step 5,000) and declined monotonically for 50,000 more updates. The root causes identified:

1. **No LR schedule** — flat LR 0.0001 causes the optimizer to overshoot and drift without any decay
2. **No label smoothing** — model becomes overconfident, hurting generalization
3. **No length penalty in decoding** — `--normalize 0` favors short hypotheses
4. **Oversized beam** — beam-12 is excessive for short G2P output sequences
5. **Tight gradient clipping** — `--clip-norm 1` excessively restricts early-training updates

#### S2S Experiment 1: LR Schedule

| Parameter | Baseline | Experiment 1 |
|---|---|---|
| `--learn-rate` | 0.0001 | **0.0003** |
| `--lr-decay-inv-sqrt` | 0 (off) | **16,000** |
| `--lr-report` | off | **on** |
| All other params | — | unchanged |

**Rationale:** Directly fixes the root cause. The inverse-sqrt schedule decays LR as `1/√step` after step 16,000, stabilizing late-stage training. Base LR is aligned with the Transformer's proven value.

#### S2S Experiment 2: LR Schedule + Label Smoothing + Decoding Fixes

| Parameter | Baseline | Experiment 2 |
|---|---|---|
| `--learn-rate` | 0.0001 | **0.0003** |
| `--lr-decay-inv-sqrt` | 0 (off) | **16,000** |
| `--label-smoothing` | 0 (off) | **0.1** |
| `--normalize` | 0 (off) | **0.6** |
| `--beam-size` | 12 | **6** |
| `--clip-norm` | 1 | **5** |
| All other params | — | unchanged |

**Rationale:** Builds on Experiment 1 and adds the full stack of corrections to align the S2S training setup with the Transformer's proven configuration. Each change addresses a specific identified deficiency.

### 6.2 Transformer — Problem Analysis

The Transformer baseline was well-configured and showed healthy convergence. Two parameters were identified for sensitivity testing:

1. **No LR warmup** — The standard Transformer recipe (Vaswani 2017) uses warmup; the baseline skipped it (`--lr-warmup 0`)
2. **High dropout (0.3)** — For a 20K-sentence dataset, 0.3 may over-regularize; the original paper used 0.1

#### Transformer Experiment 1: LR Warmup Only

| Parameter | Baseline | Experiment 1 |
|---|---|---|
| `--lr-warmup` | 0 | **4,000** |
| `--transformer-dropout` | 0.3 | 0.3 (unchanged) |

**Rationale:** Warmup linearly increases LR from 0 to 0.0003 over 4,000 steps before the inverse-sqrt decay begins, allowing parameters to stabilize before large gradient updates.

#### Transformer Experiment 2: Reduced Dropout Only

| Parameter | Baseline | Experiment 2 |
|---|---|---|
| `--lr-warmup` | 0 | 0 (unchanged) |
| `--transformer-dropout` | 0.3 | **0.1** |

**Rationale:** Dropout 0.3 may be too aggressive for a 20K training corpus. The original paper value of 0.1 should allow the model to learn more freely without over-regularization.

#### Transformer Experiment 3: Warmup + Reduced Dropout

| Parameter | Baseline | Experiment 3 |
|---|---|---|
| `--lr-warmup` | 0 | **4,000** |
| `--transformer-dropout` | 0.3 | **0.1** |

**Rationale:** Tests whether both improvements are complementary. If Experiment 3 outperforms both Experiments 1 and 2, the gains are additive.

---

## 7. Results Comparison

### 7.1 S2S Results

| Experiment | `learn-rate` | `lr-inv-sqrt` | `label-smooth` | `normalize` | `beam` | `clip-norm` | Best Dev BLEU |
|---|---|---|---|---|---|---|---|
| Baseline | 0.0001 | 0 | 0 | 0 | 12 | 1 | 66.9571 @ Up. 5,000 |
| Exp 1: LR Schedule | **0.0003** | **16,000** | 0 | 0 | 12 | 1 | 59.8028 @ Up. 5,000 |
| Exp 2: LR + Smooth | **0.0003** | **16,000** | **0.1** | **0.6** | **6** | **5** | **72.7649 @ Up. 65,000** |

### 7.2 Transformer Results

| Experiment | `lr-warmup` | `dropout` | Best Dev BLEU |
|---|---|---|---|
| Baseline | 0 | 0.3 | 73.3498 @ Up. 30,000 |
| Exp 1: Warmup | **4,000** | 0.3 | **73.8941 @ Up. 25,000** |
| Exp 2: Dropout 0.1 | 0 | **0.1** | 72.0564 @ Up. 15,000 |
| Exp 3: Warmup + Dropout | **4,000** | **0.1** | 72.0401 @ Up. 20,000 |

### 7.3 Cross-Architecture Summary

| Architecture | Experiment | Best Dev BLEU | Best at Update |
|---|---|---|---|
| S2S | Baseline | 66.9571 | 5,000 |
| S2S | Exp 1: LR Schedule | 59.8028 | 5,000 |
| S2S | Exp 2: LR + Smoothing | **72.7649** | 65,000 |
| Transformer | Baseline | 73.3498 | 30,000 |
| Transformer | Exp 1: Warmup | **73.8941** | 25,000 |
| Transformer | Exp 2: Dropout 0.1 | 72.0564 | 15,000 |
| Transformer | Exp 3: Warmup + Dropout | 72.0401 | 20,000 |

> All results collected from `valid.log` files downloaded with:
> ```bash
> kaggle kernels output <kernel-id> --file-pattern ".*\.(log|txt)$" -p <local-path>
> ```

---

## 8. Discussion

### Why Transformer Outperforms S2S

The baseline results show a **+6.4 BLEU** advantage for the Transformer. Several factors explain this:

1. **Parallelism:** The Transformer processes the entire input sequence in parallel via self-attention, while the LSTM must process tokens sequentially. This enables better capture of long-range dependencies.
2. **Better training recipe out of the box:** The Transformer baseline had LR schedule, label smoothing, and proper beam decoding — the S2S baseline had none of these.
3. **Training speed:** ~58 minutes vs ~5.5 hours for the same task and hardware, because the Transformer's attention mechanism is highly parallelizable on GPU while LSTM is inherently sequential.

### The S2S Baseline Failure Pattern

The monotonically increasing validation cross-entropy (from 2.00 at step 5,000 to 2.82 at step 55,000) is textbook evidence of a flat learning rate without decay. The model keeps overshooting the loss minimum and drifting further away with each update.

### G2P as a Monotone Task

Myanmar G2P is essentially a **monotone sequence transduction** task — the phoneme output largely follows the left-to-right order of the grapheme input with minimal reordering. Both architectures are well-suited, but the Transformer's multi-head attention captures this monotone alignment more efficiently than the LSTM's compressed hidden state.

---

## 9. Repository Structure

The repository is organized to be **GitHub-friendly** — model weights (`.npz`, ~150–300 MB each) are excluded, keeping only code, configs, logs, and hypothesis outputs.

```
/Users/tklwin/GithubRepos/AIE-F/assignment-submission/class-22/thein-kyaw-lwin-assignment7/
├── g2p-par/                                           # Dataset (train/dev/test)
├── marian-nmt-install/                                # Marian compile kernel
│   ├── install_marian.py
│   ├── kernel-metadata.json
│   └── marian-nmt-install.log
├── marian-nmt-training-g2p-s2s/                      # S2S baseline
│   ├── train_g2p_s2s.py
│   ├── kernel-metadata.json
│   ├── marian-g2p-s2s.log
│   ├── valid.log
│   └── s2s.my-ph.hyp.txt
├── marian-nmt-training-g2p-s2s-lr-schedule/          # S2S Exp 1
├── marian-nmt-training-g2p-s2s-lr-schedule-smoothing/ # S2S Exp 2
├── marian-nmt-training-g2p-transformer/              # Transformer baseline
├── marian-nmt-training-g2p-transformer-warmup/       # Transformer Exp 1
├── marian-nmt-training-g2p-transformer-dropout/      # Transformer Exp 2
├── marian-nmt-training-g2p-transformer-warmup-dropout/ # Transformer Exp 3
├── Seq2Seq-NMT-marian-ph2gp.ipynb                    # Reference notebook
├── Transformer-NMT-marian-ph2gp.ipynb                # Reference notebook
├── kaggle-linux.md                                    # Kaggle env notes
├── log-dl.md                                         # Download commands
└── README.md                                     # This report
```

**Tracked in Git:** `.py`, `.json`, `.log`, `.txt`, `.md`, `.ipynb`  
**Excluded from Git:** `model.npz`, `model.npz.best-*` (150–300 MB each)

---

## 10. References

1. Ye Kyaw Thu. *Seq2Seq-NMT-marian-ph2gp.ipynb* (24 May 2026). Language Understanding Lab., Myanmar.
2. Ye Kyaw Thu. *Transformer-NMT-marian-ph2gp.ipynb* (26 May 2026). Language Understanding Lab., Myanmar.
3. Junczys-Dowmunt, M. et al. (2018). *Marian: Fast Neural Machine Translation in C++*. ACL System Demonstrations.
4. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
5. Marian NMT documentation: https://marian-nmt.github.io/docs/
