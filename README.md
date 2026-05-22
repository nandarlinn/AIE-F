# Assignment 5 — Myanmar LM Domain Adaptation

**Student:** Aung Chan Nyein
**Course:** AIE-F (AI Fundamental Class)
**Date:** May 2026

---

## What This Submission Contains

This project demonstrates **domain shift** and **domain adaptation** for a Myanmar LSTM language model. A baseline LM trained on the general myPOS corpus is evaluated and adapted across three specialized domains: News, Medical, and Legal.

## Key Results

| Domain | Baseline PPL | Adapted PPL | Reduction |
|---|---:|---:|---:|
| News (Khit Thit) | 399.51 | 322.24 | **-19.3%** |
| Medical (Hello Sayar Won) | 675.64 | 587.87 | **-13.0%** |
| Legal (myanmar_legal) | 709.68 | 517.85 | **-27.0%** |

**Conclusion:** Domain adaptation lowered perplexity in all three domains, with the largest improvement on the most distant domain (Legal). Full analysis is in [`report/report.md`](report/report.md).

## Folder Structure

```
Assignment-5-ACN/
├── README.md                       (this file)
├── requirements.txt                (Python dependencies)
│
├── report/
│   ├── report.md                   (full analysis — start here)
│   ├── domain_shift_chart.png      (Phase 3 Step 8 visualization)
│   └── domain_adaptation_chart.png (before/after comparison)
│
├── data/
│   ├── raw_samples/                (300-syllable domain exploration samples)
│   ├── prepared/                   (standardized test + adaptation sets)
│   └── training/                   (baseline LM training corpus)
│
├── scripts/
│   ├── 01_train_baseline_lstm.py   (baseline LM trainer)
│   ├── 02_evaluate_lstm.py         (PPL/BPC evaluator)
│   ├── 03_domain_adaptation.py     (continued pre-training)
│   └── 04_plot_results_auto.py     (chart generator)
    results/
│   ├── lstm_baseline/                 ← Trained baseline model (checekpoint + Vocab)
│   │   ├── best_model.pt
│   │   ├── vocab.json
│   │   └── history.json
│   ├── lstm_medical_adapted/          ← After Medical adaptation
│   ├── lstm_news_adapted/             ← After News adaptation
│   ├── lstm_legal_adapted/            ← After Legal adaptation
│   └── ppl_results.csv                ← Final 6-number table
```

## How to Reproduce

See the "Reproducibility" section of [`report/report.md`](report/report.md) for the full command list.

Quick path:

```bash
pip install -r requirements.txt
python scripts/05_plot_results.py
```

This regenerates both required charts from the included model checkpoints.

## Method at a Glance

1. **Phase 1** — Cleaned 3 domain datasets (CSV/parquet), tokenized using a Myanmar-aware syllable regex
2. **Phase 2** — Trained 1-layer LSTM (embed=128, hidden=256) on myPOS v3; built 3 standardized test sets (10 sentences × 20 syllables each)
3. **Phase 3** — Evaluated baseline → fine-tuned on 50×20 in-domain samples (lr=1e-4, 15 epochs) → re-evaluated → produced bar charts

## Tools

- PyTorch (with Apple Silicon MPS acceleration)
- HuggingFace `datasets` for source data
- Python 3.11, pandas, regex, matplotlib

---

*For full methodology, analysis, and discussion, see [`report/report.md`](report/report.md).*
