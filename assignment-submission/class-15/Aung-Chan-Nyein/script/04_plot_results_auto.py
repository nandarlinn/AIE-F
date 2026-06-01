"""
05_plot_results_auto.py  (Fixed for your actual folder structure)
s
Generates 2 charts from your trained models:
  1. Baseline LSTM perplexity across 3 domains (Phase 3 Step 8)
  2. Before vs After domain adaptation (Phase 3 Step 10)

Results Folder Layout:
    results/
      ├── lstm_mypos_baseline/     ← the SHARED baseline (used for all 3 domains)
      ├── lstm_news_adapted/       ← adapted-on-news version
      ├── lstm_medical_adapted/    ← adapted-on-medical version
      └── lstm_legal_adapted/      ← adapted-on-legal version

Usage:
    python 04_plot_results_auto2.py
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent

DATA_DIR = WORK_DIR / "data" / "prepared"
RESULTS_DIR = WORK_DIR / "results"
OUTPUT_DIR = WORK_DIR / "report"
OUTPUT_DIR.mkdir(exist_ok=True)

EVAL_SCRIPT = SCRIPT_DIR / "02_evaluate_lstm.py"

# Domain config — for each domain, list possible baseline folder names
# The script tries them in order until one is found.
DOMAINS = {
    "News\n(Khit Thit)": {
        "test_file":           DATA_DIR / "test_news_std.txt",
        "baseline_candidates": ["lstm_mypos_baseline", "lstm_news"],
        "adapted":             RESULTS_DIR / "lstm_news_adapted",
        "color":               "#55A868",
    },
    "Medical\n(Hello Sayar Won)": {
        "test_file":           DATA_DIR / "test_medical_std.txt",
        "baseline_candidates": ["lstm_mypos_baseline", "lstm_medical"],
        "adapted":             RESULTS_DIR / "lstm_medical_adapted",
        "color":               "#4C72B0",
    },
    "Legal\n(myanmar_legal)": {
        "test_file":           DATA_DIR / "test_legal_std.txt",
        "baseline_candidates": ["lstm_mypos_baseline", "lstm_legal"],
        "adapted":             RESULTS_DIR / "lstm_legal_adapted",
        "color":               "#C44E52",
    },
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def find_baseline(candidates):
    """Find the first existing baseline folder from a list of candidates."""
    for name in candidates:
        candidate = RESULTS_DIR / name
        if candidate.exists() and (candidate / "best_model.pt").exists():
            return candidate
    return None


def run_eval(model_dir, test_file):
    """Run the eval script and return PPL from the resulting eval_results.json."""
    if not model_dir or not model_dir.exists():
        print(f"  ⚠️  Model folder missing: {model_dir}")
        return None
    if not test_file.exists():
        print(f"  ⚠️  Test file missing: {test_file}")
        return None
    if not EVAL_SCRIPT.exists():
        print(f"  ❌ Eval script not found: {EVAL_SCRIPT}")
        return None

    print(f"  Evaluating {model_dir.name} on {test_file.name}...")
    result = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT),
         "--model_dir", str(model_dir),
         "--test_file", str(test_file)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"  ❌ Eval failed:")
        print(f"     {result.stderr[:300]}")
        return None

    # Read PPL from JSON
    eval_json = model_dir / "eval_results.json"
    if not eval_json.exists():
        print(f"  ❌ No eval_results.json produced.")
        return None
    with open(eval_json) as f:
        return json.load(f)["perplexity"]


def collect_all_ppl():
    """Run all 6 evaluations. Returns dict {domain: (baseline_ppl, adapted_ppl)}."""
    results = {}
    for domain, cfg in DOMAINS.items():
        print(f"\n--- {domain.replace(chr(10), ' ')} ---")
        baseline_dir = find_baseline(cfg["baseline_candidates"])
        if baseline_dir:
            print(f"  Baseline found: {baseline_dir.name}")
        else:
            print(f"  ⚠️  No baseline folder found. Tried: {cfg['baseline_candidates']}")
        baseline_ppl = run_eval(baseline_dir, cfg["test_file"])
        adapted_ppl = run_eval(cfg["adapted"], cfg["test_file"])
        results[domain] = (baseline_ppl, adapted_ppl)
    return results


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_baseline_only(results):
    """Chart 1: Baseline PPL across domains (Phase 3 Step 8)."""
    domains = list(results.keys())
    baseline_values = [v[0] for v in results.values()]

    if any(b is None for b in baseline_values):
        print("\n⚠️  Skipping baseline chart — not all 3 baseline PPLs available.")
        missing = [d for d, v in results.items() if v[0] is None]
        print(f"   Missing: {[m.replace(chr(10), ' ') for m in missing]}")
        return

    colors = [DOMAINS[d]["color"] for d in domains]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(domains, baseline_values, color=colors)

    ax.set_title('Phase 3 Step 8 — Baseline LSTM Perplexity Across Domains',
                 fontsize=13, pad=15)
    ax.set_ylabel('Perplexity (PPL) — Lower is Better', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(baseline_values) * 1.2)

    for bar, val in zip(bars, baseline_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(baseline_values) * 0.02,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = OUTPUT_DIR / "domain_shift_chart.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Chart 1 saved: {save_path}")
    plt.close()


def plot_before_after(results):
    """Chart 2: Before vs After adaptation (Phase 3 Step 10)."""
    has_any_adaptation = any(v[1] is not None for v in results.values())
    has_any_baseline = any(v[0] is not None for v in results.values())

    if not (has_any_adaptation and has_any_baseline):
        print("\n⚠️  Skipping adaptation chart — need at least 1 baseline + 1 adapted PPL.")
        return

    domains = list(results.keys())
    baseline = [v[0] if v[0] is not None else 0 for v in results.values()]
    adapted = [v[1] if v[1] is not None else 0 for v in results.values()]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, baseline, width,
                   label='Baseline LSTM', color='#4C72B0')
    bars2 = ax.bar(x + width / 2, adapted, width,
                   label='Domain-Adapted LSTM', color='#C44E52')

    max_val = max(max(baseline), max(adapted))
    for bar, val in zip(bars1, baseline):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.02,
                    f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, adapted):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.02,
                    f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel('Perplexity (PPL) — Lower is Better', fontsize=11)
    ax.set_title('Phase 3 Step 10 — Domain Adaptation: Before vs After',
                 fontsize=13, pad=15)
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max_val * 1.2)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "domain_adaptation_chart.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Chart 2 saved: {save_path}")
    plt.close()


def print_summary(results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Domain':<25} {'Baseline':>10} {'Adapted':>10} {'Δ':>10}")
    print("-" * 60)
    for name, (baseline, adapted) in results.items():
        clean = name.replace("\n", " ")
        b = f"{baseline:.2f}" if baseline else "—"
        a = f"{adapted:.2f}" if adapted else "—"
        if baseline and adapted:
            delta = f"{((adapted - baseline) / baseline) * 100:+.1f}%"
        else:
            delta = "—"
        print(f"{clean:<25} {b:>10} {a:>10} {delta:>10}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Work dir:   {WORK_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Eval script: {EVAL_SCRIPT}")
    print("\nRunning all evaluations and collecting PPL...")

    results = collect_all_ppl()
    plot_baseline_only(results)
    plot_before_after(results)
    print_summary(results)
