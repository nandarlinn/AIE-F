"""
Evaluate the trained multi-head Myanmar syllable recognizer.

Computes:
  - Per-head: top-1 accuracy, macro-P/R/F1, weighted-P/R/F1
  - All-components-correct rate (all 7 heads simultaneously correct)
  - Syllable lookup accuracy (true syllable is among lookup candidates)
  - Per-class breakdown for the worst-performing classes (for re-collection)

Example:
    python multihead/evaluate_multihead.py \
        --checkpoint runs/mt1/best.pt \
        --image_dir image \
        --syl_file syl.txt \
        --out_json runs/mt1/eval.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Imports from the project
sys.path.insert(0, "recognizer")
from dataset import list_all_samples, split_by_copy, split_by_writer

from decompose import build_vocabularies
from model_multihead import SyllableMultiHead
from train_multihead import (
    Invert, MultiHeadDataset, collate,
    build_component_to_syllable_map, predicted_components_to_key,
)


# ---------- Vocab JSON I/O ----------
def load_vocabs_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    vocabs = {}
    for k, v in raw.items():
        out = []
        for x in v:
            if isinstance(x, list):
                out.append(tuple(x))
            else:
                out.append(x)
        vocabs[k] = out
    return vocabs


# ---------- Per-head P/R/F1 ----------
def compute_per_class_metrics(per_class):
    """
    per_class[c] = {"tp": int, "fp": int, "fn": int, "support": int}
    Returns the per-class P/R/F1 and the macro/weighted aggregates.
    """
    classes = sorted(per_class.keys())
    precisions, recalls, f1s, supports = [], [], [], []
    rows = {}
    for c in classes:
        m = per_class[c]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        n = m["support"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
        supports.append(n)
        rows[c] = {"precision": prec, "recall": rec, "f1": f1, "support": n}

    n_c = len(classes)
    macro_p  = sum(precisions) / n_c if n_c else 0.0
    macro_r  = sum(recalls)    / n_c if n_c else 0.0
    macro_f1 = sum(f1s)        / n_c if n_c else 0.0

    total = sum(supports) or 1
    weighted_p  = sum(p * s for p, s in zip(precisions, supports)) / total
    weighted_r  = sum(r * s for r, s in zip(recalls,    supports)) / total
    weighted_f1 = sum(f * s for f, s in zip(f1s,        supports)) / total

    return rows, {
        "macro":    {"precision": macro_p,  "recall": macro_r,  "f1": macro_f1},
        "weighted": {"precision": weighted_p,"recall": weighted_r,"f1": weighted_f1},
    }


def _display_label(val):
    """Pretty-print a vocab entry: codepoint int -> char, tuple -> joined chars."""
    if val is None:
        return "<none>"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return chr(val)
    if isinstance(val, (list, tuple)):
        return "".join(chr(c) for c in val) if val else "<none>"
    return str(val)


# ---------- Main eval ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--image_dir",   default="image")
    ap.add_argument("--syl_file",    default="syl.txt")
    ap.add_argument("--vocabs_json", default=None,
                    help="default: <checkpoint_dir>/vocabs.json")
    ap.add_argument("--split",       choices=["copy","writer"], default="copy")
    ap.add_argument("--test_copy",   type=int, default=3)
    ap.add_argument("--test_users",  nargs="*", default=[])
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--worst_n",     type=int, default=20,
                    help="number of worst classes to display per head")
    ap.add_argument("--out_json",    default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_sizes = ckpt["head_sizes"]
    img_size   = ckpt["img_size"]
    head_names = list(head_sizes.keys())
    print(f"Checkpoint: epoch {ckpt.get('epoch')}, "
          f"reported test syl acc: {ckpt.get('test_syl_acc', 'n/a')}")

    # ---- Load vocabs ----
    vocabs_path = args.vocabs_json or os.path.join(
        os.path.dirname(args.checkpoint), "vocabs.json"
    )
    vocabs = load_vocabs_from_json(vocabs_path)

    # ---- Load labels and build component map ----
    with open(args.syl_file, "r", encoding="utf-8") as f:
        syl_labels = [ln.strip() for ln in f if ln.strip()]
    comp_map = build_component_to_syllable_map(syl_labels, vocabs)

    # ---- Samples + split ----
    samples = list_all_samples(args.image_dir)
    if args.split == "copy":
        _, test_s = split_by_copy(samples, test_copy=args.test_copy)
    else:
        if not args.test_users:
            sys.exit("--split writer requires --test_users")
        _, test_s = split_by_writer(samples, args.test_users)
    print(f"Test samples: {len(test_s)}")

    # ---- Dataset ----
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        Invert(),
    ])
    test_ds = MultiHeadDataset(test_s, syl_labels, vocabs, tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0,
                             collate_fn=collate)

    # ---- Model ----
    model = SyllableMultiHead(head_sizes, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Accumulators ----
    # Per head: TP/FP/FN per class
    per_head_counts = {
        h: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
        for h in head_names
    }
    head_total   = {h: 0 for h in head_names}
    head_correct = {h: 0 for h in head_names}
    all_correct_count = 0
    syl_correct_count = 0
    total_samples = 0

    # For the per-syllable analysis
    per_syllable_correct = defaultdict(int)
    per_syllable_total   = defaultdict(int)

    with torch.no_grad():
        for imgs, labels, lines in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels_dev = {h: v.to(device, non_blocking=True) for h, v in labels.items()}

            logits = model(imgs)
            bs = imgs.size(0)
            total_samples += bs

            pred = {h: logits[h].argmax(dim=1) for h in head_names}

            # Per-head per-class TP/FP/FN
            all_right = torch.ones(bs, dtype=torch.bool, device=device)
            for h in head_names:
                p_cpu = pred[h].cpu().tolist()
                t_cpu = labels_dev[h].cpu().tolist()
                for p_i, t_i in zip(p_cpu, t_cpu):
                    per_head_counts[h][t_i]["support"] += 1
                    if p_i == t_i:
                        per_head_counts[h][t_i]["tp"] += 1
                    else:
                        per_head_counts[h][t_i]["fn"] += 1
                        per_head_counts[h][p_i]["fp"] += 1

                ok = (pred[h] == labels_dev[h])
                head_correct[h] += ok.sum().item()
                head_total[h]   += bs
                all_right &= ok

            all_correct_count += all_right.sum().item()

            # Syllable-level: lookup map
            pred_cpu = {h: pred[h].cpu().tolist() for h in head_names}
            true_lines = lines.tolist()
            for i in range(bs):
                indices = {h: pred_cpu[h][i] for h in head_names}
                key = predicted_components_to_key(indices, vocabs)
                cands = comp_map.get(key, [])
                line0 = true_lines[i]
                per_syllable_total[line0] += 1
                if any((line_1b - 1) == line0 for line_1b, _ in cands):
                    syl_correct_count += 1
                    per_syllable_correct[line0] += 1

    # =====================================================================
    # Report
    # =====================================================================
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Test samples : {total_samples}")
    print(f"\nAll-components-correct accuracy: "
          f"{all_correct_count/total_samples*100:6.2f}%")
    print(f"Syllable lookup accuracy       : "
          f"{syl_correct_count/total_samples*100:6.2f}%")

    # Per-head summary
    print("\n" + "=" * 70)
    print("PER-HEAD METRICS")
    print("=" * 70)
    print(f"{'head':>10}  {'top-1':>7}  "
          f"{'macro-P':>8}  {'macro-R':>8}  {'macro-F1':>8}  "
          f"{'wt-P':>7}  {'wt-R':>7}  {'wt-F1':>7}")

    per_head_report = {}
    for h in head_names:
        rows, agg = compute_per_class_metrics(per_head_counts[h])
        top1 = head_correct[h] / head_total[h]
        print(f"{h:>10}  {top1*100:6.2f}%  "
              f"{agg['macro']['precision']*100:7.2f}% "
              f"{agg['macro']['recall']*100:7.2f}% "
              f"{agg['macro']['f1']*100:7.2f}%  "
              f"{agg['weighted']['precision']*100:6.2f}% "
              f"{agg['weighted']['recall']*100:6.2f}% "
              f"{agg['weighted']['f1']*100:6.2f}%")

        per_head_report[h] = {
            "top1":     top1,
            "macro":    agg["macro"],
            "weighted": agg["weighted"],
            "per_class": [
                {
                    "index":     c,
                    "label":     _display_label(vocabs[h][c])
                                  if 0 <= c < len(vocabs[h]) else "?",
                    "precision": rows[c]["precision"],
                    "recall":    rows[c]["recall"],
                    "f1":        rows[c]["f1"],
                    "support":   rows[c]["support"],
                }
                for c in sorted(rows.keys())
            ],
        }

    # Worst classes per head
    print("\n" + "=" * 70)
    print(f"WORST {args.worst_n} CLASSES BY F1 (per head)")
    print("=" * 70)
    for h in head_names:
        rows = {pc["index"]: pc for pc in per_head_report[h]["per_class"]}
        worst = sorted(rows.values(),
                       key=lambda r: (r["f1"], -r["support"]))[:args.worst_n]
        if not worst:
            continue
        # Only show heads with more than a handful of classes
        if len(rows) < 5:
            continue
        print(f"\n--- {h} ---")
        print(f"  {'label':>8}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>4}")
        for r in worst:
            print(f"  {r['label']:>8}  "
                  f"{r['precision']*100:5.1f}% "
                  f"{r['recall']*100:5.1f}% "
                  f"{r['f1']*100:5.1f}%  "
                  f"{r['support']:>4}")

    # Per-syllable accuracy (full lookup-based)
    print("\n" + "=" * 70)
    print(f"WORST {args.worst_n} SYLLABLES BY LOOKUP ACCURACY")
    print("=" * 70)
    syl_rows = []
    for line0, n in per_syllable_total.items():
        if n == 0:
            continue
        acc = per_syllable_correct[line0] / n
        syl_rows.append({
            "line":     line0 + 1,
            "syllable": syl_labels[line0],
            "acc":      acc,
            "support":  n,
        })
    syl_rows.sort(key=lambda r: (r["acc"], -r["support"]))
    print(f"  {'line':>5}  {'syl':>4}  {'acc':>6}  {'n':>3}")
    for r in syl_rows[:args.worst_n]:
        print(f"  {r['line']:>5}  {r['syllable']:>4}  "
              f"{r['acc']*100:5.1f}%  {r['support']:>3}")

    # Save full JSON
    if args.out_json:
        report = {
            "total_test_samples":      total_samples,
            "all_components_correct":  all_correct_count / total_samples,
            "syllable_lookup_accuracy":syl_correct_count / total_samples,
            "per_head":                per_head_report,
            "per_syllable":            syl_rows,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull report saved -> {args.out_json}")


if __name__ == "__main__":
    main()
