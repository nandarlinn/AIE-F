"""
Evaluate a trained checkpoint and dump diagnostics.

Outputs:
  - overall top-1 / top-5 accuracy
  - the worst-N classes (lowest per-class accuracy) -> useful for targeted re-collection
  - the most-confused predicted-pairs              -> useful for targeted re-collection

Example:
    python evaluate.py --checkpoint runs/exp1/best.pt \
        --image_dir image --syl_file syl.txt
"""

import os
import argparse
import json
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    list_all_samples, load_labels,
    split_by_copy, split_by_writer,
    SyllableImageDataset,
)
from model import SyllableCNN
# Reuse the same Invert class as in train.py so the eval transform
# matches training exactly. Module-level so it pickles for DataLoader workers.
from train import Invert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image_dir",  default="image")
    ap.add_argument("--syl_file",   default="syl.txt")
    ap.add_argument("--split",      choices=["copy", "writer"], default="copy")
    ap.add_argument("--test_copy",  type=int, default=3)
    ap.add_argument("--test_users", nargs="*", default=[])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--worst_n",    type=int, default=30)
    ap.add_argument("--out_json",   default=None,
                    help="If set, write detailed per-class report to this JSON file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt["num_classes"]
    img_size    = ckpt.get("img_size", 64)

    labels = load_labels(args.syl_file)
    if len(labels) != num_classes:
        print(f"WARNING: label count {len(labels)} != model classes {num_classes}")

    samples = list_all_samples(args.image_dir)

    if args.split == "copy":
        _, test_s = split_by_copy(samples, test_copy=args.test_copy)
    else:
        if not args.test_users:
            raise SystemExit("--split writer needs --test_users")
        _, test_s = split_by_writer(samples, args.test_users)

    if not test_s:
        raise SystemExit("Test split is empty.")

    # Must match eval_tf in training (invert: strokes -> 1, bg -> 0)
    invert = Invert()
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        invert,
    ])
    test_ds = SyllableImageDataset(test_s, num_classes, transform=eval_tf)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = SyllableCNN(num_classes=num_classes, img_size=img_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total = 0
    hit1, hit5 = 0, 0
    per_class_total   = defaultdict(int)   # # of true samples in this class (TP+FN)
    per_class_correct = defaultdict(int)   # # correctly predicted (TP)
    per_class_predicted = defaultdict(int) # # times this class was predicted (TP+FP)
    confusion = Counter()  # (true_label, predicted_label) -> count

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            logits = model(imgs)
            _, top5 = logits.topk(5, dim=1)
            preds = top5[:, 0]

            for t, p, t5 in zip(lbls.tolist(), preds.tolist(), top5.tolist()):
                total += 1
                per_class_total[t] += 1
                per_class_predicted[p] += 1
                if t == p:
                    hit1 += 1
                    per_class_correct[t] += 1
                else:
                    confusion[(t, p)] += 1
                if t in t5:
                    hit5 += 1

    # ----- Compute precision / recall / F1 per class -----
    # Per-class metrics:
    #   precision_c = TP_c / (TP_c + FP_c) = correct[c] / predicted[c]
    #   recall_c    = TP_c / (TP_c + FN_c) = correct[c] / total[c]
    #   f1_c        = 2 * P * R / (P + R)
    classes_in_test = sorted(per_class_total.keys())
    precisions, recalls, f1s, supports = [], [], [], []
    per_class_metrics = {}
    for c in classes_in_test:
        tp = per_class_correct[c]
        n_true = per_class_total[c]
        n_pred = per_class_predicted[c]
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec  = tp / n_true if n_true > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(n_true)
        per_class_metrics[c] = {"precision": prec, "recall": rec,
                                "f1": f1, "support": n_true}

    # Macro: simple average across classes (each class counts the same)
    n_c = len(classes_in_test)
    macro_p  = sum(precisions) / n_c if n_c else 0.0
    macro_r  = sum(recalls)    / n_c if n_c else 0.0
    macro_f1 = sum(f1s)        / n_c if n_c else 0.0

    # Weighted: average weighted by class support (each sample counts the same)
    total_supp = sum(supports) if supports else 1
    weighted_p  = sum(p * s for p, s in zip(precisions, supports)) / total_supp
    weighted_r  = sum(r * s for r, s in zip(recalls,    supports)) / total_supp
    weighted_f1 = sum(f * s for f, s in zip(f1s,        supports)) / total_supp

    # ----- Print summary -----
    print(f"\nTest samples : {total}")
    print(f"Classes seen : {n_c}")
    print(f"\n--- Top-K accuracy ---")
    print(f"  Top-1 accuracy : {hit1/total*100:7.2f}%   ({hit1}/{total})")
    print(f"  Top-5 accuracy : {hit5/total*100:7.2f}%   ({hit5}/{total})")

    print(f"\n--- Macro-averaged metrics  (each class weighted equally) ---")
    print(f"  Precision      : {macro_p*100:7.2f}%")
    print(f"  Recall         : {macro_r*100:7.2f}%")
    print(f"  F1 score       : {macro_f1*100:7.2f}%")

    print(f"\n--- Weighted-averaged metrics  (weighted by class support) ---")
    print(f"  Precision      : {weighted_p*100:7.2f}%")
    print(f"  Recall         : {weighted_r*100:7.2f}%")
    print(f"  F1 score       : {weighted_f1*100:7.2f}%")

    print(f"\n  Note: with single-sample-per-class test sets, weighted recall")
    print(f"        equals top-1 accuracy by definition.")

    # Worst classes
    per_class_acc = []
    for c, n in per_class_total.items():
        per_class_acc.append((c, per_class_correct[c] / n, n))
    per_class_acc.sort(key=lambda x: (x[1], -x[2]))  # lowest acc first

    print(f"\nWorst {args.worst_n} classes (by per-class top-1 accuracy):")
    print(f"  {'line':>5}  {'syl':>4}  {'acc':>6}  {'n':>3}")
    for c, acc, n in per_class_acc[:args.worst_n]:
        syl = labels[c] if c < len(labels) else "?"
        print(f"  {c+1:>5}  {syl:>4}  {acc*100:5.1f}%  {n:>3}")

    # Most confused pairs
    print(f"\nTop {args.worst_n} most-confused (true -> predicted) pairs:")
    for (t, p), cnt in confusion.most_common(args.worst_n):
        ts = labels[t] if t < len(labels) else "?"
        ps = labels[p] if p < len(labels) else "?"
        print(f"  {ts:>4} (line {t+1:>5}) -> {ps:>4} (line {p+1:>5})   x{cnt}")

    if args.out_json:
        report = {
            "total":         total,
            "classes_seen":  n_c,
            "top1":          hit1 / total,
            "top5":          hit5 / total,
            "macro": {
                "precision": macro_p,
                "recall":    macro_r,
                "f1":        macro_f1,
            },
            "weighted": {
                "precision": weighted_p,
                "recall":    weighted_r,
                "f1":        weighted_f1,
            },
            "per_class": [
                {"line": c + 1,
                 "syllable":  labels[c] if c < len(labels) else "?",
                 "acc":       per_class_correct[c] / per_class_total[c]
                              if per_class_total[c] else 0.0,
                 "precision": per_class_metrics[c]["precision"],
                 "recall":    per_class_metrics[c]["recall"],
                 "f1":        per_class_metrics[c]["f1"],
                 "support":   per_class_metrics[c]["support"]}
                for c in classes_in_test
            ],
            "confusion_pairs": [
                {"true_line": t + 1, "true": labels[t] if t < len(labels) else "?",
                 "pred_line": p + 1, "pred": labels[p] if p < len(labels) else "?",
                 "count": cnt}
                for (t, p), cnt in confusion.most_common(200)
            ],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report -> {args.out_json}")


if __name__ == "__main__":
    main()
