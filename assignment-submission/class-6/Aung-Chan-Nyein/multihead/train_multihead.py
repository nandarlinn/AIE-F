"""
Train the multi-task Myanmar syllable recognizer
=================================================
The model predicts 7 structural components per image. We sum the
cross-entropy loss across all heads. At evaluation time we compute:

  - per-head top-1 accuracy
  - full-syllable accuracy (all 7 heads must agree with the true components)
  - look-up syllable accuracy (after mapping predicted components back to the
    closest syllable in syl.txt - some component-tuples are shared by 2-5
    syllables)

Usage:
    python train_multihead.py --epochs 60 --out_dir runs/mt1
"""
import os
import sys
import json
import time
import argparse
import random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Reuse helpers from the recognizer/ directory
sys.path.insert(0, "recognizer")
from dataset import list_all_samples, split_by_copy, split_by_writer

from decompose import (
    decompose, recompose, build_vocabularies, syllable_to_label_indices,
)
from model_multihead import SyllableMultiHead, count_parameters


# ---------- helpers ----------
class Invert:
    def __call__(self, t):
        return 1.0 - t


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- dataset ----------
class MultiHeadDataset(Dataset):
    """
    Returns (image_tensor, {head: label_idx, ...}, line_idx_0based).
    """
    def __init__(self, samples, syl_labels, vocabs, transform):
        self.samples = samples
        self.syl_labels = syl_labels    # list of syllables (index 0 = line 1)
        self.vocabs = vocabs
        self.transform = transform
        # Precompute label dicts per line
        self._label_cache = {}
        for line_idx_1based in range(1, len(syl_labels) + 1):
            syl = syl_labels[line_idx_1based - 1]
            self._label_cache[line_idx_1based] = syllable_to_label_indices(syl, vocabs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, user, line_idx, copy_idx = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        labels = self._label_cache[line_idx]
        return img, labels, line_idx - 1   # 0-based for convenience


def collate(batch):
    """Stack images, and stack each head's labels into a tensor."""
    imgs = torch.stack([b[0] for b in batch])
    lines = torch.tensor([b[2] for b in batch], dtype=torch.long)
    heads = batch[0][1].keys()
    label_tensors = {
        h: torch.tensor([b[1][h] for b in batch], dtype=torch.long)
        for h in heads
    }
    return imgs, label_tensors, lines


# ---------- vocab loading / saving ----------
def save_vocabs(vocabs, path):
    """Vocabs contain tuples (for medials/vowels/tones) - serialize as lists."""
    safe = {}
    for k, v in vocabs.items():
        out = []
        for x in v:
            if isinstance(x, tuple):
                out.append(list(x))
            else:
                out.append(x)
        safe[k] = out
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)


# ---------- inference helpers ----------
def build_component_to_syllable_map(syl_labels, vocabs):
    """
    Group syllables by their (base, medials, vowels, final, stack, tones, asat)
    tuple. For each tuple, store the list of (line_idx_1based, syllable_str)
    that share it. We pick the lowest line_idx as the canonical prediction.
    """
    groups = defaultdict(list)
    for i, syl in enumerate(syl_labels):
        d = decompose(syl)
        key = (d["base"], d["medials"], d["vowels"], d["final"],
               d["stack"], d["tones"], d["asat"])
        groups[key].append((i + 1, syl))   # 1-based line index
    return groups


def predicted_components_to_key(pred_indices, vocabs):
    """Convert {head: label_idx} -> the tuple key used in the map above."""
    def get(head):
        v = vocabs[head][pred_indices[head]]
        return v
    return (
        get("base"),
        tuple(get("medials")) if isinstance(get("medials"), list) else get("medials"),
        tuple(get("vowels"))  if isinstance(get("vowels"),  list) else get("vowels"),
        get("final"),
        get("stack"),
        tuple(get("tones"))   if isinstance(get("tones"),   list) else get("tones"),
        bool(get("asat")),
    )


# ---------- training loop ----------
def train_one_epoch(model, loader, optim, head_names, device, log_every=50):
    model.train()
    head_total   = {h: 0 for h in head_names}
    head_correct = {h: 0 for h in head_names}
    loss_sum, total = 0.0, 0
    crit = nn.CrossEntropyLoss()

    t0 = time.time()
    for step, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = {h: v.to(device, non_blocking=True) for h, v in labels.items()}

        logits = model(imgs)
        loss = sum(crit(logits[h], labels[h]) for h in head_names)

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs
        for h in head_names:
            pred = logits[h].argmax(dim=1)
            head_correct[h] += (pred == labels[h]).sum().item()
            head_total[h]   += bs

        if step % log_every == 0:
            print(f"    step {step:4d}/{len(loader)}  loss={loss.item():.4f}")

    dt = time.time() - t0
    avg_loss = loss_sum / total
    accs = {h: head_correct[h] / head_total[h] for h in head_names}
    return avg_loss, accs, dt


@torch.no_grad()
def evaluate(model, loader, head_names, vocabs, comp_map, device):
    model.eval()
    head_total   = {h: 0 for h in head_names}
    head_correct = {h: 0 for h in head_names}
    all_components_correct = 0   # all heads agree
    syllable_correct       = 0   # lookup map gives the right syllable line
    total = 0
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0

    for imgs, labels, lines in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels_dev = {h: v.to(device, non_blocking=True) for h, v in labels.items()}

        logits = model(imgs)
        loss = sum(crit(logits[h], labels_dev[h]) for h in head_names)

        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs

        # Per-head accuracy
        pred = {h: logits[h].argmax(dim=1) for h in head_names}
        all_right = torch.ones(bs, dtype=torch.bool, device=device)
        for h in head_names:
            ok = (pred[h] == labels_dev[h])
            head_correct[h] += ok.sum().item()
            head_total[h]   += bs
            all_right &= ok
        all_components_correct += all_right.sum().item()

        # Syllable-level accuracy: reconstruct predicted components, look up
        # in the comp_map, see if the true line is in the candidate set.
        pred_cpu = {h: pred[h].cpu().tolist() for h in head_names}
        true_lines = lines.tolist()  # 0-based
        for i in range(bs):
            indices = {h: pred_cpu[h][i] for h in head_names}
            key = predicted_components_to_key(indices, vocabs)
            cands = comp_map.get(key, [])
            if any((line_1b - 1) == true_lines[i] for line_1b, _ in cands):
                syllable_correct += 1

    return {
        "loss":        loss_sum / max(1, total),
        "head_acc":    {h: head_correct[h] / max(1, head_total[h])
                        for h in head_names},
        "all_correct": all_components_correct / max(1, total),
        "syl_acc":     syllable_correct       / max(1, total),
    }


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir",   default="image")
    ap.add_argument("--syl_file",    default="syl.txt")
    ap.add_argument("--split",       choices=["copy", "writer"], default="copy")
    ap.add_argument("--test_copy",   type=int, default=3)
    ap.add_argument("--test_users",  nargs="*", default=[])
    ap.add_argument("--img_size",    type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=60)
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out_dir",     default="runs/mt1")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Labels and vocabularies ----
    with open(args.syl_file, "r", encoding="utf-8") as f:
        syl_labels = [ln.strip() for ln in f if ln.strip()]
    print(f"Syllables: {len(syl_labels)}")

    vocabs = build_vocabularies(syl_labels)
    head_sizes = {
        "base":    len(vocabs["base"]),
        "medials": len(vocabs["medials"]),
        "vowels":  len(vocabs["vowels"]),
        "final":   len(vocabs["final"]),
        "stack":   len(vocabs["stack"]),
        "tones":   len(vocabs["tones"]),
        "asat":    len(vocabs["asat"]),
    }
    head_names = list(head_sizes.keys())
    print("Head sizes:", head_sizes)

    save_vocabs(vocabs, os.path.join(args.out_dir, "vocabs.json"))
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": syl_labels}, f, ensure_ascii=False, indent=2)

    comp_map = build_component_to_syllable_map(syl_labels, vocabs)
    multi_groups = sum(1 for v in comp_map.values() if len(v) > 1)
    print(f"Distinct component-tuples: {len(comp_map)}  "
          f"(of which {multi_groups} are shared by 2+ syllables)")

    # ---- Samples + split ----
    samples = list_all_samples(args.image_dir)
    if args.split == "copy":
        train_s, test_s = split_by_copy(samples, test_copy=args.test_copy)
    else:
        if not args.test_users:
            sys.exit("--split writer needs --test_users")
        train_s, test_s = split_by_writer(samples, args.test_users)
    print(f"Train: {len(train_s)}  Test: {len(test_s)}")

    # ---- Transforms ----
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        Invert(),
    ])
    train_ds = MultiHeadDataset(train_s, syl_labels, vocabs, tf)
    test_ds  = MultiHeadDataset(test_s,  syl_labels, vocabs, tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate)

    # ---- Model ----
    model = SyllableMultiHead(head_sizes, img_size=args.img_size).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Train ----
    history = []
    best_syl = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss, tr_accs, tr_time = train_one_epoch(
            model, train_loader, optim, head_names, device,
        )
        ev = evaluate(model, test_loader, head_names, vocabs, comp_map, device)
        scheduler.step()

        print(f"  train loss={tr_loss:.4f}  time={tr_time:.1f}s")
        for h in head_names:
            print(f"    head {h:>8}:  train={tr_accs[h]*100:6.2f}%  "
                  f"test={ev['head_acc'][h]*100:6.2f}%")
        print(f"  all-components-correct (test): {ev['all_correct']*100:.2f}%")
        print(f"  syllable lookup acc (test)   : {ev['syl_acc']*100:.2f}%")

        log = {
            "epoch":           epoch,
            "train_loss":      tr_loss,
            "train_head_acc":  tr_accs,
            "test_loss":       ev["loss"],
            "test_head_acc":   ev["head_acc"],
            "all_correct":     ev["all_correct"],
            "syl_acc":         ev["syl_acc"],
        }
        history.append(log)

        torch.save({
            "model_state":  model.state_dict(),
            "head_sizes":   head_sizes,
            "img_size":     args.img_size,
            "args":         vars(args),
            "epoch":        epoch,
        }, last_path)

        if ev["syl_acc"] > best_syl:
            best_syl = ev["syl_acc"]
            torch.save({
                "model_state":  model.state_dict(),
                "head_sizes":   head_sizes,
                "img_size":     args.img_size,
                "args":         vars(args),
                "epoch":        epoch,
                "test_syl_acc": ev["syl_acc"],
            }, best_path)
            print(f"  --> new best syllable accuracy, saved {best_path}")

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

    print(f"\nBest test syllable accuracy: {best_syl*100:.2f}%")


if __name__ == "__main__":
    main()
