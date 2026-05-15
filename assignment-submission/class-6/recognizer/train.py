"""
Train a Myanmar Syllable Handwriting Recognizer
===============================================
End-to-end training script.

Typical usage (single writer, you only):
    python train.py --image_dir image --syl_file syl.txt --epochs 60

Multi-writer (later, when you have more contributors):
    python train.py --split writer --test_users alice bob

Author: For Ye Kyaw Thu / LU Lab.
"""

import os
import sys
import json
import time
import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    list_all_samples, load_labels,
    split_by_copy, split_by_writer,
    SyllableImageDataset, print_dataset_stats,
)
from model import SyllableCNN, count_parameters


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# Transforms
# -------------------------------------------------------------------
class Invert:
    """
    Invert a [0,1] tensor: strokes -> 1, background -> 0.
    Defined at module level (not a lambda) so it pickles for DataLoader workers.
    """
    def __call__(self, t):
        return 1.0 - t


def build_transforms(img_size: int, augment: bool):
    """
    Train transforms suited to thin-stroke binary handwriting on small images.

    Important details for THIS dataset:
      - Images are 64x64 pure black-on-white (only 2 unique pixel values).
      - Strokes are 1-2 px wide. With NEAREST interpolation and aggressive
        affine, strokes break up between epochs. So we use BILINEAR.
      - We INVERT the image (strokes -> 1, background -> 0). For thin lines
        this gives a much better signal than the reverse.
      - Augmentation is gentle (degrees=5, translate=2%, scale=98%-102%).

    NOTE: NO horizontal flip -- Myanmar script is not flip-symmetric.
    """
    invert = Invert()

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                fill=255,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),  # -> [0, 1] float, shape (1, H, W)
            invert,                  # strokes -> bright (1), bg -> dark (0)
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            invert,
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        invert,
    ])
    return train_tf, eval_tf


# -------------------------------------------------------------------
# Top-K accuracy
# -------------------------------------------------------------------
def topk_correct(logits, labels, ks=(1, 5)):
    """Returns dict {k: num_correct_in_top_k}."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)         # (B, maxk)
    pred = pred.t()                            # (maxk, B)
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    out = {}
    for k in ks:
        out[k] = correct[:k].any(dim=0).sum().item()
    return out


# -------------------------------------------------------------------
# Train / eval loops
# -------------------------------------------------------------------
def train_one_epoch(model, loader, optim, criterion, device, log_every=50):
    model.train()
    total, hit1, hit5, loss_sum = 0, 0, 0, 0.0
    t0 = time.time()
    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        bs = labels.size(0)
        total    += bs
        loss_sum += loss.item() * bs
        c = topk_correct(logits.detach(), labels, ks=(1, 5))
        hit1 += c[1]
        hit5 += c[5]

        if step % log_every == 0:
            print(f"    step {step:4d}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"lr={optim.param_groups[0]['lr']:.5f}")

    dt = time.time() - t0
    return {
        "loss"  : loss_sum / max(1, total),
        "top1"  : hit1 / max(1, total),
        "top5"  : hit5 / max(1, total),
        "time_s": dt,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, hit1, hit5, loss_sum = 0, 0, 0, 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        bs = labels.size(0)
        total    += bs
        loss_sum += loss.item() * bs
        c = topk_correct(logits, labels, ks=(1, 5))
        hit1 += c[1]
        hit5 += c[5]

    return {
        "loss": loss_sum / max(1, total),
        "top1": hit1 / max(1, total),
        "top5": hit5 / max(1, total),
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train Myanmar syllable recognizer (image-based CNN)"
    )

    # Data
    ap.add_argument("--image_dir", default="image",
                    help="Folder produced by convert2image.py")
    ap.add_argument("--syl_file",  default="syl.txt",
                    help="Syllable label list, one per line")

    # Split strategy
    ap.add_argument("--split", choices=["copy", "writer"], default="copy",
                    help="copy: hold out one copy per syllable (single writer). "
                         "writer: hold out entire writers (multi-writer).")
    ap.add_argument("--test_copy", type=int, default=3,
                    help="Which copy index to hold out as test (used with --split copy)")
    ap.add_argument("--test_users", nargs="*", default=[],
                    help="Users held out as test (used with --split writer)")

    # Model / image
    ap.add_argument("--img_size", type=int, default=64)

    # Training
    ap.add_argument("--epochs",       type=int,   default=120,
                    help="With only 2 samples/class, the model needs many "
                         "epochs to converge. Default raised to 120.")
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",  type=int,   default=2)
    ap.add_argument("--augment",      action="store_true",
                    help="Enable data augmentation. With only 2-3 samples per "
                         "class, augmentation usually HURTS. Leave off unless "
                         "you have many samples per class.")
    ap.add_argument("--seed",         type=int,   default=42)

    # Output
    ap.add_argument("--out_dir", default="runs/exp1",
                    help="Where to save checkpoints, logs, label map")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load labels & samples ----
    labels      = load_labels(args.syl_file)
    num_classes = len(labels)
    print(f"Number of classes (syllables): {num_classes}")

    samples = list_all_samples(args.image_dir)
    print_dataset_stats(samples, "all samples")

    # ---- Split ----
    if args.split == "copy":
        train_s, test_s = split_by_copy(samples, test_copy=args.test_copy)
        print(f"\nSplit by copy: copy {args.test_copy} held out for test")
    else:
        if not args.test_users:
            sys.exit("ERROR: --split writer requires --test_users user1 user2 ...")
        train_s, test_s = split_by_writer(samples, args.test_users)
        print(f"\nSplit by writer: held out users = {args.test_users}")

    print_dataset_stats(train_s, "train")
    print_dataset_stats(test_s,  "test")

    if not train_s or not test_s:
        sys.exit("ERROR: train or test split is empty. Check your data and split args.")

    # ---- Datasets / loaders ----
    train_tf, eval_tf = build_transforms(args.img_size, augment=args.augment)
    if args.augment:
        print("Augmentation: ON")
    else:
        print("Augmentation: OFF (recommended for tiny datasets like 2 samples/class)")
    train_ds = SyllableImageDataset(train_s, num_classes, transform=train_tf)
    test_ds  = SyllableImageDataset(test_s,  num_classes, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = SyllableCNN(num_classes=num_classes, img_size=args.img_size).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs,
    )

    # ---- Save label map for inference ----
    label_map_path = os.path.join(args.out_dir, "labels.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
    print(f"Saved label map -> {label_map_path}")

    # ---- Train ----
    history = []
    best_top1 = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        tr = train_one_epoch(model, train_loader, optim, criterion, device)
        ev = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log = {
            "epoch":      epoch,
            "lr":         optim.param_groups[0]["lr"],
            "train_loss": tr["loss"],
            "train_top1": tr["top1"],
            "train_top5": tr["top5"],
            "test_loss":  ev["loss"],
            "test_top1":  ev["top1"],
            "test_top5":  ev["top5"],
            "time_s":     tr["time_s"],
        }
        history.append(log)
        print(f"  train: loss={tr['loss']:.4f}  top1={tr['top1']*100:.2f}%  "
              f"top5={tr['top5']*100:.2f}%  ({tr['time_s']:.1f}s)")
        print(f"  test : loss={ev['loss']:.4f}  top1={ev['top1']*100:.2f}%  "
              f"top5={ev['top5']*100:.2f}%")

        # Always save "last"
        torch.save({
            "model_state": model.state_dict(),
            "args":        vars(args),
            "epoch":       epoch,
            "num_classes": num_classes,
            "img_size":    args.img_size,
        }, last_path)

        # Save "best"
        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            torch.save({
                "model_state": model.state_dict(),
                "args":        vars(args),
                "epoch":       epoch,
                "num_classes": num_classes,
                "img_size":    args.img_size,
                "test_top1":   ev["top1"],
                "test_top5":   ev["top5"],
            }, best_path)
            print(f"  --> new best top1, saved {best_path}")

        # Save history each epoch (cheap, JSON)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best test top-1 = {best_top1*100:.2f}%")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()
