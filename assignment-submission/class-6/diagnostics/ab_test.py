"""
Direct A/B test: same training code on 100 classes vs all 4413 classes.

Purpose: isolate WHICH variable change breaks training. Both runs use
identical data loading, model construction, optimizer, and loop.

Run from project root:  python ab_test.py
"""
import sys
sys.path.insert(0, "recognizer")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import list_all_samples, split_by_copy, SyllableImageDataset
from model import SyllableCNN
from train import Invert


def run_experiment(name, samples, num_classes_in_model, n_epochs=8):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  samples in this run    : {len(samples)}")
    print(f"  num_classes (model out): {num_classes_in_model}")
    print(f"{'='*70}")

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        Invert(),
    ])
    ds = SyllableImageDataset(samples, num_classes_in_model, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    torch.manual_seed(42)
    model = SyllableCNN(num_classes=num_classes_in_model, img_size=64)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print(f"\n  {'epoch':>5}  {'loss':>7}  {'top1':>7}")
    for ep in range(1, n_epochs + 1):
        model.train()
        total, hit, lsum = 0, 0, 0.0
        for x, y in loader:
            logits = model(x)
            loss = crit(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            bs = y.size(0)
            total += bs
            lsum += loss.item() * bs
            hit += (logits.argmax(1) == y).sum().item()
        print(f"  {ep:>5}  {lsum/total:>7.4f}  {hit/total*100:>6.2f}%")


def main():
    all_samples = list_all_samples("image")
    train_full, _ = split_by_copy(all_samples, test_copy=3)

    # First 100 classes only - copies 1 and 2
    train_100 = [s for s in train_full if s[2] <= 100]

    # Three runs to triangulate exactly where it breaks:

    # Run A: 100 classes, model has 100 outputs (= proven working baseline)
    run_experiment(
        "A: 100 classes, model_classes=100",
        train_100,
        num_classes_in_model=100,
    )

    # Run B: 100 classes, model has 4413 outputs (only labels 0..99 ever appear)
    # If A works and B fails, the problem is the size of the output layer.
    run_experiment(
        "B: 100 classes data, model_classes=4413",
        train_100,
        num_classes_in_model=4413,
    )

    # Run C: full data, full classes
    run_experiment(
        "C: ALL classes, model_classes=4413  (the actual training)",
        train_full,
        num_classes_in_model=4413,
        n_epochs=4,   # we already know this fails - 4 is plenty
    )


if __name__ == "__main__":
    main()
