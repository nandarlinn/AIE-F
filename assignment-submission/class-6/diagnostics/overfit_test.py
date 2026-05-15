"""
Sanity test: can the model OVERFIT a tiny subset (100 classes, 3 copies each)?

A working classifier MUST be able to memorize a small training set.
If train accuracy stays low after 30 epochs on just 300 images, the model
architecture is too aggressive for our sparse handwritten data.

Run from the project root (where image/ lives):
    python overfit_test.py
"""

import os
import glob
import re
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Reuse our model
sys.path.insert(0, "recognizer")
from model import SyllableCNN

FILENAME_RE = re.compile(r"^(\d+)-(\d+)\.png$")


class TinyDS(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.tf(img), label


class Invert:
    def __call__(self, t):
        return 1.0 - t


def main():
    # Collect samples
    files = sorted(glob.glob("image/*/*.png"))
    by_class = defaultdict(list)
    for f in files:
        m = FILENAME_RE.match(os.path.basename(f))
        if m:
            cls = int(m.group(1))
            by_class[cls].append(f)

    # Take first 100 classes, ALL copies for training
    chosen = sorted(by_class.keys())[:100]
    samples = []
    for new_lbl, cls in enumerate(chosen):
        for f in by_class[cls]:
            samples.append((f, new_lbl))
    print(f"Overfitting on {len(samples)} images across {len(chosen)} classes")

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        Invert(),
    ])
    ds = TinyDS(samples, tf)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyllableCNN(num_classes=100, img_size=64).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    crit = nn.CrossEntropyLoss()

    print(f"\n{'epoch':>5}  {'loss':>7}  {'top1':>7}")
    for ep in range(1, 31):
        model.train()
        total, hit, lsum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            bs = y.size(0)
            total += bs
            lsum += loss.item() * bs
            hit += (logits.argmax(1) == y).sum().item()
        print(f"{ep:>5}  {lsum/total:>7.4f}  {hit/total*100:>6.2f}%")


if __name__ == "__main__":
    main()
