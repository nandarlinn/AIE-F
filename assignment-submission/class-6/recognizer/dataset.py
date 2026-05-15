"""
Myanmar Syllable Handwriting Dataset Loader
============================================
Loads images produced by convert2image.py.

Expected folder layout (output of convert2image.py):
    image/
        <user1>/
            1-1.png
            1-2.png
            1-3.png
            2-1.png
            ...
        <user2>/
            ...

Filename convention:  <line_number>-<copy_number>.png
where line_number is 1-based and matches the line in syl.txt (the label).

Author: For Ye Kyaw Thu / LU Lab.
"""

import os
import re
from collections import defaultdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


FILENAME_RE = re.compile(r"^(\d+)-(\d+)\.png$")


# -------------------------------------------------------------------
# Sample listing
# -------------------------------------------------------------------
def list_all_samples(image_dir: str) -> List[Tuple[str, str, int, int]]:
    """
    Walk image_dir and return a list of (filepath, user, line_idx_1based, copy_idx_1based).

    line_idx_1based corresponds to the 1-based line number in syl.txt.
    """
    samples = []
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    for user in sorted(os.listdir(image_dir)):
        user_path = os.path.join(image_dir, user)
        if not os.path.isdir(user_path):
            continue
        for fname in sorted(os.listdir(user_path)):
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            line_idx = int(m.group(1))
            copy_idx = int(m.group(2))
            samples.append((os.path.join(user_path, fname), user, line_idx, copy_idx))
    return samples


def load_labels(syl_file: str) -> List[str]:
    """Read syl.txt and return list of syllables. Index 0 == line 1 in the file."""
    with open(syl_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# -------------------------------------------------------------------
# Train / Test split
# -------------------------------------------------------------------
def split_by_copy(samples, test_copy: int = 3):
    """
    Split: every <line>-<test_copy>.png goes to test, all other copies go to train.

    Use this when you only have ONE writer (your case right now).
    Pros: simple, deterministic, every class is in both train and test.
    Cons: same writer in both splits, so accuracy is OPTIMISTIC about new writers.
    """
    train, test = [], []
    for s in samples:
        if s[3] == test_copy:
            test.append(s)
        else:
            train.append(s)
    return train, test


def split_by_writer(samples, test_users: List[str]):
    """
    Split: samples whose user is in `test_users` go to test, rest to train.

    Use this when you have MULTIPLE writers (recommended for real evaluation).
    Pros: realistic measure of how the model generalizes to new handwriting.
    Cons: needs at least a few writers; some classes may be missing from test.
    """
    test_users = set(test_users)
    train, test = [], []
    for s in samples:
        (test if s[1] in test_users else train).append(s)
    return train, test


# -------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------
class SyllableImageDataset(Dataset):
    """
    Each item is (image_tensor, label_int_0based).

    label_int_0based = line_idx_1based - 1   (so it can index into a list/Linear layer)
    """

    def __init__(self, samples, num_classes: int, transform=None):
        self.samples = samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, user, line_idx, copy_idx = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform is not None:
            img = self.transform(img)
        label = line_idx - 1  # to 0-based
        return img, label


# -------------------------------------------------------------------
# Quick stats helper
# -------------------------------------------------------------------
def print_dataset_stats(samples, name="dataset"):
    by_user = defaultdict(int)
    by_class = defaultdict(int)
    for _, user, line_idx, _ in samples:
        by_user[user] += 1
        by_class[line_idx] += 1
    print(f"\n--- {name} ---")
    print(f"  total samples : {len(samples)}")
    print(f"  unique users  : {len(by_user)}")
    print(f"  unique classes: {len(by_class)}")
    if by_class:
        cps = list(by_class.values())
        print(f"  samples/class : min={min(cps)}, max={max(cps)}, "
              f"avg={sum(cps)/len(cps):.2f}")
    print(f"  by user       : {dict(by_user)}")


# Allow `python dataset.py` to print a quick summary
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", default="image")
    ap.add_argument("--syl_file", default="syl.txt")
    args = ap.parse_args()

    samples = list_all_samples(args.image_dir)
    labels = load_labels(args.syl_file)
    print(f"Total syllables in {args.syl_file}: {len(labels)}")
    print_dataset_stats(samples, "all samples")

    tr, te = split_by_copy(samples, test_copy=3)
    print_dataset_stats(tr, "train (copies 1, 2)")
    print_dataset_stats(te, "test  (copy 3)")
