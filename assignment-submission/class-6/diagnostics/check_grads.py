"""
Direct gradient check: train ONE step and inspect gradients at every layer.

If gradients are zero or extremely small somewhere, that's where the signal
is getting stuck.

Run from project root:  python check_grads.py
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


def main():
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        Invert(),
    ])
    samples = list_all_samples("image")
    train_s, _ = split_by_copy(samples, test_copy=3)
    ds = SyllableImageDataset(train_s, num_classes=4413, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    torch.manual_seed(42)
    model = SyllableCNN(num_classes=4413, img_size=64)
    model.train()

    x, y = next(iter(loader))
    print(f"Input shape : {x.shape}")
    print(f"Input range : [{x.min():.4f}, {x.max():.4f}], mean={x.mean():.4f}")
    print(f"Labels      : {y[:8].tolist()}...")

    # Forward
    logits = model(x)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}], "
          f"mean={logits.mean():.4f}, std={logits.std():.4f}")

    crit = nn.CrossEntropyLoss()
    loss = crit(logits, y)
    print(f"Loss        : {loss.item():.6f}  (random would be {torch.log(torch.tensor(4413.0)):.4f})")

    # Backward
    loss.backward()

    print(f"\n{'layer':<60}  {'param.mean':>12}  {'param.std':>12}  "
          f"{'grad.mean':>12}  {'grad.std':>12}  {'grad.absmax':>12}")
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:<60}  *** NO GRADIENT ***")
            continue
        print(f"{name:<60}  "
              f"{p.data.mean().item():>12.6f}  "
              f"{p.data.std().item():>12.6f}  "
              f"{p.grad.mean().item():>12.3e}  "
              f"{p.grad.std().item():>12.3e}  "
              f"{p.grad.abs().max().item():>12.3e}")


if __name__ == "__main__":
    main()
