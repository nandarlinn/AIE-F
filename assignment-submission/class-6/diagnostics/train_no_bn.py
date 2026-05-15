"""
Test if BatchNorm is the problem.

Replaces the BatchNorm2d layers in SyllableCNN with GroupNorm,
which doesn't depend on batch statistics. If this trains, BatchNorm was
the culprit.

Run from project root:  python train_no_bn.py
"""
import sys
sys.path.insert(0, "recognizer")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import list_all_samples, split_by_copy, SyllableImageDataset
from train import Invert


class SyllableCNN_GN(nn.Module):
    """Same architecture as SyllableCNN but with GroupNorm replacing BatchNorm."""
    def __init__(self, num_classes, img_size=64, dropout=0.3):
        super().__init__()
        # GroupNorm with num_groups dividing the channels evenly
        def gn(c):
            return nn.GroupNorm(num_groups=8, num_channels=c)

        self.features = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), gn(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), gn(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), gn(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256, 3, padding=1), gn(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        feat = 256 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    img_size = 64
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        Invert(),
    ])

    samples = list_all_samples("image")
    train_s, _ = split_by_copy(samples, test_copy=3)
    print(f"Train samples: {len(train_s)}")

    ds = SyllableImageDataset(train_s, num_classes=4413, transform=tf)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyllableCNN_GN(num_classes=4413, img_size=64).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print(f"\n{'epoch':>5}  {'loss':>7}  {'top1':>7}")
    # Just 5 epochs - if it's working we'll see immediately
    for ep in range(1, 6):
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
