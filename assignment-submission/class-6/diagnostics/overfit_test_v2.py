"""
Same overfit test as before, but using the EXACT data pipeline (dataset.py
+ Invert + same transforms) that train.py uses.

If THIS still overfits, then the data pipeline is fine and the bug is in
something else train.py does (e.g. the optimizer setup, dataloader workers).

Run from project root:  python overfit_test_v2.py
"""
import sys
sys.path.insert(0, "recognizer")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import list_all_samples, SyllableImageDataset
from model import SyllableCNN
from train import Invert

# Same transform as train.py uses (no augment)
img_size = 64
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    Invert(),
])

# Get all samples, then keep only the first 100 classes (line 1..100)
samples = list_all_samples("image")
samples_100 = [s for s in samples if s[2] <= 100]   # s[2] is line_idx_1based
print(f"Samples in first 100 classes: {len(samples_100)}")

# Use SyllableImageDataset with num_classes=4413 so labels are 0-based 0..99
ds = SyllableImageDataset(samples_100, num_classes=4413, transform=train_tf)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

# Build model with 4413 classes (same as train.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SyllableCNN(num_classes=4413, img_size=64).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

print(f"\n{'epoch':>5}  {'loss':>7}  {'top1':>7}")
for ep in range(1, 21):
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
