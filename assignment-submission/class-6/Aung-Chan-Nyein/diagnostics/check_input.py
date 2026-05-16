"""
Diagnostic: what does ONE batch from the actual training pipeline look like?
Run from project root:  python check_input.py
"""
import sys
sys.path.insert(0, "recognizer")

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import list_all_samples, split_by_copy, SyllableImageDataset
from train import Invert  # use the exact same Invert class

# Build the exact same train transform as train.py uses (no augment)
img_size = 64
invert = Invert()
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    invert,
])

samples = list_all_samples("image")
train_s, _ = split_by_copy(samples, test_copy=3)
print(f"Train samples: {len(train_s)}")

ds = SyllableImageDataset(train_s, num_classes=4413, transform=train_tf)
loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

x, y = next(iter(loader))
print(f"\nBatch tensor shape: {x.shape}")
print(f"Batch dtype       : {x.dtype}")
print(f"Pixel value range : min={x.min().item():.4f}  max={x.max().item():.4f}")
print(f"Pixel value mean  : {x.mean().item():.4f}")
print(f"Pixel value std   : {x.std().item():.4f}")
print(f"\nLabel tensor shape: {y.shape}")
print(f"Label dtype       : {y.dtype}")
print(f"Labels in batch   : {y.tolist()}")
print(f"Label range       : min={y.min().item()}  max={y.max().item()}")

# Check that different samples are actually different
print(f"\nPer-sample stats (each row should have different values):")
for i in range(min(8, x.size(0))):
    s = x[i, 0]
    print(f"  sample {i}: label={y[i].item():>4}  "
          f"sum={s.sum().item():.1f}  "
          f"nonzero_pixels={(s > 0).sum().item():>4}  "
          f"max_pixel={s.max().item():.3f}")

# Check that same sample loaded twice gives same result
print(f"\nDeterminism check (no augment -> should be identical):")
x1, _ = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
x2, _ = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
diff = (x1 - x2).abs().max().item()
print(f"  max absolute diff between two loads: {diff}")
