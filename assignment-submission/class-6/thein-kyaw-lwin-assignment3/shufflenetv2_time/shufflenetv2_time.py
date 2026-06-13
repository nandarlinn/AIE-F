# ============================================================================
# Myanmar Syllable Handwriting — ShuffleNetV2 x1.0
# Multi-class Classification: 1000 Myanmar Syllables
# Dataset: "time" colour-mode PNGs (3 samples per class, 3000 total)
# Strategy: 3-Fold Stratified CV × 3 Seeds | Two-Stage Fine-Tuning
# Kaggle GPU (T4) + WandB logging
# ============================================================================


# %% Cell 1: Install Dependencies & WandB Setup
# -----------------------------------------------------------------------
import subprocess, sys, os
from pathlib import Path

wandb_enabled = False
WANDB_KEY_CANDIDATES = [
    Path("/kaggle/input/datasets/theinkyawlwin/wandbapi/wandb.txt"),  # Kaggle v2 path
    Path("/kaggle/input/wandbapi/wandb.txt"),                         # Classic path
]
try:
    for key_path in WANDB_KEY_CANDIDATES:
        if key_path.exists():
            api_key = key_path.read_text().strip()
            os.environ["WANDB_API_KEY"] = api_key
            wandb_enabled = True
            print(f"[✓] W&B API key loaded from: {key_path}")
            break
    else:
        print("[⚠] W&B key file not found — training will continue without logging")
except Exception as exc:
    print(f"[⚠] W&B disabled: {exc}")


# %% Cell 2: Imports
import random
import copy
import time
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    top_k_accuracy_score
)
import wandb

# GPU info — pushed with --accelerator NvidiaTeslaT4 so T4 is guaranteed.
if torch.cuda.is_available():
    _cap, _name = torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0)
    print(f"[✓] GPU: {_name} (sm_{_cap[0]}{_cap[1]})")
else:
    print("[⚠] No GPU detected — running on CPU (will be slow)")


# %% Cell 3: Configuration
CONFIG = {
    # --- Project ---
    "project_name":  "myanmar-hw-shufflenetv2",
    "model_name":    "shufflenet_v2_x1_0_time",
    "num_classes":   1000,           # one class per Myanmar syllable
    "color_mode":    "time",         # drawing-speed colour encoding

    # --- Cross-Validation ---
    # 3-fold is the natural choice: exactly 3 samples/class splits into
    # 2 train + 1 val per fold cleanly, with no class left unrepresented.
    # 5-fold cannot guarantee every class appears in every val split with n=3.
    "n_folds":       3,
    "fold_seed":     42,
    "train_seeds":   [42, 123, 456],  # 3 folds × 3 seeds = 9 runs total

    # --- Data ---
    # Native resolution is 128×128. Upscaling to 224 adds no real information
    # (bilinear interpolation on 128→224 just blurs the strokes).
    # ShuffleNetV2 works well at 128×128; pretrained weights still transfer.
    "img_size":      128,
    "batch_size":    64,              # can increase batch size at lower res
    "num_workers":   2,

    # --- Stage 1: Head Only (warm-up) ---
    # With only 2 samples/class the head converges very fast — 6 epochs is enough.
    # More epochs here risks overfitting the head to frozen ImageNet features.
    "stage1_lr":     1e-3,
    "stage1_epochs": 6,

    # --- Stage 2: Full Fine-Tuning ---
    # Discriminative LRs: backbone gets a lower LR than the head to prevent
    # catastrophic forgetting of pretrained features on tiny data.
    "stage2_lr_backbone": 5e-6,   # conservative — backbone already knows shapes
    "stage2_lr_head":     2e-5,   # head adapts faster to new 1000-class space
    "stage2_epochs": 50,
    "patience":      12,           # monitor top-5 acc (less noisy than val_loss)
    "weight_decay":  1e-4,
    "label_smoothing": 0.1,        # regularise overconfidence with 1000 classes

    # --- ImageNet Normalisation ---
    # Single-colour PNGs are saved as RGB by convert2image.py,
    # so ImageNet stats still apply.
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}


# %% Cell 4: Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# %% Cell 5: Locate & Unzip Dataset
# The Kaggle dataset uploads directories as .zip files.
# We unzip them on first run, then use the extracted folder.

DATASET_ROOT_CANDIDATES = [
    "/kaggle/input/myanmar-syllable-handwriting",
    "/kaggle/input/datasets/theinkyawlwin/myanmar-syllable-handwriting",
    "/Users/tklwin/GithubRepos/mm-hw-collector",   # local dev fallback
]

def find_dataset_root():
    for root in DATASET_ROOT_CANDIDATES:
        if os.path.isdir(root):
            return root
    raise FileNotFoundError("Dataset root not found. Check dataset_sources in kernel-metadata.json.")

def ensure_unzipped(dataset_root, color_mode):
    """
    Kaggle stores each uploaded directory as <name>.zip inside the dataset root.
    Unzip if the extracted folder doesn't already exist.
    Returns the path to the extracted ImageFolder-compatible directory.
    """
    zip_path   = os.path.join(dataset_root, f"{color_mode}.zip")
    out_dir    = os.path.join("/kaggle/working", color_mode)

    # If we're running locally and the folder already exists, return it directly
    local_dir = os.path.join(dataset_root, color_mode)
    if os.path.isdir(local_dir):
        print(f"[✓] Using local directory: {local_dir}")
        return local_dir

    if not os.path.isdir(out_dir):
        print(f"[→] Unzipping {zip_path} → {out_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(f"[✓] Unzipped to {out_dir}")
    else:
        print(f"[✓] Already unzipped: {out_dir}")

    return out_dir

dataset_root = find_dataset_root()
IMAGES_DIR   = ensure_unzipped(dataset_root, CONFIG["color_mode"])

# Verify structure: ImageFolder expects IMAGES_DIR/<user>/<class_idx>-<sample>.png
# Our layout is:   IMAGES_DIR/TheinKyawLwin/<idx>-<sample>.png  (flat per-user folder)
# We need: one subfolder per class label.
# The filenames already encode the class: "<syllable_index>-<sample>.png"
# We'll reorganise into a proper class-subfolder structure in /kaggle/working.

ORGANISED_DIR = "/kaggle/working/organised_time"

def organise_dataset(raw_dir, out_dir):
    """
    Converts the flat structure:
        raw_dir/<user>/<idx>-<sample>.png
    into an ImageFolder-compatible structure:
        out_dir/<idx>/<user>_<idx>-<sample>.png
    where <idx> is zero-padded to 4 digits so lexicographic == numeric order.
    """
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"[✓] Organised dataset already at: {out_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    import shutil, re

    pattern = re.compile(r"^(\d+)-\d+\.png$")
    moved = 0

    for user_dir in os.scandir(raw_dir):
        if not user_dir.is_dir():
            continue
        for img_file in os.scandir(user_dir.path):
            m = pattern.match(img_file.name)
            if not m:
                continue
            class_idx  = int(m.group(1))            # 1-based syllable index
            class_name = f"{class_idx:04d}"         # zero-padded → 0001 … 1000
            class_dir  = os.path.join(out_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            dst = os.path.join(class_dir, f"{user_dir.name}_{img_file.name}")
            shutil.copy2(img_file.path, dst)
            moved += 1

    print(f"[✓] Organised {moved} images into {out_dir}")

organise_dataset(IMAGES_DIR, ORGANISED_DIR)
print(f"[✓] Dataset ready at: {ORGANISED_DIR}")
print(f"    Classes: {len(os.listdir(ORGANISED_DIR))}")


# %% Cell 6: Data Transforms
# Pre-processing notes for handwriting on white background:
#
#   INVERSION: The 'single' images are black ink on white (pixel≈255 bg,
#   pixel≈0 strokes). ImageNet-pretrained models were trained on natural
#   images where foreground objects are typically brighter than background.
#   Inverting (255 - pixel) makes the ink bright on dark — closer to the
#   pretrained feature distribution and lets the model focus on strokes.
#
#   NO HORIZONTAL FLIP: Myanmar script is not horizontally symmetric.
#
#   NO COLOUR JITTER: 'single' mode uses a fixed stroke colour; jitter
#   would destroy the meaning in 'stroke'/'time' modes as well.
#
#   AFFINE / PERSPECTIVE: Captures natural writing variation — slight tilt,
#   shift, scale. Keep distortion_scale small at 128×128 to avoid
#   degrading already-small images.
#
#   NO UPSCALE: Images are native 128×128; we resize to img_size=128 (no-op
#   in practice, but keeps the pipeline consistent).

def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                fill=255,                   # white bg fill during affine
            ),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.3, fill=255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),  # invert: black ink → bright
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),  # same inversion at test time
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ])


# %% Cell 7: Model Definition
def create_model(num_classes):
    model = models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# %% Cell 8: Training & Validation Functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.concatenate(all_probs, axis=0)
    total = len(all_labels)

    top1  = accuracy_score(all_labels, all_preds)
    top5  = top_k_accuracy_score(all_labels, all_probs, k=5, labels=list(range(num_classes)))
    top10 = top_k_accuracy_score(all_labels, all_probs, k=10, labels=list(range(num_classes)))

    metrics = {
        "val_loss":         running_loss / total,
        "val_top1_acc":     top1,
        "val_top5_acc":     top5,
        "val_top10_acc":    top10,
        "val_f1_macro":     f1_score(all_labels, all_preds, average="macro",    zero_division=0),
        "val_f1_weighted":  f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }
    return metrics, all_preds, all_labels


# %% Cell 9: Early Stopping
# Monitor top-5 accuracy (higher = better) rather than val_loss.
# With only 1 val sample per class, val_loss is noisy; top-5 is more stable
# and directly reflects what we care about for this recognition task.
class EarlyStopping:
    def __init__(self, patience=12):
        self.patience  = patience
        self.counter   = 0
        self.best_score = None          # top-5 acc, higher is better
        self.best_model_state = None

    def __call__(self, top5_acc, model):
        if self.best_score is None or top5_acc > self.best_score:
            self.best_score = top5_acc
            self.counter    = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


# %% Cell 10: Two-Stage Training Run
def train_single_run(model, train_loader, val_loader, device, fold, seed):
    # Label smoothing: with 1000 classes and tiny data, cross-entropy drives
    # logits toward ±inf (overconfidence). Smoothing regularises this by
    # targeting 0.9 for the correct class and 0.0001 for all others.
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    run_name  = f"fold{fold}_seed{seed}"

    if wandb_enabled:
        wandb.init(
            project=CONFIG["project_name"],
            name=f"{CONFIG['model_name']}_{run_name}",
            group=CONFIG["model_name"],
            config={**CONFIG, "fold": fold, "seed": seed},
            reinit=True,
        )

    # ── STAGE 1: Head Only ──────────────────────────────────────────────
    print(f"\n  Stage 1: Head Only | Fold {fold} | Seed {seed}")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["stage1_lr"], weight_decay=CONFIG["weight_decay"]
    )

    for epoch in range(CONFIG["stage1_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _    = validate(model, val_loader, criterion, device, CONFIG["num_classes"])
        print(
            f"    [S1] Ep {epoch+1:02d}/{CONFIG['stage1_epochs']} "
            f"| TLoss {train_loss:.4f} TAcc {train_acc:.4f} "
            f"| VLoss {val_metrics['val_loss']:.4f} "
            f"Top1 {val_metrics['val_top1_acc']:.4f} "
            f"Top5 {val_metrics['val_top5_acc']:.4f}"
        )
        if wandb_enabled:
            wandb.log({"stage": 1, "epoch": epoch+1, "train_loss": train_loss,
                       "train_acc": train_acc, **val_metrics})

    # ── STAGE 2: Full Fine-Tuning with Discriminative LRs ───────────────
    # Backbone gets a lower LR than the head:
    #   - Backbone already knows how to detect strokes/edges from ImageNet
    #   - We want gentle adaptation to handwriting domain, not wholesale rewrite
    #   - Head needs faster movement to separate 1000 classes
    print(f"  Stage 2: Fine-Tuning (discriminative LR) | Fold {fold} | Seed {seed}")
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    head_params     = [p for n, p in model.named_parameters() if     n.startswith("fc")]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": CONFIG["stage2_lr_backbone"]},
        {"params": head_params,     "lr": CONFIG["stage2_lr_head"]},
    ], weight_decay=CONFIG["weight_decay"])

    # Cosine annealing: LR decays smoothly from initial value → ~0 over
    # stage2_epochs. Prevents oscillation late in training on tiny datasets.
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["stage2_epochs"], eta_min=1e-7
    )
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    for epoch in range(CONFIG["stage2_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _    = validate(model, val_loader, criterion, device, CONFIG["num_classes"])
        scheduler.step()
        current_lr_bb  = scheduler.get_last_lr()[0]
        current_lr_hd  = scheduler.get_last_lr()[1]
        print(
            f"    [S2] Ep {epoch+1:02d}/{CONFIG['stage2_epochs']} "
            f"| TLoss {train_loss:.4f} TAcc {train_acc:.4f} "
            f"| VLoss {val_metrics['val_loss']:.4f} "
            f"Top1 {val_metrics['val_top1_acc']:.4f} "
            f"Top5 {val_metrics['val_top5_acc']:.4f} "
            f"| LR bb={current_lr_bb:.2e} hd={current_lr_hd:.2e}"
        )
        if wandb_enabled:
            wandb.log({"stage": 2, "epoch": CONFIG["stage1_epochs"] + epoch+1,
                       "train_loss": train_loss, "train_acc": train_acc,
                       "lr_backbone": current_lr_bb, "lr_head": current_lr_hd,
                       **val_metrics})

        # Stop when top-5 accuracy stops improving (more stable than val_loss)
        if early_stopping(val_metrics["val_top5_acc"], model):
            print(f"    ⚡ Early stopping at epoch {epoch+1} "
                  f"(best top-5: {early_stopping.best_score:.4f})")
            break

    model.load_state_dict(early_stopping.best_model_state)
    final_metrics, _, _ = validate(model, val_loader, criterion, device, CONFIG["num_classes"])

    print(
        f"\n  📊 Fold {fold} | Seed {seed} "
        f"| Top1: {final_metrics['val_top1_acc']:.4f} "
        f"| Top5: {final_metrics['val_top5_acc']:.4f} "
        f"| F1-macro: {final_metrics['val_f1_macro']:.4f}"
    )
    if wandb_enabled:
        wandb.log({
            "final_top1":       final_metrics["val_top1_acc"],
            "final_top5":       final_metrics["val_top5_acc"],
            "final_top10":      final_metrics["val_top10_acc"],
            "final_f1_macro":   final_metrics["val_f1_macro"],
            "final_f1_weighted":final_metrics["val_f1_weighted"],
        })
        wandb.finish()

    return final_metrics, model


# %% Cell 11: Main Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"\n{'='*70}\n  Myanmar Syllable Handwriting — ShuffleNetV2 x1.0\n"
          f"  Mode: {CONFIG['color_mode']} | Classes: {CONFIG['num_classes']}\n{'='*70}")

    # Load dataset labels for stratified splitting
    full_ds    = datasets.ImageFolder(ORGANISED_DIR)
    all_labels = [label for _, label in full_ds.samples]
    print(f"📁 Total images: {len(full_ds)} | Classes: {len(full_ds.classes)}")

    all_results = []
    skf = StratifiedKFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["fold_seed"]
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(all_labels)), all_labels), 1
    ):
        for seed in CONFIG["train_seeds"]:
            print(f"\n{'#'*60}\n  FOLD {fold}/{CONFIG['n_folds']} | SEED {seed}\n{'#'*60}")
            set_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_ds = datasets.ImageFolder(ORGANISED_DIR, transform=get_transforms(True))
            val_ds   = datasets.ImageFolder(ORGANISED_DIR, transform=get_transforms(False))

            train_loader = DataLoader(
                Subset(train_ds, train_idx),
                batch_size=CONFIG["batch_size"], shuffle=True,
                num_workers=CONFIG["num_workers"], pin_memory=True,
                worker_init_fn=seed_worker, generator=g,
            )
            val_loader = DataLoader(
                Subset(val_ds, val_idx),
                batch_size=CONFIG["batch_size"], shuffle=False,
                num_workers=CONFIG["num_workers"], pin_memory=True,
            )

            model = create_model(CONFIG["num_classes"]).to(device)
            metrics, trained_model = train_single_run(
                model, train_loader, val_loader, device, fold, seed
            )

            all_results.append({"fold": fold, "seed": seed, **metrics})
            save_path = f"model_{CONFIG['model_name']}_fold{fold}_seed{seed}.pth"
            torch.save(trained_model.state_dict(), save_path)
            print(f"  💾 Saved: {save_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}\n  FINAL CV SUMMARY: {CONFIG['model_name']}\n{'='*70}")
    print(f"  Top-1  Accuracy:  {df['val_top1_acc'].mean():.4f} ± {df['val_top1_acc'].std():.4f}")
    print(f"  Top-5  Accuracy:  {df['val_top5_acc'].mean():.4f} ± {df['val_top5_acc'].std():.4f}")
    print(f"  Top-10 Accuracy:  {df['val_top10_acc'].mean():.4f} ± {df['val_top10_acc'].std():.4f}")
    print(f"  F1 (macro):       {df['val_f1_macro'].mean():.4f} ± {df['val_f1_macro'].std():.4f}")
    print(f"  F1 (weighted):    {df['val_f1_weighted'].mean():.4f} ± {df['val_f1_weighted'].std():.4f}")

    df.to_csv(f"results_{CONFIG['model_name']}.csv", index=False)
    print(f"\n✅ Results saved to results_{CONFIG['model_name']}.csv")

    return df


# %% Cell 12: Execution
if __name__ == "__main__":
    main()
