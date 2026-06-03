# ============================================================================
# NOTEBOOK 14: SqueezeNet v1.1 + CBAM (V8: Mid-level Texture Focus)
# Chickpea Fusarium Wilt Binary Classification
# 5-Fold Stratified Cross-Validation × 3 Random Seeds
# ============================================================================
#
# PURPOSE:
#   Train SqueezeNet v1.1 with CBAM attention module (Variant V8) on Experiment A dataset.
#   - Train Pool: 85% Turkey + 170 purely healthy (~1,895 images) -> 15 runs
#   - Test A: 15% Turkey (In-domain, ~305 images)
#   - Test B: 100% Myanmar (Cross-domain, 268 images)
#
# CBAM VARIANT V8 (Mid-level Texture Focus):
#   Places a CBAM attention block exactly at the mid-level processing stage
#   where disease textures (wilting, yellowing) are processed:
#   - After Fire4 (27x27 spatial)
#   - After Fire5 (27x27 spatial)
#
# Total runs: 5 folds × 3 seeds = 15
# Usage: Copy cells into a Kaggle Notebook (GPU P100 recommended)
# ============================================================================


# %% Cell 1: Install Dependencies & WandB Setup
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
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import wandb


# %% Cell 3: Configuration
CONFIG = {
    # --- Project ---
    "project_name": "chickpea-fusarium-wilt",
    "model_name": "squeezenet1_1_cbam_v8_exp_a",
    "num_classes": 2,
    "class_names": ["Diseased", "Healthy"],

    # --- Cross-Validation ---
    "n_folds": 5,
    "fold_seed": 42,
    "train_seeds": [42, 123, 456],

    # --- Data ---
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 2,

    # --- Stage 1: Head Only ---
    "stage1_lr": 1e-3,
    "stage1_epochs": 10,

    # --- Stage 2: Fine-Tuning ---
    "stage2_lr": 2e-5,
    "stage2_epochs": 50,
    "patience": 7,
    "weight_decay": 1e-4,

    # --- ImageNet Normalization ---
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],

    # --- CBAM ---
    "cbam_reduction": 16,
    "cbam_positions": [6, 7], # V8: Fire4, Fire5
}

CHANNEL_MAP = {
    3: 128, 4: 128,              # Group A
    6: 256, 7: 256,              # Group B
    9: 384, 10: 384,             # Group C
    11: 512, 12: 512,            # Group C (deeper)
}


# %% Cell 4: Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# %% Cell 5: Data Paths
KAGGLE_PATHS = [
    "/kaggle/input/chickpea-fusarium-wilt-exp-a",
    "/kaggle/input/datasets/theinkyawlwin/chickpea-fusarium-wilt-exp-a",
    "/Users/tklwin/GithubRepos/chickpea-mobile/kaggle_dataset_exp_a", # Local fallback
]

TRAIN_DIR, TEST_A_DIR, TEST_B_DIR = None, None, None
for p in KAGGLE_PATHS:
    if os.path.isdir(os.path.join(p, "train_pool")):
        TRAIN_DIR = os.path.join(p, "train_pool")
        TEST_A_DIR = os.path.join(p, "test_a")
        TEST_B_DIR = os.path.join(p, "test_b")
        break

if TRAIN_DIR is None:
    raise FileNotFoundError("Dataset not found! Ensure paths are correct.")

print(f"Data directories located at: {os.path.dirname(TRAIN_DIR)}")


# %% Cell 6: Data Transforms
def get_transforms(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ])


# %% Cell 7: CBAM Blocks
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# %% Cell 8: Model Definition
def create_model_cbam(cbam_positions, num_classes=2, reduction=16):
    base = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)

    new_features = nn.Sequential()
    for i, module in enumerate(base.features):
        new_features.add_module(str(i), module)
        if i in cbam_positions:
            channels = CHANNEL_MAP[i]
            new_features.add_module(f"cbam_{i}", CBAM(channels, reduction))

    base.features = new_features
    base.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    base.num_classes = num_classes

    return base


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# %% Cell 9: Training & Validation Functions
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
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    metrics = {
        "val_loss": running_loss / total,
        "val_accuracy": accuracy_score(all_labels, all_preds),
        "val_f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "val_f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "val_precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "val_recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    metrics["diseased_recall"] = float(per_class_recall[0]) if len(per_class_recall) > 0 else 0.0
    metrics["healthy_recall"]  = float(per_class_recall[1]) if len(per_class_recall) > 1 else 0.0

    return metrics, all_preds, all_labels


# %% Cell 10: Early Stopping
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


# %% Cell 11: Two-Stage Training Run
def train_single_run(model, train_loader, val_loader, device, fold, seed):
    criterion = nn.CrossEntropyLoss()
    run_name = f"fold{fold}_seed{seed}"

    if wandb_enabled:
        wandb.init(
            project=CONFIG["project_name"],
            name=f"{CONFIG['model_name']}_{run_name}",
            group=CONFIG["model_name"],
            config={**CONFIG, "fold": fold, "seed": seed},
            reinit=True,
        )

    # STAGE 1: HEAD ONLY + CBAM
    print(f"\n  Stage 1: Head + CBAM | Fold {fold} | Seed {seed}")
    
    # Freeze ONLY pretrained layers (not CBAM, not classifier)
    for name, param in model.named_parameters():
        if "cbam" not in name and "classifier" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["stage1_lr"], weight_decay=CONFIG["weight_decay"]
    )

    for epoch in range(CONFIG["stage1_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        print(f"    [S1] Epoch {epoch+1:02d}/{CONFIG['stage1_epochs']} | TLoss: {train_loss:.4f} TAcc: {train_acc:.4f} | VLoss: {val_metrics['val_loss']:.4f} VAcc: {val_metrics['val_accuracy']:.4f}")
        if wandb_enabled:
            wandb.log({"stage": 1, "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, **val_metrics})

    # STAGE 2: FINE-TUNING
    print(f"  Stage 2: Fine-Tuning | Fold {fold} | Seed {seed}")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["stage2_lr"], weight_decay=CONFIG["weight_decay"])
    early_stopping = EarlyStopping(patience=CONFIG["patience"])

    for epoch in range(CONFIG["stage2_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        print(f"    [S2] Epoch {epoch+1:02d}/{CONFIG['stage2_epochs']} | TLoss: {train_loss:.4f} TAcc: {train_acc:.4f} | VLoss: {val_metrics['val_loss']:.4f} VAcc: {val_metrics['val_accuracy']:.4f}")
        if wandb_enabled:
            wandb.log({"stage": 2, "epoch": CONFIG["stage1_epochs"] + epoch + 1, "train_loss": train_loss, "train_acc": train_acc, **val_metrics})
        
        if early_stopping(val_metrics["val_loss"], model):
            print(f"    ⚡ Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(early_stopping.best_model_state)
    final_metrics, preds, labels = validate(model, val_loader, criterion, device)
    cm = confusion_matrix(labels, preds)

    print(f"\n  📊 Fold {fold} | Seed {seed} | Acc: {final_metrics['val_accuracy']:.4f} | F1: {final_metrics['val_f1_weighted']:.4f} | Diseased Recall: {final_metrics['diseased_recall']:.4f}")
    if wandb_enabled:
        wandb.log({"final_accuracy": final_metrics["val_accuracy"],
                   "final_f1_weighted": final_metrics["val_f1_weighted"],
                   "final_diseased_recall": final_metrics["diseased_recall"],
                   "final_healthy_recall": final_metrics["healthy_recall"]})
        wandb.finish()

    return final_metrics, model


# %% Cell 12: Main Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"\n{'='*70}\n  EXPERIMENT: SqueezeNet v1.1 + CBAM V8\n  Dataset: Exp A Pre-split (Train Pool, Test A, Test B)\n{'='*70}")

    train_ds_eval = datasets.ImageFolder(TRAIN_DIR) # labels ONLY for stratify
    all_labels = [label for _, label in train_ds_eval.samples]
    
    print(f"📁 Train Pool: {len(train_ds_eval)} images, Dist: {dict(zip(train_ds_eval.classes, np.bincount(all_labels)))}")
    sample_model = create_model_cbam(CONFIG["cbam_positions"], CONFIG["num_classes"], CONFIG["cbam_reduction"])
    total_p, trainable_p = count_parameters(sample_model)
    print(f"  Total parameters: {total_p:,}")
    del sample_model


    all_results = []
    skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["fold_seed"])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), 1):
        for seed in CONFIG["train_seeds"]:
            print(f"\n{'#'*60}\n  FOLD {fold}/{CONFIG['n_folds']} | SEED {seed}\n{'#'*60}")
            set_seed(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(True))
            val_ds = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(False))

            train_loader = DataLoader(
                Subset(train_ds, train_idx), batch_size=CONFIG["batch_size"], shuffle=True,
                num_workers=CONFIG["num_workers"], pin_memory=True, worker_init_fn=seed_worker, generator=g,
            )
            val_loader = DataLoader(
                Subset(val_ds, val_idx), batch_size=CONFIG["batch_size"], shuffle=False,
                num_workers=CONFIG["num_workers"], pin_memory=True,
            )

            model = create_model_cbam(CONFIG["cbam_positions"], CONFIG["num_classes"], CONFIG["cbam_reduction"]).to(device)
            metrics, trained_model = train_single_run(model, train_loader, val_loader, device, fold, seed)

            all_results.append({"fold": fold, "seed": seed, **metrics})
            save_path = f"model_{CONFIG['model_name']}_fold{fold}_seed{seed}.pth"
            torch.save(trained_model.state_dict(), save_path)
            print(f"  💾 Saved: {save_path}")

    # SUMMARY
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}\n  FINAL CV SUMMARY: {CONFIG['model_name']}\n{'='*70}")
    print(f"  Accuracy:        {df['val_accuracy'].mean():.4f} ± {df['val_accuracy'].std():.4f}")
    print(f"  F1 (weighted):   {df['val_f1_weighted'].mean():.4f} ± {df['val_f1_weighted'].std():.4f}")
    print(f"  Diseased Recall: {df['diseased_recall'].mean():.4f} ± {df['diseased_recall'].std():.4f}")
    print(f"  Healthy Recall:  {df['healthy_recall'].mean():.4f} ± {df['healthy_recall'].std():.4f}")

    df.to_csv(f"results_{CONFIG['model_name']}.csv", index=False)

    return df


# %% Cell 13: External Evaluation (Test A & Test B)
def evaluate_final_test_sets(df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\n  EXTERNAL EVALUATION (TEST A & B)\n{'='*70}")

    test_a_ds = datasets.ImageFolder(TEST_A_DIR, transform=get_transforms(False))
    test_b_ds = datasets.ImageFolder(TEST_B_DIR, transform=get_transforms(False))
    
    test_a_loader = DataLoader(test_a_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    test_b_loader = DataLoader(test_b_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"📁 Test A (Turkey 15%): {len(test_a_ds)} images")
    print(f"📁 Test B (Myanmar): {len(test_b_ds)} images")

    # Find Best and Median checkpoints based on CV Validation Accuracy
    df_sorted = df.sort_values(by="val_accuracy", ascending=False).reset_index(drop=True)
    best_run = df_sorted.iloc[0]
    median_run = df_sorted.iloc[len(df_sorted) // 2]
    
    checkpoints_to_eval = [
        ("🥇 BEST Checkpoint", best_run),
        ("🎯 MEDIAN Checkpoint", median_run)
    ]
    
    criterion = nn.CrossEntropyLoss()
    eval_results = []
    
    for label, run_info in checkpoints_to_eval:
        fold, seed = int(run_info['fold']), int(run_info['seed'])
        model_path = f"model_{CONFIG['model_name']}_fold{fold}_seed{seed}.pth"
        
        print(f"\n➤ Loading {label} — Fold: {fold}, Seed: {seed} (CV Val Acc: {run_info['val_accuracy']:.4f})")
        model = create_model_cbam(CONFIG["cbam_positions"], CONFIG["num_classes"], CONFIG["cbam_reduction"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        metrics_a, preds_a, labels_a = validate(model, test_a_loader, criterion, device)
        print(f"  [Test A - Turkey 15%] Acc: {metrics_a['val_accuracy']:.4f} | F1: {metrics_a['val_f1_weighted']:.4f} | Diseased R: {metrics_a['diseased_recall']:.4f} | Healthy R: {metrics_a['healthy_recall']:.4f}")
        
        metrics_b, preds_b, labels_b = validate(model, test_b_loader, criterion, device)
        print(f"  [Test B - Myanmar]    Acc: {metrics_b['val_accuracy']:.4f} | F1: {metrics_b['val_f1_weighted']:.4f} | Diseased R: {metrics_b['diseased_recall']:.4f} | Healthy R: {metrics_b['healthy_recall']:.4f}")
        
        eval_results.append({
            "type": label.split()[1],
            "fold": fold, "seed": seed,
            "test_a_acc": metrics_a['val_accuracy'],
            "test_b_acc": metrics_b['val_accuracy'],
        })
    
    pd.DataFrame(eval_results).to_csv(f"test_evaluation_results_{CONFIG['model_name']}.csv", index=False)
    print("\n✅ Evaluation complete! Results saved.")


# %% Cell 14: Execution
if __name__ == "__main__":
    df_results = main()
    evaluate_final_test_sets(df_results)