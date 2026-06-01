"""
02_evaluate_lstm.py
===================
Evaluate a trained LSTM model on the shared test set.
Reports loss, perplexity, and bits-per-character (BPC) for fair comparison.

Usage for FULL EVALUATION - Benchmark for KENLM, SRILM & LSTM:
python 02_evaluate_lstm.py --model_dir ../results/lstm_wikipedia --test_file ../data/prepared/test_shared.txt 
python 02_evaluate_lstm.py --model_dir ../results/lstm_freococo --test_file ../data/prepared/test_shared.txt
python 02_evaluate_lstm.py --model_dir ../results/lstm_myanmar_literature --test_file ../data/prepared/test_shared.txt

Usage for DOMAIN-SPECIFIC EVALUATION:
python 02_evaluate_lstm.py --model_dir ../results/lstm_mypos_baseline --test_file ../data/prepared/test_medical_std.txt
python 02_evaluate_lstm.py --model_dir ../results/lstm_mypos_baseline --test_file ../data/prepared/test_news_std.txt
python 02_evaluate_lstm.py --model_dir ../results/lstm_mypos_baseline --test_file ../data/prepared/test_legal_std.txt

Domain Adaption
python 02_evaluate_lstm.py --model_dir ../results/lstm_legal_adapted --test_file ../data/prepared/test_legal_std.txt
python 02_evaluate_lstm.py --model_dir ../results/lstm_medical_adapted --test_file ../data/prepared/test_medical_std.txt
python 02_evaluate_lstm.py --model_dir ../results/lstm_news_adapted --test_file ../data/prepared/test_news_std.txt
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse classes from the trainer (sits next to this script)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module


# Simpler: just import the classes directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "lstm_trainer",
    Path(__file__).resolve().parent / "01_train_lstm.py"
)
lstm_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lstm_trainer)

Vocabulary = lstm_trainer.Vocabulary
TextDataset = lstm_trainer.TextDataset
LSTM_LM = lstm_trainer.LSTM_LM
collate_fn = lstm_trainer.collate_fn
get_device = lstm_trainer.get_device


def count_chars(file_path):
    """Count total characters in test file (used for BPC)."""
    with open(file_path, encoding="utf-8") as f:
        return sum(len(line.strip()) for line in f if line.strip())


def evaluate_model(model, loader, criterion, device):
    """Compute loss, perplexity, and total tokens."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            mask = (targets != 0)
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss), total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_dir = Path(args.model_dir)

    # Load vocabulary
    vocab = Vocabulary.load(model_dir / "vocab.json")
    print(f"Vocabulary size: {len(vocab):,}")

    # Load checkpoint to get config
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Build model with same config
    model = LSTM_LM(
        vocab_size=len(vocab),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Build test dataset
    test_ds = TextDataset(args.test_file, vocab, max_len=cfg.get("max_len", 64))
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Evaluate
    loss, ppl, n_tokens = evaluate_model(model, test_loader, criterion, device)

    # Compute BPC
    n_chars = count_chars(args.test_file)
    # BPC = (loss_in_nats × n_tokens) / (n_chars × ln(2))
    bpc = (loss * n_tokens) / (n_chars * math.log(2))

    # Print results
    print("\n" + "="*60)
    print(f"RESULTS for {model_dir.name}")
    print("="*60)
    print(f"  Test file:      {args.test_file}")
    print(f"  Total tokens:   {n_tokens:,}")
    print(f"  Total chars:    {n_chars:,}")
    print(f"  Loss (nats):    {loss:.4f}")
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Bits-Per-Char:  {bpc:.4f}")

    # Save results
    results = {
        "model_dir": str(model_dir),
        "test_file": str(args.test_file),
        "loss_nats": loss,
        "perplexity": ppl,
        "bpc": bpc,
        "n_tokens": n_tokens,
        "n_chars": n_chars,
    }
    with open(model_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {model_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
