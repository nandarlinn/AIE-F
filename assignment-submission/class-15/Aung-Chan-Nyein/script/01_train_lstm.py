"""
01_train_lstm.py
s
LSTM trainer optimized for Apple Silicon (MPS) and CPU.

Key speed optimizations vs the tutorial's original:
  • Smaller model (embedding 128, hidden 256, 1 layer)
  • Vocabulary cap (top 20k words)
  • Larger batch size (64)
  • Fewer epochs (10)
  • Pin memory + workers for CPU loading
  • MPS

Usage:
python 01_train_lstm.py --train_file ../data/prepared/test_legal_std.txt --save_dir ../results/lstm_legal
python 01_train_lstm.py --train_file ../data/prepared/test_medical_std.txt --save_dir ../results/lstm_medical
python 01_train_lstm.py --train_file ../data/prepared/test_news_std.txt --save_dir ../results/lstm_news

Baseline
python 01_train_lstm.py --train_file ../data/mypos_v3.word.clean --save_dir ../results/lstm_mypos_baseline

For saving model + vocabulary + final loss/PPL into save_dir.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────

class Vocabulary:
    """Word vocabulary with frequency-based pruning."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<s>"
    EOS = "</s>"

    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self._add(self.PAD)
        self._add(self.UNK)
        self._add(self.BOS)
        self._add(self.EOS)

    def _add(self, w):
        if w not in self.word2id:
            i = len(self.word2id)
            self.word2id[w] = i
            self.id2word[i] = w

    def build(self, sentences, max_size=20000, min_freq=2):
        counter = Counter()
        for sent in sentences:
            counter.update(sent.split())
        most_common = counter.most_common(max_size - 4)  # reserve 4 special tokens
        for word, freq in most_common:
            if freq < min_freq:
                break
            self._add(word)

    def encode(self, sentence):
        unk = self.word2id[self.UNK]
        return [self.word2id.get(w, unk) for w in sentence.split()]

    def __len__(self):
        return len(self.word2id)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, encoding="utf-8") as f:
            w2i = json.load(f)
        v.word2id = w2i
        v.id2word = {i: w for w, i in w2i.items()}
        return v


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Reads sentences, adds <s>...</s>, encodes as token IDs."""

    def __init__(self, file_path, vocab, max_len=64):
        self.examples = []
        bos = vocab.word2id[Vocabulary.BOS]
        eos = vocab.word2id[Vocabulary.EOS]
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = [bos] + vocab.encode(line)[: max_len - 2] + [eos]
                if len(ids) >= 3:
                    self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_id=0):
    """Pad to max length in batch."""
    lengths = [len(x) for x in batch]
    max_len = max(lengths)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, : len(x)] = x
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    return inputs, targets


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class LSTM_LM(nn.Module):
    """Compact LSTM language model — fast on MPS/CPU."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=1, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        return self.fc(self.dropout(out))


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))
            mask = (targets != 0)
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


def train(args):
    device = get_device()
    print(f"Device: {device}")

    # Load sentences
    with open(args.train_file, encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(sentences):,} training sentences from {args.train_file}")

    # Train/val split (95/5)
    split = int(len(sentences) * 0.95)
    train_sents = sentences[:split]
    val_sents = sentences[split:]

    # Build vocabulary on training data only
    vocab = Vocabulary()
    vocab.build(train_sents, max_size=args.vocab_size, min_freq=2)
    print(f"Vocabulary size: {len(vocab):,}")

    # Save dir
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(save_dir / "vocab.json")

    # Datasets
    train_ds = TextDataset.__new__(TextDataset)
    train_ds.examples = []
    bos = vocab.word2id[Vocabulary.BOS]
    eos = vocab.word2id[Vocabulary.EOS]
    for s in train_sents:
        ids = [bos] + vocab.encode(s)[: args.max_len - 2] + [eos]
        if len(ids) >= 3:
            train_ds.examples.append(torch.tensor(ids, dtype=torch.long))

    val_ds = TextDataset.__new__(TextDataset)
    val_ds.examples = []
    for s in val_sents:
        ids = [bos] + vocab.encode(s)[: args.max_len - 2] + [eos]
        if len(ids) >= 3:
            val_ds.examples.append(torch.tensor(ids, dtype=torch.long))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Model
    model = LSTM_LM(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    history = []
    best_val = float("inf")
    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_tokens = 0.0, 0
        for step, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mask = (targets != 0)
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            if step % 100 == 0:
                cur = total_loss / max(total_tokens, 1)
                print(f"  Epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={cur:.4f} ppl={math.exp(cur):.2f}", end="\r")

        train_loss = total_loss / max(total_tokens, 1)
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_ppl={math.exp(train_loss):.2f}  "
              f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
              f"time={elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": math.exp(train_loss),
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "seconds": elapsed,
        })

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "vocab_size": len(vocab),
            }, save_dir / "best_model.pt")

    # Save final history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete. Best val PPL: {math.exp(best_val):.2f}")
    print(f"   Model + history saved to: {save_dir}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--vocab_size", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
