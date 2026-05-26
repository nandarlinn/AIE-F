"""
04_domain_adaptation.py

usage
python 04_domain_adaptation.py 


# Medical adaptation
python 04_domain_adaptation.py \
    --base_model_dir ../results/lstm_mypos_baseline \
    --train_file ../data/prepared/train_medical_adapt.txt \
    --save_dir ../results/lstm_medical_adapted \
    --epochs 15

# News adaptation
python 04_domain_adaptation.py \
    --base_model_dir ../results/lstm_mypos_baseline \
    --train_file ../data/prepared/train_news_adapt.txt \
    --save_dir ../results/lstm_news_adapted \
    --epochs 15

# Legal adaptation
python 04_domain_adaptation.py \
    --base_model_dir ../results/lstm_mypos_baseline \
    --train_file ../data/prepared/train_legal_adapt.txt \
    --save_dir ../results/lstm_legal_adapted \
    --epochs 15
"""

import argparse
import json
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# --- 1. Define the identical architecture ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.fc(out)
        return out, hidden


# --- 2. Smarter Vocabulary Loader ---
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.unk_idx = 0 

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Check if it's a nested dictionary or a flat dictionary
            if 'word2idx' in data:
                self.word2idx = data['word2idx']
                self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            else:
                # If it's a flat dictionary, it IS the word2idx mapping
                self.word2idx = data
                self.idx2word = {int(v): k for k, v in data.items()}
                
            # Safely find the UNK token
            self.unk_idx = self.word2idx.get("<unk>", self.word2idx.get("<UNK>", 0))

    def __len__(self):
        return len(self.word2idx)

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Set paths
    base_model_dir = Path(args.base_model_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the BASELINE Vocabulary
    vocab = Vocabulary()
    vocab.load(base_model_dir / "vocab.json")
    print(f"Loaded Baseline Vocabulary size: {len(vocab)}")

    # 2. Process the Adaptation Data using the OLD vocabulary
    data = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            # Convert words to existing IDs, use <unk> if it's a completely new legal word
            indices = [vocab.word2idx.get(w, vocab.unk_idx) for w in tokens]
            if indices:
                data.append(torch.tensor(indices, dtype=torch.long))

    if not data:
        print("❌ Error: No training data found.")
        return

    # Create sequences (X, y)
    X, y = [], []
    for seq in data:
        if len(seq) > 1:
            X.append(seq[:-1])
            y.append(seq[1:])
    
    if not X:
        print("❌ Error: Sequences too short.")
        return

    X = torch.stack(X).to(device)
    y = torch.stack(y).to(device)

    # 3. Load the BASELINE Model
    model = LSTMModel(vocab_size=len(vocab)).to(device)
    
    # Load the checkpoint bundle
    checkpoint = torch.load(base_model_dir / "best_model.pt", map_location=device)
    
    # Extract just the weights from the bundle
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    print("✅ Baseline model weights loaded successfully.")

    # 4. Fine-Tune Setup (Small Learning Rate is critical!)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 10x smaller than normal training

    print("\nStarting Domain Adaptation (Fine-Tuning)...")
    model.train()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        ppl = math.exp(loss.item())
        print(f"Epoch {epoch+1:2d}/{args.epochs} | Adaptation Loss: {loss.item():.4f} | PPL: {ppl:.2f}")


    # 5. Save the ADAPTED model (packaged as a bundle for the evaluator)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 1,
            "dropout": 0.0,
        }
    }, save_dir / "best_model.pt")
    
    
    # Save the exact same flat vocabulary so evaluation script doesn't crash
    with open(save_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab.word2idx, f)
        
    print(f"\n✅ Adaptation complete! Model saved to {save_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_dir", default="../results/lstm_mypos_baseline")
    p.add_argument("--train_file", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--epochs", type=int, default=15)
    main(p.parse_args())