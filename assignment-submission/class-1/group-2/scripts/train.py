#!/usr/bin/env python3

"""This module trains the BiLSTM emotion model using prepared training/validation data.

This module depends on:
- src/prep_data.py
- src/model.py
"""

import os
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import EmotionalBiLSTM
from src.prep_data import prepare_train_val_data


# main
def main():
    DATA_PATH = "../data/merged/Combined.csv" ## CHANGE HERE
    TOKENIZED_OUTPUT = "../data/merged/Combined_tokenized.csv" ## CHANGE HERE
    MODEL_PATH = "../models/BiLSTM_model.pth" ## CHANGE HERE
    STOPWORDS_PATH = "../data/stopwords.txt"
    TEXT_COL = "text"
    LABEL_COL = "label"

    SEED = 42

    ## CHANGE HERE: tune training parameters
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 0.001
    VAL_SPLIT = 0.1
    MAX_LEN = 50

    # set True only if enough RAM; False if dataset is large
    MATERIALIZE_SPLITS_FOR_SHAPE_CHECK = False ## CHANGE HERE

    # set seed for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare loaders and artifacts so train.py stays minimal
    train_loader, val_loader, train_ds, val_ds, word2id, id2label, label2id, class_weights = (
        prepare_train_val_data(
            data_path=DATA_PATH,
            text_col=TEXT_COL,
            label_col=LABEL_COL,
            stopwords_path=STOPWORDS_PATH,
            seed=SEED,
            val_split=VAL_SPLIT,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
            tokenized_output_path=TOKENIZED_OUTPUT,
        )
    )

    # check batch shapes of train and val; same as model sees each step
    xb, yb = next(iter(train_loader))
    xvb, yvb = next(iter(val_loader))
    print(f"[shapes] train batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
    print(f"[shapes] val batch: x {tuple(xvb.shape)}, y {tuple(yvb.shape)}")

    # check full shapes of stacked train and val; high ram cost
    if MATERIALIZE_SPLITS_FOR_SHAPE_CHECK:
        train_X, train_y = zip(*[(x, y) for x, y in train_ds])
        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)
        val_X, val_y = zip(*[(x, y) for x, y in val_ds])
        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)
        print(f"[shapes] train_X {tuple(train_X.shape)}, train_y {tuple(train_y.shape)}")
        print(f"[shapes] val_X {tuple(val_X.shape)}, val_y {tuple(val_y.shape)}")

    # choose device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU device name: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

    # initialize model (load from src/model.py)
    model = EmotionalBiLSTM(vocab_size=len(word2id)).to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # compute class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {acc:.2%}")

    # save model & vocab
    os.makedirs("models", exist_ok=True)

    torch.save({
        "state": model.state_dict(),
        "vocab": word2id,
        "label2id": label2id,
        "id2label": id2label,
        "text_col": TEXT_COL,
        "label_col": LABEL_COL,
        "max_len": MAX_LEN,
    }, MODEL_PATH)

    print(f"[+] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()