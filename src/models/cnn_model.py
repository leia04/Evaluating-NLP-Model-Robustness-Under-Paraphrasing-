"""
cnn_model.py
Text CNN for AG News topic classification.
Architecture: GloVe embeddings -> parallel conv layers (filter sizes 3,4,5)
              -> max-over-time pooling -> dropout -> linear classifier
"""

import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_NAMES   = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
GLOVE_PATH    = "glove.6B.100d.txt"   # download separately if needed
EMBED_DIM     = 100
MAX_LEN       = 64                    # tokens per sample (AG News titles are short)
FILTER_SIZES  = [3, 4, 5]
NUM_FILTERS   = 128
DROPOUT       = 0.5
BATCH_SIZE    = 64
EPOCHS        = 5
LR            = 1e-3
MIN_FREQ      = 2                     # min word frequency to keep in vocab
DEVICE        = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)


# ── Vocabulary ────────────────────────────────────────────────────────────────
class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, texts, min_freq=MIN_FREQ):
        from collections import Counter
        counts = Counter(w for text in texts for w in text.lower().split())
        for word, freq in counts.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word
        print(f"Vocabulary size: {len(self.word2idx):,}")

    def encode(self, text, max_len=MAX_LEN):
        tokens = text.lower().split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # pad / truncate to max_len
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


def load_glove(vocab, glove_path=GLOVE_PATH, embed_dim=EMBED_DIM):
    """Load GloVe vectors for words in vocab. Unknown words get random init."""
    matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    matrix[0] = 0  # PAD token -> zeros

    if not os.path.exists(glove_path):
        print(f"[WARNING] GloVe file not found at '{glove_path}'. "
              "Using random embeddings. Download glove.6B.zip from "
              "https://nlp.stanford.edu/projects/glove/ and unzip.")
        return matrix

    found = 0
    print(f"Loading GloVe from '{glove_path}'...")
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            if word in vocab.word2idx:
                matrix[vocab.word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1
    print(f"GloVe coverage: {found}/{len(vocab):,} words "
          f"({found/len(vocab)*100:.1f}%)")
    return matrix


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.data   = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx],   dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes, num_filters, dropout,
                 pretrained_embeddings=None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(pretrained_embeddings, dtype=torch.float32)
            )

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = emb.permute(0, 2, 1)       # (batch, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))        # (batch, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            pooled.append(c)

        out = torch.cat(pooled, dim=1)   # (batch, num_filters * len(filter_sizes))
        out = self.dropout(out)
        return self.fc(out)              # (batch, num_classes)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_ag_news():
    print("Loading AG News dataset...")
    dataset  = load_dataset("fancyzhx/ag_news")
    train_df = pd.DataFrame({"text": dataset["train"]["text"],
                              "label": dataset["train"]["label"]})
    test_df  = pd.DataFrame({"text": dataset["test"]["text"],
                              "label": dataset["test"]["label"]})
    print(f"  Train: {len(train_df):,} / Test: {len(test_df):,}")
    return train_df, test_df


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, train_loader, epochs=EPOCHS, lr=LR):
    print(f"Training TextCNN on {DEVICE} for {epochs} epoch(s)...")
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    criterion   = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {acc:.4f}")
    print("  Done.\n")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_on_texts(model, vocab, texts, batch_size=BATCH_SIZE):
    dataset    = TextDataset(texts, [0] * len(texts), vocab)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_preds  = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(DEVICE))
            all_preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(all_preds)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, vocab, texts, labels, split_name="Test"):
    preds    = predict_on_texts(model, vocab, texts)
    acc      = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_per   = f1_score(labels, preds, average=None)
    print(f"\n{'='*50}\n  {split_name}\n{'='*50}")
    print(f"  Accuracy: {acc:.4f} | F1 macro: {f1_macro:.4f}\n")
    print(classification_report(
        labels, preds,
        target_names=[LABEL_NAMES[i] for i in range(4)]
    ))
    return {
        "split":        split_name,
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_per_class": {LABEL_NAMES[i]: float(f1_per[i]) for i in range(4)},
        "predictions":  preds.tolist(),
    }


# ── Save / load ───────────────────────────────────────────────────────────────
def save_model(model, vocab, path="cnn_model.pt"):
    torch.save({"model_state": model.state_dict(),
                "vocab":       vocab}, path)
    print(f"Model saved → {path}")


def load_model(path="cnn_model.pt",
               embed_dim=EMBED_DIM, filter_sizes=FILTER_SIZES,
               num_filters=NUM_FILTERS, dropout=DROPOUT):
    checkpoint = torch.load(path, map_location=DEVICE)
    vocab      = checkpoint["vocab"]
    model      = TextCNN(
        vocab_size=len(vocab), embed_dim=embed_dim,
        num_classes=4, filter_sizes=filter_sizes,
        num_filters=num_filters, dropout=dropout
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Model loaded ← {path}")
    return model, vocab


def save_metrics(metrics_list, path="results_cnn.json"):
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved → {path}")


# ── Main (train + eval on full test set) ──────────────────────────────────────
if __name__ == "__main__":
    train_df, test_df = load_ag_news()

    # build vocab from training data
    vocab = Vocabulary()
    vocab.build(train_df["text"].tolist())

    # load GloVe (falls back to random if file missing)
    embed_matrix = load_glove(vocab)

    # datasets & loaders
    train_ds     = TextDataset(train_df["text"].tolist(),
                               train_df["label"].tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # build & train model
    model = TextCNN(
        vocab_size=len(vocab), embed_dim=EMBED_DIM,
        num_classes=4, filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS, dropout=DROPOUT,
        pretrained_embeddings=embed_matrix
    ).to(DEVICE)

    model = train(model, train_loader)

    # evaluate on full test set
    results = evaluate(model, vocab,
                       test_df["text"].tolist(),
                       test_df["label"].tolist(),
                       split_name="AG News Test Set")
    save_model(model, vocab)
    save_metrics([results])