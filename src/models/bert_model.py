import json
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else
                            "mps"  if torch.backends.mps.is_available() else "cpu")


def load_ag_news():
    print("Loading AG News dataset...")
    dataset = load_dataset("fancyzhx/ag_news")
    def to_df(split):
        return pd.DataFrame({"text": split["text"], "label": split["label"]})
    train_df = to_df(dataset["train"])
    test_df  = to_df(dataset["test"])
    print(f"  Train: {len(train_df):,} / Test: {len(test_df):,}")
    return train_df, test_df


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def build_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    model     = model.to(DEVICE)
    return tokenizer, model


def train(model, tokenizer, train_df):
    print(f"Training BERT on {DEVICE}...")
    dataset    = AGNewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer  = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS
    scheduler  = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
    print("  Done.\n")
    return model


def evaluate(model, tokenizer, texts, labels, split_name="Test"):
    preds    = predict_on_texts(model, tokenizer, texts)
    acc      = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_per   = f1_score(labels, preds, average=None)
    print(f"\n{'='*50}\n  {split_name}\n{'='*50}")
    print(f"  Accuracy: {acc:.4f} | F1 macro: {f1_macro:.4f}\n")
    print(classification_report(labels, preds, target_names=[LABEL_NAMES[i] for i in range(4)]))
    return {
        "split": split_name, "accuracy": acc, "f1_macro": f1_macro,
        "f1_per_class": {LABEL_NAMES[i]: float(f1_per[i]) for i in range(4)},
        "predictions": preds.tolist(),
    }


def predict_on_texts(model, tokenizer, texts):
    dataset    = AGNewsDataset(texts, [0] * len(texts), tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
            )
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)


def save_model(model, tokenizer, path="bert_model"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model saved → {path}/")


def load_model(path="bert_model"):
    tokenizer = BertTokenizer.from_pretrained(path)
    model     = BertForSequenceClassification.from_pretrained(path)
    model     = model.to(DEVICE)
    print(f"Model loaded ← {path}/")
    return tokenizer, model


def save_metrics(metrics_list, path="results_bert.json"):
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved → {path}")
