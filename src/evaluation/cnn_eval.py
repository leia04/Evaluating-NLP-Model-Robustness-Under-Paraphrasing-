import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader

from cnn_model import (
    Vocabulary, TextCNN, TextDataset,
    load_glove, load_ag_news, train as cnn_train,
    predict_on_texts as cnn_predict,
    save_model as cnn_save_model,
    load_model as cnn_load_model,
    LABEL_NAMES, EMBED_DIM, FILTER_SIZES, NUM_FILTERS, DROPOUT,
    BATCH_SIZE, DEVICE,
)

LABEL_NAMES_LIST = [LABEL_NAMES[i] for i in range(4)]

PARAPHRASE_SOURCES = {
    "backtranslation": "paraphrase_backtranslation.csv",
    "bart":            "paraphrase_bart.csv",
    "t5_chatgpt":      "paraphrase_t5_chatgpt.csv",
}

CNN_CKPT_PATH = "cnn_model.pt"

def get_cnn_model():
    """Load CNN checkpoint if available, otherwise train from scratch."""
    if os.path.exists(CNN_CKPT_PATH):
        print(f"Loading CNN checkpoint ← {CNN_CKPT_PATH}")
        model, vocab = cnn_load_model(CNN_CKPT_PATH)
        return model, vocab

    print(f"'{CNN_CKPT_PATH}' not found — training CNN from scratch...")
    train_df, _ = load_ag_news()

    vocab = Vocabulary()
    vocab.build(train_df["text"].tolist())

    embed_matrix = load_glove(vocab)

    train_ds     = TextDataset(train_df["text"].tolist(),
                               train_df["label"].tolist(), vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN(
        vocab_size=len(vocab), embed_dim=EMBED_DIM,
        num_classes=4, filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS, dropout=DROPOUT,
        pretrained_embeddings=embed_matrix,
    ).to(DEVICE)

    model = cnn_train(model, train_loader)
    cnn_save_model(model, vocab, CNN_CKPT_PATH)
    return model, vocab


def load_paraphrase_csv(path):
    df = pd.read_csv(path)
    required = {"original_text", "label", "paraphrase", "similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df = df.dropna(subset=["original_text", "paraphrase", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} rows from '{path}' "
          f"(avg sim {df['similarity'].mean():.4f})")
    return df

def compute_robustness(orig_preds, para_preds, labels):
    orig_preds = np.array(orig_preds)
    para_preds = np.array(para_preds)
    labels     = np.array(labels)

    consistency = float(np.mean(orig_preds == para_preds))
    orig_acc    = float(accuracy_score(labels, orig_preds))
    para_acc    = float(accuracy_score(labels, para_preds))
    orig_f1     = float(f1_score(labels, orig_preds, average="macro"))
    para_f1     = float(f1_score(labels, para_preds, average="macro"))

    per_class = {}
    for cls_id, cls_name in LABEL_NAMES.items():
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        per_class[cls_name] = {
            "n_samples": int(mask.sum()),
            "flip_rate": float(np.mean(orig_preds[mask] != para_preds[mask])),
        }

    return {
        "consistency":   consistency,
        "flip_rate":     1.0 - consistency,
        "orig_acc":      orig_acc,
        "para_acc":      para_acc,
        "acc_drop":      orig_acc - para_acc,
        "orig_f1_macro": orig_f1,
        "para_f1_macro": para_f1,
        "f1_drop":       orig_f1 - para_f1,
        "per_class":     per_class,
    }


def print_robustness(metrics, title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print(f"  Consistency : {metrics['consistency']:.4f}  "
          f"({metrics['consistency']*100:.2f}%)")
    print(f"  Flip rate   : {metrics['flip_rate']:.4f}")
    print(f"  Acc  orig → para : {metrics['orig_acc']:.4f} → "
          f"{metrics['para_acc']:.4f}  ({metrics['acc_drop']:+.4f})")
    print(f"  F1   orig → para : {metrics['orig_f1_macro']:.4f} → "
          f"{metrics['para_f1_macro']:.4f}  ({metrics['f1_drop']:+.4f})")
    print("\n  Per-class flip rate:")
    for cls_name, info in metrics["per_class"].items():
        bar = "█" * int(info["flip_rate"] * 20)
        print(f"    {cls_name:<16} {info['flip_rate']:.4f}  {bar}")


def similarity_bucket_analysis(df, orig_preds, para_preds):
    df = df.copy()
    df["flipped"] = (np.array(orig_preds) != np.array(para_preds)).astype(int)
    bins   = [0.0, 0.7, 0.8, 0.9, 0.95, 1.01]
    labels = ["<0.70", "0.70–0.80", "0.80–0.90", "0.90–0.95", "≥0.95"]
    df["sim_bucket"] = pd.cut(df["similarity"], bins=bins, labels=labels, right=False)
    return (
        df.groupby("sim_bucket", observed=True)["flipped"]
          .agg(n="count", flip_rate="mean")
          .reset_index()
    )


def plot_confusion_matrices(labels, orig_preds, para_preds, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, preds, title in zip(axes, [orig_preds, para_preds],
                                 ["Original", "Paraphrased"]):
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_NAMES_LIST,
                    yticklabels=LABEL_NAMES_LIST, ax=ax)
        ax.set_title(f"Confusion Matrix — {title}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_comparison_across_sources(summary, save_path="cnn_comparison_overview.png"):
    rows = [{
        "source": m["paraphrase_source"],
        "consistency": m["consistency"],
        "acc_drop": m["acc_drop"],
        "f1_drop": m["f1_drop"],
    } for m in summary.values()]
    df = pd.DataFrame(rows).set_index("source")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title, color in zip(
        axes,
        ["consistency", "acc_drop", "f1_drop"],
        ["Consistency (↑)", "Accuracy Drop (↓)", "F1 Drop (↓)"],
        ["#4C72B0", "#C44E52", "#55A868"],
    ):
        df[col].plot(kind="bar", ax=ax, rot=0, color=color, edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.set_xlabel("Paraphrase source")
        for i, v in enumerate(df[col]):
            ax.text(i, v + (0.002 if v >= 0 else -0.01),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    fig.suptitle("TextCNN — Robustness Across Paraphrase Sources", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved overview → {save_path}")

def evaluate_source(model, vocab, source_name, df, out_dir):
    combo = f"cnn_{source_name}"
    print(f"\n{'#'*60}\n#  {combo}\n{'#'*60}")

    labels = df["label"].tolist()
    print("  Predicting on originals...")
    orig_preds = cnn_predict(model, vocab, df["original_text"].tolist())
    print("  Predicting on paraphrases...")
    para_preds = cnn_predict(model, vocab, df["paraphrase"].tolist())

    metrics = compute_robustness(orig_preds, para_preds, labels)
    print_robustness(metrics, f"CNN × {source_name}")

    buckets = similarity_bucket_analysis(df, orig_preds, para_preds)
    print("\n  Flip rate by similarity bucket:")
    for _, row in buckets.iterrows():
        bar = "█" * int(row["flip_rate"] * 20)
        print(f"    {str(row['sim_bucket']):<12} n={row['n']:>4}  "
              f"flip={row['flip_rate']:.4f}  {bar}")

    plot_confusion_matrices(labels, orig_preds, para_preds,
                             os.path.join(out_dir, f"confusion_{combo}.png"))

    result = {
        "model": "cnn",
        "paraphrase_source": source_name,
        "n_samples": len(df),
        **metrics,
        "similarity_buckets": buckets.to_dict(orient="records"),
    }
    out_json = os.path.join(out_dir, f"robustness_{combo}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved → {out_json}")
    return combo, result


def main(csv_dir=".", out_dir="."):
    os.makedirs(out_dir, exist_ok=True)

    model, vocab = get_cnn_model()

    summary = {}
    for source, fname in PARAPHRASE_SOURCES.items():
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        df = load_paraphrase_csv(path)
        combo, result = evaluate_source(model, vocab, source, df, out_dir)
        summary[combo] = result

    summary_path = os.path.join(out_dir, "robustness_summary_cnn.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nUnified summary → {summary_path}")

    plot_comparison_across_sources(
        summary, os.path.join(out_dir, "cnn_comparison_overview.png")
    )

    print(f"\n{'='*66}\n  Summary\n{'='*66}")
    print(f"  {'Source':<16} {'Consist.':>9} {'OrigAcc':>8} "
          f"{'ParaAcc':>8} {'AccDrop':>8} {'F1Drop':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in summary.values():
        print(f"  {m['paraphrase_source']:<16} "
              f"{m['consistency']:>9.4f} "
              f"{m['orig_acc']:>8.4f} {m['para_acc']:>8.4f} "
              f"{m['acc_drop']:>+8.4f} {m['f1_drop']:>+8.4f}")


if __name__ == "__main__":
    main(csv_dir=".", out_dir=".")