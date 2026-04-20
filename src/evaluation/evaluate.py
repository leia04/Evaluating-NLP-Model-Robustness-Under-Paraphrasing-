import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from tfidf_svm_model import (
    load_ag_news as svm_load_ag_news,
    build_pipeline as svm_build_pipeline,
    train as svm_train,
    save_pipeline, load_pipeline,
    LABEL_NAMES,
)
from bert_model import (
    build_model as bert_build_model,
    train as bert_train,
    predict_on_texts as bert_predict,
    save_model as bert_save_model,
    load_model as bert_load_model,
    load_ag_news as bert_load_ag_news,
)
import bert_model as bert_mod

LABEL_NAMES_LIST = [LABEL_NAMES[i] for i in range(4)]

PARAPHRASE_SOURCES = {
    "backtranslation": "paraphrase_backtranslation.csv",
    "bart":            "paraphrase_bart.csv",
    "t5_chatgpt":      "paraphrase_t5_chatgpt.csv",
}

BERT_SUBSAMPLE   = 40_000   
BERT_EPOCHS      = 2        
BERT_MODEL_DIR   = "bert_model"

def load_paraphrase_csv(path):
    df = pd.read_csv(path)
    required = {"original_text", "label", "paraphrase", "similarity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["original_text", "paraphrase", "label"]).reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN")

    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} rows from '{path}'")
    print(f"  Avg similarity: {df['similarity'].mean():.4f}  "
          f"(min {df['similarity'].min():.4f})")
    return df

def get_svm_predictor(bert_unused=None):
    """Return a callable predict(texts) -> np.ndarray for the SVM pipeline."""
    path = "svm_pipeline.pkl"
    try:
        pipeline = load_pipeline(path)
    except FileNotFoundError:
        print(f"'{path}' not found — training SVM from scratch...")
        train_df, _ = svm_load_ag_news()
        pipeline = svm_train(svm_build_pipeline(), train_df)
        save_pipeline(pipeline, path)

    def predict(texts):
        return np.array(pipeline.predict(list(texts)))
    return predict


def get_bert_predictor():
    """Load from disk if available, otherwise train on a subsample."""
    if os.path.isdir(BERT_MODEL_DIR) and os.listdir(BERT_MODEL_DIR):
        print(f"Loading BERT from '{BERT_MODEL_DIR}/'...")
        tokenizer, model = bert_load_model(BERT_MODEL_DIR)
    else:
        print(f"'{BERT_MODEL_DIR}/' not found — training BERT...")
        train_df, _ = bert_load_ag_news()
        if BERT_SUBSAMPLE and BERT_SUBSAMPLE < len(train_df):
            train_df = (
                train_df.groupby("label", group_keys=False)
                        .apply(lambda g: g.sample(
                            n=BERT_SUBSAMPLE // 4, random_state=42))
                        .reset_index(drop=True)
            )
            print(f"  Subsampled to {len(train_df):,} "
                  f"(stratified, {BERT_SUBSAMPLE//4}/class)")
        bert_mod.EPOCHS = BERT_EPOCHS
        print(f"  Epochs: {BERT_EPOCHS}")
        tokenizer, model = bert_build_model()
        model = bert_train(model, tokenizer, train_df)
        bert_save_model(model, tokenizer, BERT_MODEL_DIR)

    def predict(texts):
        return np.array(bert_predict(model, tokenizer, list(texts)))
    return predict

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


def plot_comparison_bar(summary, save_path="comparison_overview.png"):
    """Grouped bar chart: consistency & acc_drop across model × source."""
    rows = []
    for combo, m in summary.items():
        rows.append({
            "combo": combo,
            "model": m["model"],
            "source": m["paraphrase_source"],
            "consistency": m["consistency"],
            "acc_drop": m["acc_drop"],
            "f1_drop": m["f1_drop"],
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, title in zip(
        axes,
        ["consistency", "acc_drop", "f1_drop"],
        ["Prediction Consistency (↑)", "Accuracy Drop (↓)", "F1 Drop (↓)"],
    ):
        pivot = df.pivot(index="source", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=0,
                   color=["#4C72B0", "#C44E52"], edgecolor="black")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xlabel("Paraphrase source")
        ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved overview → {save_path}")

def evaluate_combination(model_name, predictor, source_name, df, out_dir):
    combo = f"{model_name}_{source_name}"
    print(f"\n{'#'*60}\n#  {combo}\n{'#'*60}")

    texts_orig = df["original_text"].tolist()
    texts_para = df["paraphrase"].tolist()
    labels     = df["label"].tolist()

    print("  Predicting on originals...")
    orig_preds = predictor(texts_orig)
    print("  Predicting on paraphrases...")
    para_preds = predictor(texts_para)

    metrics = compute_robustness(orig_preds, para_preds, labels)
    print_robustness(metrics, f"{model_name.upper()} × {source_name}")

    buckets = similarity_bucket_analysis(df, orig_preds, para_preds)
    print("\n  Flip rate by similarity bucket:")
    for _, row in buckets.iterrows():
        bar = "█" * int(row["flip_rate"] * 20)
        print(f"    {str(row['sim_bucket']):<12} n={row['n']:>4}  "
              f"flip={row['flip_rate']:.4f}  {bar}")

    cm_path = os.path.join(out_dir, f"confusion_{combo}.png")
    plot_confusion_matrices(labels, orig_preds, para_preds, cm_path)

    result = {
        "model": model_name,
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

    datasets = {}
    for source, fname in PARAPHRASE_SOURCES.items():
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        datasets[source] = load_paraphrase_csv(path)

    if not datasets:
        raise RuntimeError("No paraphrase CSVs found")

    summary = {}

    print("\n" + "="*60 + "\n  SVM\n" + "="*60)
    svm_predict = get_svm_predictor()
    for source, df in datasets.items():
        combo, result = evaluate_combination(
            "svm", svm_predict, source, df, out_dir
        )
        summary[combo] = result

    print("\n" + "="*60 + "\n  BERT\n" + "="*60)
    bert_predict_fn = get_bert_predictor()
    for source, df in datasets.items():
        combo, result = evaluate_combination(
            "bert", bert_predict_fn, source, df, out_dir
        )
        summary[combo] = result

    summary_path = os.path.join(out_dir, "robustness_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nUnified summary → {summary_path}")

    plot_comparison_bar(summary, os.path.join(out_dir, "comparison_overview.png"))

    print(f"\n{'='*72}")
    print(f"  Summary")
    print(f"{'='*72}")
    print(f"  {'Model':<6} {'Source':<16} {'Consist.':>9} "
          f"{'OrigAcc':>8} {'ParaAcc':>8} {'AccDrop':>8} {'F1Drop':>8}")
    print(f"  {'-'*6} {'-'*16} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for combo, m in summary.items():
        print(f"  {m['model']:<6} {m['paraphrase_source']:<16} "
              f"{m['consistency']:>9.4f} "
              f"{m['orig_acc']:>8.4f} {m['para_acc']:>8.4f} "
              f"{m['acc_drop']:>+8.4f} {m['f1_drop']:>+8.4f}")


if __name__ == "__main__":
    main(csv_dir=".", out_dir=".")
