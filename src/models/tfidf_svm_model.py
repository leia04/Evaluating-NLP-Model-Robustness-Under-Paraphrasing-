import json
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}


def load_ag_news():
    print("Loading AG News dataset...")
    dataset = load_dataset("fancyzhx/ag_news")
    def to_df(split):
        return pd.DataFrame({"text": split["text"], "label": split["label"]})
    train_df = to_df(dataset["train"])
    test_df  = to_df(dataset["test"])
    print(f"  Train: {len(train_df):,} / Test: {len(test_df):,}")
    return train_df, test_df


def build_pipeline():
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=100_000,
        sublinear_tf=True, min_df=2, strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    classifier = CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=42)
    )
    return Pipeline([("tfidf", vectorizer), ("svm", classifier)])


def train(pipeline, train_df):
    print("Training...")
    pipeline.fit(train_df["text"], train_df["label"])
    print("  Done.\n")
    return pipeline


def evaluate(pipeline, texts, labels, split_name="Test"):
    preds    = pipeline.predict(texts)
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


def predict_on_texts(pipeline, texts):
    return pipeline.predict(texts)


def save_pipeline(pipeline, path="svm_pipeline.pkl"):
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved → {path}")


def load_pipeline(path="svm_pipeline.pkl"):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded ← {path}")
    return pipeline


def save_metrics(metrics_list, path="results_svm.json"):
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved → {path}")
