"""Train and use a transparent text baseline on Fakeddit titles.

This is explicitly a TF-IDF/logistic-regression baseline, not FDHN. Fakeddit labels
are subreddit-derived and are weak supervision; scores are ranking scores, not calibrated probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from fuzzy_fusion import infer


class TextBranchUnavailable(RuntimeError):
    pass


class TextPredictor:
    def __init__(self, model_path: str | Path = "text_fake_baseline.joblib"):
        path = Path(model_path)
        if not path.is_file():
            raise TextBranchUnavailable(f"text model not found: {path}. Train it with train_text.py first.")
        bundle = joblib.load(path)
        self.model: Pipeline = bundle["model"]
        self.metadata: dict[str, Any] = bundle["metadata"]

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        score = float(self.model.predict_proba([text])[0][1])
        return {
            "fake_score": score,
            "score_kind": "uncalibrated model ranking score (not a verified probability)",
            "model": self.metadata,
        }


def train(data_dir: str | Path, output: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    splits = {name: pd.read_csv(data_dir / f"{name}.csv") for name in ("train", "val", "test")}
    for name, df in splits.items():
        if not {"clean_title", "is_fake"}.issubset(df.columns):
            raise ValueError(f"{name}.csv must contain clean_title and is_fake columns")
        if df.empty:
            raise ValueError(f"{name}.csv is empty")
    train_df, val_df, test_df = (splits[k] for k in ("train", "val", "test"))
    # Do not fit vocabulary or classifier on validation/test data.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                  strip_accents="unicode", sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    x_train = train_df.clean_title.fillna("").astype(str)
    y_train = train_df.is_fake.astype(int)
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(val_df.clean_title.fillna("").astype(str))[:, 1]
    test_scores = model.predict_proba(test_df.clean_title.fillna("").astype(str))[:, 1]
    def metrics(df: pd.DataFrame, scores) -> dict[str, Any]:
        y = df.is_fake.astype(int).to_numpy()
        pred = (scores >= 0.5).astype(int)
        fuzzy_outcomes = [infer(text_fake=float(s)).outcome for s in scores]
        return {
            "n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())),
            "false_negative_rate": float(((pred == 0) & (y == 1)).sum() / max(1, (y == 1).sum())),
            "fuzzy_outcome_counts_text_only": {name: fuzzy_outcomes.count(name)
                for name in ("Likely Real", "Likely Fake", "Needs Verification")},
            "needs_verification_rate_text_only": float(fuzzy_outcomes.count("Needs Verification") / len(fuzzy_outcomes)),
            "classification_report": classification_report(y, pred, labels=[0, 1],
                target_names=["real_label_0", "fake_label_1"], output_dict=True, zero_division=0),
        }
    report = {
        "model": "TF-IDF word unigram/bigram + balanced logistic regression",
        "label_source": "Fakeddit 2-way label; distant supervision from subreddit, not human fact checking",
        "score_interpretation": "predict_proba ranking score; not evaluated/calibrated as a real-world probability",
        "threshold": 0.5,
        "validation": metrics(val_df, val_scores),
        "test": metrics(test_df, test_scores),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: report[k] for k in ("model", "label_source", "score_interpretation", "threshold")}
    joblib.dump({"model": model, "metadata": metadata}, output)
    return {"checkpoint": str(output), "evaluation": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="fakeddit_sample")
    parser.add_argument("--out", default="text_fake_baseline.joblib")
    parser.add_argument("--report", default="evaluation/text_baseline.json")
    args = parser.parse_args()
    result = train(args.data, args.out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result["evaluation"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
