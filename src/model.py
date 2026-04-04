from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

def _train_test_split_stratified(df: pd.DataFrame, target_col: str, test_ratio: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    test_parts = []

    for _, group in df.groupby(target_col):
        n_test = max(1, int(len(group) * test_ratio))
        sampled_idx = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_parts.append(df.loc[sampled_idx])

    test_df = pd.concat(test_parts).sort_index()
    train_df = df.drop(index=test_df.index)
    return train_df, test_df


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _amount_iqr_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    q1 = train_df["Amount"].quantile(0.25)
    q3 = train_df["Amount"].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return (test_df["Amount"] > threshold).astype(int).to_numpy()


def _mad_score_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    feature_cols = [c for c in train_df.columns if c != "Class"]

    med = train_df[feature_cols].median()
    mad = (train_df[feature_cols] - med).abs().median().replace(0, 1e-9)

    robust_z = ((test_df[feature_cols] - med).abs() / mad).mean(axis=1)
    threshold = float(np.percentile(robust_z, 99.5))
    return (robust_z > threshold).astype(int).to_numpy()


def train_and_evaluate(df: pd.DataFrame, output_dir: Path) -> Tuple[Dict[str, object], str]:
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column in dataset for fraud classification.")

    train_df, test_df = _train_test_split_stratified(df, target_col="Class", test_ratio=0.2, random_state=42)

    y_test = test_df["Class"].to_numpy()
    pred_amount_iqr = _amount_iqr_classifier(train_df, test_df)
    pred_mad_score = _mad_score_classifier(train_df, test_df)

    metrics = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "fraud_ratio_train_percent": float(train_df["Class"].mean() * 100),
            "fraud_ratio_test_percent": float(test_df["Class"].mean() * 100),
        },
        "amount_iqr_rule": _binary_metrics(y_test, pred_amount_iqr),
        "mad_score_rule": _binary_metrics(y_test, pred_mad_score),
    }

    best_model = "mad_score_rule" if metrics["mad_score_rule"]["f1_score"] >= metrics["amount_iqr_rule"]["f1_score"] else "amount_iqr_rule"

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, best_model
