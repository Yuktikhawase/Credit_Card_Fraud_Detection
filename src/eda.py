from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    if "Class" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x="Class", hue="Class", palette="Set2", legend=False)
        plt.title("Class Distribution (0: Non-Fraud, 1: Fraud)")
        class_plot = output_dir / "class_distribution.png"
        _save_plot(class_plot)
        saved["class_distribution"] = str(class_plot)

    if {"Amount", "Class"}.issubset(df.columns):
        plt.figure(figsize=(9, 5))
        sns.histplot(data=df, x="Amount", hue="Class", bins=60, kde=True, stat="density", common_norm=False)
        plt.title("Transaction Amount Distribution by Class")
        amount_plot = output_dir / "amount_distribution_by_class.png"
        _save_plot(amount_plot)
        saved["amount_distribution"] = str(amount_plot)

    if "Time" in df.columns:
        ts_df = df.copy()
        ts_df["hour"] = (ts_df["Time"] / 3600).astype(int)
        grp = ts_df.groupby(["hour", "Class"], as_index=False).size() if "Class" in ts_df.columns else ts_df.groupby("hour", as_index=False).size()

        plt.figure(figsize=(10, 5))
        if "Class" in ts_df.columns:
            sns.lineplot(data=grp, x="hour", y="size", hue="Class", marker="o")
            plt.legend(title="Class")
        else:
            sns.lineplot(data=grp, x="hour", y="size", marker="o")
        plt.title("Hourly Transaction Activity")
        plt.xlabel("Hour")
        plt.ylabel("Number of Transactions")
        ts_plot = output_dir / "hourly_transaction_activity.png"
        _save_plot(ts_plot)
        saved["hourly_activity"] = str(ts_plot)

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        corr_plot = output_dir / "correlation_heatmap.png"
        _save_plot(corr_plot)
        saved["correlation_heatmap"] = str(corr_plot)

    return saved


def variable_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    analysis: Dict[str, Dict[str, float]] = {}
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = int(((series < low) | (series > high)).sum())

        analysis[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "outlier_count_iqr": float(outliers),
        }

    return analysis


def relationship_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    numeric_df = df.select_dtypes(include=["number"])

    if "Class" in numeric_df.columns:
        class_corr = numeric_df.corr(numeric_only=True)["Class"].sort_values(ascending=False)
        results["correlation_with_class"] = {k: float(v) for k, v in class_corr.to_dict().items()} # type: ignore

    if {"Amount", "Time"}.issubset(numeric_df.columns):
        corr = numeric_df[["Amount", "Time"]].corr(numeric_only=True).iloc[0, 1]
        results["amount_time_correlation"] = {"pearson": float(corr)} # type: ignore

    return results
