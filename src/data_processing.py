from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class CleaningSummary:
    original_rows: int
    original_columns: int
    duplicate_rows_removed: int
    missing_values_before: int
    missing_values_after: int
    outliers_capped_amount: int


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    return pd.read_csv(csv_path)


def _cap_outliers_iqr(series: pd.Series, whisker: float = 1.5) -> Tuple[pd.Series, int]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - whisker * iqr
    high = q3 + whisker * iqr
    clipped = series.clip(lower=low, upper=high)
    outliers = int(((series < low) | (series > high)).sum())
    return clipped, outliers


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningSummary]:
    clean_df = df.copy()

    original_rows, original_columns = clean_df.shape
    missing_before = int(clean_df.isna().sum().sum())

    duplicate_count = int(clean_df.duplicated().sum())
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    numeric_columns = clean_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if clean_df[col].isna().any():
            clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    outliers_capped = 0
    if "Amount" in clean_df.columns:
        clean_df["Amount"], outliers_capped = _cap_outliers_iqr(clean_df["Amount"])

    missing_after = int(clean_df.isna().sum().sum())

    summary = CleaningSummary(
        original_rows=original_rows,
        original_columns=original_columns,
        duplicate_rows_removed=duplicate_count,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
        outliers_capped_amount=outliers_capped,
    )
    return clean_df, summary


def basic_profile(df: pd.DataFrame) -> Dict[str, object]:
    profile: Dict[str, object] = {
        "shape": tuple(df.shape),
        "columns": df.columns.tolist(),
        "class_distribution": (
            df["Class"].value_counts(normalize=True).rename("percentage").mul(100).round(4).to_dict()
            if "Class" in df.columns
            else {}
        ),
        "describe": df.describe().to_dict(),
    }
    return profile
