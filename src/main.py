from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.data_processing import basic_profile, clean_data, load_data
from src.eda import create_visualizations, relationship_analysis, variable_analysis
from src.model import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Analytics Project")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("creditcard.csv"),
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    figures_dir = args.output_dir / "figures"
    reports_dir = args.output_dir / "reports"
    models_dir = args.output_dir / "models"

    raw_df = load_data(args.data_path)
    clean_df, cleaning_summary = clean_data(raw_df)

    profile = basic_profile(clean_df)
    variable_stats = variable_analysis(clean_df)
    relation_stats = relationship_analysis(clean_df)
    visualization_files = create_visualizations(clean_df, figures_dir)
    model_metrics, best_model = train_and_evaluate(clean_df, models_dir)

    reports_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = reports_dir / "analysis_summary.json"
    report_path = reports_dir / "project_report.md"

    with analysis_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "cleaning_summary": cleaning_summary.__dict__,
                "dataset_profile": profile,
                "variable_analysis": variable_stats,
                "relationship_analysis": relation_stats,
                "visualizations": visualization_files,
                "model_metrics_file": str(models_dir / "model_metrics.json"),
                "best_model": best_model,
            },
            f,
            indent=2,
        )

    class_distribution = profile.get("class_distribution", {})
    if not isinstance(class_distribution, dict):
        class_distribution = {}

    fraud_pct = class_distribution.get(1, class_distribution.get("1", None))
    non_fraud_pct = class_distribution.get(0, class_distribution.get("0", None))

    report_text = f"""# Credit Card Fraud Detection Project\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## 1. Data Collection and Cleaning\n- Source dataset: `{args.data_path}`\n- Original shape: {cleaning_summary.original_rows} rows, {cleaning_summary.original_columns} columns\n- Duplicate rows removed: {cleaning_summary.duplicate_rows_removed}\n- Missing values before cleaning: {cleaning_summary.missing_values_before}\n- Missing values after cleaning: {cleaning_summary.missing_values_after}\n- Amount outliers capped using IQR: {cleaning_summary.outliers_capped_amount}\n\n## 2. Data Visualization\nGenerated figures:\n- Class distribution\n- Amount distribution by class\n- Hourly transaction activity (time-series style)\n- Correlation heatmap\n\nFigure directory: `{figures_dir}`\n\n## 3. Variable Analysis\n- Numeric variables analyzed for mean, median, std, min, max, and IQR-based outlier count\n- Full details in `{analysis_path}` under `variable_analysis`\n\n## 4. Relationship Analysis\n- Correlation with target class computed for all numeric variables\n- Additional Pearson correlation between Amount and Time\n- Full details in `{analysis_path}` under `relationship_analysis`\n\n## 5. Fraud Detection Modeling\nModels trained:\n- Amount IQR Rule-Based Classifier\n- MAD-Score Rule-Based Classifier\n\nBest model by F1 score: **{best_model}**\nDetailed metrics file: `{models_dir / 'model_metrics.json'}`\n\n## 6. Key Class Imbalance Insight\n- Non-fraud percentage: {non_fraud_pct}\n- Fraud percentage: {fraud_pct}\n\n## 7. Assumptions and Limitations\n- Outlier capping applied to `Amount` only for robust scaling of extreme transaction values\n- Baseline models are rule-based anomaly detectors and may have lower recall than advanced machine learning methods\n- For real-time production detection, threshold tuning, drift monitoring, and periodic retraining are recommended\n"""

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    print("Project pipeline completed successfully.")
    print(f"Analysis summary: {analysis_path}")
    print(f"Project report: {report_path}")
    print(f"Model metrics: {models_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
