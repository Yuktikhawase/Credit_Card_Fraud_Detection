from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "creditcard.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(data_path: Path, output_dir: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    combined = "\n".join(x for x in [result.stdout, result.stderr] if x.strip())
    return result.returncode == 0, combined or "Pipeline finished with no terminal output."


def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

    st.title("Credit Card Fraud Detection Dashboard")
    st.caption("Run analysis, explore EDA outputs, and inspect fraud-detection metrics.")

    with st.sidebar:
        st.header("Run Settings")
        data_path_text = st.text_input("Dataset path", value=str(DEFAULT_DATASET))
        output_dir_text = st.text_input("Output directory", value=str(OUTPUTS_DIR))

        run_clicked = st.button("Run Full Pipeline", type="primary", use_container_width=True)

    data_path = Path(data_path_text)
    output_dir = Path(output_dir_text)

    if run_clicked:
        if not data_path.exists():
            st.error(f"Dataset not found: {data_path}")
        else:
            with st.spinner("Running analysis pipeline..."):
                ok, logs = run_pipeline(data_path, output_dir)
            if ok:
                st.success("Pipeline completed successfully.")
            else:
                st.error("Pipeline failed. Check logs below.")
            st.text_area("Execution log", logs, height=180)

    st.subheader("Dataset Preview")
    if data_path.exists():
        df = pd.read_csv(data_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        fraud_rate = float(df["Class"].mean() * 100) if "Class" in df.columns else 0.0
        c3.metric("Fraud Rate", f"{fraud_rate:.4f}%")

        st.dataframe(df.head(15), use_container_width=True)
    else:
        st.info("Dataset path does not exist yet.")

    metrics_path = output_dir / "models" / "model_metrics.json"
    report_path = output_dir / "reports" / "project_report.md"

    st.subheader("Model Metrics")
    metrics = read_json_if_exists(metrics_path)
    if metrics:
        model_names = [k for k in metrics.keys() if k != "dataset"]
        for model_name in model_names:
            model_metrics = metrics.get(model_name, {})
            st.markdown(f"### {model_name}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{model_metrics.get('accuracy', 0):.4f}")
            m2.metric("Precision", f"{model_metrics.get('precision', 0):.4f}")
            m3.metric("Recall", f"{model_metrics.get('recall', 0):.4f}")
            m4.metric("F1", f"{model_metrics.get('f1_score', 0):.4f}")
            st.write("Confusion matrix [ [TN, FP], [FN, TP] ]:")
            st.write(model_metrics.get("confusion_matrix", []))
    else:
        st.info("No metrics found yet. Run the pipeline first.")

    st.subheader("EDA Visualizations")
    figure_files = [
        output_dir / "figures" / "class_distribution.png",
        output_dir / "figures" / "amount_distribution_by_class.png",
        output_dir / "figures" / "hourly_transaction_activity.png",
        output_dir / "figures" / "correlation_heatmap.png",
    ]

    for fig in figure_files:
        if fig.exists():
            st.image(str(fig), caption=fig.name, use_container_width=True)

    if report_path.exists():
        st.subheader("Project Report")
        st.markdown(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
