# 2. Credit Card Fraud Detection Project

This project performs end-to-end data analytics and baseline fraud detection modeling on `creditcard.csv`.

## Project Objectives

- Data Collection and Cleaning
- Data Visualization
- Variable Analysis
- Relationship Analysis
- Summarization and Documentation

## Folder Structure

```text
Credit_Card_Fraud_Detection/
|-- creditcard.csv
|-- requirements.txt
|-- README.md
|-- src/
|   |-- __init__.py
|   |-- data_processing.py
|   |-- eda.py
|   |-- model.py
|   |-- main.py
|-- outputs/
|   |-- figures/
|   |-- models/
|   |-- reports/
```

## Setup

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Project

```powershell
python -m src.main --data-path creditcard.csv --output-dir outputs
```

## Run User Interface

```powershell
streamlit run app.py
```

The dashboard allows you to:
- Run the full pipeline from the browser
- Preview dataset size and sample rows
- View fraud model metrics
- View generated EDA charts
- Read the generated project report

## Generated Outputs

- `outputs/figures/`
  - `class_distribution.png`
  - `amount_distribution_by_class.png`
  - `hourly_transaction_activity.png`
  - `correlation_heatmap.png`
- `outputs/models/model_metrics.json`
- `outputs/reports/analysis_summary.json`
- `outputs/reports/project_report.md`

## Modeling Notes

The pipeline trains two baseline classifiers:

- Amount IQR Rule-Based Classifier
- MAD-Score Rule-Based Classifier

The best model is selected by F1 score and reported in the generated report.

## Real-Time Extension Idea

For production-grade banking systems, this batch pipeline can be extended with streaming ingestion (Kafka/Flink), real-time scoring API, thresholding rules, and alert/reversal workflows.
