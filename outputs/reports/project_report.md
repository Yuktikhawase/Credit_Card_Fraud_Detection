# Credit Card Fraud Detection Project

Generated: 2026-04-04 22:32:38

## 1. Data Collection and Cleaning
- Source dataset: `C:\Users\rakes\Desktop\Credit_Card_Fraud_Detection\creditcard.csv`
- Original shape: 284807 rows, 31 columns
- Duplicate rows removed: 1081
- Missing values before cleaning: 0
- Missing values after cleaning: 0
- Amount outliers capped using IQR: 31685

## 2. Data Visualization
Generated figures:
- Class distribution
- Amount distribution by class
- Hourly transaction activity (time-series style)
- Correlation heatmap

Figure directory: `C:\Users\rakes\Desktop\Credit_Card_Fraud_Detection\outputs\figures`

## 3. Variable Analysis
- Numeric variables analyzed for mean, median, std, min, max, and IQR-based outlier count
- Full details in `C:\Users\rakes\Desktop\Credit_Card_Fraud_Detection\outputs\reports\analysis_summary.json` under `variable_analysis`

## 4. Relationship Analysis
- Correlation with target class computed for all numeric variables
- Additional Pearson correlation between Amount and Time
- Full details in `C:\Users\rakes\Desktop\Credit_Card_Fraud_Detection\outputs\reports\analysis_summary.json` under `relationship_analysis`

## 5. Fraud Detection Modeling
Models trained:
- Amount IQR Rule-Based Classifier
- MAD-Score Rule-Based Classifier

Best model by F1 score: **mad_score_rule**
Detailed metrics file: `C:\Users\rakes\Desktop\Credit_Card_Fraud_Detection\outputs\models\model_metrics.json`

## 6. Key Class Imbalance Insight
- Non-fraud percentage: 99.8333
- Fraud percentage: 0.1667

## 7. Assumptions and Limitations
- Outlier capping applied to `Amount` only for robust scaling of extreme transaction values
- Baseline models are rule-based anomaly detectors and may have lower recall than advanced machine learning methods
- For real-time production detection, threshold tuning, drift monitoring, and periodic retraining are recommended
