# Fraud Detection with Random Forest

Automatic detection of fraudulent bank transactions using a Random Forest classifier optimized with GridSearchCV.

## Results

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 0.625 |
| F1 Score | 0.769 |
| Accuracy | 99.1% |
| Best params | max_depth=5, n_estimators=20 |
| Threshold | 0.19 (adjusted for class imbalance) |
| Dataset | 1743 transactions, 46 fraud cases (2.6%) |
| ROC AUC | 0.99 |

## Pipeline

1. Data loading and validation
2. Categorical encoding (merchant state, city, card type)
3. Outlier removal (Z-score, threshold = 3)
4. Train/test split (80/20)
5. Hyperparameter tuning (n_estimators, max_depth)
6. Evaluation — Precision, Recall, F1, ROC-AUC, Confusion Matrix

## Dataset

Bank transaction records with features including transaction amount, merchant info, card type, and purchased items.
Target variable: fraud_flag (binary).

## Setup

pip install -r requirements.txt
python scripts/script.py

## License

MIT