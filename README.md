# Fraud Detection with Random Forest

Automatic detection of fraudulent bank transactions using a Random Forest classifier optimized with GridSearchCV.

## Results

| Metric | Value |
|---|---|
| Optimizer | GridSearchCV (5-fold CV) |
| Scoring | Precision |
| Threshold | 0.19 (adjusted for class imbalance) |

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