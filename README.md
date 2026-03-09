# 💳 Credit Card Fraud Detection — AI/ML Project

A complete **Python AI & Machine Learning project** that predicts fraudulent credit card transactions using three classification models.

---

## 📁 Project Structure

```
credit_card_fraud_predictor/
│
├── main.py                  ← ✅ Run this to start everything
│
├── src/
│   ├── generate_dataset.py  ← Generates synthetic transaction data
│   ├── preprocess.py        ← Data cleaning & feature engineering
│   ├── train_model.py       ← Trains 3 ML models & picks the best
│   ├── visualize.py         ← Creates all charts & graphs
│   └── predict.py           ← Predict new transactions
│
├── data/
│   └── creditcard_data.csv  ← Auto-generated dataset (10,000 rows)
│
├── models/
│   ├── best_model.pkl               ← Best performing model
│   ├── scaler.pkl                   ← Fitted StandardScaler
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
└── outputs/
    ├── 1_class_distribution.png
    ├── 2_correlation_heatmap.png
    ├── 3_amount_distribution.png
    ├── 4_roc_curves.png
    ├── 5_confusion_matrices.png
    ├── 6_feature_importance.png
    └── 7_model_comparison.png
```

---

## ⚙️ Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## ▶️ How to Run

```bash
cd credit_card_fraud_predictor
python main.py
```

That's it! The pipeline runs all 7 steps automatically.

---

## 🤖 ML Models Used

| Model                  | Purpose                                      |
|------------------------|----------------------------------------------|
| Logistic Regression    | Baseline linear classifier                  |
| Random Forest          | Ensemble of decision trees                  |
| Gradient Boosting      | Sequential boosting for high accuracy        |

---

## 📊 Features Used

| Feature               | Description                                  |
|-----------------------|----------------------------------------------|
| Amount                | Transaction amount in dollars                |
| Hour                  | Hour of day (0–23)                           |
| V1–V5                 | PCA-style anonymized transaction features    |
| Is_Online             | Whether transaction was online (1/0)         |
| Is_Foreign            | Whether transaction was foreign (1/0)        |
| Transactions_Last1Hr  | Number of transactions in past hour          |
| Log_Amount            | Log-transformed amount (engineered)          |
| Suspicious_Hour       | 1 if transaction was between 12AM–5AM        |
| Risk_Score            | Composite risk score (engineered)            |

---

## 📈 Evaluation Metrics

- **ROC-AUC** — overall model discrimination ability
- **F1 Score** — balance between precision and recall
- **Precision** — of all flagged fraud, how many were real?
- **Recall** — of all real fraud, how many did we catch?

---

## 🔮 Predicting a New Transaction

```python
from src.predict import predict_transaction, load_best_model
import pickle

# Load model and scaler
model, name = load_best_model()
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define transaction
transaction = {
    "Amount": 3200.00,
    "Hour": 2,
    "V1": -4.1, "V2": 3.5, "V3": -5.2, "V4": 4.0, "V5": -3.0,
    "Is_Online": 1,
    "Is_Foreign": 1,
    "Transactions_Last1Hr": 12
}

result = predict_transaction(transaction, scaler, model)
print(result)
# → {'prediction': '⚠️  FRAUD', 'fraud_prob': 97.3, 'risk_level': '🔴 HIGH'}
```

---

## 🧠 AI/ML Concepts Covered

- Supervised Learning (Classification)
- Feature Engineering & Scaling
- Class Imbalance handling (`class_weight="balanced"`)
- Model Selection & Hyperparameter Tuning
- Cross-validation via train/test split
- Evaluation: ROC-AUC, Confusion Matrix, F1
- Model Persistence with `pickle`
