"""
predict.py
----------
Loads the best trained model and predicts whether a new
credit card transaction is FRAUD or LEGITIMATE.

File Location: credit_card_fraud_predictor/src/predict.py
"""

import pickle
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_best_model():
    path = MODELS_DIR / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}.\n"
                                "Run main.py first to train the model.")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[✓] Loaded model : {bundle['name']}")
    return bundle["model"], bundle["name"]


def predict_transaction(transaction: dict, scaler, model) -> dict:
    """
    Predict a single transaction.

    Parameters
    ----------
    transaction : dict with keys:
        Amount, Hour, V1, V2, V3, V4, V5,
        Is_Online, Is_Foreign, Transactions_Last1Hr
    scaler : fitted StandardScaler
    model  : trained classifier

    Returns
    -------
    dict with prediction, probability, and risk level
    """
    # Feature engineering (mirrors preprocess.py)
    log_amount      = np.log1p(transaction["Amount"])
    suspicious_hour = 1 if transaction["Hour"] < 6 else 0
    risk_score      = (transaction["Is_Online"]  * 0.3 +
                       transaction["Is_Foreign"] * 0.4 +
                       suspicious_hour           * 0.3)

    features = np.array([[
        log_amount,
        transaction["Hour"],
        transaction["V1"],
        transaction["V2"],
        transaction["V3"],
        transaction["V4"],
        transaction["V5"],
        transaction["Is_Online"],
        transaction["Is_Foreign"],
        transaction["Transactions_Last1Hr"],
        suspicious_hour,
        risk_score
    ]])

    features_scaled = scaler.transform(features)
    pred   = model.predict(features_scaled)[0]
    prob   = model.predict_proba(features_scaled)[0][1]

    risk_level = ("🔴 HIGH"   if prob > 0.7  else
                  "🟡 MEDIUM" if prob > 0.4  else
                  "🟢 LOW")

    return {
        "prediction":   "⚠️  FRAUD"    if pred == 1 else "✅ LEGITIMATE",
        "fraud_prob":   round(float(prob) * 100, 2),
        "risk_level":   risk_level,
        "raw_pred":     int(pred)
    }


def demo_predictions(scaler, model):
    """Run a few sample transactions to demonstrate the predictor."""
    samples = [
        {
            "label":   "Normal daytime purchase",
            "Amount": 45.00, "Hour": 14,
            "V1": 0.1, "V2": -0.2, "V3": 0.3, "V4": -0.1, "V5": 0.2,
            "Is_Online": 0, "Is_Foreign": 0, "Transactions_Last1Hr": 2
        },
        {
            "label":   "Suspicious large midnight transfer",
            "Amount": 3200.00, "Hour": 2,
            "V1": -4.1, "V2": 3.5, "V3": -5.2, "V4": 4.0, "V5": -3.0,
            "Is_Online": 1, "Is_Foreign": 1, "Transactions_Last1Hr": 12
        },
        {
            "label":   "Foreign online purchase (medium risk)",
            "Amount": 180.00, "Hour": 21,
            "V1": -1.5, "V2": 1.2, "V3": -1.8, "V4": 1.0, "V5": -0.8,
            "Is_Online": 1, "Is_Foreign": 1, "Transactions_Last1Hr": 3
        },
    ]

    print("\n" + "="*55)
    print("  DEMO PREDICTIONS")
    print("="*55)
    for s in samples:
        txn   = {k: v for k, v in s.items() if k != "label"}
        result = predict_transaction(txn, scaler, model)
        print(f"\n  Transaction : {s['label']}")
        print(f"  Amount      : ${s['Amount']:.2f}  |  Hour: {s['Hour']:02d}:00")
        print(f"  Online: {'Yes' if s['Is_Online'] else 'No'}  "
              f"Foreign: {'Yes' if s['Is_Foreign'] else 'No'}  "
              f"Freq: {s['Transactions_Last1Hr']}")
        print(f"  ─── Result ───────────────────────────────")
        print(f"  Prediction  : {result['prediction']}")
        print(f"  Fraud Prob  : {result['fraud_prob']}%")
        print(f"  Risk Level  : {result['risk_level']}")
    print("="*55)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocess import load_data, preprocess
    from sklearn.preprocessing import StandardScaler
    import pickle

    # Load scaler
    scaler_path = Path(__file__).parent.parent / "models" / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model, name = load_best_model()
    demo_predictions(scaler, model)
