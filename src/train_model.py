"""
train_model.py
--------------
Trains multiple ML models for credit card fraud detection,
evaluates them, selects the best, and saves it to disk.

Models trained:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Gradient Boosting Classifier

File Location: credit_card_fraud_predictor/src/train_model.py
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics       import (classification_report, confusion_matrix,
                                   roc_auc_score, f1_score, precision_score, recall_score)


MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _print_metrics(name: str, y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n{'─'*50}")
    print(f"  Model : {name}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {pre:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred, target_names=["Legit","Fraud"]))
    return {"name": name, "auc": auc, "f1": f1,
            "precision": pre, "recall": rec,
            "model": None}  # model set after


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test):
    """
    Trains all models, evaluates them, picks the best by F1,
    saves every model + the best model separately.

    Returns: dict with results and best_model
    """
    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    candidates = [
        ("Logistic Regression",    train_logistic_regression),
        ("Random Forest",          train_random_forest),
        ("Gradient Boosting",      train_gradient_boosting),
    ]

    results = []
    for name, train_fn in candidates:
        print(f"\n[►] Training {name} …")
        model   = train_fn(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        metrics = _print_metrics(name, y_test, y_pred, y_prob)
        metrics["model"] = model
        results.append(metrics)

        # Save individual model
        pkl_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  [✓] Saved → {pkl_path}")

    # Pick best by F1
    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'='*55}")
    print(f"  🏆  BEST MODEL  →  {best['name']}")
    print(f"      F1={best['f1']:.4f}   AUC={best['auc']:.4f}")
    print(f"{'='*55}\n")

    best_path = MODELS_DIR / "best_model.pkl"
    with open(best_path, "wb") as f:
        pickle.dump({"model": best["model"], "name": best["name"]}, f)
    print(f"[✓] Best model saved → {best_path}")

    return results, best


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_dataset import generate_fraud_dataset
    from preprocess import load_data, explore_data, preprocess

    df = generate_fraud_dataset()
    explore_data(df)
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
    train_all(X_train, X_test, y_train, y_test)
