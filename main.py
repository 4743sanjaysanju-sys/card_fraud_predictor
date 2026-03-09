"""
main.py
-------
Entry point for the Credit Card Fraud Detection project.
Runs the full ML pipeline end-to-end:
  1. Generate synthetic dataset
  2. Explore & preprocess data
  3. Train multiple ML models
  4. Evaluate & compare models
  5. Generate all visualizations
  6. Save best model
  7. Run demo predictions

File Location: credit_card_fraud_predictor/main.py

HOW TO RUN:
    cd credit_card_fraud_predictor
    python main.py
"""

import sys
import pickle
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_dataset import generate_fraud_dataset
from preprocess       import load_data, explore_data, preprocess
from train_model      import train_all
from visualize        import (plot_class_distribution, plot_correlation_heatmap,
                               plot_amount_distribution, plot_roc_curves,
                               plot_confusion_matrices, plot_feature_importance,
                               plot_model_comparison)
from predict          import load_best_model, demo_predictions


BANNER = """
╔══════════════════════════════════════════════════════╗
║   💳  CREDIT CARD FRAUD DETECTION SYSTEM             ║
║   AI & Machine Learning Project — Python             ║
║   Models: Logistic Regression | Random Forest |      ║
║           Gradient Boosting                          ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── STEP 1: Generate Dataset ──────────────────────────────
    print("STEP 1/7 ► Generating synthetic credit card dataset …")
    df = generate_fraud_dataset(n_samples=10_000, fraud_ratio=0.02)

    # ── STEP 2: Explore Data ──────────────────────────────────
    print("\nSTEP 2/7 ► Exploring dataset …")
    explore_data(df)

    # ── STEP 3: Preprocess ────────────────────────────────────
    print("\nSTEP 3/7 ► Preprocessing & feature engineering …")
    X_train, X_test, y_train, y_test, scaler, features = preprocess(df)

    # Save scaler for future use
    scaler_path = Path(__file__).parent / "models" / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[✓] Scaler saved → {scaler_path}")

    # ── STEP 4: Train Models ──────────────────────────────────
    print("\nSTEP 4/7 ► Training ML models …")
    results, best = train_all(X_train, X_test, y_train, y_test)

    # ── STEP 5: Visualizations ────────────────────────────────
    print("\nSTEP 5/7 ► Generating visualizations …")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_amount_distribution(df)
    plot_roc_curves(results, X_test, y_test)
    plot_confusion_matrices(results, X_test, y_test)
    plot_feature_importance(results, features)
    plot_model_comparison(results)

    # ── STEP 6: Demo Predictions ──────────────────────────────
    print("\nSTEP 6/7 ► Running demo predictions …")
    model, name = load_best_model()
    demo_predictions(scaler, model)

    # ── STEP 7: Summary ───────────────────────────────────────
    print("\nSTEP 7/7 ► Project Summary")
    print("="*55)
    print(f"  Best Model  : {best['name']}")
    print(f"  F1 Score    : {best['f1']:.4f}")
    print(f"  ROC-AUC     : {best['auc']:.4f}")
    print(f"  Precision   : {best['precision']:.4f}")
    print(f"  Recall      : {best['recall']:.4f}")
    print("="*55)
    print("\n  Output files saved to:  credit_card_fraud_predictor/outputs/")
    print("  Trained models saved to: credit_card_fraud_predictor/models/")
    print("\n✅ Pipeline complete!\n")


if __name__ == "__main__":
    main()
