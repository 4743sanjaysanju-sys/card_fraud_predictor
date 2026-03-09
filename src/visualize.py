"""
visualize.py
------------
Generates all evaluation charts and saves them to outputs/.

Charts produced:
  1. Class distribution (bar)
  2. Correlation heatmap
  3. Amount distribution (fraud vs legit)
  4. ROC curves (all models)
  5. Confusion matrices (all models)
  6. Feature importance (Random Forest)

File Location: credit_card_fraud_predictor/src/visualize.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (roc_curve, auc, confusion_matrix)

OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

PALETTE = {"Legit": "#2ecc71", "Fraud": "#e74c3c"}
sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Class Distribution ─────────────────────────────────────
def plot_class_distribution(df):
    counts = df["Class"].value_counts().rename({0: "Legit", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE["Legit"], PALETTE["Fraud"]], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"{val:,}\n({val/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Class Distribution: Legit vs Fraud", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.tight_layout()
    p = OUTPUTS / "1_class_distribution.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")


# ── 2. Correlation Heatmap ────────────────────────────────────
def plot_correlation_heatmap(df):
    num_cols = ["Amount", "V1", "V2", "V3", "V4", "V5",
                "Is_Online", "Is_Foreign", "Transactions_Last1Hr", "Class"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = OUTPUTS / "2_correlation_heatmap.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")


# ── 3. Amount Distribution ────────────────────────────────────
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axes, [False, True]):
        legit = df[df["Class"]==0]["Amount"]
        fraud = df[df["Class"]==1]["Amount"]
        data  = [np.log1p(legit), np.log1p(fraud)] if log else [legit, fraud]
        labels = ["Legit", "Fraud"]
        colors = [PALETTE["Legit"], PALETTE["Fraud"]]
        for d, lbl, c in zip(data, labels, colors):
            ax.hist(d, bins=50, alpha=0.6, color=c, label=lbl, edgecolor="white")
        ax.set_title(f"Amount Distribution {'(Log Scale)' if log else ''}", fontweight="bold")
        ax.set_xlabel("log(1 + Amount)" if log else "Amount ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    p = OUTPUTS / "3_amount_distribution.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")


# ── 4. ROC Curves ─────────────────────────────────────────────
def plot_roc_curves(results, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#3498db", "#e67e22", "#9b59b6"]
    for res, color in zip(results, colors):
        model  = res["model"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{res['name']} (AUC = {roc_auc:.3f})")
    ax.plot([0,1], [0,1], "k--", lw=1, label="Random Classifier")
    ax.fill_between([0,1], [0,1], alpha=0.05, color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUTPUTS / "4_roc_curves.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")


# ── 5. Confusion Matrices ─────────────────────────────────────
def plot_confusion_matrices(results, X_test, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["model"].predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit","Fraud"],
                    yticklabels=["Legit","Fraud"], ax=ax,
                    linewidths=1, linecolor="white")
        ax.set_title(res["name"], fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = OUTPUTS / "5_confusion_matrices.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[✓] Saved → {p}")


# ── 6. Feature Importance (RF) ────────────────────────────────
def plot_feature_importance(results, feature_names):
    rf_res = next((r for r in results if "Random Forest" in r["name"]), None)
    if rf_res is None:
        return
    model = rf_res["model"]
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp)))
    bars   = ax.barh([feature_names[i] for i in idx[::-1]],
                     imp[idx[::-1]], color=colors, edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, imp[idx[::-1]]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    p = OUTPUTS / "6_feature_importance.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")


# ── 7. Model Comparison Bar Chart ────────────────────────────
def plot_model_comparison(results):
    names   = [r["name"] for r in results]
    metrics = {"F1 Score": [r["f1"]  for r in results],
               "ROC-AUC":  [r["auc"] for r in results],
               "Precision":[r["precision"] for r in results],
               "Recall":   [r["recall"] for r in results]}
    x  = np.arange(len(names))
    w  = 0.2
    colors = ["#3498db","#e67e22","#9b59b6","#1abc9c"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, vals) in enumerate(metrics.items()):
        ax.bar(x + i*w, vals, w, label=metric, color=colors[i], edgecolor="white")
    ax.set_xticks(x + w*1.5); ax.set_xticklabels(names, rotation=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = OUTPUTS / "7_model_comparison.png"
    fig.savefig(p, dpi=150); plt.close()
    print(f"[✓] Saved → {p}")
