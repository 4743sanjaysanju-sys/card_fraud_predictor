"""
preprocess.py
-------------
Handles data loading, cleaning, feature engineering,
and train/test splitting for the fraud detection pipeline.

File Location: credit_card_fraud_predictor/src/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str = None) -> pd.DataFrame:
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "creditcard_data.csv"
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame):
    print("\n" + "="*55)
    print("  DATASET OVERVIEW")
    print("="*55)
    print(df.dtypes.to_string())
    print(f"\nMissing values : {df.isnull().sum().sum()}")
    print(f"\nClass distribution:")
    counts = df["Class"].value_counts()
    for label, cnt in counts.items():
        tag = "FRAUD" if label == 1 else "Legit"
        print(f"  {tag:6s} ({label}): {cnt:,}  ({cnt/len(df)*100:.2f}%)")
    print("="*55)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to improve model performance."""
    df = df.copy()
    # Log-transform skewed Amount
    df["Log_Amount"] = np.log1p(df["Amount"])
    # Flag suspicious hours (midnight – 5 AM)
    df["Suspicious_Hour"] = df["Hour"].apply(lambda h: 1 if h < 6 else 0)
    # Combined risk score
    df["Risk_Score"] = (df["Is_Online"] * 0.3 +
                        df["Is_Foreign"] * 0.4 +
                        df["Suspicious_Hour"] * 0.3)
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = engineer_features(df)

    FEATURES = ["Log_Amount", "Hour", "V1", "V2", "V3", "V4", "V5",
                "Is_Online", "Is_Foreign", "Transactions_Last1Hr",
                "Suspicious_Hour", "Risk_Score"]
    TARGET = "Class"

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"[✓] Train size : {X_train.shape[0]:,}")
    print(f"[✓] Test  size : {X_test.shape[0]:,}")
    print(f"[✓] Features   : {FEATURES}")
    return X_train, X_test, y_train, y_test, scaler, FEATURES


if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    preprocess(df)
