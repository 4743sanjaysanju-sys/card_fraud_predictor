"""
generate_dataset.py
-------------------
Generates a synthetic credit card transaction dataset
and saves it to the data/ folder.

File Location: credit_card_fraud_predictor/src/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02, random_state=42):
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def make_transactions(n, is_fraud):
        label = 1 if is_fraud else 0
        if is_fraud:
            amount     = np.random.exponential(scale=400, size=n).clip(1, 5000)
            hour       = np.random.choice(range(0, 6), size=n)           # odd hours
            v1         = np.random.normal(-3.5, 1.5, n)
            v2         = np.random.normal(2.8, 1.2, n)
            v3         = np.random.normal(-4.0, 1.8, n)
            v4         = np.random.normal(3.2, 1.0, n)
            v5         = np.random.normal(-2.5, 1.3, n)
            online     = np.random.choice([0, 1], size=n, p=[0.2, 0.8])
            foreign    = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
            freq       = np.random.randint(5, 20, size=n)
        else:
            amount     = np.random.exponential(scale=80, size=n).clip(1, 2000)
            hour       = np.random.choice(range(6, 23), size=n)
            v1         = np.random.normal(0.0, 1.0, n)
            v2         = np.random.normal(0.0, 1.0, n)
            v3         = np.random.normal(0.0, 1.0, n)
            v4         = np.random.normal(0.0, 1.0, n)
            v5         = np.random.normal(0.0, 1.0, n)
            online     = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
            foreign    = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
            freq       = np.random.randint(1, 6, size=n)

        return pd.DataFrame({
            "Amount":             np.round(amount, 2),
            "Hour":               hour,
            "V1":                 np.round(v1, 4),
            "V2":                 np.round(v2, 4),
            "V3":                 np.round(v3, 4),
            "V4":                 np.round(v4, 4),
            "V5":                 np.round(v5, 4),
            "Is_Online":          online,
            "Is_Foreign":         foreign,
            "Transactions_Last1Hr": freq,
            "Class":              label
        })

    df = pd.concat([
        make_transactions(n_legit, is_fraud=False),
        make_transactions(n_fraud, is_fraud=True)
    ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    out_path = Path(__file__).parent.parent / "data" / "creditcard_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✓] Dataset saved → {out_path}")
    print(f"    Total rows : {len(df):,}")
    print(f"    Fraud rows : {df['Class'].sum():,}  ({df['Class'].mean()*100:.1f}%)")
    return df

if __name__ == "__main__":
    generate_fraud_dataset()
