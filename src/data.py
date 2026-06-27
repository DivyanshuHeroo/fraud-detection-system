"""Data loading and splitting for the credit-card fraud dataset.

Dataset: Kaggle "Credit Card Fraud Detection" (ULB / Worldline). 284,807
European card transactions over two days in Sep 2013, 492 of them fraudulent
(~0.172%). Features V1..V28 are PCA components (anonymised); `Time` and
`Amount` are raw. `Class` is the label (1 = fraud).
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Load the raw dataset, with a friendly error if it is missing."""
    if not C.DATA_FILE.exists():
        sys.exit(
            f"\nDataset not found at {C.DATA_FILE}\n"
            "Download the real dataset, e.g.:\n"
            "  curl -sL -o data/creditcard.csv \\\n"
            "    https://huggingface.co/datasets/David-Egea/"
            "Creditcard-fraud-detection/resolve/main/creditcard.csv\n"
        )
    df = pd.read_csv(C.DATA_FILE)
    if C.TARGET not in df.columns:
        sys.exit(f"Expected target column '{C.TARGET}' not found in dataset.")
    if C.DROP_DUPLICATES:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"   dropped {removed:,} duplicate rows (anti-leakage)")
    return df


def describe(df: pd.DataFrame) -> dict:
    """Quick dataset summary used for logging / the dashboard."""
    n = len(df)
    n_fraud = int(df[C.TARGET].sum())
    return {
        "n_transactions": n,
        "n_fraud": n_fraud,
        "n_legit": n - n_fraud,
        "fraud_rate": n_fraud / n,
        "n_features": df.shape[1] - 1,
        "amount_total": float(df["Amount"].sum()),
        "amount_fraud_total": float(df.loc[df[C.TARGET] == 1, "Amount"].sum()),
    }


def split(df: pd.DataFrame):
    """Stratified train / validation / test split.

    The validation split is held out *before* model fitting and is used only
    for threshold / decision-band tuning, so the reported test metrics stay
    honest (thresholds never see the test set).
    """
    X = df.drop(columns=[C.TARGET])
    y = df[C.TARGET].astype(int)

    if C.SPLIT_MODE == "temporal":
        # Past -> future: train on the earliest transactions, validate on the
        # next slice, test on the most recent. No row from the "future" is ever
        # seen during fitting or threshold tuning.
        order = X["Time"].sort_values().index
        n = len(order)
        n_test = int(n * C.TEST_SIZE)
        n_val = int((n - n_test) * C.VAL_SIZE)
        train_idx = order[: n - n_test - n_val]
        val_idx = order[n - n_test - n_val: n - n_test]
        test_idx = order[n - n_test:]
        return (X.loc[train_idx], X.loc[val_idx], X.loc[test_idx],
                y.loc[train_idx], y.loc[val_idx], y.loc[test_idx])

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y,
        test_size=C.TEST_SIZE,
        stratify=y,
        random_state=C.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=C.VAL_SIZE,
        stratify=y_tr_full,
        random_state=C.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
