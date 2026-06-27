"""Feature preprocessing.

The V1..V28 columns are already PCA components and roughly standardised, so the
main work is scaling the raw `Time` and `Amount` columns and deriving a couple
of cheap, interpretable engineered features. Everything is wrapped in a
scikit-learn ColumnTransformer so the exact same transform is reused at
training, evaluation and inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    """Add a few interpretable engineered features.

    These are intentionally simple and leakage-free (each row depends only on
    its own values), which keeps the pipeline serialisable and fast at scoring
    time.
    """
    X = X.copy()
    # Hour-of-day proxy: Time is seconds from the first transaction over a
    # ~2-day window. Fraud has a known diurnal pattern.
    X["hour"] = (X["Time"] / 3600.0) % 24.0
    # Log-amount tames the heavy right tail of transaction amounts.
    X["log_amount"] = np.log1p(X["Amount"])
    return X


# Columns produced after `engineer`.
RAW_SCALE_COLS = ["Time", "Amount", "hour", "log_amount"]
PCA_COLS = [f"V{i}" for i in range(1, 29)]


def build_preprocessor() -> ColumnTransformer:
    """RobustScaler for raw/engineered cols, light StandardScaler for V*."""
    return ColumnTransformer(
        transformers=[
            ("raw", RobustScaler(), RAW_SCALE_COLS),
            ("pca", StandardScaler(), PCA_COLS),
        ],
        remainder="drop",
    )


def feature_names() -> list[str]:
    return RAW_SCALE_COLS + PCA_COLS
