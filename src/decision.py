"""Risk-decisioning engine: fraud probability -> APPROVE / MANUAL_REVIEW / BLOCK.

Wraps a fitted model + preprocessor + a persisted decision policy (the two
cost-optimised thresholds) into a single object that takes raw transaction rows
and returns an action plus the score and a short reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import features as F


@dataclass
class Decision:
    score: float
    action: str
    reason: str


class RiskDecisioner:
    def __init__(self, model, preprocessor, t_review: float, t_block: float):
        self.model = model
        self.preprocessor = preprocessor
        self.t_review = t_review
        self.t_block = t_block

    # -- persistence --------------------------------------------------------
    @classmethod
    def load(cls, model_name: str = "xgboost") -> "RiskDecisioner":
        bundle = joblib.load(C.MODELS_DIR / f"{model_name}.joblib")
        policy = json.loads((C.MODELS_DIR / "decision_policy.json").read_text())
        return cls(
            model=bundle["model"],
            preprocessor=bundle["preprocessor"],
            t_review=policy["t_review"],
            t_block=policy["t_block"],
        )

    # -- scoring ------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xe = F.engineer(X)
        Xt = self.preprocessor.transform(Xe)
        return self.model.predict_proba(Xt)[:, 1]

    def decide_one(self, score: float) -> Decision:
        if score >= self.t_block:
            return Decision(score, "BLOCK",
                            f"score {score:.3f} >= block {self.t_block:.3f}")
        if score >= self.t_review:
            return Decision(score, "MANUAL_REVIEW",
                            f"review {self.t_review:.3f} <= score {score:.3f} "
                            f"< block {self.t_block:.3f}")
        return Decision(score, "APPROVE",
                        f"score {score:.3f} < review {self.t_review:.3f}")

    def decide(self, X: pd.DataFrame) -> pd.DataFrame:
        scores = self.score(X)
        actions = np.where(
            scores >= self.t_block, "BLOCK",
            np.where(scores >= self.t_review, "MANUAL_REVIEW", "APPROVE"),
        )
        return pd.DataFrame({"fraud_probability": scores, "action": actions},
                            index=X.index)
