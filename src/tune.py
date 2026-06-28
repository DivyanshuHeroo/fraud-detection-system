"""Hyperparameter tuning, done without leakage.

Search is run with RandomizedSearchCV optimising **PR-AUC (average_precision)**
via stratified CV on the TRAIN split only. Preprocessing (engineer + scalers)
lives inside the pipeline, so it is refit on each CV fold — no information from
the validation/test rows leaks into tuning. We then compare baseline vs tuned
PR-AUC on the held-out validation split (the honest model-selection signal) and
finally on the untouched test split.

    python -m src.tune
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from . import config as C
from . import data as D
from . import features as F

warnings.filterwarnings("ignore")
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)


def _pipe(clf):
    return Pipeline([
        ("engineer", FunctionTransformer(F.engineer, validate=False)),
        ("pre", F.build_preprocessor()),
        ("clf", clf),
    ])


def _spaces(spw):
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": loguniform(1e-3, 1e2)},
            12,
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                   random_state=C.RANDOM_STATE),
            {"clf__n_estimators": [200, 300, 400],
             "clf__max_depth": [None, 8, 16, 24],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", 0.3, 0.5]},
            8,
        ),
        "xgboost": (
            XGBClassifier(scale_pos_weight=spw, eval_metric="aucpr",
                          tree_method="hist", n_jobs=-1,
                          random_state=C.RANDOM_STATE),
            {"clf__n_estimators": [300, 500, 700],
             "clf__max_depth": [4, 6, 8, 10],
             "clf__learning_rate": loguniform(0.02, 0.3),
             "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0],
             "clf__min_child_weight": [1, 5, 10],
             "clf__reg_lambda": [1, 5, 10]},
            25,
        ),
        "lightgbm": (
            LGBMClassifier(class_weight="balanced", n_jobs=-1,
                           random_state=C.RANDOM_STATE, verbose=-1),
            {"clf__n_estimators": [300, 500, 700],
             "clf__num_leaves": [31, 63, 127],
             "clf__learning_rate": loguniform(0.02, 0.2),
             "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.7, 0.85, 1.0],
             "clf__min_child_samples": [20, 50, 100],
             "clf__reg_lambda": [0, 1, 5, 10]},
            25,
        ),
    }


# Current production defaults, to measure the lift against.
def _baselines(spw):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=C.RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
        "lightgbm": LGBMClassifier(
            n_estimators=400, num_leaves=31, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_samples=50,
            class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE,
            verbose=-1),
    }


def _pr(clf, Xtr, ytr, X, y):
    p = clone(clf).fit(Xtr, ytr).predict_proba(X)[:, 1]
    return float(average_precision_score(y, p))


def main():
    df = D.load_raw()
    Xtr, Xva, Xte, ytr, yva, yte = D.split(df)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    baselines = _baselines(spw)

    print(f"{'model':22s}{'base val':>9s}{'tuned val':>10s}"
          f"{'Δ val':>8s}{'base test':>10s}{'tuned test':>11s}{'best params'}")
    print("-" * 100)
    results = {}
    for name, (estimator, space, n_iter) in _spaces(spw).items():
        tic = time.time()
        # --- baseline (current defaults) on the SAME pipeline -------------
        base_pipe = _pipe(baselines[name])
        base_val = _pr(base_pipe, Xtr, ytr, Xva, yva)
        base_test = _pr(base_pipe, Xtr, ytr, Xte, yte)

        # --- tuned: CV search on train, scoring PR-AUC -------------------
        search = RandomizedSearchCV(
            _pipe(estimator), space, n_iter=n_iter, scoring="average_precision",
            cv=CV, n_jobs=-1, random_state=C.RANDOM_STATE, refit=True)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        tuned_val = float(average_precision_score(yva, best.predict_proba(Xva)[:, 1]))
        tuned_test = float(average_precision_score(yte, best.predict_proba(Xte)[:, 1]))

        params = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v)
                  for k, v in search.best_params_.items()}
        results[name] = {"base_val": base_val, "tuned_val": tuned_val,
                         "base_test": base_test, "tuned_test": tuned_test,
                         "cv_pr_auc": float(search.best_score_),
                         "best_params": params, "secs": round(time.time() - tic)}
        flag = "↑" if tuned_val > base_val else "↓" if tuned_val < base_val else "="
        print(f"{name:22s}{base_val:9.4f}{tuned_val:10.4f}"
              f"{tuned_val-base_val:+8.4f}{base_test:10.4f}{tuned_test:11.4f}  {flag} {params}")

    # Verdict on the deployed-by-validation model.
    best_base = max(results, key=lambda k: results[k]["base_val"])
    best_tuned = max(results, key=lambda k: results[k]["tuned_val"])
    print("\nValidation-selected model:")
    print(f"  baseline -> {best_base}  (val PR-AUC {results[best_base]['base_val']:.4f}, "
          f"test {results[best_base]['base_test']:.4f})")
    print(f"  tuned    -> {best_tuned} (val PR-AUC {results[best_tuned]['tuned_val']:.4f}, "
          f"test {results[best_tuned]['tuned_test']:.4f})")
    import json
    (C.REPORTS_DIR / "tuning.json").write_text(json.dumps(results, indent=2))
    print("\nSaved reports/tuning.json")


if __name__ == "__main__":
    main()
