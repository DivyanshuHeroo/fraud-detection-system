"""Train the fraud models, tune cost-based thresholds, and persist everything.

Models (all handling class imbalance internally rather than resampling):
  * Logistic Regression  (class_weight="balanced")
  * Random Forest        (class_weight="balanced")
  * XGBoost              (scale_pos_weight = #neg / #pos)
  * LightGBM             (scale_pos_weight = #neg / #pos)

Outputs (in models/ and reports/):
  * <model>.joblib            fitted model + preprocessor bundle, per model
  * decision_policy.json      cost-optimised (t_review, t_block) for best model
  * metrics.json              all metrics for every model
  * curves.png, cost.png      evaluation plots for the best model
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from . import config as C
from . import data as D
from . import evaluate as E
from . import features as F
from . import threshold as T


def _build_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=C.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE,
        ),
        # NOTE: an extreme scale_pos_weight (~578) destroys LightGBM's ranking
        # here (PR-AUC collapses to ~0.05); class_weight="balanced" recovers a
        # PR-AUC on par with XGBoost. See README "Modelling notes".
        "lightgbm": LGBMClassifier(
            n_estimators=400, num_leaves=31, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_samples=50,
            class_weight="balanced", n_jobs=-1,
            random_state=C.RANDOM_STATE, verbose=-1,
        ),
    }


def _plot_curves(y_test, score, name, path):
    prec, rec, _ = E.pr_curve(y_test, score)
    fpr, tpr, _ = E.roc_curve_pts(y_test, score)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(rec, prec, color="#c0392b")
    ax[0].set(xlabel="Recall", ylabel="Precision",
              title=f"Precision-Recall ({name})")
    ax[0].grid(alpha=0.3)
    ax[1].plot(fpr, tpr, color="#2980b9")
    ax[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
              title=f"ROC ({name})")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_cost(grid, costs, t_star, baseline, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid, costs, color="#8e44ad", label="expected cost")
    ax.axvline(t_star, color="#c0392b", ls="--",
               label=f"cost-optimal thr = {t_star:.3f}")
    ax.axhline(baseline, color="grey", ls=":",
               label=f"approve-all baseline = {baseline:,.0f}")
    ax.set(xlabel="Block threshold", ylabel="Expected cost (validation)",
           title="Cost-based threshold tuning")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    print(">> Loading data ...")
    df = D.load_raw()
    info = D.describe(df)
    print(f"   {info['n_transactions']:,} transactions | "
          f"{info['n_fraud']:,} fraud ({info['fraud_rate']*100:.3f}%)")

    X_train, X_val, X_test, y_train, y_val, y_test = D.split(df)
    amt_val = X_val["Amount"].to_numpy()
    amt_test = X_test["Amount"].to_numpy()

    # Fit preprocessor on training data only.
    pre = F.build_preprocessor()
    Xtr = pre.fit_transform(F.engineer(X_train))
    Xva = pre.transform(F.engineer(X_val))
    Xte = pre.transform(F.engineer(X_test))

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f">> Training models (scale_pos_weight={spw:.1f}) ...")

    all_metrics, fitted, val_pr_auc = {}, {}, {}
    for name, model in _build_models(spw).items():
        tic = time.time()
        model.fit(Xtr, y_train)
        # Model SELECTION uses the validation split only — the test set stays
        # untouched until the final report, so it can't leak into model choice.
        val_pr_auc[name] = float(
            average_precision_score(y_val, model.predict_proba(Xva)[:, 1]))
        # Test metrics are recorded for the final report table only.
        score_test = model.predict_proba(Xte)[:, 1]
        rep = E.full_report(y_test, score_test, threshold=0.5)
        rep["val_pr_auc"] = val_pr_auc[name]
        rep["fit_seconds"] = round(time.time() - tic, 1)
        all_metrics[name] = rep
        fitted[name] = model
        print(f"   {name:20s} val-PR={val_pr_auc[name]:.4f} "
              f"test-PR={rep['pr_auc']:.4f} "
              f"test-ROC={rep['roc_auc']:.4f} ({rep['fit_seconds']}s)")
        joblib.dump({"model": model, "preprocessor": pre},
                    C.MODELS_DIR / f"{name}.joblib")

    # Pick the best model by *validation* PR-AUC (right metric, no test leakage).
    best_name = max(val_pr_auc, key=val_pr_auc.get)
    best_model = fitted[best_name]
    print(f">> Best model by validation PR-AUC: {best_name}")

    # --- Cost-based threshold tuning on the validation split ---------------
    val_score = best_model.predict_proba(Xva)[:, 1]
    t_star, cost_star, grid, costs = T.best_cost_threshold(
        y_val, val_score, amounts=amt_val)
    t_review, t_block, band_cost = T.best_decision_bands(
        y_val, val_score, amounts=amt_val)
    baseline = T.baseline_cost_no_model(y_val, amounts=amt_val)
    print(f"   single cost-optimal block threshold = {t_star:.4f} "
          f"(val cost {cost_star:,.0f} vs approve-all {baseline:,.0f})")
    print(f"   decision bands: review>={t_review:.4f} block>={t_block:.4f} "
          f"(val cost {band_cost:,.0f})")

    # --- Evaluate the deployed policy on the untouched test set -----------
    test_score = best_model.predict_proba(Xte)[:, 1]
    test_at_block = E.metrics_at_threshold(y_test, test_score, t_block)
    test_cost_bands = T.expected_cost_bands(
        y_test, test_score, t_review, t_block, amounts=amt_test)
    test_baseline = T.baseline_cost_no_model(y_test, amounts=amt_test)

    # --- Persist policy + metrics + plots --------------------------------
    policy = {
        "best_model": best_name,
        "t_review": t_review,
        "t_block": t_block,
        "single_block_threshold": t_star,
        "cost_model": {
            "FP_COST": C.FP_COST, "REVIEW_COST": C.REVIEW_COST,
            "FN_COST_IS_AMOUNT": C.FN_COST_IS_AMOUNT,
            "FN_COST_FLAT": C.FN_COST_FLAT,
        },
        "validation": {
            "approve_all_cost": baseline,
            "single_threshold_cost": cost_star,
            "decision_band_cost": band_cost,
        },
        "test": {
            "approve_all_cost": test_baseline,
            "decision_band_cost": test_cost_bands,
            "savings_vs_approve_all": test_baseline - test_cost_bands,
            "metrics_at_block_threshold": test_at_block,
        },
    }
    (C.MODELS_DIR / "decision_policy.json").write_text(json.dumps(policy, indent=2))

    report = {"dataset": info, "models": all_metrics, "policy": policy}
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    _plot_curves(y_test, test_score, best_name, C.REPORTS_DIR / "curves.png")
    _plot_cost(grid, costs, t_star, baseline, C.REPORTS_DIR / "cost.png")

    print(f">> Test decision-band cost {test_cost_bands:,.0f} vs "
          f"approve-all {test_baseline:,.0f} "
          f"(saved {test_baseline - test_cost_bands:,.0f})")
    print(f">> Saved models/, reports/. Done in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
