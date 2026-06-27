"""Streamlit dashboard for the Transaction Fraud Detection & Risk Decisioning
system.

    streamlit run app/dashboard.py

Tabs:
  1. Overview        — dataset stats + model leaderboard
  2. Model metrics   — PR/ROC curves, per-model PR-AUC/ROC-AUC/recall/precision
  3. Decisioning     — live scoring of transactions into APPROVE/REVIEW/BLOCK,
                       with an interactive cost simulator over the thresholds
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C            # noqa: E402
from src import data as D              # noqa: E402
from src import evaluate as E          # noqa: E402
from src import features as F          # noqa: E402
from src import threshold as T         # noqa: E402
from src.decision import RiskDecisioner  # noqa: E402

st.set_page_config(page_title="Fraud Detection & Risk Decisioning",
                   layout="wide", page_icon="🛡️")

ACTION_COLORS = {"APPROVE": "#2ecc71", "MANUAL_REVIEW": "#f1c40f",
                 "BLOCK": "#e74c3c"}


@st.cache_data(show_spinner=False)
def load_report():
    p = C.REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=True)
def load_test_sample(n=4000):
    df = D.load_raw()
    _, _, X_test, _, _, y_test = D.split(df)
    X = X_test.copy()
    X["Class"] = y_test.values
    return X.sample(min(n, len(X)), random_state=0).reset_index(drop=True)


@st.cache_resource(show_spinner=True)
def load_decisioner(model_name):
    return RiskDecisioner.load(model_name)


report = load_report()
if report is None:
    st.error("No reports found. Run `python run_pipeline.py` first.")
    st.stop()

policy = report["policy"]
best_model = policy["best_model"]

st.title("🛡️ Transaction Fraud Detection & Risk Decisioning")
st.caption("Real Kaggle/ULB credit-card dataset · 284,807 transactions · "
           "cost-based APPROVE / MANUAL_REVIEW / BLOCK decisioning")

tab1, tab2, tab3 = st.tabs(["Overview", "Model metrics", "Live decisioning"])

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------
with tab1:
    info = report["dataset"]
    c = st.columns(4)
    c[0].metric("Transactions", f"{info['n_transactions']:,}")
    c[1].metric("Frauds", f"{info['n_fraud']:,}")
    c[2].metric("Fraud rate", f"{info['fraud_rate']*100:.3f}%")
    c[3].metric("Fraud $ exposure", f"${info['amount_fraud_total']:,.0f}")

    st.subheader("Model leaderboard (test set)")
    rows = []
    for name, m in report["models"].items():
        rows.append({
            "model": name,
            "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
            "recall": m["recall"], "precision": m["precision"], "F1": m["f1"],
            "fit (s)": m["fit_seconds"],
        })
    lb = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    st.dataframe(
        lb.style.format({"PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}",
                         "recall": "{:.3f}", "precision": "{:.3f}",
                         "F1": "{:.3f}"})
        .highlight_max(subset=["PR-AUC", "ROC-AUC"], color="#1e3a1e"),
        use_container_width=True)
    st.success(f"Deployed model: **{best_model}** "
               f"(selected by PR-AUC — the right metric for imbalanced fraud).")

    st.subheader("Cost impact (test set)")
    t = policy["test"]
    cc = st.columns(3)
    cc[0].metric("Approve-all loss", f"${t['approve_all_cost']:,.0f}")
    cc[1].metric("With decisioning", f"${t['decision_band_cost']:,.0f}")
    cc[2].metric("Savings", f"${t['savings_vs_approve_all']:,.0f}",
                 delta=f"{t['savings_vs_approve_all']/max(t['approve_all_cost'],1)*100:.1f}%")

# ---------------------------------------------------------------------------
# Tab 2 — Model metrics
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Threshold-free metrics")
    st.caption("PR-AUC is the headline number; with 0.17% fraud, ROC-AUC alone "
               "can look great while precision is poor.")
    for name, m in sorted(report["models"].items(),
                          key=lambda kv: -kv[1]["pr_auc"]):
        cols = st.columns([2, 1, 1, 1, 1, 1])
        cols[0].write(f"**{name}**")
        cols[1].metric("PR-AUC", f"{m['pr_auc']:.4f}")
        cols[2].metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        cols[3].metric("Recall", f"{m['recall']:.3f}")
        cols[4].metric("Precision", f"{m['precision']:.3f}")
        cols[5].metric("F1", f"{m['f1']:.3f}")

    st.subheader(f"Curves — {best_model}")
    img_cols = st.columns(2)
    if (C.REPORTS_DIR / "curves.png").exists():
        img_cols[0].image(str(C.REPORTS_DIR / "curves.png"),
                          caption="Precision-Recall & ROC")
    if (C.REPORTS_DIR / "cost.png").exists():
        img_cols[1].image(str(C.REPORTS_DIR / "cost.png"),
                          caption="Cost-based threshold tuning")

# ---------------------------------------------------------------------------
# Tab 3 — Live decisioning + cost simulator
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Score real transactions into actions")
    dec = load_decisioner(best_model)
    sample = load_test_sample()
    X = sample.drop(columns=["Class"])
    y = sample["Class"].to_numpy()
    scores = dec.score(X)
    amounts = X["Amount"].to_numpy()

    st.caption("Adjust the decision bands and watch action mix + business cost "
               "move. Defaults are the cost-optimised thresholds from training.")
    c1, c2 = st.columns(2)
    t_review = c1.slider("Review threshold", 0.0, 1.0,
                         float(policy["t_review"]), 0.005)
    t_block = c2.slider("Block threshold", 0.0, 1.0,
                        float(policy["t_block"]), 0.005)
    if t_block < t_review:
        t_block = t_review

    actions = np.where(scores >= t_block, "BLOCK",
                       np.where(scores >= t_review, "MANUAL_REVIEW", "APPROVE"))
    res = X.copy()
    res["fraud_probability"] = scores
    res["action"] = actions
    res["actual"] = np.where(y == 1, "FRAUD", "legit")

    counts = pd.Series(actions).value_counts().reindex(C.ACTIONS).fillna(0)
    cols = st.columns(3)
    for col, act in zip(cols, C.ACTIONS):
        col.markdown(
            f"<div style='border-left:6px solid {ACTION_COLORS[act]};"
            f"padding-left:10px'><b>{act}</b><br>"
            f"<span style='font-size:1.6em'>{int(counts[act]):,}</span></div>",
            unsafe_allow_html=True)

    cost = T.expected_cost_bands(y, scores, t_review, t_block, amounts=amounts)
    base = T.baseline_cost_no_model(y, amounts=amounts)
    caught = ((scores >= t_review) & (y == 1)).sum()
    cc = st.columns(3)
    cc[0].metric("Frauds caught (review+block)", f"{int(caught)}/{int((y==1).sum())}")
    cc[1].metric("Policy cost (sample)", f"${cost:,.0f}")
    cc[2].metric("vs approve-all", f"${base-cost:,.0f}", delta="saved")

    st.subheader("Highest-risk transactions in sample")
    show = res.sort_values("fraud_probability", ascending=False).head(25)
    st.dataframe(
        show[["fraud_probability", "action", "actual", "Amount", "hour"]
             if "hour" in show else
             ["fraud_probability", "action", "actual", "Amount"]]
        .style.format({"fraud_probability": "{:.4f}", "Amount": "${:.2f}"}),
        use_container_width=True)
