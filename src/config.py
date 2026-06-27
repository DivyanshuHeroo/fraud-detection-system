"""Central configuration for the fraud detection / risk decisioning system."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

DATA_FILE = DATA_DIR / "creditcard.csv"

for _d in (MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data schema (Kaggle "Credit Card Fraud Detection", mlg-ulb / ULB)
#   Time, V1..V28 (PCA-anonymised), Amount, Class (1 = fraud)
# ---------------------------------------------------------------------------
TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20          # carved out of the train split for threshold tuning

# This dataset contains 1,081 exact duplicate transactions; left in, ~800 of
# them straddle the train/test split and leak (a row memorised in train is
# scored again in test). Drop them before splitting.
DROP_DUPLICATES = True

# "random"   : stratified random split (Kaggle convention; headline numbers).
# "temporal" : past -> future split on `Time` (more realistic; ~0.08 lower
#              PR-AUC because the random split leaks future information).
SPLIT_MODE = "random"

# ---------------------------------------------------------------------------
# Cost model (used for cost-based threshold tuning + decisioning)
#
#   Catching fraud is valuable; reviewing/blocking good customers is costly.
#   These are illustrative business numbers — tune them to a real portfolio.
# ---------------------------------------------------------------------------
# Cost of a *missed* fraud (false negative): we assume the full transaction
#   amount is charged back / lost. Handled per-transaction using `Amount`.
FN_COST_IS_AMOUNT = True
# Fallback flat cost of a missed fraud if amount is unavailable.
FN_COST_FLAT = 150.0
# Cost of a false positive: friction / lost goodwill / ops cost of bothering a
#   legitimate customer (blocking a good transaction).
FP_COST = 15.0
# Cost of routing a transaction to a human analyst (manual review).
REVIEW_COST = 4.0
# Manual review is capacity-constrained: analysts can only handle a small
# fraction of all transactions, so the policy cannot simply "review everything".
# This forces high-confidence fraud to be BLOCKED outright.
REVIEW_CAPACITY = 0.005          # at most 0.5% of traffic to manual review
# Manual review is good but not perfect at catching fraud.
REVIEW_DETECTION = 0.95          # prob a reviewed fraud is actually stopped

# ---------------------------------------------------------------------------
# Decisioning bands. A model score p (fraud probability) maps to an action:
#   p <  t_review            -> APPROVE
#   t_review <= p < t_block  -> MANUAL_REVIEW
#   p >= t_block             -> BLOCK
# Defaults here are overwritten by cost-optimised thresholds at train time and
# persisted to models/decision_policy.json.
# ---------------------------------------------------------------------------
DEFAULT_REVIEW_THRESHOLD = 0.30
DEFAULT_BLOCK_THRESHOLD = 0.80

ACTIONS = ("APPROVE", "MANUAL_REVIEW", "BLOCK")
