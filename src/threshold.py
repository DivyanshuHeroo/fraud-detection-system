"""Cost-based threshold tuning and decision-band selection.

Standard accuracy/F1 optimisation ignores that a missed fraud and a false
alarm cost wildly different amounts. Here we minimise *expected business cost*:

    cost = sum over transactions of
             FN_cost  if fraud and not flagged
             FP_cost  if legit and blocked
             REVIEW_cost if routed to manual review

Two things are tuned, both on a held-out validation split:
  1. `best_cost_threshold` — single APPROVE/BLOCK cutoff that minimises cost.
  2. `best_decision_bands` — a (review, block) pair for the three-way
     APPROVE / MANUAL_REVIEW / BLOCK policy.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def _fn_costs(y_true, amounts) -> np.ndarray:
    """Per-row cost of *missing* a fraud (only meaningful where y_true == 1)."""
    if C.FN_COST_IS_AMOUNT and amounts is not None:
        return np.maximum(np.asarray(amounts, dtype=float), C.FP_COST)
    return np.full(len(y_true), C.FN_COST_FLAT)


def expected_cost_binary(y_true, y_score, threshold, amounts=None) -> float:
    """Total cost of a binary APPROVE/BLOCK policy at `threshold`."""
    y_true = np.asarray(y_true)
    blocked = np.asarray(y_score) >= threshold
    fn_cost = _fn_costs(y_true, amounts)

    missed_fraud = (y_true == 1) & (~blocked)
    false_alarm = (y_true == 0) & (blocked)
    cost = fn_cost[missed_fraud].sum() + C.FP_COST * false_alarm.sum()
    return float(cost)


def best_cost_threshold(y_true, y_score, amounts=None, n_grid=300):
    """Grid-search the single threshold that minimises expected cost."""
    grid = np.unique(np.quantile(y_score, np.linspace(0, 1, n_grid)))
    costs = [expected_cost_binary(y_true, y_score, t, amounts) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i]), grid, np.asarray(costs)


def expected_cost_bands(y_true, y_score, t_review, t_block, amounts=None) -> float:
    """Total cost of the three-way decision policy.

    APPROVE  -> pay full FN_cost if it was actually fraud
    REVIEW   -> pay REVIEW_cost on every reviewed item; an analyst stops a
                fraud with prob REVIEW_DETECTION, so reviewed frauds keep a
                residual (1 - REVIEW_DETECTION) * FN_cost
    BLOCK    -> pay FP_cost if it was actually legit (good customer blocked)
    """
    y_true = np.asarray(y_true)
    s = np.asarray(y_score)
    fn_cost = _fn_costs(y_true, amounts)

    approve = s < t_review
    review = (s >= t_review) & (s < t_block)
    block = s >= t_block

    cost = 0.0
    cost += fn_cost[(y_true == 1) & approve].sum()        # frauds let through
    cost += C.REVIEW_COST * review.sum()                  # ops cost of review
    cost += (1 - C.REVIEW_DETECTION) * \
        fn_cost[(y_true == 1) & review].sum()             # frauds review misses
    cost += C.FP_COST * ((y_true == 0) & block).sum()     # legit we blocked
    return float(cost)


def review_rate(y_score, t_review, t_block) -> float:
    s = np.asarray(y_score)
    return float(((s >= t_review) & (s < t_block)).mean())


def best_decision_bands(y_true, y_score, amounts=None, n_grid=60):
    """Grid-search (t_review, t_block) minimising three-way expected cost,
    subject to the manual-review capacity constraint. The capacity cap is what
    forces high-confidence fraud to be BLOCKED rather than endlessly reviewed.
    """
    # Dense quantile grid concentrated near the top of the score distribution:
    # feasible review bands are narrow (<= REVIEW_CAPACITY of traffic), so we
    # need fine resolution there or every candidate pair looks infeasible.
    qs = np.unique(np.concatenate([
        np.linspace(0.80, 0.99, n_grid, endpoint=False),
        np.linspace(0.99, 0.99999, 2 * n_grid),
    ]))
    grid = np.unique(np.quantile(y_score, qs))
    best = (C.DEFAULT_REVIEW_THRESHOLD, C.DEFAULT_BLOCK_THRESHOLD)
    best_cost = np.inf
    for i, t_review in enumerate(grid):
        for t_block in grid[i:]:
            if t_block <= t_review:
                continue
            if review_rate(y_score, t_review, t_block) > C.REVIEW_CAPACITY:
                continue  # infeasible: exceeds analyst capacity
            c = expected_cost_bands(y_true, y_score, t_review, t_block, amounts)
            if c < best_cost:
                best_cost, best = c, (float(t_review), float(t_block))
    return best[0], best[1], float(best_cost)


def baseline_cost_no_model(y_true, amounts=None) -> float:
    """Cost of approving everything (no fraud system) — the thing to beat."""
    y_true = np.asarray(y_true)
    fn_cost = _fn_costs(y_true, amounts)
    return float(fn_cost[y_true == 1].sum())
