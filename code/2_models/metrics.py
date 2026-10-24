"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Evaluation metrics. R2 / MSE / MAE alone reward matching persistence
          (IV barely moves bar-to-bar), so skill is measured against the
          random-walk forecast, plus a cost-adjusted vega P&L measure.
"""

import numpy as np


def hit_rate(now: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of rows where the predicted direction of IV change (relative
    to the current/anchor IV) matches the actual direction."""
    return float(np.mean(np.sign(pred - now) == np.sign(actual - now)))


def r2_oos_rw(now: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> float:
    """Campbell-Thompson out-of-sample R² relative to the random-walk
    (no-change) benchmark: 1 - SSE_model / SSE_random_walk. Positive means
    the model beats the naive 'IV stays the same' forecast; zero or negative
    means it adds no skill over persistence."""
    sse_model = np.sum((actual - pred) ** 2)
    sse_rw    = np.sum((actual - now) ** 2)
    return float(1.0 - sse_model / sse_rw) if sse_rw > 0 else float("nan")


# C_IV / P_IV are decimals; C_VEGA / P_VEGA are per 1 vol point (0.01 of IV).
# Multiplying the IV change by this factor expresses it in vol points, so
# vega * ΔIV is in option price units ($ per share), like the spread cost.
VOL_POINTS_PER_UNIT_IV = 100.0


def vega_sharpe(now: np.ndarray, pred: np.ndarray, actual: np.ndarray,
                vega: np.ndarray, mid: np.ndarray, spread_rel: np.ndarray) -> float:
    """Per-decision Sharpe ratio of a directional single-option vega trade,
    net of half the bid-ask spread (the cost of crossing the spread to
    enter). direction = sign(pred - now); PnL = direction * vega * 100 *
    (actual - now) - 0.5 * spread_rel * mid, with IV in decimals and vega per
    vol point. Unannualized — each row is treated as one trading decision.
    The cost is floored at zero: a momentarily crossed quote (bid > ask,
    spread_rel < 0) cannot pay the trader to enter."""
    direction = np.sign(pred - now)
    raw_pnl   = direction * vega * VOL_POINTS_PER_UNIT_IV * (actual - now)
    cost      = 0.5 * np.maximum(spread_rel, 0.0) * mid
    net_pnl   = raw_pnl - cost
    sd = net_pnl.std()
    return float(net_pnl.mean() / sd) if sd > 0 else float("nan")


def compute_metrics(now: np.ndarray, pred: np.ndarray, actual: np.ndarray,
                    vega: np.ndarray, mid: np.ndarray, spread_rel: np.ndarray) -> dict:
    return {
        "hit_rate":    hit_rate(now, pred, actual),
        "r2_oos_rw":   r2_oos_rw(now, pred, actual),
        "vega_sharpe": vega_sharpe(now, pred, actual, vega, mid, spread_rel),
    }
