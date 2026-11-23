"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Sanity tests for the vega-scaled Sharpe ratio. Synthetic data mimics
          the 2023 test set (ATM vega ~0.17 per vol point, IV ~0.14, mid ~2.6,
          relative spread ~0.9%).
Run     : python -m unittest test_metrics
"""

import unittest

import numpy as np

import metrics as mcc


def _synthetic(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    now        = rng.uniform(0.09, 0.20, n)
    actual     = now + rng.normal(0.0, 0.006, n)   # ~0.6 vol-point moves
    vega       = rng.uniform(0.10, 0.25, n)        # per vol point
    mid        = rng.uniform(1.5, 4.0, n)
    spread_rel = rng.uniform(0.004, 0.014, n)
    return now, actual, vega, mid, spread_rel


class TestCorrectedVegaSharpe(unittest.TestCase):

    def test_oracle_forecast_is_positive(self):
        """Perfect foresight (pred = actual) must earn a positive Sharpe."""
        now, actual, vega, mid, spread_rel = _synthetic()
        self.assertGreater(mcc.vega_sharpe(now, actual, actual, vega, mid, spread_rel), 0.5)

    def test_random_direction_is_near_zero_without_cost(self):
        """A coin-flip direction has no edge: Sharpe ~ 0 before costs."""
        now, actual, vega, mid, spread_rel = _synthetic()
        rng = np.random.default_rng(7)
        pred = now + rng.choice([-1.0, 1.0], len(now)) * 1e-3
        sharpe = mcc.vega_sharpe(now, pred, actual, vega, mid, 0.0 * spread_rel)
        self.assertLess(abs(sharpe), 0.03)

    def test_random_direction_is_negative_with_cost(self):
        """With costs, a coin-flip direction loses roughly the spread."""
        now, actual, vega, mid, spread_rel = _synthetic()
        rng = np.random.default_rng(7)
        pred = now + rng.choice([-1.0, 1.0], len(now)) * 1e-3
        self.assertLess(mcc.vega_sharpe(now, pred, actual, vega, mid, spread_rel), 0.0)

    def test_zero_cost_is_symmetric(self):
        """With zero cost, flipping every direction flips the Sharpe's sign."""
        now, actual, vega, mid, spread_rel = _synthetic()
        rng = np.random.default_rng(3)
        pred = actual + rng.normal(0.0, 0.006, len(now))
        flipped = 2.0 * now - pred
        zero = 0.0 * spread_rel
        s = mcc.vega_sharpe(now, pred, actual, vega, mid, zero)
        s_flipped = mcc.vega_sharpe(now, flipped, actual, vega, mid, zero)
        self.assertAlmostEqual(s, -s_flipped, places=10)

    def test_gross_pnl_is_in_price_units(self):
        """A correctly called 1 vol-point IV rise (0.15 -> 0.16) on vega 0.15
        must be worth 0.15 per share, the units of the spread cost."""
        gross = 0.15 * mcc.VOL_POINTS_PER_UNIT_IV * (0.16 - 0.15)
        self.assertAlmostEqual(gross, 0.15, places=12)

    def test_crossed_quote_costs_nothing(self):
        """A crossed quote (spread_rel < 0) is charged zero, never a credit:
        Sharpe with crossed spreads equals Sharpe with zero spreads."""
        now, actual, vega, mid, spread_rel = _synthetic()
        rng = np.random.default_rng(11)
        pred = actual + rng.normal(0.0, 0.006, len(now))
        crossed = -np.abs(spread_rel)
        self.assertEqual(mcc.vega_sharpe(now, pred, actual, vega, mid, crossed),
                         mcc.vega_sharpe(now, pred, actual, vega, mid, 0.0 * spread_rel))

    def test_unit_error_fails_oracle(self):
        """Without the vol-point conversion (the bug in the submitted thesis),
        perfect foresight scores below zero: gross P&L is 100x too small."""
        now, actual, vega, mid, spread_rel = _synthetic()
        pnl = vega * np.abs(actual - now) - 0.5 * spread_rel * mid
        self.assertLess(pnl.mean() / pnl.std(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
