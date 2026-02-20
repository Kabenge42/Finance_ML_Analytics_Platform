"""
Tests for deriving actual outcomes from eps_surprise_pct for model confidence
metrics in market_analytics.py.

Validates that the confidence estimation block correctly:
1. Derives binary actual outcomes from eps_surprise_pct (positive → 1, else → 0).
2. Aligns predicted probabilities with matched outcomes via ticker join.
3. Falls back gracefully when eps_surprise_pct is missing or insufficient.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from finance_ml.analytics.probability_analytics import (
    ModelConfidenceEstimator,
)


class TestDeriveActualOutcomes(unittest.TestCase):
    """Test the logic for deriving actual_outcomes from eps_surprise_pct."""

    def _derive_outcomes(
        self, df: pd.DataFrame, probability_results: pd.DataFrame
    ) -> tuple:
        """
        Mirror the derivation logic from market_analytics.py so we can
        unit-test it in isolation.

        Returns (actual_outcomes, predicted_probs_aligned) or (None, None).
        """
        actual_outcomes = None
        predicted_probs_aligned = None
        ticker_col = "ticker" if "ticker" in probability_results.columns else "isin"

        if (
            "eps_surprise_pct" in df.columns
            and ticker_col in probability_results.columns
            and ticker_col in df.columns
        ):
            surprise_map = (
                df[[ticker_col, "eps_surprise_pct"]]
                .dropna(subset=["eps_surprise_pct"])
                .drop_duplicates(subset=[ticker_col])
                .set_index(ticker_col)["eps_surprise_pct"]
            )
            matched_surprise = probability_results[ticker_col].map(surprise_map)
            valid_mask = matched_surprise.notna()

            if valid_mask.sum() > 10:
                actual_outcomes = (matched_surprise[valid_mask].values > 0).astype(
                    np.float64
                )
                predicted_probs_aligned = probability_results.loc[
                    valid_mask, "posterior_beat_prob"
                ].values

        return actual_outcomes, predicted_probs_aligned

    def _make_df(self, n: int = 20) -> pd.DataFrame:
        """Create a source DataFrame with ticker and eps_surprise_pct."""
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(n)],
                "eps_surprise_pct": rng.uniform(-10, 10, n),
            }
        )

    def _make_prob_results(self, n: int = 20) -> pd.DataFrame:
        """Create probability_results with ticker and posterior_beat_prob."""
        rng = np.random.RandomState(99)
        return pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(n)],
                "posterior_beat_prob": rng.uniform(0.2, 0.9, n),
                "beat_classification": ["likely_beat"] * n,
            }
        )

    def test_derives_outcomes_when_surprise_available(self):
        """Outcomes should be derived when eps_surprise_pct exists and > 10 matches."""
        df = self._make_df(25)
        prob = self._make_prob_results(25)
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        self.assertIsNotNone(pred)
        self.assertEqual(len(actual), len(pred))
        self.assertTrue(len(actual) > 10)
        # All values should be 0 or 1
        self.assertTrue(set(np.unique(actual)).issubset({0.0, 1.0}))

    def test_positive_surprise_maps_to_beat(self):
        """Positive eps_surprise_pct should produce actual_outcome = 1."""
        df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(15)],
                "eps_surprise_pct": [5.0] * 15,  # all positive
            }
        )
        prob = self._make_prob_results(15)
        actual, _ = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual, np.ones(len(actual)))

    def test_negative_surprise_maps_to_miss(self):
        """Negative eps_surprise_pct should produce actual_outcome = 0."""
        df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(15)],
                "eps_surprise_pct": [-3.0] * 15,  # all negative
            }
        )
        prob = self._make_prob_results(15)
        actual, _ = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual, np.zeros(len(actual)))

    def test_returns_none_when_surprise_missing(self):
        """Should return None when eps_surprise_pct column is absent."""
        df = pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(20)]})
        prob = self._make_prob_results(20)
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNone(actual)
        self.assertIsNone(pred)

    def test_returns_none_when_too_few_matches(self):
        """Should return None when fewer than 11 tickers match."""
        df = self._make_df(5)  # only 5 tickers
        prob = self._make_prob_results(20)  # 20 tickers, only 5 overlap
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNone(actual)
        self.assertIsNone(pred)

    def test_handles_nan_surprise_values(self):
        """NaN surprise values should be excluded, not crash."""
        df = self._make_df(25)
        df.loc[0:4, "eps_surprise_pct"] = np.nan  # 5 NaN values → 20 valid
        prob = self._make_prob_results(25)
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), 20)

    def test_aligned_arrays_feed_confidence_estimator(self):
        """Derived arrays should be accepted by ModelConfidenceEstimator."""
        df = self._make_df(30)
        prob = self._make_prob_results(30)
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        estimator = ModelConfidenceEstimator(n_bins=5)
        result = estimator.compute_confidence_metrics(
            predicted_probs=pred,
            actual_outcomes=actual,
            model_name="Test Model",
        )

        self.assertGreaterEqual(result.brier_score, 0.0)
        self.assertLessEqual(result.brier_score, 1.0)
        self.assertGreaterEqual(result.calibration_error, 0.0)
        self.assertGreaterEqual(result.overall_confidence, 0.0)
        self.assertLessEqual(result.overall_confidence, 100.0)

    def test_isin_fallback(self):
        """Should fall back to 'isin' when 'ticker' is not in probability_results."""
        df = pd.DataFrame(
            {
                "isin": [f"ISIN{i:03d}" for i in range(15)],
                "eps_surprise_pct": [2.0] * 15,
            }
        )
        prob = pd.DataFrame(
            {
                "isin": [f"ISIN{i:03d}" for i in range(15)],
                "posterior_beat_prob": np.random.uniform(0.3, 0.8, 15),
                "beat_classification": ["likely_beat"] * 15,
            }
        )
        actual, pred = self._derive_outcomes(df, prob)

        self.assertIsNotNone(actual)
        self.assertEqual(len(actual), 15)


if __name__ == "__main__":
    unittest.main()
