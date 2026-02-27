"""Tests for historical price target drift enrichment in expected_returns_v3."""
import unittest

import numpy as np
import pandas as pd

from expected_returns_v3 import (
    ALL_HISTORICAL_PRICE_TARGET_COLS,
    FEATURE_CATEGORIES,
    _HISTORICAL_PRICE_COLS,
    _HISTORICAL_PRICE_TARGET_COLS,
    _HISTORICAL_PRICE_TARGET_HIGH_COLS,
    _HISTORICAL_PRICE_TARGET_LOW_COLS,
    _HISTORICAL_PRICE_TARGET_MEDIAN_COLS,
    _enrich_with_historical_target_drift,
    _log_historical_coverage,
    _resolve_available_historical_cols,
    run_kalman_filter,
    run_monte_carlo_analysis,
    run_price_target_achievement,
)


def _make_sample_df(**extra_cols):
    """Create a minimal DataFrame with core price target columns."""
    data = {
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "name": ["Apple", "Microsoft", "Alphabet"],
        "industry": ["Tech", "Tech", "Tech"],
        "last_price": [150.0, 300.0, 140.0],
        "price_target": [180.0, 350.0, 170.0],
        "price_target_high": [200.0, 400.0, 190.0],
        "price_target_low": [160.0, 300.0, 150.0],
        "price_target_median": [175.0, 340.0, 165.0],
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


class TestColumnRegistries(unittest.TestCase):
    """Verify column registry lists are correctly defined."""

    def test_historical_price_cols_count(self):
        self.assertEqual(len(_HISTORICAL_PRICE_COLS), 9)

    def test_historical_price_target_cols_count(self):
        self.assertEqual(len(_HISTORICAL_PRICE_TARGET_COLS), 7)

    def test_historical_price_target_high_cols_count(self):
        self.assertEqual(len(_HISTORICAL_PRICE_TARGET_HIGH_COLS), 8)

    def test_historical_price_target_low_cols_count(self):
        self.assertEqual(len(_HISTORICAL_PRICE_TARGET_LOW_COLS), 8)

    def test_historical_price_target_median_cols_count(self):
        self.assertEqual(len(_HISTORICAL_PRICE_TARGET_MEDIAN_COLS), 8)

    def test_all_historical_cols_total(self):
        self.assertEqual(len(ALL_HISTORICAL_PRICE_TARGET_COLS), 40)

    def test_all_historical_cols_is_union(self):
        expected = (
            _HISTORICAL_PRICE_COLS
            + _HISTORICAL_PRICE_TARGET_COLS
            + _HISTORICAL_PRICE_TARGET_HIGH_COLS
            + _HISTORICAL_PRICE_TARGET_LOW_COLS
            + _HISTORICAL_PRICE_TARGET_MEDIAN_COLS
        )
        self.assertEqual(ALL_HISTORICAL_PRICE_TARGET_COLS, expected)


class TestResolveAvailableHistoricalCols(unittest.TestCase):
    """Test _resolve_available_historical_cols helper."""

    def test_empty_dataframe(self):
        df = pd.DataFrame({"ticker": ["A"]})
        result = _resolve_available_historical_cols(df)
        self.assertEqual(sum(len(v) for v in result.values()), 0)

    def test_partial_columns(self):
        df = _make_sample_df(
            price_1m_ago=[145.0, 290.0, 135.0],
            price_target_1m_ago=[170.0, 340.0, 160.0],
        )
        result = _resolve_available_historical_cols(df)
        self.assertEqual(len(result["historical_prices"]), 1)
        self.assertEqual(len(result["historical_targets"]), 1)
        self.assertEqual(len(result["historical_targets_high"]), 0)

    def test_all_categories_present(self):
        extras = {
            "price_1m_ago": [145.0, 290.0, 135.0],
            "price_target_1m_ago": [170.0, 340.0, 160.0],
            "price_target_high_1m_ago": [190.0, 390.0, 180.0],
            "price_target_low_1m_ago": [150.0, 290.0, 140.0],
            "price_target_median_1m_ago": [165.0, 330.0, 155.0],
        }
        df = _make_sample_df(**extras)
        result = _resolve_available_historical_cols(df)
        for key in result:
            self.assertGreaterEqual(len(result[key]), 1, f"{key} should have >=1 col")


class TestEnrichWithHistoricalTargetDrift(unittest.TestCase):
    """Test _enrich_with_historical_target_drift function."""

    def test_no_historical_cols_returns_unchanged(self):
        df = _make_sample_df()
        hist = _resolve_available_historical_cols(df)
        original_cols = set(df.columns)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertEqual(set(result.columns), original_cols)

    def test_consensus_target_drift_1m(self):
        df = _make_sample_df(price_target_1m_ago=[170.0, 340.0, 160.0])
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("pt_drift_1m", result.columns)
        # AAPL: (180-170)/170 * 100 = 5.88%
        self.assertAlmostEqual(result.loc[0, "pt_drift_1m"], 5.882, places=2)

    def test_consensus_target_drift_multiple_horizons(self):
        df = _make_sample_df(
            price_target_1m_ago=[170.0, 340.0, 160.0],
            price_target_3m_ago=[160.0, 320.0, 150.0],
            price_target_6m_ago=[155.0, 310.0, 145.0],
            price_target_1y_ago=[150.0, 300.0, 140.0],
        )
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        for h in ["1m", "3m", "6m", "1y"]:
            self.assertIn(f"pt_drift_{h}", result.columns)

    def test_spread_change(self):
        df = _make_sample_df(
            price_target_high_1m_ago=[190.0, 390.0, 180.0],
            price_target_low_1m_ago=[150.0, 290.0, 140.0],
        )
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("pt_spread_change_1m", result.columns)
        # AAPL: current spread = 200-160=40, prev spread = 190-150=40, change=0
        self.assertAlmostEqual(result.loc[0, "pt_spread_change_1m"], 0.0, places=2)

    def test_median_drift(self):
        df = _make_sample_df(price_target_median_1m_ago=[165.0, 330.0, 155.0])
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("pt_median_drift_1m", result.columns)

    def test_historical_price_anchor_fallback(self):
        df = _make_sample_df(price_5d_ago=[148.0, 295.0, 138.0])
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("historical_price_anchor", result.columns)
        self.assertAlmostEqual(result.loc[0, "historical_price_anchor"], 148.0)

    def test_price_vs_historical(self):
        df = _make_sample_df(
            price_1m_ago=[145.0, 290.0, 135.0],
            price_3m_ago=[140.0, 280.0, 130.0],
        )
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("price_vs_historical_1m", result.columns)
        self.assertIn("price_vs_historical_3m", result.columns)

    def test_convergence_signal(self):
        df = _make_sample_df(
            price_target_1m_ago=[170.0, 340.0, 160.0],
            price_1m_ago=[145.0, 290.0, 135.0],
        )
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("target_vs_price_convergence_1m", result.columns)
        # Should be pt_drift_1m - price_vs_historical_1m
        expected = result["pt_drift_1m"] - result["price_vs_historical_1m"]
        pd.testing.assert_series_equal(
            result["target_vs_price_convergence_1m"], expected, check_names=False,
        )

    def test_handles_nan_values(self):
        df = _make_sample_df(price_target_1m_ago=[170.0, np.nan, 160.0])
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertIn("pt_drift_1m", result.columns)
        self.assertTrue(pd.isna(result.loc[1, "pt_drift_1m"]))

    def test_handles_zero_denominator(self):
        df = _make_sample_df(price_target_1m_ago=[0.0, 340.0, 160.0])
        hist = _resolve_available_historical_cols(df)
        result = _enrich_with_historical_target_drift(df.copy(), hist)
        self.assertTrue(pd.isna(result.loc[0, "pt_drift_1m"]))


class TestFeatureCategories(unittest.TestCase):
    """Verify FEATURE_CATEGORIES includes historical columns."""

    def test_price_target_dynamics_has_historical_targets(self):
        ptd = FEATURE_CATEGORIES["Price Target Dynamics"]
        for col in _HISTORICAL_PRICE_TARGET_COLS:
            self.assertIn(col, ptd, f"{col} missing from Price Target Dynamics")

    def test_price_target_dynamics_has_historical_high(self):
        ptd = FEATURE_CATEGORIES["Price Target Dynamics"]
        for col in _HISTORICAL_PRICE_TARGET_HIGH_COLS:
            self.assertIn(col, ptd, f"{col} missing from Price Target Dynamics")

    def test_price_target_dynamics_has_historical_low(self):
        ptd = FEATURE_CATEGORIES["Price Target Dynamics"]
        for col in _HISTORICAL_PRICE_TARGET_LOW_COLS:
            self.assertIn(col, ptd, f"{col} missing from Price Target Dynamics")

    def test_price_target_dynamics_has_historical_median(self):
        ptd = FEATURE_CATEGORIES["Price Target Dynamics"]
        for col in _HISTORICAL_PRICE_TARGET_MEDIAN_COLS:
            self.assertIn(col, ptd, f"{col} missing from Price Target Dynamics")

    def test_momentum_technical_has_historical_prices(self):
        mt = FEATURE_CATEGORIES["Momentum & Technical"]
        for col in _HISTORICAL_PRICE_COLS:
            self.assertIn(col, mt, f"{col} missing from Momentum & Technical")


class TestModelRunnerSignatures(unittest.TestCase):
    """Verify model runners accept use_historical_targets parameter."""

    def test_mc_accepts_use_historical_targets_false(self):
        df = _make_sample_df()
        # With use_historical_targets=False, should still work (no enrichment)
        # Won't actually run MC (needs more data), but should not raise TypeError
        result = run_monte_carlo_analysis(df, use_historical_targets=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_pt_accepts_use_historical_targets_false(self):
        df = _make_sample_df()
        result = run_price_target_achievement(df, use_historical_targets=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_kalman_accepts_use_historical_targets_false(self):
        df = _make_sample_df()
        result = run_kalman_filter(df, use_historical_targets=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_mc_missing_cols_returns_empty(self):
        df = pd.DataFrame({"ticker": ["A"], "last_price": [100.0]})
        result = run_monte_carlo_analysis(df, use_historical_targets=True)
        self.assertTrue(result.empty)

    def test_kalman_missing_cols_returns_empty(self):
        df = pd.DataFrame({"ticker": ["A"]})
        result = run_kalman_filter(df, use_historical_targets=True)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
