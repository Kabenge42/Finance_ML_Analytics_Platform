"""Test that build_expected_returns_summary merges market data columns."""

import unittest
from unittest.mock import patch

import pandas as pd


MARKET_DATA_COLS = [
    "market_cap",
    "enterprise_value",
    "last_price",
    "price_target",
    "price_target_high",
    "price_target_low",
    "price_target_median",
    "volume_shrs",
    "shares_outstanding",
]

ID_COLS = ["isin", "ticker", "name", "region", "country", "sector", "industry"]


def _make_mc():
    """Minimal Monte Carlo result — only has last_price from market cols."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "name": ["Apple", "Microsoft", "Google"],
        "industry": ["Tech", "Tech", "Tech"],
        "last_price": [150.0, 300.0, 130.0],
        "expected_upside_pct": [10.0, 5.0, 15.0],
        "price_target_mc": [165.0, 315.0, 149.5],
        "prob_positive_upside": [70.0, 60.0, 80.0],
        "var_5_pct": [-5.0, -8.0, -3.0],
        "risk_reward_ratio": [1.2, 0.8, 1.5],
    })


def _make_kal():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "filtered_upside": [8.0, 4.0, 12.0],
        "kalman_estimate": [162.0, 312.0, 145.6],
    })


def _make_pt():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "expected_return_prob_weighted": [9.0, 3.0, 14.0],
        "price_target_prob_weighted": [163.5, 309.0, 148.2],
        "achievement_probability": [0.7, 0.5, 0.8],
        "confidence_level": ["High", "Medium", "High"],
        "analyst_conviction": [0.8, 0.5, 0.9],
        "eps_revision_momentum": [0.1, -0.1, 0.2],
        "analyst_rating_normalized": [0.9, 0.6, 0.95],
    })


def _make_earn():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "posterior_beat_prob": [0.7, 0.4, 0.8],
        "confidence_score": [0.8, 0.5, 0.9],
        "beat_classification": ["Likely Beat", "Uncertain", "Likely Beat"],
    })


def _make_full_source_df():
    """Source DataFrame with identifier AND market data columns (mv_all_stock_features)."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "isin": ["US0378331005", "US5949181045", "US02079K3059"],
        "name": ["Apple", "Microsoft", "Google"],
        "region": ["Americas", "Americas", "Americas"],
        "country": ["US", "US", "US"],
        "sector": ["Technology", "Technology", "Technology"],
        "industry": ["Tech", "Tech", "Tech"],
        "market_cap": [2500e9, 2200e9, 1800e9],
        "enterprise_value": [2600e9, 2300e9, 1900e9],
        "last_price": [150.0, 300.0, 130.0],
        "price_target": [170.0, 320.0, 150.0],
        "price_target_high": [200.0, 380.0, 180.0],
        "price_target_low": [140.0, 280.0, 120.0],
        "price_target_median": [168.0, 315.0, 148.0],
        "volume_shrs": [50e6, 30e6, 20e6],
        "shares_outstanding": [15e9, 7.5e9, 6e9],
    })


def _make_feature_views_only_source_df():
    """Source DataFrame mimicking feature views — has identifiers but NO market data."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOG"],
        "isin": ["US0378331005", "US5949181045", "US02079K3059"],
        "name": ["Apple", "Microsoft", "Google"],
        "region": ["Americas", "Americas", "Americas"],
        "country": ["US", "US", "US"],
        "sector": ["Technology", "Technology", "Technology"],
        "industry": ["Tech", "Tech", "Tech"],
        # Feature columns only — no market_cap, enterprise_value, etc.
        "pe_ratio": [25.0, 30.0, 22.0],
        "debt_to_equity": [1.5, 0.8, 0.3],
    })


class TestExpectedReturnsSummaryMerge(unittest.TestCase):
    """Verify market-data columns are merged into the expected_returns_summary."""

    @patch("expected_returns_v3.load_identifier_columns")
    @patch("expected_returns_v3.get_identifier_cols_set")
    def test_market_data_columns_merged_from_full_source_df(
        self, mock_id_set, mock_id_cols
    ):
        """Market data columns from a full source_df must appear in the summary."""
        mock_id_cols.return_value = ID_COLS
        mock_id_set.return_value = set(ID_COLS)

        from expected_returns_v3 import build_expected_returns_summary

        summary = build_expected_returns_summary(
            _make_mc(), _make_kal(), _make_pt(), _make_earn(),
            source_df=_make_full_source_df(),
        )

        self.assertFalse(summary.empty, "Summary should not be empty")

        missing = [c for c in MARKET_DATA_COLS if c not in summary.columns]
        self.assertEqual(
            missing, [],
            f"Market data columns missing from summary: {missing}. "
            f"Summary columns: {sorted(summary.columns.tolist())}",
        )

    @patch("expected_returns_v3.load_identifier_columns")
    @patch("expected_returns_v3.get_identifier_cols_set")
    def test_market_data_columns_missing_with_feature_views_source(
        self, mock_id_set, mock_id_cols
    ):
        """When source_df has only features (no market data), columns are missing.

        This demonstrates the root-cause: passing feature-views-only data
        as source_df fails to enrich the summary with market data.
        """
        mock_id_cols.return_value = ID_COLS
        mock_id_set.return_value = set(ID_COLS)

        from expected_returns_v3 import build_expected_returns_summary

        summary = build_expected_returns_summary(
            _make_mc(), _make_kal(), _make_pt(), _make_earn(),
            source_df=_make_feature_views_only_source_df(),
        )

        self.assertFalse(summary.empty, "Summary should not be empty")
        # last_price comes from mc, but other market data cols are absent
        self.assertIn("last_price", summary.columns)
        market_cols_no_last = [c for c in MARKET_DATA_COLS if c != "last_price"]
        present = [c for c in market_cols_no_last if c in summary.columns]
        self.assertEqual(
            present, [],
            "Market data columns should NOT be present when source_df "
            "only has feature-view data (no market data).",
        )

    @patch("expected_returns_v3.load_identifier_columns")
    @patch("expected_returns_v3.get_identifier_cols_set")
    def test_market_data_columns_without_source_df(
        self, mock_id_set, mock_id_cols
    ):
        """Without source_df, only columns present in mc should appear."""
        mock_id_cols.return_value = ID_COLS
        mock_id_set.return_value = set(ID_COLS)

        from expected_returns_v3 import build_expected_returns_summary

        summary = build_expected_returns_summary(
            _make_mc(), _make_kal(), _make_pt(), _make_earn(),
            source_df=None,
        )

        self.assertFalse(summary.empty, "Summary should not be empty")
        self.assertIn("last_price", summary.columns)

    @patch("expected_returns_v3.load_identifier_columns")
    @patch("expected_returns_v3.get_identifier_cols_set")
    def test_summary_has_expected_model_columns(
        self, mock_id_set, mock_id_cols
    ):
        """Verify the summary contains all model output columns."""
        mock_id_cols.return_value = ID_COLS
        mock_id_set.return_value = set(ID_COLS)

        from expected_returns_v3 import build_expected_returns_summary

        summary = build_expected_returns_summary(
            _make_mc(), _make_kal(), _make_pt(), _make_earn(),
            source_df=_make_full_source_df(),
        )

        expected_model_cols = [
            "expected_upside_pct", "filtered_upside",
            "expected_return_prob_weighted", "posterior_beat_prob",
            "agreement_score", "signal", "weighted_agreement",
            "mc_bullish", "kal_bullish", "pt_bullish", "earn_bullish",
        ]
        for col in expected_model_cols:
            self.assertIn(col, summary.columns, f"Missing model column: {col}")

    @patch("expected_returns_v3.load_identifier_columns")
    @patch("expected_returns_v3.get_identifier_cols_set")
    def test_market_data_values_are_correct(
        self, mock_id_set, mock_id_cols
    ):
        """Verify that merged market data values match the source."""
        mock_id_cols.return_value = ID_COLS
        mock_id_set.return_value = set(ID_COLS)

        from expected_returns_v3 import build_expected_returns_summary

        source = _make_full_source_df()
        summary = build_expected_returns_summary(
            _make_mc(), _make_kal(), _make_pt(), _make_earn(),
            source_df=source,
        )

        aapl_row = summary[summary["ticker"] == "AAPL"].iloc[0]
        self.assertAlmostEqual(aapl_row["market_cap"], 2500e9)
        self.assertAlmostEqual(aapl_row["enterprise_value"], 2600e9)
        self.assertAlmostEqual(aapl_row["price_target"], 170.0)
        self.assertAlmostEqual(aapl_row["shares_outstanding"], 15e9)


if __name__ == "__main__":
    unittest.main()
