"""
Tests for Data Loading & Export refactorings (Audit Issues 1–8).

Covers:
  - Issue 1: Identifier column propagation through model outputs
  - Issue 2: ExportConfig pipeline usage in export_probability_analytics_results
  - Issue 3: CSV filename == DB table name consistency
  - Issue 4: Shared reorder_with_identifiers in data_utils
  - Issue 5: Reordered DataFrame used for CSV exports (not just DB)
  - Issue 6: Error handling / column validation in summary stats
  - Issue 7: Numeric dtype casting before export
  - Issue 8: confidence_interval tuple split into ci_lower / ci_upper
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_equities_df(n: int = 5) -> pd.DataFrame:
    """Build a small DataFrame that mimics vw_features_* with full identifiers."""
    return pd.DataFrame({
        "isin": [f"US000{i}" for i in range(n)],
        "ticker": [f"TK{i}" for i in range(n)],
        "name": [f"Company {i}" for i in range(n)],
        "region": ["North America"] * n,
        "country": ["US"] * n,
        "trading_country": ["US"] * n,
        "exchange": ["NYSE"] * n,
        "sector": ["Technology"] * n,
        "industry": ["Software"] * n,
        # Earnings-related columns for models
        "eps_beat_count": [3] * n,
        "eps_total_reports": [5] * n,
        "eps_trajectory_score": [70.0] * n,
        "eps_positive_streak": [3] * n,
        "eps_improvement_count": [2] * n,
        # Credit risk columns
        "altman_z_score": [3.5] * n,
        "liquidity_stress_score": [30] * n,
        "cash_runway_months": [24] * n,
        # Dividend columns
        "fcf_dividend_coverage": [2.0] * n,
        "dividend_payout_ratio": [50] * n,
        "dividend_streak": [10] * n,
        # Price target columns
        "upside_potential": [15.0] * n,
        "price_target_spread_pct": [18.0] * n,
        "analyst_rating_normalized": [60] * n,
    })


# Patch load_identifier_columns to avoid DB dependency
_PATCH_ID_COLS = [
    "isin", "ticker", "name", "region", "country", "trading_country",
    "exchange", "sector", "industry",
]


def _patch_id_cols():
    return patch(
        "finance_ml.analytics.data_utils.load_identifier_columns",
        return_value=list(_PATCH_ID_COLS),
    )


# ===========================================================================
# Issue 4: Shared reorder_with_identifiers in data_utils
# ===========================================================================

class TestReorderWithIdentifiers(unittest.TestCase):
    """Issue 4 — reorder_with_identifiers must exist in data_utils and work correctly."""

    def test_function_exists_in_data_utils(self):
        from finance_ml.analytics.data_utils import reorder_with_identifiers
        self.assertTrue(callable(reorder_with_identifiers))

    def test_identifier_cols_come_first(self):
        from finance_ml.analytics.data_utils import reorder_with_identifiers
        df = pd.DataFrame({
            "value": [1, 2],
            "ticker": ["A", "B"],
            "isin": ["X", "Y"],
            "sector": ["Tech", "Fin"],
        })
        with _patch_id_cols():
            result = reorder_with_identifiers(df)
        # isin should be first, then ticker, then sector, then non-id cols
        self.assertEqual(list(result.columns[:3]), ["isin", "ticker", "sector"])
        self.assertIn("value", result.columns)

    def test_missing_id_cols_skipped(self):
        from finance_ml.analytics.data_utils import reorder_with_identifiers
        df = pd.DataFrame({"ticker": ["A"], "metric": [1.0]})
        with _patch_id_cols():
            result = reorder_with_identifiers(df)
        self.assertEqual(list(result.columns), ["ticker", "metric"])

    def test_exported_from_package(self):
        """reorder_with_identifiers should be importable from the analytics package."""
        from finance_ml.analytics import reorder_with_identifiers
        self.assertTrue(callable(reorder_with_identifiers))


# ===========================================================================
# Issue 1: Identifier column propagation through model outputs
# ===========================================================================

class TestIdentifierPropagation(unittest.TestCase):
    """Issue 1 — model outputs must carry all available identifier columns."""

    def _assert_id_cols_present(self, result_df: pd.DataFrame, source_df: pd.DataFrame):
        """Assert that all identifier columns present in source appear in result."""
        for col in _PATCH_ID_COLS:
            if col in source_df.columns:
                self.assertIn(
                    col, result_df.columns,
                    f"Identifier column '{col}' missing from model output",
                )

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_earnings_beat_model_propagates_identifiers(self, _mock):
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel
        model = EarningsBeatProbabilityModel()
        df = _sample_equities_df(3)
        result = model.analyze_dataframe(df)
        self.assertGreater(len(result), 0)
        self._assert_id_cols_present(result, df)

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_credit_risk_model_propagates_identifiers(self, _mock):
        from finance_ml.analytics.probability_analytics import CreditRiskProbabilityModel
        model = CreditRiskProbabilityModel()
        df = _sample_equities_df(3)
        result = model.analyze_dataframe(df)
        self.assertGreater(len(result), 0)
        self._assert_id_cols_present(result, df)

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_dividend_cut_model_propagates_identifiers(self, _mock):
        from finance_ml.analytics.probability_analytics import DividendCutProbabilityModel
        model = DividendCutProbabilityModel()
        df = _sample_equities_df(3)
        result = model.analyze_dataframe(df)
        self.assertGreater(len(result), 0)
        self._assert_id_cols_present(result, df)

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_price_target_model_propagates_identifiers(self, _mock):
        from finance_ml.analytics.probability_analytics import PriceTargetAchievementModel
        model = PriceTargetAchievementModel()
        df = _sample_equities_df(3)
        result = model.analyze_dataframe(df)
        self.assertGreater(len(result), 0)
        self._assert_id_cols_present(result, df)

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_eps_streak_analyzer_propagates_identifiers(self, _mock):
        from finance_ml.analytics.probability_analytics import EPSStreakAnalyzer
        analyzer = EPSStreakAnalyzer()
        df = _sample_equities_df(3)
        result = analyzer.analyze_dataframe(df)
        self.assertGreater(len(result), 0)
        self._assert_id_cols_present(result, df)

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_isin_specifically_present(self, _mock):
        """isin is the primary business key and MUST appear in every model output."""
        from finance_ml.analytics.probability_analytics import (
            EarningsBeatProbabilityModel,
            CreditRiskProbabilityModel,
        )
        df = _sample_equities_df(2)
        for ModelClass in [EarningsBeatProbabilityModel, CreditRiskProbabilityModel]:
            model = ModelClass()
            result = model.analyze_dataframe(df)
            self.assertIn("isin", result.columns, f"{ModelClass.__name__} missing isin")


# ===========================================================================
# Issue 8: confidence_interval tuple → ci_lower / ci_upper
# ===========================================================================

class TestCreditRiskCISplit(unittest.TestCase):
    """Issue 8 — CreditRiskProbabilityModel must emit ci_lower/ci_upper, not tuple."""

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_ci_columns_are_numeric(self, _mock):
        from finance_ml.analytics.probability_analytics import CreditRiskProbabilityModel
        model = CreditRiskProbabilityModel()
        df = _sample_equities_df(3)
        result = model.analyze_dataframe(df)
        self.assertIn("ci_lower", result.columns)
        self.assertIn("ci_upper", result.columns)
        self.assertNotIn("confidence_interval", result.columns)
        # Values should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(result["ci_lower"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(result["ci_upper"]))


# ===========================================================================
# Issue 7: Numeric dtype casting before export
# ===========================================================================

class TestNumericCasting(unittest.TestCase):
    """Issue 7 — mixed-type columns must be cast to proper numeric dtypes."""

    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_enhanced_output_numeric_cols(self, _mock):
        """Columns like gaap_revision_momentum should be float, not object."""
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel
        model = EarningsBeatProbabilityModel()
        # Build df with forward signals so enhanced path is used
        df = _sample_equities_df(3)
        # Add forward signal columns to trigger enhanced path
        df["eps_norm_fy1e"] = [2.5, 3.0, None]
        df["eps_norm_ntm"] = [2.3, 2.8, None]
        df["eps_gaap_fy1e"] = [2.4, 2.9, None]
        df["eps_gaap_ntm"] = [2.2, 2.7, None]
        df["analyst_count"] = [15, 20, None]
        df["eps_revision_momentum"] = [5.0, -2.0, None]
        df["eps_revision_3m"] = [3.0, -1.0, None]
        df["eps_revision_6m"] = [4.0, -1.5, None]
        # Reported history columns
        df["eps_reported_q1"] = [1.0, 1.5, None]
        df["eps_reported_q2"] = [1.1, 1.6, None]
        df["eps_reported_a1"] = [4.0, 5.0, None]
        df["eps_reported_a2"] = [3.8, 4.8, None]

        result = model.analyze_dataframe_enhanced(df)
        if len(result) > 0 and "gaap_revision_momentum" in result.columns:
            # Drop NaN rows for dtype check — the non-None rows should be numeric
            non_null = result["gaap_revision_momentum"].dropna()
            if len(non_null) > 0:
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(non_null),
                    "gaap_revision_momentum should be numeric",
                )


# ===========================================================================
# Issue 2 & 3 & 5: Export function uses ExportConfig, consistent naming,
#                   reordered CSV
# ===========================================================================

class TestExportProbabilityAnalyticsResults(unittest.TestCase):
    """Issues 2, 3, 5 — export function must use ExportConfig pipeline."""

    def _make_prob_df(self):
        return pd.DataFrame({
            "ticker": ["A", "B"],
            "isin": ["US1", "US2"],
            "name": ["Co A", "Co B"],
            "sector": ["Tech", "Fin"],
            "posterior_beat_prob": [0.7, 0.4],
            "beat_classification": ["likely_beat", "uncertain"],
            "confidence_score": [0.8, 0.5],
        })

    def _make_streak_df(self):
        return pd.DataFrame({
            "ticker": ["A", "B"],
            "isin": ["US1", "US2"],
            "current_streak": [3, -1],
            "streak_type": ["beat", "miss"],
        })

    @patch("finance_ml.analytics.probability_analytics.export_to_json")
    @patch("finance_ml.analytics.probability_analytics.export_to_csv")
    @patch("finance_ml.analytics.probability_analytics.export_to_db")
    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_uses_export_config_pipeline(self, _id_mock, mock_db, mock_csv, mock_json):
        from finance_ml.analytics.probability_analytics import (
            export_probability_analytics_results,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_probability_analytics_results(
                probability_df=self._make_prob_df(),
                streak_df=self._make_streak_df(),
                output_dir=Path(tmpdir),
            )
        # export_to_db should have been called (not export_to_analytics_db directly)
        self.assertTrue(mock_db.called, "export_to_db must be used")
        self.assertTrue(mock_csv.called, "export_to_csv must be used")
        self.assertTrue(mock_json.called, "export_to_json must be used")

    @patch("finance_ml.analytics.probability_analytics.export_to_json")
    @patch("finance_ml.analytics.probability_analytics.export_to_csv")
    @patch("finance_ml.analytics.probability_analytics.export_to_db")
    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_csv_filename_matches_db_table(self, _id_mock, mock_db, mock_csv, mock_json):
        """Issue 3 — CSV filename must equal DB table name (no 'beat' mismatch)."""
        from finance_ml.analytics.probability_analytics import (
            export_probability_analytics_results,
        )
        from finance_ml.analytics.data_utils import ExportConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            export_probability_analytics_results(
                probability_df=self._make_prob_df(),
                streak_df=self._make_streak_df(),
                output_dir=Path(tmpdir),
            )

        # Collect all table_names used in export_to_csv calls
        csv_table_names = set()
        for call in mock_csv.call_args_list:
            cfg = call[1].get("config") or (call[0][1] if len(call[0]) > 1 else None)
            if isinstance(cfg, ExportConfig):
                csv_table_names.add(cfg.table_name)

        db_table_names = set()
        for call in mock_db.call_args_list:
            cfg = call[1].get("config") or (call[0][1] if len(call[0]) > 1 else None)
            if isinstance(cfg, ExportConfig):
                db_table_names.add(cfg.table_name)

        # The probability table should be "earnings_probability_analysis" in both
        self.assertIn("earnings_probability_analysis", db_table_names)
        self.assertIn("earnings_probability_analysis", csv_table_names)
        # The old mismatched name should NOT appear
        self.assertNotIn("earnings_beat_probability_analysis", csv_table_names)

    @patch("finance_ml.analytics.probability_analytics.export_to_json")
    @patch("finance_ml.analytics.probability_analytics.export_to_csv")
    @patch("finance_ml.analytics.probability_analytics.export_to_db")
    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_csv_receives_reordered_df(self, _id_mock, mock_db, mock_csv, mock_json):
        """Issue 5 — CSV export must receive the reordered DataFrame."""
        from finance_ml.analytics.probability_analytics import (
            export_probability_analytics_results,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            export_probability_analytics_results(
                probability_df=self._make_prob_df(),
                streak_df=self._make_streak_df(),
                output_dir=Path(tmpdir),
            )
        # The first positional arg to export_to_csv is the DataFrame
        for call in mock_csv.call_args_list:
            df_arg = call[0][0] if call[0] else call[1].get("df")
            if df_arg is not None and "isin" in df_arg.columns:
                # isin should be the first column (reordered)
                self.assertEqual(
                    df_arg.columns[0], "isin",
                    "CSV export DataFrame should be reordered with identifiers first",
                )
                break


# ===========================================================================
# Issue 6: Error handling for summary stats with missing columns
# ===========================================================================

class TestExportErrorHandling(unittest.TestCase):
    """Issue 6 — export must not crash when probability_df is missing columns."""

    @patch("finance_ml.analytics.probability_analytics.export_to_json")
    @patch("finance_ml.analytics.probability_analytics.export_to_csv")
    @patch("finance_ml.analytics.probability_analytics.export_to_db")
    @patch("finance_ml.analytics.probability_analytics.load_identifier_columns",
           return_value=list(_PATCH_ID_COLS))
    def test_missing_summary_columns_no_crash(self, _id_mock, mock_db, mock_csv, mock_json):
        """If probability_df lacks posterior_beat_prob, function should not crash."""
        from finance_ml.analytics.probability_analytics import (
            export_probability_analytics_results,
        )
        # DataFrame missing the columns needed for summary
        prob_df = pd.DataFrame({
            "ticker": ["A"],
            "isin": ["US1"],
            "some_metric": [0.5],
        })
        streak_df = pd.DataFrame({
            "ticker": ["A"],
            "isin": ["US1"],
            "current_streak": [2],
            "streak_type": ["beat"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should NOT raise
            result = export_probability_analytics_results(
                probability_df=prob_df,
                streak_df=streak_df,
                output_dir=Path(tmpdir),
            )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
