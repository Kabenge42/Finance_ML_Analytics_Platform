"""Tests for the clean finance_ml.api facade (Phase 6.1).

These tests intentionally stay *lightweight* and focus on importability
and basic callability of the documented public symbols. Heavy model
training, database access, or large-data operations are explicitly
avoided to keep the test quick and deterministic.
"""

from __future__ import annotations

import unittest

import pandas as pd


class TestPublicAPISymbols(unittest.TestCase):
    def test_api_imports_and_all(self):
        import finance_ml.api as api

        required = [
            # Data / preprocessing
            "load_from_csv",
            "load_from_db",
            "normalize_columns",
            "prepare_phase91_data",
            # Features
            "build_features",
            "PresetName",
            # Classification
            "create_enhanced_event_labels",
            "prepare_classification_data",
            "train_xgboost_classifier",
            "train_lightgbm_classifier",
            "train_catboost_classifier",
            "train_neural_network_classifier",
            "train_stacking_classifier",
            "compare_classifiers",
            # Regression
            "create_event_labels",
            "train_event_classifier",
            "prepare_regression_data",
            "train_stacking_regressor",
            "compare_regressors",
            "train_quantile_regressor",
            "train_sector_specific_models",
            # Analytics
            "calculate_mispricing_score",
            "calculate_risk_adjusted_mispricing",
            "rank_undervalued_stocks",
            "rank_overvalued_stocks",
            "rank_stocks_by_sector",
            "optimize_portfolio",
            # Reporting
            "calculate_financial_metrics_dashboard",
            "generate_data_quality_alerts",
            "prepare_plotly_dashboard_data",
        ]

        for name in required:
            with self.subTest(name=name):
                self.assertTrue(hasattr(api, name), f"finance_ml.api missing {name}")

        # __all__ should include at least these names for explicit export
        self.assertTrue(hasattr(api, "__all__"))
        for name in required:
            self.assertIn(name, api.__all__)


class TestPublicAPIBasicCalls(unittest.TestCase):
    """Very small smoke tests on a subset of API functions.

    We only test functions that can operate on tiny in-memory DataFrames
    without external services. The goal is to catch obvious wiring
    issues (wrong imports, missing parameters) rather than exhaustively
    validate ML behavior (covered elsewhere).
    """

    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Finance"],
                "last_price": [10.0, 20.0, 30.0],
                "price_target": [12.0, 21.0, 25.0],
            }
        )

    def test_build_features_and_mispricing(self):
        import finance_ml.api as api

        # build_features should accept a small DataFrame and return a DataFrame
        feats = api.build_features(self.df.copy(), preset="basic")
        self.assertIsInstance(feats, pd.DataFrame)
        self.assertGreaterEqual(len(feats), len(self.df))

        # calculate_mispricing_score should work with default columns
        mis = api.calculate_mispricing_score(
            feats.assign(predicted_price_target=feats.get("price_target", self.df["price_target"]))
        )
        self.assertIsInstance(mis, pd.DataFrame)
        self.assertIn("mispricing_score", mis.columns)

        # ranking helpers should return subset DataFrames
        top_under = api.rank_undervalued_stocks(mis, top_n=2)
        self.assertLessEqual(len(top_under), 2)

    def test_prepare_phase91_data_basic(self):
        import finance_ml.api as api

        # prepare_phase91_data should at least return a DataFrame for tiny input
        out = api.prepare_phase91_data(
            self.df.copy(), apply_outlier_detection=False, apply_winsorization=False
        )
        self.assertIsInstance(out, pd.DataFrame)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
