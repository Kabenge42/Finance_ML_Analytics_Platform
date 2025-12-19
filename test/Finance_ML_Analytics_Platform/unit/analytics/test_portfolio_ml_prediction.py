"""Tests for stock selection and ML-related helpers (Phase 1–2).

This module now contains concrete TDD tests derived from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md.
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.analytics.stock_selection import (
    rank_stocks_balanced,
    rank_stocks_multi_metric,
    select_portfolio_candidates,
)

from finance_ml.ml_workflow.analytics.ml_returns import (
    create_ml_return_features,
    train_linear_return_predictor,
    create_ensemble_return_predictions,
    evaluate_return_predictions,
)


def create_sample_portfolio_data() -> pd.DataFrame:
    """Small helper to build a deterministic sample universe for tests.

    This mirrors the helper used in test_portfolio_selection_enhancements,
    but extends it with the columns required for composite scoring.
    """

    rng = np.random.RandomState(42)

    data = {
        "ticker": [f"T{i}" for i in range(10)],
        "sector": [
            "Technology",
            "Technology",
            "Healthcare",
            "Healthcare",
            "Finance",
            "Finance",
            "Energy",
            "Energy",
            "Utilities",
            "Utilities",
        ],
        "region": ["US", "EU", "US", "EU", "US", "EU", "US", "EU", "US", "EU"],
        "market_cap": np.linspace(5e9, 50e9, 10),
        "mispricing_score": np.linspace(1.0, 10.0, 10),
        "expected_return": rng.normal(0.10, 0.02, size=10),
        "return_1y": rng.normal(0.08, 0.03, size=10),
    }
    return pd.DataFrame(data)


def create_sample_return_series(n: int = 100) -> pd.DataFrame:
    """Create a deterministic time series with 1d returns and prices.

    The pattern has a mild positive drift so that simple ML models trained on
    the engineered features can achieve a small but positive correlation with
    the target returns without requiring heavy training.
    """

    rng = np.random.RandomState(123)
    # Daily returns with mean ~0.001 and small noise
    returns_1d = rng.normal(0.001, 0.01, size=n)
    prices = 100 * (1 + returns_1d).cumprod()

    return pd.DataFrame(
        {
            "return_1d": returns_1d,
            "last_price": prices,
        }
    )


class TestRankStocksMultiMetric(unittest.TestCase):
    """Phase 1.2.1 – ranking by composite score."""

    def test_rank_by_composite_score(self):
        df = create_sample_portfolio_data()

        top_stocks = rank_stocks_multi_metric(
            df,
            metrics=["mispricing_score", "expected_return", "return_1y"],
            weights=[0.5, 0.3, 0.2],
            top_n=5,
        )

        self.assertLessEqual(len(top_stocks), 5)
        self.assertIn("composite_score", top_stocks.columns)
        # Composite score should be monotonically non‑increasing
        self.assertTrue(top_stocks["composite_score"].is_monotonic_decreasing)


class TestRankStocksBalanced(unittest.TestCase):
    """Phase 1.2.2 – sector‑balanced ranking."""

    def test_sector_balanced_ranking(self):
        df = create_sample_portfolio_data()

        top_stocks = rank_stocks_balanced(
            df,
            top_n=10,
            max_sector_weight=0.3,
            ranking_col="mispricing_score",
        )

        sector_counts = top_stocks["sector"].value_counts(normalize=True)
        self.assertTrue((sector_counts <= 0.3 + 1e-9).all())


class TestSelectPortfolioCandidates(unittest.TestCase):
    """Phase 1.3.1 – integration helper for notebook Step 3.5."""

    def test_notebook_portfolio_candidate_selection(self):
        all_stocks = create_sample_portfolio_data()

        candidates = select_portfolio_candidates(
            all_stocks, min_market_cap=1, top_n=5, max_sector_weight=0.4
        )

        self.assertLessEqual(len(candidates), 5)
        self.assertIn("composite_score", candidates.columns)

        sector_weights = candidates.groupby("sector").size() / len(candidates)
        self.assertTrue((sector_weights <= 0.4 + 1e-9).all())


class TestCreateMlReturnFeatures(unittest.TestCase):
    """Phase 2.1.1 – ML feature engineering for return prediction."""

    def test_create_ml_features_for_returns(self):
        base_df = create_sample_return_series(n=60)

        features_df = create_ml_return_features(
            base_df,
            lags=[5, 10, 20],
            technical_indicators=["sma", "momentum", "volatility"],
        )

        expected_cols = [
            "return_lag_5",
            "return_lag_10",
            "return_lag_20",
            "sma_20",
            "momentum_10",
            "volatility_20",
        ]

        for col in expected_cols:
            self.assertIn(col, features_df.columns)

        # All rows in the returned feature frame should be fully populated
        self.assertFalse(features_df.isnull().any().any())


class TestMlReturnPredictor(unittest.TestCase):
    """Phase 2.2 – compact ML-based return predictor.

    Uses a lightweight linear model instead of a heavy DNN to keep tests fast
    and dependency-light while still validating basic predictive behaviour.
    """

    def test_train_linear_return_predictor(self):
        base_df = create_sample_return_series(n=200)
        features_df = create_ml_return_features(base_df)

        # Align target with features index
        y = base_df.loc[features_df.index, "return_1d"].to_numpy()
        X = features_df[[c for c in features_df.columns if c != "return_1d"]].to_numpy()

        # Simple train/test split
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = train_linear_return_predictor(X_train, y_train)
        y_pred = model.predict(X_test)

        # Shape check
        self.assertEqual(y_pred.shape, y_test.shape)

        # Predictions should be finite and in a reasonable range for daily returns
        self.assertTrue(np.isfinite(y_pred).all())
        self.assertGreater(y_pred.min(), -0.2)
        self.assertLess(y_pred.max(), 0.2)

        # Phase 2 review checkpoint: evaluate correlation and error metrics
        metrics = evaluate_return_predictions(y_test, y_pred)

        # Basic positive correlation with actuals
        self.assertGreater(metrics["correlation"], 0.1)

        # Error metrics should be finite and reasonably small for synthetic data
        self.assertTrue(np.isfinite(metrics["mae"]))
        self.assertTrue(np.isfinite(metrics["rmse"]))
        self.assertLess(metrics["mae"], 0.05)
        self.assertLess(metrics["rmse"], 0.05)


class TestEnsembleReturnPrediction(unittest.TestCase):
    """Phase 2.3 – ensemble of multiple return prediction sources."""

    def test_ensemble_return_prediction(self):
        rng = np.random.RandomState(7)
        df = pd.DataFrame(
            {
                "ml_prediction": rng.normal(0.05, 0.01, size=20),
                "target_prediction": rng.normal(0.06, 0.01, size=20),
                "analyst_consensus": rng.normal(0.055, 0.005, size=20),
            }
        )

        ensemble_df = create_ensemble_return_predictions(
            df,
            models=["ml_prediction", "target_prediction", "analyst_consensus"],
            weights=[0.4, 0.4, 0.2],
        )

        self.assertIn("ensemble_return", ensemble_df.columns)
        self.assertFalse(ensemble_df["ensemble_return"].isnull().any())

        expected = (
            0.4 * ensemble_df["ml_prediction"]
            + 0.4 * ensemble_df["target_prediction"]
            + 0.2 * ensemble_df["analyst_consensus"]
        )

        np.testing.assert_array_almost_equal(
            ensemble_df["ensemble_return"].to_numpy(), expected.to_numpy(), decimal=6
        )


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
