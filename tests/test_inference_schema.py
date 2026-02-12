"""
Tests for finance_ml.analytics.inference_schema module.

Strict TDD: tests written first, then minimal code to pass.
Covers EquityCoordinates, FeatureCoordinates, all 4 InferenceData factory
functions, summarize_inference_data, and edge cases.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestEquityCoordinates(unittest.TestCase):
    """Tests for EquityCoordinates dataclass."""

    def _make_df(self, n=3):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "isin": [f"US000{i}" for i in range(n)],
                "name": [f"Company {i}" for i in range(n)],
                "sector": ["Tech", "Health", "Finance"][:n],
                "industry": ["Software", "Pharma", "Banking"][:n],
                "country": ["US", "US", "UK"][:n],
                "exchange": ["NYSE", "NASDAQ", "LSE"][:n],
            }
        )

    def test_from_dataframe_basic(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        df = self._make_df()
        coords = EquityCoordinates.from_dataframe(df)
        np.testing.assert_array_equal(coords.tickers, ["T0", "T1", "T2"])
        np.testing.assert_array_equal(coords.sectors, ["Tech", "Health", "Finance"])

    def test_from_dataframe_missing_ticker_raises(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        df = pd.DataFrame({"isin": ["US0001"]})
        with self.assertRaises(ValueError, msg="DataFrame must contain a 'ticker' column"):
            EquityCoordinates.from_dataframe(df)

    def test_from_dataframe_minimal(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        df = pd.DataFrame({"ticker": ["AAPL", "GOOG"]})
        coords = EquityCoordinates.from_dataframe(df)
        self.assertEqual(len(coords.tickers), 2)
        self.assertEqual(len(coords.sectors), 0)

    def test_to_xarray_coords_full(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        df = self._make_df()
        coords = EquityCoordinates.from_dataframe(df)
        xr_coords = coords.to_xarray_coords()

        self.assertIn("equity", xr_coords)
        np.testing.assert_array_equal(xr_coords["equity"], ["T0", "T1", "T2"])
        self.assertIn("sector", xr_coords)
        self.assertIn("industry", xr_coords)
        self.assertIn("isin", xr_coords)

    def test_to_xarray_coords_partial(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        df = pd.DataFrame({"ticker": ["AAPL"]})
        coords = EquityCoordinates.from_dataframe(df)
        xr_coords = coords.to_xarray_coords()

        self.assertIn("equity", xr_coords)
        self.assertNotIn("sector", xr_coords)

    def test_frozen_dataclass(self):
        from finance_ml.analytics.inference_schema import EquityCoordinates

        coords = EquityCoordinates(tickers=np.array(["AAPL"]))
        with self.assertRaises(AttributeError):
            coords.tickers = np.array(["GOOG"])


class TestFeatureCoordinates(unittest.TestCase):
    """Tests for FeatureCoordinates dataclass."""

    def _make_registry_df(self):
        return pd.DataFrame(
            {
                "feature_key": ["roe", "pe_ratio", "fcf_yield"],
                "feature_alias": ["Return on Equity", "P/E Ratio", "FCF Yield"],
                "category": ["Profitability", "Valuation", "Cash Flow"],
                "source_function": ["calc_roe", "calc_pe", "calc_fcf"],
                "primary_source_col": ["net_income", "earnings", "fcf"],
            }
        )

    def test_from_dataframe(self):
        from finance_ml.analytics.inference_schema import FeatureCoordinates

        df = self._make_registry_df()
        coords = FeatureCoordinates.from_dataframe(df)
        np.testing.assert_array_equal(coords.feature_keys, ["roe", "pe_ratio", "fcf_yield"])
        np.testing.assert_array_equal(
            coords.categories, ["Profitability", "Valuation", "Cash Flow"]
        )

    def test_to_xarray_coords(self):
        from finance_ml.analytics.inference_schema import FeatureCoordinates

        df = self._make_registry_df()
        coords = FeatureCoordinates.from_dataframe(df)
        xr_coords = coords.to_xarray_coords()

        self.assertIn("feature", xr_coords)
        self.assertIn("category", xr_coords)
        self.assertIn("feature_alias", xr_coords)

    def test_from_dataframe_minimal(self):
        from finance_ml.analytics.inference_schema import FeatureCoordinates

        df = pd.DataFrame({"feature_key": ["roe"]})
        coords = FeatureCoordinates.from_dataframe(df)
        self.assertEqual(len(coords.feature_keys), 1)
        self.assertEqual(len(coords.categories), 0)

    def test_frozen_dataclass(self):
        from finance_ml.analytics.inference_schema import FeatureCoordinates

        coords = FeatureCoordinates(feature_keys=np.array(["roe"]))
        with self.assertRaises(AttributeError):
            coords.feature_keys = np.array(["pe"])


class TestBuildBeatProbabilityInferenceData(unittest.TestCase):
    """Tests for build_beat_probability_inference_data factory."""

    def _make_beat_results_df(self, n=5):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "sector": ["Tech"] * n,
                "posterior_alpha": rng.uniform(2, 10, n),
                "posterior_beta": rng.uniform(2, 10, n),
                "prior_alpha": np.full(n, 2.0),
                "prior_beta": np.full(n, 2.0),
                "historical_beat_rate": rng.uniform(0.3, 0.8, n),
                "confidence_score": rng.uniform(0.5, 1.0, n),
            }
        )

    def _make_observed_df(self, n=5):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "last_price": np.random.default_rng(42).uniform(10, 200, n),
            }
        )

    def test_returns_inference_data_or_dataset(self):
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            ARVIZ_AVAILABLE,
        )
        import xarray as xr

        beat_df = self._make_beat_results_df()
        obs_df = self._make_observed_df()
        result = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=100, n_chains=2, random_seed=42
        )

        if ARVIZ_AVAILABLE:
            import arviz as az

            self.assertIsInstance(result, az.InferenceData)
            self.assertIn("posterior", result.groups())
            self.assertIn("posterior_predictive", result.groups())
            self.assertIn("observed_data", result.groups())
            self.assertIn("log_likelihood", result.groups())
        else:
            self.assertIsInstance(result, xr.Dataset)
            self.assertIn("beat_probability", result.data_vars)

    def test_posterior_shape(self):
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            ARVIZ_AVAILABLE,
        )

        n_equities = 5
        n_chains = 2
        n_draws = 100
        beat_df = self._make_beat_results_df(n_equities)
        obs_df = self._make_observed_df(n_equities)

        result = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=n_draws, n_chains=n_chains, random_seed=42
        )

        if ARVIZ_AVAILABLE:
            post = result.posterior
            self.assertEqual(post.sizes["chain"], n_chains)
            self.assertEqual(post.sizes["draw"], n_draws)
            self.assertEqual(post.sizes["equity"], n_equities)
        else:
            self.assertEqual(result.sizes["chain"], n_chains)
            self.assertEqual(result.sizes["draw"], n_draws)
            self.assertEqual(result.sizes["equity"], n_equities)

    def test_reproducibility(self):
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            ARVIZ_AVAILABLE,
        )

        beat_df = self._make_beat_results_df()
        obs_df = self._make_observed_df()

        r1 = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=50, n_chains=2, random_seed=123
        )
        r2 = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=50, n_chains=2, random_seed=123
        )

        if ARVIZ_AVAILABLE:
            np.testing.assert_array_equal(
                r1.posterior["beat_probability"].values,
                r2.posterior["beat_probability"].values,
            )
        else:
            np.testing.assert_array_equal(
                r1["beat_probability"].values,
                r2["beat_probability"].values,
            )

    def test_posterior_values_in_range(self):
        """Beta posterior samples should be in (0, 1)."""
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            ARVIZ_AVAILABLE,
        )

        beat_df = self._make_beat_results_df()
        obs_df = self._make_observed_df()
        result = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=200, n_chains=2, random_seed=42
        )

        if ARVIZ_AVAILABLE:
            vals = result.posterior["beat_probability"].values
        else:
            vals = result["beat_probability"].values

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))


class TestBuildCreditRiskInferenceData(unittest.TestCase):
    """Tests for build_credit_risk_inference_data factory."""

    def _make_ruin_df(self, n=4):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "ruin_probability": [0.05, 0.3, 0.7, 0.95],
                "capital": [1e6, 5e5, 2e5, 1e4],
                "cash_burn": [1e4, 2e4, 5e4, 1e4],
                "volatility": [0.2, 0.3, 0.5, 0.8],
                "risk_tier": ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
            }
        )

    def _make_observed_df(self, n=4):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "distress_risk_score": [80, 50, 30, 10],
                "altman_z_score": [3.5, 2.0, 1.5, 0.5],
            }
        )

    def test_returns_valid_result(self):
        from finance_ml.analytics.inference_schema import (
            build_credit_risk_inference_data,
            ARVIZ_AVAILABLE,
        )
        import xarray as xr

        ruin_df = self._make_ruin_df()
        obs_df = self._make_observed_df()
        result = build_credit_risk_inference_data(
            ruin_df, obs_df, n_posterior_samples=100, n_chains=2, random_seed=42
        )

        if ARVIZ_AVAILABLE:
            import arviz as az

            self.assertIsInstance(result, az.InferenceData)
            self.assertIn("posterior", result.groups())
        else:
            self.assertIsInstance(result, xr.Dataset)
            self.assertIn("ruin_probability", result.data_vars)

    def test_posterior_shape(self):
        from finance_ml.analytics.inference_schema import (
            build_credit_risk_inference_data,
            ARVIZ_AVAILABLE,
        )

        n_equities = 4
        n_chains = 2
        n_draws = 50
        result = build_credit_risk_inference_data(
            self._make_ruin_df(n_equities),
            self._make_observed_df(n_equities),
            n_posterior_samples=n_draws,
            n_chains=n_chains,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            post = result.posterior
            self.assertEqual(post.sizes["chain"], n_chains)
            self.assertEqual(post.sizes["draw"], n_draws)
            self.assertEqual(post.sizes["equity"], n_equities)
        else:
            self.assertEqual(result.sizes["chain"], n_chains)
            self.assertEqual(result.sizes["draw"], n_draws)
            self.assertEqual(result.sizes["equity"], n_equities)

    def test_ruin_probability_in_range(self):
        from finance_ml.analytics.inference_schema import (
            build_credit_risk_inference_data,
            ARVIZ_AVAILABLE,
        )

        result = build_credit_risk_inference_data(
            self._make_ruin_df(),
            self._make_observed_df(),
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            vals = result.posterior["ruin_probability"].values
        else:
            vals = result["ruin_probability"].values

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))


class TestBuildMonteCarloInferenceData(unittest.TestCase):
    """Tests for build_monte_carlo_inference_data factory."""

    def _make_mc_df(self, n=3):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "last_price": [100.0, 200.0, 50.0],
                "pt_median": [120.0, 220.0, 60.0],
            }
        )

    def _make_observed_df(self, n=3):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "last_price": [100.0, 200.0, 50.0],
            }
        )

    def test_returns_valid_result(self):
        from finance_ml.analytics.inference_schema import (
            build_monte_carlo_inference_data,
            ARVIZ_AVAILABLE,
        )
        import xarray as xr

        result = build_monte_carlo_inference_data(
            self._make_mc_df(),
            self._make_observed_df(),
            n_simulations=500,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            import arviz as az

            self.assertIsInstance(result, az.InferenceData)
            self.assertIn("posterior_predictive", result.groups())
            self.assertIn("observed_data", result.groups())
        else:
            self.assertIsInstance(result, xr.Dataset)
            self.assertIn("simulated_price", result.data_vars)

    def test_simulation_shape(self):
        from finance_ml.analytics.inference_schema import (
            build_monte_carlo_inference_data,
            ARVIZ_AVAILABLE,
        )

        n_equities = 3
        n_sims = 200
        result = build_monte_carlo_inference_data(
            self._make_mc_df(n_equities),
            self._make_observed_df(n_equities),
            n_simulations=n_sims,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            pp = result.posterior_predictive
            self.assertEqual(pp.sizes["chain"], 1)
            self.assertEqual(pp.sizes["draw"], n_sims)
            self.assertEqual(pp.sizes["equity"], n_equities)
        else:
            self.assertEqual(result.sizes["chain"], 1)
            self.assertEqual(result.sizes["draw"], n_sims)
            self.assertEqual(result.sizes["equity"], n_equities)

    def test_prices_positive(self):
        from finance_ml.analytics.inference_schema import (
            build_monte_carlo_inference_data,
            ARVIZ_AVAILABLE,
        )

        result = build_monte_carlo_inference_data(
            self._make_mc_df(),
            self._make_observed_df(),
            n_simulations=500,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            vals = result.posterior_predictive["simulated_price"].values
        else:
            vals = result["simulated_price"].values

        self.assertTrue(np.all(vals > 0))


class TestBuildCategoryAnalysisInferenceData(unittest.TestCase):
    """Tests for build_category_analysis_inference_data factory."""

    def _make_analysis_results(self):
        return {
            "roe": {
                "posterior_mean": 15.0,
                "posterior_std": 2.5,
                "n_obs": 100,
                "sample_mean": 14.8,
            },
            "roa": {
                "posterior_mean": 8.0,
                "posterior_std": 1.2,
                "n_obs": 100,
                "sample_mean": 7.9,
            },
        }

    def _make_observed_df(self, n=10):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "roe": rng.normal(15, 3, n),
                "roa": rng.normal(8, 2, n),
            }
        )

    def test_returns_valid_result(self):
        from finance_ml.analytics.inference_schema import (
            build_category_analysis_inference_data,
            ARVIZ_AVAILABLE,
        )
        import xarray as xr

        result = build_category_analysis_inference_data(
            self._make_analysis_results(),
            self._make_observed_df(),
            category_name="Profitability",
            features=["roe", "roa"],
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            import arviz as az

            self.assertIsInstance(result, az.InferenceData)
            self.assertIn("posterior", result.groups())
        else:
            self.assertIsInstance(result, xr.Dataset)
            self.assertIn("feature_mean", result.data_vars)

    def test_posterior_shape(self):
        from finance_ml.analytics.inference_schema import (
            build_category_analysis_inference_data,
            ARVIZ_AVAILABLE,
        )

        n_chains = 2
        n_draws = 50
        features = ["roe", "roa"]
        result = build_category_analysis_inference_data(
            self._make_analysis_results(),
            self._make_observed_df(),
            category_name="Profitability",
            features=features,
            n_posterior_samples=n_draws,
            n_chains=n_chains,
            random_seed=42,
        )

        if ARVIZ_AVAILABLE:
            post = result.posterior
            self.assertEqual(post.sizes["chain"], n_chains)
            self.assertEqual(post.sizes["draw"], n_draws)
            self.assertEqual(post.sizes["feature"], len(features))
        else:
            self.assertEqual(result.sizes["chain"], n_chains)
            self.assertEqual(result.sizes["draw"], n_draws)
            self.assertEqual(result.sizes["feature"], len(features))

    def test_no_analysed_features_raises(self):
        from finance_ml.analytics.inference_schema import (
            build_category_analysis_inference_data,
        )

        with self.assertRaises(ValueError):
            build_category_analysis_inference_data(
                analysis_results={
                    "roe": {"posterior_mean": 1, "posterior_std": 1, "n_obs": 10, "sample_mean": 1}
                },
                observed_df=self._make_observed_df(),
                category_name="Test",
                features=["nonexistent_feature"],
            )

    def test_with_feature_coordinates(self):
        from finance_ml.analytics.inference_schema import (
            build_category_analysis_inference_data,
            FeatureCoordinates,
            ARVIZ_AVAILABLE,
        )

        fc = FeatureCoordinates(
            feature_keys=np.array(["roe", "roa"]),
            feature_aliases=np.array(["Return on Equity", "Return on Assets"]),
            categories=np.array(["Profitability", "Profitability"]),
        )

        result = build_category_analysis_inference_data(
            self._make_analysis_results(),
            self._make_observed_df(),
            category_name="Profitability",
            features=["roe", "roa"],
            n_posterior_samples=50,
            n_chains=2,
            random_seed=42,
            feature_coords=fc,
        )

        if ARVIZ_AVAILABLE:
            self.assertIn("posterior", result.groups())
        else:
            self.assertIn("feature_mean", result.data_vars)


class TestSummarizeInferenceData(unittest.TestCase):
    """Tests for summarize_inference_data utility."""

    def test_summarize_xr_dataset(self):
        import xarray as xr
        from finance_ml.analytics.inference_schema import summarize_inference_data

        ds = xr.Dataset(
            {"value": (["chain", "draw"], np.random.randn(2, 100))},
            coords={"chain": [0, 1], "draw": np.arange(100)},
        )
        summary = summarize_inference_data(ds)
        self.assertIn("groups", summary)
        self.assertEqual(summary["n_chains"], 2)
        self.assertEqual(summary["n_draws"], 100)
        self.assertIn("value", summary["variables"])

    def test_summarize_inference_data_object(self):
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            summarize_inference_data,
            ARVIZ_AVAILABLE,
        )

        if not ARVIZ_AVAILABLE:
            self.skipTest("ArviZ not available")

        rng = np.random.default_rng(42)
        beat_df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "posterior_alpha": [5.0, 3.0],
                "posterior_beta": [2.0, 4.0],
            }
        )
        obs_df = pd.DataFrame({"ticker": ["A", "B"]})
        result = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=50, n_chains=2, random_seed=42
        )
        summary = summarize_inference_data(result)

        self.assertIn("groups", summary)
        self.assertEqual(summary["n_chains"], 2)
        self.assertEqual(summary["n_draws"], 50)
        self.assertIn("beat_probability", summary["variables"])

    def test_summarize_unknown_type(self):
        from finance_ml.analytics.inference_schema import summarize_inference_data

        summary = summarize_inference_data("not_valid")
        self.assertIn("error", summary)


class TestModuleConstants(unittest.TestCase):
    """Tests for module-level constants and availability flags."""

    def test_arviz_available_flag(self):
        from finance_ml.analytics.inference_schema import ARVIZ_AVAILABLE

        self.assertIsInstance(ARVIZ_AVAILABLE, bool)

    def test_role_to_coord_dim(self):
        from finance_ml.analytics.inference_schema import _ROLE_TO_COORD_DIM

        self.assertIn("id", _ROLE_TO_COORD_DIM)
        self.assertEqual(_ROLE_TO_COORD_DIM["id"], "equity")

    def test_feature_dim_constant(self):
        from finance_ml.analytics.inference_schema import _FEATURE_DIM

        self.assertEqual(_FEATURE_DIM, "feature")


class TestDBLoadersValidation(unittest.TestCase):
    """Tests for DB loaders — error handling without real DB."""

    def test_load_equity_coordinates_no_db_url(self):
        from finance_ml.analytics.inference_schema import load_equity_coordinates_from_db
        import os

        # Ensure DB_URL is not set
        old = os.environ.pop("DB_URL", None)
        try:
            with self.assertRaises(ValueError):
                load_equity_coordinates_from_db(db_url=None)
        finally:
            if old is not None:
                os.environ["DB_URL"] = old

    def test_load_feature_coordinates_no_db_url(self):
        from finance_ml.analytics.inference_schema import load_feature_coordinates_from_db
        import os

        old = os.environ.pop("DB_URL", None)
        try:
            with self.assertRaises(ValueError):
                load_feature_coordinates_from_db(db_url=None)
        finally:
            if old is not None:
                os.environ["DB_URL"] = old


class TestResolveCol(unittest.TestCase):
    """Tests for _resolve_col helper in screening.py."""

    def test_resolve_first_match(self):
        from finance_ml.analytics.screening import _resolve_col

        df = pd.DataFrame({"interest_coverage": [1], "interest_coverage_ratio": [2]})
        self.assertEqual(
            _resolve_col(df, "interest_coverage", "interest_coverage_ratio"), "interest_coverage"
        )

    def test_resolve_second_match(self):
        from finance_ml.analytics.screening import _resolve_col

        df = pd.DataFrame({"interest_coverage_ratio": [2]})
        self.assertEqual(
            _resolve_col(df, "interest_coverage", "interest_coverage_ratio"),
            "interest_coverage_ratio",
        )

    def test_resolve_none(self):
        from finance_ml.analytics.screening import _resolve_col

        df = pd.DataFrame({"other_col": [1]})
        self.assertIsNone(_resolve_col(df, "interest_coverage", "interest_coverage_ratio"))


class TestOptimizedOpsArvizFlag(unittest.TestCase):
    """Test that get_optimization_status includes arviz_available."""

    def test_arviz_in_status(self):
        from finance_ml.analytics.optimized_ops import get_optimization_status

        status = get_optimization_status()
        self.assertIn("arviz_available", status)
        self.assertIsInstance(status["arviz_available"], bool)


class TestValidateFeatureRegistryAlignment(unittest.TestCase):
    """Tests for validate_feature_registry_alignment in data_utils."""

    def test_no_db_url_returns_error(self):
        from finance_ml.analytics.data_utils import validate_feature_registry_alignment
        import os

        old = os.environ.pop("DB_URL", None)
        try:
            result = validate_feature_registry_alignment(db_url=None)
            self.assertIn("error", result)
        finally:
            if old is not None:
                os.environ["DB_URL"] = old


class TestInitExports(unittest.TestCase):
    """Test that __init__.py exports the new symbols."""

    def test_inference_schema_exports(self):
        from finance_ml.analytics import __all__

        expected = [
            "ARVIZ_AVAILABLE",
            "EquityCoordinates",
            "FeatureCoordinates",
            "build_beat_probability_inference_data",
            "build_credit_risk_inference_data",
            "build_monte_carlo_inference_data",
            "build_category_analysis_inference_data",
            "load_equity_coordinates_from_db",
            "load_feature_coordinates_from_db",
            "summarize_inference_data",
        ]
        for name in expected:
            self.assertIn(name, __all__, f"{name} not in __all__")

    def test_can_import_from_analytics(self):
        from finance_ml.analytics import (
            EquityCoordinates,
            FeatureCoordinates,
            build_beat_probability_inference_data,
            summarize_inference_data,
        )

        self.assertIsNotNone(EquityCoordinates)
        self.assertIsNotNone(FeatureCoordinates)


class TestIntegrationBeatToSummary(unittest.TestCase):
    """Integration test: build inference data then summarize."""

    def test_end_to_end_beat_probability(self):
        from finance_ml.analytics.inference_schema import (
            build_beat_probability_inference_data,
            summarize_inference_data,
            ARVIZ_AVAILABLE,
        )

        beat_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOG", "MSFT"],
                "sector": ["Tech", "Tech", "Tech"],
                "posterior_alpha": [8.0, 6.0, 9.0],
                "posterior_beta": [3.0, 4.0, 2.0],
                "prior_alpha": [2.0, 2.0, 2.0],
                "prior_beta": [2.0, 2.0, 2.0],
                "historical_beat_rate": [0.7, 0.5, 0.8],
            }
        )
        obs_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOG", "MSFT"],
                "last_price": [150.0, 2800.0, 300.0],
            }
        )

        idata = build_beat_probability_inference_data(
            beat_df, obs_df, n_posterior_samples=200, n_chains=2, random_seed=42
        )
        summary = summarize_inference_data(idata)

        self.assertIn("n_chains", summary)
        self.assertEqual(summary["n_chains"], 2)
        self.assertEqual(summary["n_draws"], 200)

        if ARVIZ_AVAILABLE:
            self.assertIn("r_hat", summary)
            self.assertIn("beat_probability", summary["variables"])


if __name__ == "__main__":
    unittest.main()
