"""
Tests for finance_ml.analytics.visualizations.probability_viz module.

Covers all 6 public functions plus helpers, using DataFrame fallback paths
(no ArviZ dependency required for tests to pass).
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from finance_ml.analytics.visualizations.probability_viz import (
    create_posterior_return_forest,
    create_beat_probability_posterior,
    create_ruin_probability_diagnostic,
    create_mcse_convergence_panel,
    create_bayesian_category_ridge,
    create_tri_model_posterior_comparison,
    float_array,
    _ruin_color,
    _tier_color,
    ARVIZ_AVAILABLE,
)

if ARVIZ_AVAILABLE:
    import arviz as az
    import xarray as xr


class TestFloatArray(unittest.TestCase):
    """Tests for the float_array helper."""

    def test_returns_float64(self):
        result = float_array([1, 2, 3])
        self.assertEqual(result.dtype, np.float64)

    def test_writable(self):
        result = float_array(np.array([1, 2], dtype=np.int32))
        result[0] = 99.0
        self.assertEqual(result[0], 99.0)


class TestRuinColor(unittest.TestCase):
    def test_critical(self):
        self.assertEqual(_ruin_color(0.7), "#e74c3c")

    def test_high(self):
        self.assertEqual(_ruin_color(0.4), "#f39c12")

    def test_moderate(self):
        self.assertEqual(_ruin_color(0.15), "#3498db")

    def test_low(self):
        self.assertEqual(_ruin_color(0.05), "#00bc8c")


class TestTierColor(unittest.TestCase):
    def test_critical(self):
        self.assertEqual(_tier_color("Critical Risk"), "#e74c3c")

    def test_high(self):
        self.assertEqual(_tier_color("High Risk"), "#f39c12")

    def test_moderate(self):
        self.assertEqual(_tier_color("Moderate Risk"), "#3498db")

    def test_low(self):
        self.assertEqual(_tier_color("Low Risk"), "#00bc8c")


class TestCreatePosteriorReturnForest(unittest.TestCase):
    """Tests for create_posterior_return_forest (DataFrame fallback)."""

    def _make_df(self, n=10):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "expected_upside_pct": np.random.default_rng(42).uniform(-10, 40, n),
                "upside_std": np.random.default_rng(42).uniform(1, 5, n),
            }
        )

    def test_returns_figure(self):
        fig = create_posterior_return_forest(self._make_df())
        self.assertIsInstance(fig, go.Figure)

    def test_top_n_limits_traces(self):
        fig = create_posterior_return_forest(self._make_df(20), top_n=5)
        # 5 HDI lines + 1 marker trace
        self.assertEqual(len(fig.data), 6)

    def test_missing_columns_returns_no_data(self):
        fig = create_posterior_return_forest(pd.DataFrame({"a": [1]}))
        # Should be a no-data figure with annotation
        self.assertTrue(
            any(
                "No data" in a.text or "missing" in a.text
                for a in fig.layout.annotations
                if hasattr(a, "text")
            )
        )

    def test_empty_after_dropna(self):
        df = pd.DataFrame({"ticker": ["A"], "expected_upside_pct": [np.nan]})
        fig = create_posterior_return_forest(df)
        self.assertIsInstance(fig, go.Figure)

    def test_no_upside_std_uses_fallback(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "expected_upside_pct": [10.0, 20.0],
            }
        )
        fig = create_posterior_return_forest(df)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_invalid_input_returns_no_data(self):
        fig = create_posterior_return_forest("not a df")
        self.assertIsInstance(fig, go.Figure)

    def test_custom_title(self):
        fig = create_posterior_return_forest(self._make_df(), title="Custom")
        self.assertIn("Custom", fig.layout.title.text)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_path(self):
        # Create mock InferenceData
        coords = {"equity": ["AAPL", "MSFT"], "chain": [0, 1], "draw": np.arange(10)}
        data = np.random.randn(2, 10, 2)
        posterior = xr.Dataset(
            {"expected_return": (("chain", "draw", "equity"), data)}, coords=coords
        )
        idata = az.InferenceData(posterior=posterior)

        fig = create_posterior_return_forest(idata, var_name="expected_return", top_n=2)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        self.assertIn("AAPL", [t.y[0] for t in fig.data if hasattr(t, "y") and len(t.y) > 0])

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_missing_var(self):
        posterior = xr.Dataset(
            {"other": (("chain", "draw"), np.random.randn(2, 10))},
            coords={"chain": [0, 1], "draw": np.arange(10)},
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_posterior_return_forest(idata, var_name="missing")
        self.assertTrue(
            any("No data available" in a.text for a in fig.layout.annotations if hasattr(a, "text"))
        )


class TestCreateBeatProbabilityPosterior(unittest.TestCase):
    """Tests for create_beat_probability_posterior (DataFrame fallback)."""

    def test_beta_density_path(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "posterior_alpha": [10.0, 8.0, 12.0],
                "posterior_beta": [3.0, 5.0, 2.0],
            }
        )
        fig = create_beat_probability_posterior(df)
        self.assertIsInstance(fig, go.Figure)
        # Should have traces for each ticker
        self.assertEqual(len(fig.data), 3)

    def test_bar_fallback_path(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "posterior_beat_prob": [0.8, 0.4, 0.6],
            }
        )
        fig = create_beat_probability_posterior(df)
        self.assertIsInstance(fig, go.Figure)

    def test_missing_columns(self):
        fig = create_beat_probability_posterior(pd.DataFrame({"x": [1]}))
        self.assertIsInstance(fig, go.Figure)

    def test_specific_tickers(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "posterior_alpha": [10.0, 8.0, 12.0],
                "posterior_beta": [3.0, 5.0, 2.0],
            }
        )
        fig = create_beat_probability_posterior(df, tickers=["A", "C"])
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)

    def test_invalid_alpha_beta_skipped(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "posterior_alpha": [-1.0, 8.0],
                "posterior_beta": [3.0, 0.0],
            }
        )
        fig = create_beat_probability_posterior(df)
        self.assertEqual(len(fig.data), 0)

    def test_invalid_input(self):
        fig = create_beat_probability_posterior(42)
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_path(self):
        coords = {"equity": ["A", "B"], "chain": [0], "draw": np.arange(10)}
        data = np.random.uniform(0, 1, (1, 10, 2))
        posterior = xr.Dataset(
            {"beat_probability": (("chain", "draw", "equity"), data)}, coords=coords
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_beat_probability_posterior(idata, tickers=["A"])
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_missing_var(self):
        posterior = xr.Dataset(
            {"other": (("chain", "draw"), np.random.randn(1, 10))},
            coords={"chain": [0], "draw": np.arange(10)},
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_beat_probability_posterior(idata)
        self.assertTrue(
            any("No data available" in a.text for a in fig.layout.annotations if hasattr(a, "text"))
        )


class TestCreateRuinProbabilityDiagnostic(unittest.TestCase):
    """Tests for create_ruin_probability_diagnostic (DataFrame path)."""

    def _make_df(self, n=30):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "ruin_probability": rng.uniform(0, 1, n),
                "risk_tier": rng.choice(
                    ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"], n
                ),
                "distress_risk_score": rng.uniform(0, 10, n),
                "sector": rng.choice(["Tech", "Finance", "Health"], n),
            }
        )

    def test_returns_figure(self):
        fig = create_ruin_probability_diagnostic(self._make_df())
        self.assertIsInstance(fig, go.Figure)

    def test_four_panels(self):
        fig = create_ruin_probability_diagnostic(self._make_df())
        # Should have multiple traces across 4 subplots
        self.assertGreater(len(fig.data), 2)

    def test_missing_ruin_col(self):
        fig = create_ruin_probability_diagnostic(pd.DataFrame({"x": [1]}))
        self.assertIsInstance(fig, go.Figure)

    def test_distress_probability_alias(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "distress_probability": [0.3, 0.7],
            }
        )
        fig = create_ruin_probability_diagnostic(df)
        self.assertIsInstance(fig, go.Figure)

    def test_invalid_input(self):
        fig = create_ruin_probability_diagnostic("bad")
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_path(self):
        coords = {"equity": ["A", "B"], "chain": [0, 1], "draw": np.arange(10)}
        data = np.random.uniform(0, 1, (2, 10, 2))
        posterior = xr.Dataset(
            {"ruin_probability": (("chain", "draw", "equity"), data)}, coords=coords
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_ruin_probability_diagnostic(idata)
        self.assertIsInstance(fig, go.Figure)
        # 1 Bar + 1 Pie + 1 Scatter + 1 Bar = 4 traces
        self.assertGreaterEqual(len(fig.data), 2)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_missing_var(self):
        posterior = xr.Dataset(
            {"other": (("chain", "draw"), np.random.randn(2, 10))},
            coords={"chain": [0, 1], "draw": np.arange(10)},
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_ruin_probability_diagnostic(idata)
        self.assertTrue(
            any("No data available" in a.text for a in fig.layout.annotations if hasattr(a, "text"))
        )


class TestCreateMcseConvergencePanel(unittest.TestCase):
    """Tests for create_mcse_convergence_panel (requires ArviZ)."""

    def test_no_arviz_returns_no_data(self):
        # Pass a non-InferenceData object; should return no-data figure
        fig = create_mcse_convergence_panel(pd.DataFrame())
        self.assertIsInstance(fig, go.Figure)

    def test_none_input(self):
        fig = create_mcse_convergence_panel(None)
        self.assertIsInstance(fig, go.Figure)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_path(self):
        # Need more draws for draw_counts to work properly (100+)
        draws = np.arange(120)
        coords = {"equity": ["A"], "chain": [0, 1], "draw": draws}
        data = np.random.randn(2, 120, 1)
        posterior = xr.Dataset(
            {"expected_return": (("chain", "draw", "equity"), data)}, coords=coords
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_mcse_convergence_panel(idata, var_name="expected_return")
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    @unittest.skipUnless(ARVIZ_AVAILABLE, "ArviZ not installed")
    def test_idata_missing_var(self):
        posterior = xr.Dataset(
            {"other": (("chain", "draw"), np.random.randn(1, 10))},
            coords={"chain": [0], "draw": np.arange(10)},
        )
        idata = az.InferenceData(posterior=posterior)
        fig = create_mcse_convergence_panel(idata, var_name="missing")
        self.assertTrue(
            any("No data available" in a.text for a in fig.layout.annotations if hasattr(a, "text"))
        )


class TestCreateBayesianCategoryRidge(unittest.TestCase):
    """Tests for create_bayesian_category_ridge."""

    def _make_results(self):
        return {
            "roe": {
                "posterior_mean": 0.15,
                "posterior_std": 0.03,
                "ci_95_low": 0.09,
                "ci_95_high": 0.21,
                "prob_positive": 0.99,
            },
            "roa": {
                "posterior_mean": 0.08,
                "posterior_std": 0.02,
                "ci_95_low": 0.04,
                "ci_95_high": 0.12,
                "prob_positive": 0.98,
            },
            "roic": {
                "posterior_mean": -0.01,
                "posterior_std": 0.05,
                "ci_95_low": -0.11,
                "ci_95_high": 0.09,
                "prob_positive": 0.42,
            },
        }

    def test_returns_figure(self):
        fig = create_bayesian_category_ridge(self._make_results())
        self.assertIsInstance(fig, go.Figure)

    def test_trace_count(self):
        fig = create_bayesian_category_ridge(self._make_results())
        self.assertEqual(len(fig.data), 3)

    def test_empty_dict(self):
        fig = create_bayesian_category_ridge({})
        self.assertIsInstance(fig, go.Figure)

    def test_zero_std_skipped(self):
        results = {"feat": {"posterior_mean": 1.0, "posterior_std": 0.0}}
        fig = create_bayesian_category_ridge(results)
        self.assertEqual(len(fig.data), 0)

    def test_custom_title(self):
        fig = create_bayesian_category_ridge(self._make_results(), title="My Title")
        self.assertIn("My Title", fig.layout.title.text)


class TestCreateTriModelPosteriorComparison(unittest.TestCase):
    """Tests for create_tri_model_posterior_comparison."""

    def _make_df(self, n=4):
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "expected_upside_pct": np.linspace(5, 30, n),
                "filtered_upside": np.linspace(3, 25, n),
                "expected_return_prob_weighted": np.linspace(4, 28, n),
                "agreement_score": np.linspace(0.5, 0.9, n),
                "upside_std": [5.0] * n,
            }
        )

    def test_returns_figure(self):
        fig = create_tri_model_posterior_comparison(self._make_df())
        self.assertIsInstance(fig, go.Figure)

    def test_specific_tickers(self):
        fig = create_tri_model_posterior_comparison(self._make_df(), tickers=["T0", "T1"])
        self.assertIsInstance(fig, go.Figure)

    def test_missing_columns(self):
        fig = create_tri_model_posterior_comparison(pd.DataFrame({"x": [1]}))
        self.assertIsInstance(fig, go.Figure)

    def test_custom_title(self):
        fig = create_tri_model_posterior_comparison(self._make_df(), title="Custom Tri")
        self.assertIn("Custom Tri", fig.layout.title.text)


class TestModuleImportFromPackage(unittest.TestCase):
    """Test that probability_viz functions are importable from the package."""

    def test_import_from_visualizations(self):
        from finance_ml.analytics.visualizations import (
            create_posterior_return_forest,
            create_beat_probability_posterior,
            create_ruin_probability_diagnostic,
            create_mcse_convergence_panel,
            create_bayesian_category_ridge,
            create_tri_model_posterior_comparison,
        )

        self.assertTrue(callable(create_posterior_return_forest))


if __name__ == "__main__":
    unittest.main()
