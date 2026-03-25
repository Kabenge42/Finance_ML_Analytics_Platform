"""
Tests for Bayesian Resampling Enhancements.

Covers:
- BayesianTechnicalResampler (statistical_analysis.py)
- ResampledBeatProbabilityModel (probability_analytics.py)
- build_resampled_technical_inference_data (inference_schema.py)
- resampled_posterior_returns convenience function
"""

from __future__ import annotations

import unittest
from dataclasses import fields

import numpy as np
import pandas as pd
from scipy import stats


def _make_equities_df(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic equities DataFrame with price snapshot columns."""
    rng = np.random.default_rng(seed)
    tickers = [f"TK{i:03d}" for i in range(n)]
    last_prices = rng.uniform(10, 500, n)
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "name": [f"Company {t}" for t in tickers],
            "sector": rng.choice(["Tech", "Finance", "Health"], n),
            "last_price": last_prices,
            "price_1m_ago": last_prices * rng.uniform(0.90, 1.10, n),
            "price_3m_ago": last_prices * rng.uniform(0.80, 1.20, n),
            "price_6m_ago": last_prices * rng.uniform(0.70, 1.30, n),
            "price_1y_ago": last_prices * rng.uniform(0.60, 1.40, n),
            "price_3y_ago": last_prices * rng.uniform(0.40, 1.60, n),
            "price_5y_ago": last_prices * rng.uniform(0.30, 1.70, n),
            # Momentum features
            "price_momentum_1m": rng.uniform(-50, 50, n),
            "price_momentum_3m": rng.uniform(-50, 50, n),
            # Technical features
            "ema_slope_20d": rng.uniform(-1, 1, n),
            "volatility_compression": rng.uniform(0, 1, n),
            # Earnings features for beat model
            "eps_beat_count": rng.integers(1, 8, n),
            "eps_total_reports": rng.integers(4, 12, n),
            "eps_trajectory_score": rng.uniform(-1, 1, n),
            "eps_positive_streak": rng.integers(0, 6, n),
            # Temporal
            "earnings_season_flag": rng.choice([0, 1], n),
            "pre_earnings_window": rng.choice([0, 1], n),
        }
    )
    # Ensure eps_beat_count <= eps_total_reports
    df["eps_beat_count"] = df[["eps_beat_count", "eps_total_reports"]].min(axis=1)
    return df


# =============================================================================
# BayesianTechnicalResampler Tests
# =============================================================================


class TestBayesianTechnicalResampler(unittest.TestCase):
    """Tests for BayesianTechnicalResampler in statistical_analysis.py."""

    def setUp(self):
        from finance_ml.analytics.statistical_analysis import BayesianTechnicalResampler

        self.resampler = BayesianTechnicalResampler(prior_return_mean=0.08, prior_return_std=0.20,
                                                    n_posterior_samples=200, n_chains=2, random_seed=42)
        self.df = _make_equities_df(n=10)

    def test_init_defaults(self):
        from finance_ml.analytics.statistical_analysis import BayesianTechnicalResampler

        r = BayesianTechnicalResampler()
        self.assertEqual(r.prior_return_mean, 0.08)
        self.assertEqual(r.prior_return_std, 0.20)
        self.assertEqual(r.n_posterior_samples, 4000)
        self.assertEqual(r.n_chains, 4)

    def test_compute_historical_returns_nonempty(self):
        returns_df = self.resampler._compute_historical_returns(self.df)
        self.assertFalse(returns_df.empty)
        self.assertIn("ticker", returns_df.columns)
        self.assertIn("annualised_return", returns_df.columns)
        self.assertIn("period", returns_df.columns)

    def test_compute_historical_returns_empty_no_price(self):
        df_no_price = self.df.drop(columns=["last_price"])
        returns_df = self.resampler._compute_historical_returns(df_no_price)
        self.assertTrue(returns_df.empty)

    def test_resample_returns_produces_dataframe(self):
        result = self.resampler.resample_returns(self.df, freq="1ME")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        expected_cols = [
            "ticker",
            "frequency",
            "n_periods",
            "sample_mean",
            "sample_std",
            "posterior_mean",
            "posterior_std",
            "credible_interval_90",
            "credible_interval_95",
            "prob_positive_return",
            "skewness",
            "kurtosis",
            "var_5",
            "cvar_5",
        ]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_resample_returns_frequency_stored(self):
        result = self.resampler.resample_returns(self.df, freq="1QE")
        self.assertTrue((result["frequency"] == "1QE").all())

    def test_posterior_shrinks_toward_prior(self):
        """Posterior mean should be between sample mean and prior mean."""
        result = self.resampler.resample_returns(self.df)
        for _, row in result.iterrows():
            pm = row["posterior_mean"]
            sm = row["sample_mean"]
            prior = self.resampler.prior_return_mean
            low, high = sorted([sm, prior])
            self.assertGreaterEqual(pm, low - 0.5, "Posterior too far from range")
            self.assertLessEqual(pm, high + 0.5, "Posterior too far from range")

    def test_credible_intervals_ordered(self):
        result = self.resampler.resample_returns(self.df)
        for _, row in result.iterrows():
            ci90 = row["credible_interval_90"]
            ci95 = row["credible_interval_95"]
            self.assertLess(ci90[0], ci90[1])
            self.assertLess(ci95[0], ci95[1])
            # 95% CI should be wider than 90%
            self.assertLessEqual(ci95[0], ci90[0])
            self.assertGreaterEqual(ci95[1], ci90[1])

    def test_prob_positive_return_range(self):
        result = self.resampler.resample_returns(self.df)
        for p in result["prob_positive_return"]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_var_cvar_relationship(self):
        """CVaR should be <= VaR (both are losses at 5th percentile)."""
        result = self.resampler.resample_returns(self.df)
        for _, row in result.iterrows():
            self.assertLessEqual(row["cvar_5"], row["var_5"] + 1e-10)

    def test_enrichment_with_technical_features(self):
        result = self.resampler.resample_returns(self.df)
        # Should have merged momentum features
        self.assertIn("price_momentum_1m", result.columns)

    def test_empty_dataframe_returns_empty(self):
        empty_df = pd.DataFrame(columns=self.df.columns)
        result = self.resampler.resample_returns(empty_df)
        self.assertTrue(result.empty)

    def test_single_ticker_insufficient_data(self):
        """A ticker with only one period should be skipped (need >= 2)."""
        df = pd.DataFrame(
            {
                "ticker": ["SOLO"],
                "last_price": [100.0],
                "price_1m_ago": [95.0],
                "sector": ["Tech"],
            }
        )
        result = self.resampler.resample_returns(df)
        # Only 1 period → skipped
        self.assertTrue(result.empty)

    def test_sql_style_column_names(self):
        """Columns with SQL-style names (e.g. 'Price (1M Ago)') should work."""
        rng = np.random.default_rng(99)
        n = 5
        last_prices = rng.uniform(50, 200, n)
        df = pd.DataFrame(
            {
                "ticker": [f"SQL{i}" for i in range(n)],
                "Last Price": last_prices,
                "Price (1M Ago)": last_prices * rng.uniform(0.90, 1.10, n),
                "Price (3M Ago)": last_prices * rng.uniform(0.80, 1.20, n),
                "Price (1Y Ago)": last_prices * rng.uniform(0.60, 1.40, n),
                "sector": rng.choice(["Tech", "Finance"], n),
            }
        )
        result = self.resampler.resample_returns(df)
        self.assertFalse(result.empty, "Should find returns from SQL-style columns")
        self.assertIn("ticker", result.columns)
        self.assertIn("posterior_mean", result.columns)

    def test_compute_historical_returns_sql_columns(self):
        """_compute_historical_returns should handle SQL-style price columns."""
        rng = np.random.default_rng(101)
        n = 3
        last_prices = rng.uniform(100, 300, n)
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "Last Price": last_prices,
                "Price (6M Ago)": last_prices * rng.uniform(0.70, 1.30, n),
                "Price (3Y Ago)": last_prices * rng.uniform(0.40, 1.60, n),
            }
        )
        returns_df = self.resampler._compute_historical_returns(df)
        self.assertFalse(returns_df.empty)
        periods = set(returns_df["period"].unique())
        self.assertTrue(periods & {"6M", "3Y"}, "Should contain 6M and/or 3Y periods")


class TestBayesianTechnicalResamplerInferenceData(unittest.TestCase):
    """Tests for build_inference_data on BayesianTechnicalResampler."""

    def setUp(self):
        from finance_ml.analytics.statistical_analysis import BayesianTechnicalResampler

        self.resampler = BayesianTechnicalResampler(n_posterior_samples=100, n_chains=2, random_seed=42)
        self.df = _make_equities_df(n=5)

    def test_build_inference_data_not_none(self):
        idata = self.resampler.build_inference_data(self.df)
        self.assertIsNotNone(idata)

    def test_inference_data_has_posterior(self):
        import arviz as az

        idata = self.resampler.build_inference_data(self.df)
        self.assertIsInstance(idata, az.InferenceData)
        self.assertTrue(hasattr(idata, "posterior"))
        self.assertIn("expected_return", idata.posterior.data_vars)

    def test_inference_data_shape(self):
        idata = self.resampler.build_inference_data(self.df)
        post = idata.posterior["expected_return"]
        self.assertEqual(post.sizes["chain"], 2)
        self.assertEqual(post.sizes["draw"], 100)
        self.assertGreater(post.sizes["equity"], 0)

    def test_inference_data_has_posterior_predictive(self):
        idata = self.resampler.build_inference_data(self.df)
        self.assertTrue(hasattr(idata, "posterior_predictive"))
        self.assertIn("future_return", idata.posterior_predictive.data_vars)

    def test_inference_data_has_observed_data(self):
        idata = self.resampler.build_inference_data(self.df)
        self.assertTrue(hasattr(idata, "observed_data"))
        self.assertIn("observed_return", idata.observed_data.data_vars)

    def test_inference_data_has_log_likelihood(self):
        idata = self.resampler.build_inference_data(self.df)
        self.assertTrue(hasattr(idata, "log_likelihood"))

    def test_inference_data_has_constant_data(self):
        idata = self.resampler.build_inference_data(self.df)
        self.assertTrue(hasattr(idata, "constant_data"))

    def test_build_inference_data_empty_returns_none(self):
        empty_df = pd.DataFrame(columns=["ticker", "last_price"])
        idata = self.resampler.build_inference_data(empty_df)
        self.assertIsNone(idata)


# =============================================================================
# ResampledReturnDistribution dataclass Tests
# =============================================================================


class TestResampledReturnDistribution(unittest.TestCase):
    def test_dataclass_fields(self):
        from finance_ml.analytics.statistical_analysis import ResampledReturnDistribution

        field_names = {f.name for f in fields(ResampledReturnDistribution)}
        expected = {
            "ticker",
            "frequency",
            "n_periods",
            "sample_mean",
            "sample_std",
            "posterior_mean",
            "posterior_std",
            "credible_interval_90",
            "credible_interval_95",
            "prob_positive_return",
            "skewness",
            "kurtosis",
            "var_5",
            "cvar_5",
        }
        self.assertEqual(field_names, expected)


# =============================================================================
# resampled_posterior_returns convenience function Tests
# =============================================================================


class TestResampledPosteriorReturns(unittest.TestCase):
    def test_returns_tuple(self):
        from finance_ml.analytics.statistical_analysis import resampled_posterior_returns

        df = _make_equities_df(n=5)
        result_df, idata = resampled_posterior_returns(df, freq="1ME", n_posterior_samples=100, n_chains=2)
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertFalse(result_df.empty)
        self.assertIsNotNone(idata)

    def test_empty_input(self):
        from finance_ml.analytics.statistical_analysis import resampled_posterior_returns

        empty_df = pd.DataFrame(columns=["ticker", "last_price"])
        result_df, idata = resampled_posterior_returns(empty_df)
        self.assertTrue(result_df.empty)
        self.assertIsNone(idata)


# =============================================================================
# ResampledBeatProbabilityModel Tests
# =============================================================================


class TestResampledBeatProbabilityModel(unittest.TestCase):
    """Tests for ResampledBeatProbabilityModel in probability_analytics.py."""

    def setUp(self):
        from finance_ml.analytics.probability_analytics import (
            EarningsBeatProbabilityModel,
            ResampledBeatProbabilityModel,
        )

        self.base_model = EarningsBeatProbabilityModel()
        self.model = ResampledBeatProbabilityModel(
            base_model=self.base_model,
            momentum_weight=0.3,
            volatility_weight=0.2,
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )
        self.df = _make_equities_df(n=8)

    def test_init_clips_weights(self):
        from finance_ml.analytics.probability_analytics import ResampledBeatProbabilityModel

        m = ResampledBeatProbabilityModel(momentum_weight=1.5, volatility_weight=-0.5)
        self.assertEqual(m.momentum_weight, 1.0)
        self.assertEqual(m.volatility_weight, 0.0)

    def test_compute_momentum_signal_range(self):
        row = pd.Series({"price_momentum_1m": 50.0, "price_momentum_3m": -30.0})
        signal = self.model._compute_momentum_signal(row)
        self.assertGreaterEqual(signal, -1.0)
        self.assertLessEqual(signal, 1.0)

    def test_compute_momentum_signal_no_data(self):
        row = pd.Series(dtype=float)
        signal = self.model._compute_momentum_signal(row)
        self.assertEqual(signal, 0.0)

    def test_compute_volatility_regime_range(self):
        row = pd.Series({"volatility_compression": 0.7})
        score = self.model._compute_volatility_regime(row)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compute_volatility_regime_default(self):
        row = pd.Series(dtype=float)
        score = self.model._compute_volatility_regime(row)
        self.assertEqual(score, 0.5)

    def test_adjust_prior_positive_momentum(self):
        """Positive momentum should increase alpha relative to beta."""
        a, b = self.model._adjust_prior(5.0, 5.0, momentum_signal=1.0, vol_regime=0.5)
        self.assertGreater(a, 5.0)
        self.assertLess(b, 5.0)

    def test_adjust_prior_negative_momentum(self):
        a, b = self.model._adjust_prior(5.0, 5.0, momentum_signal=-1.0, vol_regime=0.5)
        self.assertLess(a, 5.0)
        self.assertGreater(b, 5.0)

    def test_adjust_prior_floor(self):
        """Alpha and beta should never go below 0.5."""
        a, b = self.model._adjust_prior(0.5, 0.5, momentum_signal=-1.0, vol_regime=0.0)
        self.assertGreaterEqual(a, 0.5)
        self.assertGreaterEqual(b, 0.5)

    def test_analyze_returns_dataframe(self):
        result = self.model.analyze_dataframe(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            expected_cols = [
                "ticker",
                "base_posterior_mean",
                "resampled_posterior_mean",
                "technical_adjustment",
                "momentum_signal",
                "credible_interval_90",
                "credible_interval_95",
                "prob_beat_given_momentum",
            ]
            for col in expected_cols:
                self.assertIn(col, result.columns)

    def test_analyze_posterior_means_in_range(self):
        result = self.model.analyze_dataframe(self.df)
        if not result.empty:
            for _, row in result.iterrows():
                self.assertGreaterEqual(row["resampled_posterior_mean"], 0.0)
                self.assertLessEqual(row["resampled_posterior_mean"], 1.0)
                self.assertGreaterEqual(row["base_posterior_mean"], 0.0)
                self.assertLessEqual(row["base_posterior_mean"], 1.0)

    def test_technical_adjustment_sign(self):
        """Technical adjustment should equal resampled - base."""
        result = self.model.analyze_dataframe(self.df)
        if not result.empty:
            for _, row in result.iterrows():
                expected = row["resampled_posterior_mean"] - row["base_posterior_mean"]
                self.assertAlmostEqual(row["technical_adjustment"], expected, places=10)

    def test_build_inference_data(self):
        idata = self.model.build_inference_data(self.df)
        if idata is not None:
            import arviz as az

            self.assertIsInstance(idata, az.InferenceData)
            self.assertTrue(hasattr(idata, "posterior"))
            self.assertIn("beat_probability", idata.posterior.data_vars)
            # Beat probability samples should be in [0, 1]
            samples = idata.posterior["beat_probability"].values
            self.assertTrue(np.all(samples >= 0))
            self.assertTrue(np.all(samples <= 1))

    def test_build_inference_data_has_predictive(self):
        idata = self.model.build_inference_data(self.df)
        if idata is not None:
            self.assertTrue(hasattr(idata, "posterior_predictive"))
            pp = idata.posterior_predictive["beat_outcome"].values
            # Bernoulli: 0 or 1
            self.assertTrue(np.all(np.isin(pp, [0, 1])))


# =============================================================================
# ResampledBeatEstimate dataclass Tests
# =============================================================================


class TestResampledBeatEstimate(unittest.TestCase):
    def test_dataclass_fields(self):
        from finance_ml.analytics.probability_analytics import ResampledBeatEstimate

        field_names = {f.name for f in fields(ResampledBeatEstimate)}
        self.assertIn("ticker", field_names)
        self.assertIn("resampled_posterior_mean", field_names)
        self.assertIn("technical_adjustment", field_names)
        self.assertIn("earnings_season_flag", field_names)


# =============================================================================
# build_resampled_technical_inference_data Tests
# =============================================================================


class TestBuildResampledTechnicalInferenceData(unittest.TestCase):
    def test_factory_returns_inference_data(self):
        from finance_ml.analytics.inference_schema import (
            build_resampled_technical_inference_data,
        )

        df = _make_equities_df(n=5)
        idata = build_resampled_technical_inference_data(
            df,
            freq="1ME",
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )
        self.assertIsNotNone(idata)

    def test_factory_empty_input(self):
        from finance_ml.analytics.inference_schema import (
            build_resampled_technical_inference_data,
        )

        empty_df = pd.DataFrame(columns=["ticker", "last_price"])
        idata = build_resampled_technical_inference_data(empty_df)
        self.assertIsNone(idata)

    def test_factory_consistent_with_direct(self):
        from finance_ml.analytics.inference_schema import (
            build_resampled_technical_inference_data,
            summarize_inference_data,
        )

        df = _make_equities_df(n=5)
        idata = build_resampled_technical_inference_data(
            df,
            freq="1ME",
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )
        summary = summarize_inference_data(idata)
        self.assertIn("posterior", summary.get("groups", []))
        self.assertEqual(summary["n_chains"], 2)
        self.assertEqual(summary["n_draws"], 100)


# =============================================================================
# __init__.py export Tests
# =============================================================================


class TestExports(unittest.TestCase):
    def test_statistical_analysis_exports(self):
        from finance_ml.analytics import (
            BayesianTechnicalResampler,
            ResampledReturnDistribution,
            resampled_posterior_returns,
        )

        self.assertIsNotNone(BayesianTechnicalResampler)
        self.assertIsNotNone(ResampledReturnDistribution)
        self.assertIsNotNone(resampled_posterior_returns)

    def test_probability_analytics_exports(self):
        from finance_ml.analytics import (
            ResampledBeatProbabilityModel,
            ResampledBeatEstimate,
        )

        self.assertIsNotNone(ResampledBeatProbabilityModel)
        self.assertIsNotNone(ResampledBeatEstimate)

    def test_inference_schema_exports(self):
        from finance_ml.analytics import build_resampled_technical_inference_data

        self.assertIsNotNone(build_resampled_technical_inference_data)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationFullWorkflow(unittest.TestCase):
    """End-to-end integration test for the full resampling workflow."""

    def test_full_resampled_returns_workflow(self):
        from finance_ml.analytics.statistical_analysis import (
            BayesianTechnicalResampler,
            resampled_posterior_returns,
        )
        from finance_ml.analytics.inference_schema import summarize_inference_data

        df = _make_equities_df(n=8)
        result_df, idata = resampled_posterior_returns(df, freq="1ME", n_posterior_samples=100, n_chains=2)

        # Verify result_df
        self.assertFalse(result_df.empty)
        self.assertIn("posterior_mean", result_df.columns)
        self.assertIn("prob_positive_return", result_df.columns)

        # Verify idata
        self.assertIsNotNone(idata)
        summary = summarize_inference_data(idata)
        self.assertIn("expected_return", summary.get("variables", []))

    def test_full_beat_probability_workflow(self):
        from finance_ml.analytics.probability_analytics import (
            EarningsBeatProbabilityModel,
            ResampledBeatProbabilityModel,
        )

        df = _make_equities_df(n=8)
        base = EarningsBeatProbabilityModel()
        model = ResampledBeatProbabilityModel(
            base_model=base,
            n_posterior_samples=100,
            n_chains=2,
            random_seed=42,
        )

        beat_df = model.analyze_dataframe(df)
        if not beat_df.empty:
            self.assertIn("resampled_posterior_mean", beat_df.columns)
            self.assertIn("technical_adjustment", beat_df.columns)

            idata = model.build_inference_data(df)
            if idata is not None:
                import arviz as az

                self.assertIsInstance(idata, az.InferenceData)


if __name__ == "__main__":
    unittest.main()
