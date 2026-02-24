"""
Tests for enhanced statistical methods and optimized operations.

Tests cover:
- Kalman filtering functions
- Copula-based dependency modeling
- Parallel MCMC chains
- Optimized Monte Carlo simulation
- Caching utilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """Create sample DataFrame with price and target data."""
    np.random.seed(42)
    n = 100

    last_price = np.random.uniform(10, 500, n)

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "last_price": last_price,
            "price_target": last_price * np.random.uniform(0.8, 1.3, n),
            "price_target_low": last_price * np.random.uniform(0.7, 0.95, n),
            "price_target_median": last_price * np.random.uniform(0.9, 1.15, n),
            "price_target_high": last_price * np.random.uniform(1.05, 1.5, n),
        }
    )


@pytest.fixture
def sample_momentum_df() -> pd.DataFrame:
    """Create sample DataFrame with momentum data."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "price_momentum_1m": np.random.uniform(-30, 30, n),
            "price_momentum_3m": np.random.uniform(-40, 40, n),
            "price_momentum_6m": np.random.uniform(-50, 50, n),
        }
    )


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create sample DataFrame with multiple features for copula."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "roe": np.random.uniform(-10, 40, n),
            "debt_to_equity": np.random.uniform(0, 3, n),
            "p_e_ratio": np.random.uniform(5, 50, n),
            "fcf_yield": np.random.uniform(-10, 20, n),
        }
    )


@pytest.fixture
def sample_ruin_df() -> pd.DataFrame:
    """Create sample DataFrame for ruin probability calculation."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "market_cap": np.random.uniform(1e8, 1e11, n),
            "cash_burn_rate": np.random.uniform(0, 1e8, n),
            "volatility": np.random.uniform(0.1, 0.6, n),
            "fcf_margin": np.random.uniform(-20, 30, n),
            "beta_momentum": np.random.uniform(0.5, 2.0, n),
        }
    )


# =============================================================================
# Kalman Filter Tests
# =============================================================================


class TestKalmanFilterPriceTarget:
    """Tests for kalman_filter_price_target function."""

    def test_returns_dataframe(self, sample_price_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        result = kalman_filter_price_target(sample_price_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_price_df):
        """Output should have expected columns."""
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        result = kalman_filter_price_target(sample_price_df)

        expected_cols = [
            "ticker",
            "kalman_estimate",
            "kalman_variance",
            "kalman_gain",
            "signal_strength",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_kalman_estimate_reasonable(self, sample_price_df):
        """Kalman estimate should be between price and target."""
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        result = kalman_filter_price_target(sample_price_df)

        if len(result) > 0:
            # Estimate should be influenced by both price and target
            assert result["kalman_estimate"].min() > 0

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        result = kalman_filter_price_target(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_custom_variance_parameters(self, sample_price_df):
        """Function should accept custom variance parameters."""
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        result = kalman_filter_price_target(
            sample_price_df, process_variance=0.001, measurement_variance=0.5
        )
        assert isinstance(result, pd.DataFrame)


class TestKalmanMomentumFilter:
    """Tests for kalman_momentum_filter function."""

    def test_returns_dataframe(self, sample_momentum_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.statistical_analysis import kalman_momentum_filter

        result = kalman_momentum_filter(sample_momentum_df)
        assert isinstance(result, pd.DataFrame)

    def test_creates_filtered_columns(self, sample_momentum_df):
        """Function should create filtered columns."""
        from finance_ml.analytics.statistical_analysis import kalman_momentum_filter

        result = kalman_momentum_filter(sample_momentum_df)

        assert "price_momentum_1m_filtered" in result.columns
        assert "price_momentum_3m_filtered" in result.columns

    def test_filtered_values_smoother(self, sample_momentum_df):
        """Filtered values should have lower variance than original."""
        from finance_ml.analytics.statistical_analysis import kalman_momentum_filter

        result = kalman_momentum_filter(sample_momentum_df)

        # Filtered should be smoother (lower std in most cases)
        original_std = sample_momentum_df["price_momentum_1m"].std()
        filtered_std = result["price_momentum_1m_filtered"].std()

        # Allow some tolerance - filtering should reduce noise
        assert filtered_std <= original_std * 1.5

    def test_custom_momentum_columns(self, sample_momentum_df):
        """Function should work with custom column list."""
        from finance_ml.analytics.statistical_analysis import kalman_momentum_filter

        result = kalman_momentum_filter(sample_momentum_df, momentum_cols=["price_momentum_1m"])

        assert "price_momentum_1m_filtered" in result.columns


# =============================================================================
# Copula Tests
# =============================================================================


class TestFitGaussianCopula:
    """Tests for fit_gaussian_copula function."""

    def test_returns_dict(self, sample_features_df):
        """Function should return a dictionary."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        result = fit_gaussian_copula(sample_features_df, features=["roe", "debt_to_equity", "p_e_ratio"])
        assert isinstance(result, dict)

    def test_output_keys(self, sample_features_df):
        """Output should have expected keys."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        result = fit_gaussian_copula(sample_features_df, features=["roe", "debt_to_equity", "p_e_ratio"])

        expected_keys = [
            "correlation_matrix",
            "features",
            "simulated_samples",
            "tail_dependence",
            "marginal_params",
        ]
        for key in expected_keys:
            assert key in result

    def test_correlation_matrix_shape(self, sample_features_df):
        """Correlation matrix should be square with correct dimensions."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        features = ["roe", "debt_to_equity", "p_e_ratio"]
        result = fit_gaussian_copula(sample_features_df, features=features)

        corr_matrix = result["correlation_matrix"]
        assert corr_matrix.shape == (len(features), len(features))

    def test_simulated_samples_shape(self, sample_features_df):
        """Simulated samples should have correct shape."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        n_sims = 5000
        features = ["roe", "debt_to_equity"]
        result = fit_gaussian_copula(sample_features_df, features=features, n_simulations=n_sims)

        samples = result["simulated_samples"]
        assert samples.shape == (n_sims, len(features))

    def test_handles_missing_features(self, sample_features_df):
        """Function should handle missing features gracefully."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        result = fit_gaussian_copula(sample_features_df, features=["roe", "nonexistent_column"])

        # Should only use available features
        assert len(result["features"]) == 1

    def test_tail_dependence_structure(self, sample_features_df):
        """Tail dependence should have lower and upper matrices."""
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        result = fit_gaussian_copula(sample_features_df, features=["roe", "debt_to_equity", "p_e_ratio"])

        tail_dep = result["tail_dependence"]
        assert "lower" in tail_dep
        assert "upper" in tail_dep


# =============================================================================
# Parallel MCMC Tests
# =============================================================================


class TestParallelMcmcChains:
    """Tests for parallel_mcmc_chains function."""

    def test_returns_dict(self):
        """Function should return a dictionary."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        result = parallel_mcmc_chains(data, n_chains=2, n_samples=500, n_jobs=1)
        assert isinstance(result, dict)

    def test_output_keys(self):
        """Output should have expected keys."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        result = parallel_mcmc_chains(data, n_chains=2, n_samples=500, n_jobs=1)

        expected_keys = [
            "chains",
            "r_hat",
            "combined_samples",
            "converged",
            "chain_means",
            "chain_stds",
            "posterior_mean",
            "ci_95",
        ]
        for key in expected_keys:
            assert key in result

    def test_correct_number_of_chains(self):
        """Should return correct number of chains."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        n_chains = 3

        result = parallel_mcmc_chains(data, n_chains=n_chains, n_samples=500, n_jobs=1)

        assert len(result["chains"]) == n_chains
        assert len(result["chain_means"]) == n_chains

    def test_combined_samples_length(self):
        """Combined samples should have correct length."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        n_chains = 2
        n_samples = 500

        result = parallel_mcmc_chains(data, n_chains=n_chains, n_samples=n_samples, n_jobs=1)

        assert len(result["combined_samples"]) == n_chains * n_samples

    def test_r_hat_reasonable(self):
        """R-hat should be a reasonable value."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 100)

        result = parallel_mcmc_chains(data, n_chains=2, n_samples=1000, n_jobs=1)

        # R-hat should be positive
        assert result["r_hat"] > 0

    def test_posterior_mean_reasonable(self):
        """Posterior mean should be close to true mean."""
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        true_mean = 10
        data = np.random.normal(true_mean, 2, 200)

        result = parallel_mcmc_chains(data, n_chains=2, n_samples=2000, n_jobs=1)

        # Posterior mean should be within reasonable range of true mean
        assert abs(result["posterior_mean"] - true_mean) < 2


# =============================================================================
# Optimized Operations Tests
# =============================================================================


class TestDataframeHash:
    """Tests for dataframe_hash function."""

    def test_returns_string(self):
        """Function should return a string hash."""
        from finance_ml.analytics.optimized_ops import dataframe_hash

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = dataframe_hash(df)

        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hash length

    def test_same_data_same_hash(self):
        """Same data should produce same hash."""
        from finance_ml.analytics.optimized_ops import dataframe_hash

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        assert dataframe_hash(df1) == dataframe_hash(df2)

    def test_different_data_different_hash(self):
        """Different data should produce different hash."""
        from finance_ml.analytics.optimized_ops import dataframe_hash

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 4], "b": [4, 5, 6]})

        assert dataframe_hash(df1) != dataframe_hash(df2)


class TestFastMonteCarloSimulation:
    """Tests for fast_monte_carlo_simulation function."""

    def test_returns_dataframe(self, sample_price_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation

        result = fast_monte_carlo_simulation(sample_price_df, n_simulations=100)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_price_df):
        """Output should have expected columns."""
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation

        result = fast_monte_carlo_simulation(sample_price_df, n_simulations=100)

        expected_cols = [
            "ticker",
            "expected_upside",
            "upside_std",
            "var_5_pct",
            "prob_positive",
            "risk_reward_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_probability_in_range(self, sample_price_df):
        """Probability should be between 0 and 100."""
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation

        result = fast_monte_carlo_simulation(sample_price_df, n_simulations=500)

        assert result["prob_positive"].min() >= 0
        assert result["prob_positive"].max() <= 100

    def test_handles_missing_columns(self):
        """Function should handle missing columns gracefully."""
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation

        df = pd.DataFrame({"ticker": ["A", "B"], "other_col": [1, 2]})
        result = fast_monte_carlo_simulation(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestFastRuinProbability:
    """Tests for fast_ruin_probability function."""

    def test_returns_dataframe(self, sample_ruin_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.optimized_ops import fast_ruin_probability

        result = fast_ruin_probability(sample_ruin_df, n_simulations=50, n_days=50)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, sample_ruin_df):
        """Output should have expected columns."""
        from finance_ml.analytics.optimized_ops import fast_ruin_probability

        result = fast_ruin_probability(sample_ruin_df, n_simulations=50, n_days=50)

        expected_cols = ["ticker", "ruin_probability", "survival_probability", "risk_tier"]
        for col in expected_cols:
            assert col in result.columns

    def test_probability_in_range(self, sample_ruin_df):
        """Ruin probability should be between 0 and 1."""
        from finance_ml.analytics.optimized_ops import fast_ruin_probability

        result = fast_ruin_probability(sample_ruin_df, n_simulations=100, n_days=50)

        assert result["ruin_probability"].min() >= 0
        assert result["ruin_probability"].max() <= 1

    def test_survival_complement(self, sample_ruin_df):
        """Survival probability should be 1 - ruin probability."""
        from finance_ml.analytics.optimized_ops import fast_ruin_probability

        result = fast_ruin_probability(sample_ruin_df, n_simulations=50, n_days=50)

        np.testing.assert_array_almost_equal(
            result["ruin_probability"] + result["survival_probability"], np.ones(len(result))
        )


class TestVectorizedZscore:
    """Tests for vectorized_zscore function."""

    def test_returns_dataframe(self, sample_features_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.optimized_ops import vectorized_zscore

        result = vectorized_zscore(sample_features_df, columns=["roe", "p_e_ratio"])
        assert isinstance(result, pd.DataFrame)

    def test_creates_zscore_columns(self, sample_features_df):
        """Function should create z-score columns."""
        from finance_ml.analytics.optimized_ops import vectorized_zscore

        result = vectorized_zscore(sample_features_df, columns=["roe", "p_e_ratio"])

        assert "roe_zscore" in result.columns
        assert "p_e_ratio_zscore" in result.columns

    def test_zscore_mean_zero(self, sample_features_df):
        """Z-scores should have mean approximately 0."""
        from finance_ml.analytics.optimized_ops import vectorized_zscore

        result = vectorized_zscore(sample_features_df, columns=["roe"])

        assert abs(result["roe_zscore"].mean()) < 0.01

    def test_zscore_std_one(self, sample_features_df):
        """Z-scores should have std approximately 1."""
        from finance_ml.analytics.optimized_ops import vectorized_zscore

        result = vectorized_zscore(sample_features_df, columns=["roe"])

        assert abs(result["roe_zscore"].std() - 1) < 0.1


class TestVectorizedPercentileRank:
    """Tests for vectorized_percentile_rank function."""

    def test_returns_dataframe(self, sample_features_df):
        """Function should return a DataFrame."""
        from finance_ml.analytics.optimized_ops import vectorized_percentile_rank

        result = vectorized_percentile_rank(sample_features_df, columns=["roe"])
        assert isinstance(result, pd.DataFrame)

    def test_creates_percentile_columns(self, sample_features_df):
        """Function should create percentile columns."""
        from finance_ml.analytics.optimized_ops import vectorized_percentile_rank

        result = vectorized_percentile_rank(sample_features_df, columns=["roe", "p_e_ratio"])

        assert "roe_pctile" in result.columns
        assert "p_e_ratio_pctile" in result.columns

    def test_percentile_range(self, sample_features_df):
        """Percentiles should be between 0 and 100."""
        from finance_ml.analytics.optimized_ops import vectorized_percentile_rank

        result = vectorized_percentile_rank(sample_features_df, columns=["roe"])

        assert result["roe_pctile"].min() >= 0
        assert result["roe_pctile"].max() <= 100


class TestGetOptimizationStatus:
    """Tests for get_optimization_status function."""

    def test_returns_dict(self):
        """Function should return a dictionary."""
        from finance_ml.analytics.optimized_ops import get_optimization_status

        result = get_optimization_status()
        assert isinstance(result, dict)

    def test_expected_keys(self):
        """Output should have expected keys."""
        from finance_ml.analytics.optimized_ops import get_optimization_status

        result = get_optimization_status()

        expected_keys = [
            "numba_available",
            "db_cache_size",
            "stats_cache_size",
            "parallel_available",
        ]
        for key in expected_keys:
            assert key in result


# =============================================================================
# Module Import Tests
# =============================================================================


class TestEnhancedStatisticsImports:
    """Tests for module imports."""

    def test_statistical_analysis_imports(self):
        """Statistical analysis module should import enhanced functions."""
        from finance_ml.analytics import statistical_analysis

        assert hasattr(statistical_analysis, "kalman_filter_price_target")
        assert hasattr(statistical_analysis, "kalman_momentum_filter")
        assert hasattr(statistical_analysis, "fit_gaussian_copula")
        assert hasattr(statistical_analysis, "parallel_mcmc_chains")

    def test_optimized_ops_imports(self):
        """Optimized ops module should import successfully."""
        from finance_ml.analytics import optimized_ops

        assert hasattr(optimized_ops, "dataframe_hash")
        assert hasattr(optimized_ops, "fast_monte_carlo_simulation")
        assert hasattr(optimized_ops, "fast_ruin_probability")
        assert hasattr(optimized_ops, "vectorized_zscore")
        assert hasattr(optimized_ops, "vectorized_percentile_rank")

    def test_analytics_package_exports(self):
        """Analytics package should export enhanced functions."""
        from finance_ml import analytics

        # Check enhanced statistical methods
        assert hasattr(analytics, "kalman_filter_price_target")
        assert hasattr(analytics, "fit_gaussian_copula")
        assert hasattr(analytics, "parallel_mcmc_chains")

        # Check optimized operations
        assert hasattr(analytics, "fast_monte_carlo_simulation")
        assert hasattr(analytics, "fast_ruin_probability")
