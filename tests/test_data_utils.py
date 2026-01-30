"""
Unit tests for finance_ml.analytics.data_utils module.

TDD tests for data loading and preprocessing utilities following strict TDD approach:
1. Write failing tests first (Red)
2. Implement minimal code to pass (Green)
3. Refactor while keeping tests passing (Refactor)

Tests cover:
- load_feature_data_from_db (mock-based testing)
- backfill_feature_columns
- compute_metric_statistics
- validate_feature_alignment
- safe_get_column
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# =============================================================================
# Fixtures for Data Utils Tests
# =============================================================================


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create a sample DataFrame for data utils testing."""
    np.random.seed(42)
    n = 30

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "sector": ["Technology", "Healthcare", "Financials"] * 10,
            # Price columns
            "last_price": np.random.uniform(10, 500, n).round(2),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            # Valuation columns
            "p_e_ratio": np.random.uniform(5, 50, n).round(2),
            "p_b_ratio": np.random.uniform(0.5, 10, n).round(2),
            # Quality metrics
            "piotroski_f_score": np.random.randint(0, 10, n),
            "distress_risk_score": np.random.uniform(10, 95, n).round(1),
            # Earnings columns
            "eps_trajectory_score": np.random.uniform(20, 90, n).round(1),
            "fcf_positive_years": np.random.randint(0, 6, n),
        }
    )

    return df


@pytest.fixture
def sample_df_for_backfill() -> pd.DataFrame:
    """Create a DataFrame with missing columns that need backfilling."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "sector": ["Technology", "Technology", "Technology"],
            # Has tangible_book_value but not tangible_book_value_ltm
            "tangible_book_value": [100.0, 150.0, 200.0],
            # Has goodwill_to_equity but not goodwill_concentration
            "goodwill_to_equity": [0.15, 0.20, 0.10],
            # Has p_e_ratio and p_b_ratio
            "p_e_ratio": [25.0, 30.0, 22.0],
            "p_b_ratio": [5.0, 8.0, 4.0],
        }
    )


@pytest.fixture
def sample_numeric_series() -> pd.Series:
    """Create a sample numeric series for statistics testing."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(-10, 100, 100))


@pytest.fixture
def sample_series_with_nulls() -> pd.Series:
    """Create a series with null values."""
    data = [10.0, 20.0, np.nan, 30.0, np.nan, 40.0, 50.0]
    return pd.Series(data)


@pytest.fixture
def sample_feature_categories() -> dict:
    """Sample feature categories for validation testing."""
    return {
        "Valuation Ratios": ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio"],
        "Quality & Risk": ["piotroski_f_score", "distress_risk_score", "altman_z_score"],
        "Earnings Quality": ["eps_trajectory_score", "eps_surprise_pct"],
    }


# =============================================================================
# Tests for compute_metric_statistics
# =============================================================================


class TestComputeMetricStatistics:
    """Tests for compute_metric_statistics function."""

    def test_returns_dict(self, sample_numeric_series):
        """Function should return a dictionary."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_numeric_series)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_numeric_series):
        """Result should contain all required statistical keys."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_numeric_series)

        required_keys = ["count", "mean", "median", "std", "min", "max", "q25", "q75"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_count_is_correct(self, sample_numeric_series):
        """Count should match non-null values."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_numeric_series)

        assert result["count"] == len(sample_numeric_series.dropna())

    def test_mean_is_correct(self, sample_numeric_series):
        """Mean should be calculated correctly."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_numeric_series)

        expected_mean = sample_numeric_series.mean()
        assert abs(result["mean"] - expected_mean) < 0.001

    def test_handles_series_with_nulls(self, sample_series_with_nulls):
        """Should handle series with null values."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_series_with_nulls)

        assert result is not None
        assert result["count"] == 5  # 7 total - 2 nulls

    def test_returns_none_for_empty_series(self):
        """Should return None for empty series."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(pd.Series([], dtype=float))

        assert result is None

    def test_returns_none_for_all_null_series(self):
        """Should return None when all values are null."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(pd.Series([np.nan, np.nan, np.nan]))

        assert result is None

    def test_positive_pct_calculation(self):
        """Should correctly calculate percentage of positive values."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        # 3 positive, 2 negative = 60% positive
        series = pd.Series([10, 20, 30, -10, -20])
        result = compute_metric_statistics(series)

        assert result["positive_pct"] == 60.0

    def test_missing_pct_calculation(self, sample_series_with_nulls):
        """Should correctly calculate percentage of missing values."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        result = compute_metric_statistics(sample_series_with_nulls)

        # 2 nulls out of 7 = ~28.57%
        expected_missing = (2 / 7) * 100
        assert abs(result["missing_pct"] - expected_missing) < 0.1

    def test_quartiles_correct(self):
        """Quartiles should be calculated correctly."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        # Simple series for easy quartile verification
        series = pd.Series([0, 25, 50, 75, 100])
        result = compute_metric_statistics(series)

        assert result["q25"] == 25.0
        assert result["q75"] == 75.0


# =============================================================================
# Tests for validate_feature_alignment
# =============================================================================


class TestValidateFeatureAlignment:
    """Tests for validate_feature_alignment function."""

    def test_returns_dict(self, sample_feature_df, sample_feature_categories):
        """Function should return a dictionary."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        assert isinstance(result, dict)

    def test_returns_entry_per_category(self, sample_feature_df, sample_feature_categories):
        """Should return an entry for each category."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        for category in sample_feature_categories:
            assert category in result

    def test_contains_coverage_pct(self, sample_feature_df, sample_feature_categories):
        """Each category should have coverage_pct field."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        for category, info in result.items():
            assert "coverage_pct" in info

    def test_coverage_pct_correct(self, sample_feature_df, sample_feature_categories):
        """Coverage percentage should be calculated correctly."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        # Valuation Ratios: p_e_ratio, p_b_ratio exist (2/3 = 66.67%)
        valuation_coverage = result["Valuation Ratios"]["coverage_pct"]
        assert abs(valuation_coverage - 66.67) < 1

    def test_available_count_correct(self, sample_feature_df, sample_feature_categories):
        """Available count should match columns present in DataFrame."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        # Quality & Risk: piotroski_f_score, distress_risk_score exist (2 available)
        assert result["Quality & Risk"]["available_count"] == 2

    def test_missing_count_correct(self, sample_feature_df, sample_feature_categories):
        """Missing count should match columns not present in DataFrame."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        result = validate_feature_alignment(sample_feature_df, sample_feature_categories)

        # Quality & Risk: altman_z_score is missing (1 missing)
        assert result["Quality & Risk"]["missing_count"] == 1

    def test_full_coverage(self):
        """Should report 100% coverage when all features exist."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [4, 5, 6],
            }
        )
        categories = {"Test Category": ["feature_a", "feature_b"]}

        result = validate_feature_alignment(df, categories)

        assert result["Test Category"]["coverage_pct"] == 100.0

    def test_zero_coverage(self):
        """Should report 0% coverage when no features exist."""
        from finance_ml.analytics.data_utils import validate_feature_alignment

        df = pd.DataFrame({"other_col": [1, 2, 3]})
        categories = {"Test Category": ["feature_a", "feature_b"]}

        result = validate_feature_alignment(df, categories)

        assert result["Test Category"]["coverage_pct"] == 0.0


# =============================================================================
# Tests for safe_get_column
# =============================================================================


class TestSafeGetColumn:
    """Tests for safe_get_column function."""

    def test_returns_first_existing_column(self, sample_feature_df):
        """Should return the first column that exists."""
        from finance_ml.analytics.data_utils import safe_get_column

        result = safe_get_column(sample_feature_df, "p_e_ratio", "p_b_ratio")

        assert result is not None
        assert list(result) == list(sample_feature_df["p_e_ratio"])

    def test_returns_second_column_if_first_missing(self, sample_feature_df):
        """Should return second column if first doesn't exist."""
        from finance_ml.analytics.data_utils import safe_get_column

        result = safe_get_column(sample_feature_df, "nonexistent", "p_e_ratio")

        assert result is not None
        assert list(result) == list(sample_feature_df["p_e_ratio"])

    def test_returns_default_if_none_exist(self, sample_feature_df):
        """Should return default value if no columns exist."""
        from finance_ml.analytics.data_utils import safe_get_column

        result = safe_get_column(
            sample_feature_df, "nonexistent1", "nonexistent2", default="default_value"
        )

        assert result == "default_value"

    def test_returns_none_by_default(self, sample_feature_df):
        """Should return None by default if no columns exist."""
        from finance_ml.analytics.data_utils import safe_get_column

        result = safe_get_column(sample_feature_df, "nonexistent1", "nonexistent2")

        assert result is None

    def test_industry_sector_fallback(self, sample_feature_df):
        """Common use case: industry with sector fallback."""
        from finance_ml.analytics.data_utils import safe_get_column

        # sample_feature_df has 'sector' not 'industry'
        result = safe_get_column(sample_feature_df, "industry", "sector")

        assert result is not None
        assert list(result) == list(sample_feature_df["sector"])


# =============================================================================
# Tests for backfill_feature_columns
# =============================================================================


class TestBackfillFeatureColumns:
    """Tests for backfill_feature_columns function."""

    def test_returns_dataframe(self, sample_df_for_backfill):
        """Function should return a DataFrame."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        result = backfill_feature_columns(sample_df_for_backfill)

        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, sample_df_for_backfill):
        """Original columns should be preserved."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        original_cols = set(sample_df_for_backfill.columns)
        result = backfill_feature_columns(sample_df_for_backfill)

        for col in original_cols:
            assert col in result.columns

    def test_backfills_tangible_book_value_ltm(self, sample_df_for_backfill):
        """Should backfill tangible_book_value_ltm from tangible_book_value."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        result = backfill_feature_columns(sample_df_for_backfill)

        assert "tangible_book_value_ltm" in result.columns
        assert list(result["tangible_book_value_ltm"]) == list(
            sample_df_for_backfill["tangible_book_value"]
        )

    def test_backfills_goodwill_concentration(self, sample_df_for_backfill):
        """Should backfill goodwill_concentration from goodwill_to_equity."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        result = backfill_feature_columns(sample_df_for_backfill)

        assert "goodwill_concentration" in result.columns
        assert list(result["goodwill_concentration"]) == list(
            sample_df_for_backfill["goodwill_to_equity"]
        )

    def test_backfills_industry_from_sector(self, sample_df_for_backfill):
        """Should backfill industry from sector if missing."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        result = backfill_feature_columns(sample_df_for_backfill)

        assert "industry" in result.columns
        assert list(result["industry"]) == list(sample_df_for_backfill["sector"])

    def test_does_not_overwrite_existing_columns(self):
        """Should not overwrite columns that already exist."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "sector": ["Technology"],
                "industry": ["Information Technology"],  # Already exists
            }
        )

        result = backfill_feature_columns(df)

        assert result["industry"].iloc[0] == "Information Technology"

    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        from finance_ml.analytics.data_utils import backfill_feature_columns

        df = pd.DataFrame()
        result = backfill_feature_columns(df)

        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Tests for load_feature_data_from_db (Mock-based)
# =============================================================================


class TestLoadFeatureDataFromDb:
    """Tests for load_feature_data_from_db function (using mocks)."""

    def test_raises_error_without_db_url(self):
        """Should raise ValueError when db_url is not provided and env var not set."""
        from finance_ml.analytics.data_utils import load_feature_data_from_db

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="db_url parameter not provided"):
                load_feature_data_from_db(db_url=None)

    def test_uses_env_var_for_db_url(self):
        """Should use DB_URL environment variable when db_url not provided."""
        from finance_ml.analytics.data_utils import load_feature_data_from_db

        mock_df = pd.DataFrame({"ticker": ["AAPL"], "price": [150.0]})

        with patch.dict("os.environ", {"DB_URL": "postgresql://test"}):
            with patch("finance_ml.analytics.data_utils.create_engine") as mock_engine:
                with patch("pandas.read_sql") as mock_read_sql:
                    mock_read_sql.return_value = mock_df

                    result = load_feature_data_from_db()

                    # Verify engine was created with env var URL
                    mock_engine.assert_called_once_with("postgresql://test")

    def test_applies_earnings_date_filter(self):
        """Should apply earnings date filter in SQL query."""
        from finance_ml.analytics.data_utils import load_feature_data_from_db

        mock_df = pd.DataFrame({"ticker": ["AAPL"], "next_earnings": ["2026-02-01"]})

        with patch.dict("os.environ", {"DB_URL": "postgresql://test"}):
            with patch("finance_ml.analytics.data_utils.create_engine"):
                with patch("pandas.read_sql") as mock_read_sql:
                    mock_read_sql.return_value = mock_df

                    load_feature_data_from_db(earnings_date_filter="2026-03-01")

                    # Check that the SQL query contains the date filter
                    call_args = mock_read_sql.call_args
                    sql_query = call_args[0][0]
                    assert "2026-03-01" in sql_query

    def test_applies_limit_parameter(self):
        """Should apply LIMIT clause when limit parameter is provided."""
        from finance_ml.analytics.data_utils import load_feature_data_from_db

        mock_df = pd.DataFrame({"ticker": ["AAPL"], "price": [150.0]})

        with patch.dict("os.environ", {"DB_URL": "postgresql://test"}):
            with patch("finance_ml.analytics.data_utils.create_engine"):
                with patch("pandas.read_sql") as mock_read_sql:
                    mock_read_sql.return_value = mock_df

                    load_feature_data_from_db(limit=100)

                    # Check that LIMIT is in the query
                    call_args = mock_read_sql.call_args
                    sql_query = call_args[0][0]
                    assert "LIMIT 100" in sql_query

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        from finance_ml.analytics.data_utils import load_feature_data_from_db

        mock_df = pd.DataFrame({"ticker": ["AAPL"], "price": [150.0]})

        with patch.dict("os.environ", {"DB_URL": "postgresql://test"}):
            with patch("finance_ml.analytics.data_utils.create_engine"):
                with patch("pandas.read_sql", return_value=mock_df):
                    result = load_feature_data_from_db()

                    assert isinstance(result, pd.DataFrame)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataUtilsIntegration:
    """Integration tests for data_utils module."""

    def test_all_functions_importable(self):
        """All data utils functions should be importable."""
        from finance_ml.analytics.data_utils import (
            load_feature_data_from_db,
            backfill_feature_columns,
            compute_metric_statistics,
            validate_feature_alignment,
            safe_get_column,
        )

        assert callable(load_feature_data_from_db)
        assert callable(backfill_feature_columns)
        assert callable(compute_metric_statistics)
        assert callable(validate_feature_alignment)
        assert callable(safe_get_column)

    def test_backfill_then_validate_workflow(
        self, sample_df_for_backfill, sample_feature_categories
    ):
        """Test backfilling and then validating features."""
        from finance_ml.analytics.data_utils import (
            backfill_feature_columns,
            validate_feature_alignment,
        )

        # Backfill missing columns
        df = backfill_feature_columns(sample_df_for_backfill)

        # Validate coverage
        validation = validate_feature_alignment(df, sample_feature_categories)

        assert isinstance(validation, dict)
        # Should have coverage info for valuation (has p_e_ratio, p_b_ratio)
        assert validation["Valuation Ratios"]["coverage_pct"] >= 60

    def test_compute_statistics_on_multiple_columns(self, sample_feature_df):
        """Test computing statistics on multiple columns."""
        from finance_ml.analytics.data_utils import compute_metric_statistics

        columns_to_analyze = ["p_e_ratio", "piotroski_f_score", "distress_risk_score"]

        stats_results = {}
        for col in columns_to_analyze:
            stats = compute_metric_statistics(sample_feature_df[col])
            stats_results[col] = stats

        # All should have valid statistics
        for col, stats in stats_results.items():
            assert stats is not None
            assert "mean" in stats
            assert "median" in stats
