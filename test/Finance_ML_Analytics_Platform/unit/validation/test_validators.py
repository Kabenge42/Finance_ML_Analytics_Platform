"""
Unit tests for validation/validators.py module.

Phase 8: Create missing documented modules.
TDD approach: Tests written first, implementation follows.

Run with pytest:
    pytest tests/unit/validation/test_validators.py -v

Run with unittest:
    python -m unittest tests.unit.validation.test_validators -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestValidateDataQuality:
    """Tests for validate_data_quality function."""

    def test_validate_data_quality_clean_data(self):
        """Test that clean data passes validation."""
        from finance_ml.ml_workflow.validation.validators import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        result = validate_data_quality(df)

        assert result["is_valid"] is True
        assert result["has_nulls"] is False
        assert result["has_inf"] is False
        assert result["has_extreme_values"] is False
        assert len(result["issues"]) == 0

    def test_validate_data_quality_with_nulls(self):
        """Test that NaN values are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, 3.0],
                "col2": [10.0, 20.0, np.nan],
            }
        )
        result = validate_data_quality(df)

        assert result["is_valid"] is False
        assert result["has_nulls"] is True
        assert len(result["issues"]) >= 2  # At least 2 columns with NaN

    def test_validate_data_quality_with_infinity(self):
        """Test that infinite values are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1.0, np.inf, 3.0],
                "col2": [10.0, -np.inf, 30.0],
            }
        )
        result = validate_data_quality(df)

        assert result["is_valid"] is False
        assert result["has_inf"] is True

    def test_validate_data_quality_with_extreme_values(self):
        """Test that extremely large values are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1.0, 1e15, 3.0],
                "col2": [10.0, 20.0, 30.0],
            }
        )
        result = validate_data_quality(df)

        assert result["is_valid"] is False
        assert result["has_extreme_values"] is True

    def test_validate_data_quality_specific_columns(self):
        """Test validation of specific columns only."""
        from finance_ml.ml_workflow.validation.validators import validate_data_quality

        df = pd.DataFrame(
            {
                "clean_col": [1.0, 2.0, 3.0],
                "dirty_col": [1.0, np.nan, 3.0],
            }
        )
        # Only validate clean column
        result = validate_data_quality(df, columns=["clean_col"])

        assert result["is_valid"] is True
        assert result["has_nulls"] is False


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_schema_all_columns_present(self):
        """Test that schema validation passes when all required columns present."""
        from finance_ml.ml_workflow.validation.validators import validate_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "last_price": [150.0, 300.0],
            }
        )
        required = ["ticker", "sector", "last_price"]
        result = validate_schema(df, required_columns=required)

        assert result["is_valid"] is True
        assert len(result["missing_columns"]) == 0

    def test_validate_schema_missing_columns(self):
        """Test that missing columns are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
            }
        )
        required = ["ticker", "sector", "last_price", "market_cap"]
        result = validate_schema(df, required_columns=required)

        assert result["is_valid"] is False
        assert "last_price" in result["missing_columns"]
        assert "market_cap" in result["missing_columns"]

    def test_validate_schema_with_dtypes(self):
        """Test that dtype validation works."""
        from finance_ml.ml_workflow.validation.validators import validate_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "price": [150.0, 300.0],
            }
        )
        expected_dtypes = {"ticker": "object", "price": "float64"}
        result = validate_schema(df, expected_dtypes=expected_dtypes)

        assert result["is_valid"] is True


class TestValidateNumericRange:
    """Tests for validate_numeric_range function."""

    def test_validate_numeric_range_within_bounds(self):
        """Test that values within range pass validation."""
        from finance_ml.ml_workflow.validation.validators import validate_numeric_range

        df = pd.DataFrame(
            {
                "price": [10.0, 50.0, 100.0],
                "volume": [1000, 5000, 10000],
            }
        )
        result = validate_numeric_range(df, column="price", min_val=0, max_val=200)

        assert result["is_valid"] is True
        assert result["out_of_range_count"] == 0

    def test_validate_numeric_range_below_min(self):
        """Test that values below minimum are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_numeric_range

        df = pd.DataFrame(
            {
                "price": [-5.0, 10.0, 50.0],
            }
        )
        result = validate_numeric_range(df, column="price", min_val=0)

        assert result["is_valid"] is False
        assert result["out_of_range_count"] >= 1

    def test_validate_numeric_range_above_max(self):
        """Test that values above maximum are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_numeric_range

        df = pd.DataFrame(
            {
                "price": [10.0, 50.0, 1000.0],
            }
        )
        result = validate_numeric_range(df, column="price", max_val=100)

        assert result["is_valid"] is False
        assert result["out_of_range_count"] >= 1


class TestValidatePredictions:
    """Tests for validate_predictions function."""

    def test_validate_predictions_valid(self):
        """Test that valid predictions pass validation."""
        from finance_ml.ml_workflow.validation.validators import validate_predictions

        predictions = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "y_true": [150.0, 300.0, 100.0],
                "y_pred": [155.0, 290.0, 105.0],
                "pred_p10": [140.0, 270.0, 90.0],
                "pred_p50": [155.0, 290.0, 105.0],
                "pred_p90": [170.0, 310.0, 120.0],
            }
        )
        result = validate_predictions(predictions)

        assert result["is_valid"] is True

    def test_validate_predictions_negative_prices(self):
        """Test that negative price predictions are flagged."""
        from finance_ml.ml_workflow.validation.validators import validate_predictions

        predictions = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "y_true": [150.0, 300.0],
                "y_pred": [-10.0, 290.0],  # Negative prediction
            }
        )
        result = validate_predictions(predictions)

        assert result["is_valid"] is False
        assert "negative_predictions" in result["issues"]

    def test_validate_predictions_quantile_crossing(self):
        """Test that quantile crossing is detected (p10 > p50 or p50 > p90)."""
        from finance_ml.ml_workflow.validation.validators import validate_predictions

        predictions = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "y_true": [150.0],
                "y_pred": [155.0],
                "pred_p10": [160.0],  # p10 > p50 (crossing!)
                "pred_p50": [155.0],
                "pred_p90": [170.0],
            }
        )
        result = validate_predictions(predictions)

        assert result["is_valid"] is False
        assert "quantile_crossing" in result["issues"]


class TestValidateFeatures:
    """Tests for validate_features function."""

    def test_validate_features_no_constant_columns(self):
        """Test that constant columns are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_features

        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0],
                "constant": [5.0, 5.0, 5.0, 5.0],  # Constant column
                "feature2": [10.0, 20.0, 30.0, 40.0],
            }
        )
        result = validate_features(df)

        assert "constant" in result["constant_columns"]

    def test_validate_features_high_correlation(self):
        """Test that highly correlated features are detected."""
        from finance_ml.ml_workflow.validation.validators import validate_features

        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfectly correlated
                "feature3": [10.0, 8.0, 6.0, 4.0, 2.0],
            }
        )
        result = validate_features(df, correlation_threshold=0.95)

        assert len(result["high_correlation_pairs"]) >= 1


# Backward compatibility with unittest
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
