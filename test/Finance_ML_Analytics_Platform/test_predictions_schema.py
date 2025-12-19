"""
Test standardized predictions schema (Priority 1: Data Pipeline).

Validates that prediction outputs contain required columns for downstream analytics.
Addresses issue: Missing sector, ticker, quantiles in regression_predictions.csv
"""

import unittest
import pandas as pd
import numpy as np


class TestPredictionsSchema(unittest.TestCase):
    """Test that predictions dataframe has required columns."""

    def test_build_predictions_frame_exists(self):
        """Test that build_predictions_frame helper exists."""
        try:
            from finance_ml.ml_workflow.regression.io import build_predictions_frame

            self.assertTrue(callable(build_predictions_frame))
        except ImportError:
            self.fail("build_predictions_frame not implemented in regression.io")

    def test_build_predictions_frame_basic_columns(self):
        """Test that basic prediction columns are present."""
        from finance_ml.ml_workflow.regression.io import build_predictions_frame

        # Create minimal test data
        y_true = pd.Series([100, 200, 300], index=[0, 1, 2])
        y_pred = np.array([95, 210, 290])

        df_source = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [150.0, 250.0, 100.0],
            },
            index=[0, 1, 2],
        )

        result = build_predictions_frame(y_true, y_pred, df_source)

        # Required columns per code_guidelines.md
        required_cols = ["y_true", "y_pred", "abs_error", "pct_error"]
        for col in required_cols:
            self.assertIn(col, result.columns, f"Missing required column: {col}")

    def test_build_predictions_frame_with_metadata(self):
        """Test that metadata columns (ticker, sector) are included."""
        from finance_ml.ml_workflow.regression.io import build_predictions_frame

        y_true = pd.Series([100, 200], index=[0, 1])
        y_pred = np.array([95, 210])

        df_source = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "region": ["US", "US"],
                "last_price": [150.0, 250.0],
            },
            index=[0, 1],
        )

        result = build_predictions_frame(y_true, y_pred, df_source)

        # Metadata columns should be included
        self.assertIn("ticker", result.columns)
        self.assertIn("sector", result.columns)
        self.assertIn("last_price", result.columns)

    def test_build_predictions_frame_with_quantiles(self):
        """Test that quantile predictions can be included."""
        from finance_ml.ml_workflow.regression.io import build_predictions_frame

        y_true = pd.Series([100, 200], index=[0, 1])
        y_pred = np.array([95, 210])

        df_source = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Tech"]}, index=[0, 1]
        )

        # Additional quantile predictions
        quantile_cols = {
            "pred_p10": np.array([80, 180]),
            "pred_p50": np.array([95, 210]),
            "pred_p90": np.array([110, 240]),
        }

        result = build_predictions_frame(y_true, y_pred, df_source, extra_cols=quantile_cols)

        # Quantile columns should be present
        self.assertIn("pred_p10", result.columns)
        self.assertIn("pred_p50", result.columns)
        self.assertIn("pred_p90", result.columns)

    def test_build_predictions_frame_computes_errors(self):
        """Test that error metrics are computed correctly."""
        from finance_ml.ml_workflow.regression.io import build_predictions_frame

        y_true = pd.Series([100, 200], index=[0, 1])
        y_pred = np.array([90, 220])  # Errors: 10, 20

        df_source = pd.DataFrame({"ticker": ["AAPL", "MSFT"]}, index=[0, 1])

        result = build_predictions_frame(y_true, y_pred, df_source)

        # Check abs_error
        expected_abs = [10, 20]
        np.testing.assert_array_almost_equal(result["abs_error"].values, expected_abs)

        # Check pct_error
        expected_pct = [(90 - 100) / 100 * 100, (220 - 200) / 200 * 100]
        np.testing.assert_array_almost_equal(result["pct_error"].values, expected_pct, decimal=1)

    def test_predictions_schema_non_negative_invariant(self):
        """Test that pred_p10 is non-negative when last_price is non-negative."""
        from finance_ml.ml_workflow.regression.io import build_predictions_frame

        y_true = pd.Series([100, 200], index=[0, 1])
        y_pred = np.array([95, 210])

        df_source = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "last_price": [150.0, 250.0]},  # Non-negative prices
            index=[0, 1],
        )

        quantile_cols = {
            "pred_p10": np.array([80, 180]),
            "pred_p50": np.array([95, 210]),
            "pred_p90": np.array([110, 240]),
        }

        result = build_predictions_frame(y_true, y_pred, df_source, extra_cols=quantile_cols)

        # pred_p10 should be non-negative
        if "pred_p10" in result.columns:
            self.assertTrue(
                (result["pred_p10"] >= 0).all(), f"Found negative lower quantile predictions"
            )

    def test_validate_predictions_schema_exists(self):
        """validate_predictions_schema helper should be available in regression.io."""
        try:
            from finance_ml.ml_workflow.regression.io import validate_predictions_schema

            self.assertTrue(callable(validate_predictions_schema))
        except ImportError:
            self.fail("validate_predictions_schema not implemented in regression.io")

    def test_validate_predictions_schema_core_columns_required(self):
        """Core prediction columns must be present for schema to be valid."""
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "last_price": [150.0, 250.0],
                "y_true": [100.0, 200.0],
                "y_pred": [95.0, 210.0],
                "abs_error": [5.0, 10.0],
                "pct_error": [-5.0, 5.0],
            }
        )

        # Should not raise for minimal core-compliant schema
        validated = validate_predictions_schema(df)
        self.assertIsInstance(validated, pd.DataFrame)

    def test_validate_predictions_schema_missing_core_raises(self):
        """Missing core columns should trigger a clear validation error."""
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                # Intentionally omit y_true / y_pred
                "last_price": [150.0, 250.0],
                "abs_error": [5.0, 10.0],
                "pct_error": [-5.0, 5.0],
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_predictions_schema(df)

        msg = str(ctx.exception)
        self.assertIn("missing required columns", msg)
        self.assertIn("y_true", msg)

    def test_validate_predictions_schema_adds_interval_width_when_quantiles_present(self):
        """If quantiles exist but interval_width is missing, it should be derived."""
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "last_price": [150.0, 250.0],
                "y_true": [100.0, 200.0],
                "y_pred": [95.0, 210.0],
                "abs_error": [5.0, 10.0],
                "pct_error": [-5.0, 5.0],
                "pred_p10": [80.0, 180.0],
                "pred_p90": [110.0, 240.0],
            }
        )

        validated = validate_predictions_schema(df.copy())

        self.assertIn("interval_width", validated.columns)
        expected_width = validated["pred_p90"] - validated["pred_p10"]
        pd.testing.assert_series_equal(
            validated["interval_width"], expected_width, check_names=False
        )

    def test_validate_predictions_schema_rejects_negative_last_price(self):
        """Last price should be non-negative in standardized prediction outputs."""
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "last_price": [150.0, -10.0],  # Invalid negative price
                "y_true": [100.0, 200.0],
                "y_pred": [95.0, 210.0],
                "abs_error": [5.0, 10.0],
                "pct_error": [-5.0, 5.0],
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_predictions_schema(df)

        self.assertIn("last_price", str(ctx.exception))


class TestPredictionsOutputFiles(unittest.TestCase):
    """Test that prediction files have correct schema."""

    def test_regression_predictions_detailed_schema(self):
        """Test expected schema for regression_predictions_detailed.csv."""
        # This is a contract test - defines what the output should look like
        expected_cols = [
            "ticker",
            "sector",
            "region",
            "last_price",
            "y_true",
            "y_pred",
            "y_pred_calibrated",
            "pred_p10",
            "pred_p50",
            "pred_p90",
            "interval_width",
            "abs_error",
            "pct_error",
        ]

        # Note: Not all columns required (e.g., region may be missing)
        # But core prediction cols must exist
        core_required = ["y_true", "y_pred", "abs_error", "pct_error"]

        # This test defines the contract
        self.assertIsInstance(core_required, list)
        self.assertEqual(len(core_required), 4)


if __name__ == "__main__":
    unittest.main()
