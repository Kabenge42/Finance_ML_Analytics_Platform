"""
Test: Regression Function Return Type Alignment (TDD)

This test module validates that train_* functions follow the standardized
return format specified in docs/code_guidelines.md Section 1.1:

Expected return format:
{
    'model': fitted estimator or pipeline,
    'metrics': Dict[str, float] with evaluation metrics,
    'y_pred': 1D array-like or Series of predictions (or None if not computed),
    'y_proba': Optional - for classifiers only (None for regressors),
    'artifacts': Optional Dict[str, Any] with auxiliary items
}

This test is written BEFORE implementation (TDD approach) and should FAIL initially.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class TestTrainStackingRegressorReturnAlignment(unittest.TestCase):
    """Test train_stacking_regressor returns standardized dict format."""

    @classmethod
    def setUpClass(cls):
        """Create sample regression dataset once for all tests."""
        # Create small but sufficient dataset
        X, y = make_regression(n_samples=100, n_features=10, random_state=42, noise=10)
        cls.X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        cls.y = pd.Series(y, name="target")

    def test_stacking_returns_dict_not_tuple(self):
        """Test that train_stacking_regressor returns dict, not tuple."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        # Primary assertion: must return dict
        self.assertIsInstance(
            result,
            dict,
            "train_stacking_regressor must return dict, not tuple per code_guidelines.md Section 1.1",
        )

    def test_stacking_has_required_keys(self):
        """Test that result dict contains all required keys."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        # Required keys per code_guidelines.md Section 1.1
        required_keys = {"model", "metrics", "y_pred"}

        self.assertIsInstance(result, dict, "Result must be dict")
        for key in required_keys:
            self.assertIn(
                key, result, f"Result must contain '{key}' key per code_guidelines.md Section 1.1"
            )

    def test_stacking_model_key_contains_fitted_estimator(self):
        """Test that result['model'] is a fitted estimator."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        self.assertIn("model", result)
        model = result["model"]

        # Model must have predict method
        self.assertTrue(hasattr(model, "predict"), "Model must have predict method")

        # Model should be able to make predictions
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), "Predictions length mismatch")

    def test_stacking_metrics_key_is_dict(self):
        """Test that result['metrics'] is a dict with float values."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        self.assertIn("metrics", result)
        metrics = result["metrics"]

        self.assertIsInstance(metrics, dict, "metrics must be dict")

        # Should contain common regression metrics
        # At minimum should have some numeric metrics
        self.assertGreater(len(metrics), 0, "metrics dict should not be empty")

        # All metric values should be numeric
        for key, value in metrics.items():
            self.assertTrue(
                isinstance(value, (int, float, np.number)),
                f"Metric '{key}' value must be numeric, got {type(value)}",
            )

    def test_stacking_y_pred_key_exists(self):
        """Test that result['y_pred'] exists (may be None)."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        self.assertIn("y_pred", result, "Result must contain 'y_pred' key")

        # y_pred can be None (if not computed during training) or array-like
        y_pred = result["y_pred"]
        if y_pred is not None:
            # If provided, should be array-like with correct length
            self.assertEqual(len(y_pred), len(self.y), "y_pred length must match target length")

    def test_stacking_artifacts_key_optional(self):
        """Test that result may contain optional 'artifacts' key."""
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        result = train_stacking_regressor(self.X, self.y, cv=3, random_state=42)

        # artifacts is optional but if present should be dict
        if "artifacts" in result:
            self.assertIsInstance(result["artifacts"], dict, "artifacts must be dict if present")


class TestTrainQuantileRegressorReturnAlignment(unittest.TestCase):
    """Test train_quantile_regressor returns standardized dict format."""

    @classmethod
    def setUpClass(cls):
        """Create sample regression dataset once for all tests."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42, noise=10)
        cls.X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
        cls.y = pd.Series(y, name="target")

    def test_quantile_returns_dict_not_tuple(self):
        """Test that train_quantile_regressor returns dict, not tuple."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        result = train_quantile_regressor(
            self.X, self.y, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Primary assertion: must return dict
        self.assertIsInstance(
            result,
            dict,
            "train_quantile_regressor must return dict, not tuple per code_guidelines.md Section 1.1",
        )

    def test_quantile_has_required_keys(self):
        """Test that result dict contains all required keys."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        result = train_quantile_regressor(
            self.X, self.y, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Required keys per code_guidelines.md Section 1.1
        required_keys = {"model", "metrics", "y_pred"}

        self.assertIsInstance(result, dict, "Result must be dict")
        for key in required_keys:
            self.assertIn(
                key, result, f"Result must contain '{key}' key per code_guidelines.md Section 1.1"
            )

    def test_quantile_model_key_contains_models_list(self):
        """Test that result['model'] contains the list of quantile models."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        quantiles = [0.1, 0.5, 0.9]
        result = train_quantile_regressor(self.X, self.y, quantiles=quantiles, random_state=42)

        self.assertIn("model", result)
        models = result["model"]

        # For quantile regression, model should be a list
        self.assertIsInstance(models, list, "Quantile models should be returned as list")
        self.assertEqual(len(models), len(quantiles), "Should have one model per quantile")

        # Each model should have predict method
        for idx, model in enumerate(models):
            self.assertTrue(hasattr(model, "predict"), f"Model {idx} must have predict method")

    def test_quantile_metrics_key_is_dict(self):
        """Test that result['metrics'] is a dict with float values."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        result = train_quantile_regressor(
            self.X, self.y, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        self.assertIn("metrics", result)
        metrics = result["metrics"]

        self.assertIsInstance(metrics, dict, "metrics must be dict")
        self.assertGreater(len(metrics), 0, "metrics dict should not be empty")

        # All metric values should be numeric
        for key, value in metrics.items():
            self.assertTrue(
                isinstance(value, (int, float, np.number)),
                f"Metric '{key}' value must be numeric, got {type(value)}",
            )

    def test_quantile_y_pred_key_exists(self):
        """Test that result['y_pred'] exists (may be None)."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        result = train_quantile_regressor(
            self.X, self.y, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        self.assertIn("y_pred", result, "Result must contain 'y_pred' key")
        # For quantile regression, y_pred is typically None during training
        # (predictions made per quantile model individually)

    def test_quantile_artifacts_contains_models_and_metadata(self):
        """Test that result['artifacts'] contains quantile models and metadata."""
        from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

        quantiles = [0.1, 0.5, 0.9]
        result = train_quantile_regressor(self.X, self.y, quantiles=quantiles, random_state=42)

        self.assertIn("artifacts", result, "Result should contain 'artifacts' key")
        artifacts = result["artifacts"]

        self.assertIsInstance(artifacts, dict, "artifacts must be dict")

        # Should contain quantile metadata
        expected_artifact_keys = {"models", "quantiles", "n_models", "quantile_results"}
        for key in expected_artifact_keys:
            self.assertIn(
                key,
                artifacts,
                f"artifacts should contain '{key}' for backward compatibility and metadata",
            )


if __name__ == "__main__":
    unittest.main()
