"""
Tests for learning curve and bias-variance functions extracted from analytics/eval.py

TDD Step 5: Testing learning curve functions (lines 3452-3863 of eval.py)
These tests validate the learning curve, validation curve, and bias-variance
functions that will be extracted to evaluation/learning_curves.py module.

Coverage Target: 80% for learning_curves module
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


class TestGenerateLearningCurve(unittest.TestCase):
    """Tests for generate_learning_curve function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
        self.y = np.random.randn(100)

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = generate_learning_curve(model, self.X, self.y, cv=3)

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = generate_learning_curve(model, self.X, self.y, cv=3)

            expected_keys = [
                "train_sizes",
                "train_scores",
                "train_scores_mean",
                "train_scores_std",
                "val_scores",
                "val_scores_mean",
                "val_scores_std",
            ]
            for key in expected_keys:
                self.assertIn(key, result)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_default_train_sizes(self):
        """Test with default training sizes."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = generate_learning_curve(model, self.X, self.y, cv=3)

            # Default has 6 train sizes
            self.assertEqual(len(result["train_sizes"]), 6)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_custom_train_sizes(self):
        """Test with custom training sizes."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            custom_sizes = [0.2, 0.5, 0.8]
            result = generate_learning_curve(model, self.X, self.y, train_sizes=custom_sizes, cv=3)

            self.assertEqual(len(result["train_sizes"]), 3)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_scores_are_lists(self):
        """Test that scores are returned as lists."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = generate_learning_curve(model, self.X, self.y, cv=3)

            self.assertIsInstance(result["train_scores_mean"], list)
            self.assertIsInstance(result["val_scores_mean"], list)
        except ImportError:
            self.skipTest("sklearn not available")


class TestPlotLearningCurve(unittest.TestCase):
    """Tests for plot_learning_curve function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
        self.y = np.random.randn(100)

    def test_creates_plot_without_error(self):
        """Test that plot is created without error."""
        from finance_ml.ml_workflow.analytics.eval import plot_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            # Should not raise error
            plot_learning_curve(model, self.X, self.y, cv=3)
        except ImportError:
            self.skipTest("Required libraries not available")

    def test_saves_to_file(self):
        """Test that plot is saved to file when path provided."""
        from finance_ml.ml_workflow.analytics.eval import plot_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "learning_curve.png"
                plot_learning_curve(model, self.X, self.y, output_path=output_path, cv=3)

                self.assertTrue(output_path.exists())
        except ImportError:
            self.skipTest("Required libraries not available")


class TestGenerateValidationCurve(unittest.TestCase):
    """Tests for generate_validation_curve function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
        self.y = np.random.randn(100)

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import generate_validation_curve

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)

            result = generate_validation_curve(
                model, self.X, self.y, param_name="max_depth", param_range=[2, 5, 10], cv=3
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        from finance_ml.ml_workflow.analytics.eval import generate_validation_curve

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=10, random_state=42)

            result = generate_validation_curve(
                model, self.X, self.y, param_name="max_depth", param_range=[2, 5], cv=3
            )

            expected_keys = [
                "param_range",
                "train_scores",
                "train_scores_mean",
                "train_scores_std",
                "val_scores",
                "val_scores_mean",
                "val_scores_std",
            ]
            for key in expected_keys:
                self.assertIn(key, result)
        except ImportError:
            self.skipTest("sklearn not available")


class TestDiagnoseBiasVariance(unittest.TestCase):
    """Tests for diagnose_bias_variance function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_train = 80
        n_val = 20

        self.X_train = pd.DataFrame(
            {"feature1": np.random.randn(n_train), "feature2": np.random.randn(n_train)}
        )
        self.y_train = np.random.randn(n_train)

        self.X_val = pd.DataFrame(
            {"feature1": np.random.randn(n_val), "feature2": np.random.randn(n_val)}
        )
        self.y_val = np.random.randn(n_val)

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import diagnose_bias_variance

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(self.X_train, self.y_train)

            result = diagnose_bias_variance(
                model, self.X_train, self.y_train, self.X_val, self.y_val
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        from finance_ml.ml_workflow.analytics.eval import diagnose_bias_variance

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(self.X_train, self.y_train)

            result = diagnose_bias_variance(
                model, self.X_train, self.y_train, self.X_val, self.y_val
            )

            self.assertIn("train_score", result)
            self.assertIn("val_score", result)
            self.assertIn("score_gap", result)
            self.assertIn("diagnosis", result)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_diagnosis_is_string(self):
        """Test that diagnosis is a string."""
        from finance_ml.ml_workflow.analytics.eval import diagnose_bias_variance

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(self.X_train, self.y_train)

            result = diagnose_bias_variance(
                model, self.X_train, self.y_train, self.X_val, self.y_val
            )

            self.assertIsInstance(result["diagnosis"], str)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_score_gap_calculation(self):
        """Test that score gap is calculated correctly."""
        from finance_ml.ml_workflow.analytics.eval import diagnose_bias_variance

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(self.X_train, self.y_train)

            result = diagnose_bias_variance(
                model, self.X_train, self.y_train, self.X_val, self.y_val
            )

            expected_gap = result["train_score"] - result["val_score"]
            self.assertAlmostEqual(result["score_gap"], expected_gap, places=5)
        except ImportError:
            self.skipTest("sklearn not available")


class TestBiasVarianceDecomposition(unittest.TestCase):
    """Tests for bias_variance_decomposition function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_train = 80
        n_val = 20

        self.X_train = pd.DataFrame(
            {"feature1": np.random.randn(n_train), "feature2": np.random.randn(n_train)}
        )
        self.y_train = np.random.randn(n_train)

        self.X_val = pd.DataFrame(
            {"feature1": np.random.randn(n_val), "feature2": np.random.randn(n_val)}
        )
        self.y_val = np.random.randn(n_val)

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import bias_variance_decomposition

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = bias_variance_decomposition(
                model, self.X_train, self.y_train, self.X_val, self.y_val, n_bootstraps=10
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        from finance_ml.ml_workflow.analytics.eval import bias_variance_decomposition

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = bias_variance_decomposition(
                model, self.X_train, self.y_train, self.X_val, self.y_val, n_bootstraps=10
            )

            self.assertIn("bias_squared", result)
            self.assertIn("variance", result)
            self.assertIn("mse", result)
            self.assertIn("n_bootstraps", result)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_values_are_non_negative(self):
        """Test that bias, variance, and MSE are non-negative."""
        from finance_ml.ml_workflow.analytics.eval import bias_variance_decomposition

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = bias_variance_decomposition(
                model, self.X_train, self.y_train, self.X_val, self.y_val, n_bootstraps=10
            )

            self.assertGreaterEqual(result["bias_squared"], 0)
            self.assertGreaterEqual(result["variance"], 0)
            self.assertGreaterEqual(result["mse"], 0)
        except ImportError:
            self.skipTest("sklearn not available")


class TestIdentifyOptimalComplexity(unittest.TestCase):
    """Tests for identify_optimal_complexity function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_train = 80
        n_val = 20

        self.X_train = pd.DataFrame(
            {"feature1": np.random.randn(n_train), "feature2": np.random.randn(n_train)}
        )
        self.y_train = np.random.randn(n_train)

        self.X_val = pd.DataFrame(
            {"feature1": np.random.randn(n_val), "feature2": np.random.randn(n_val)}
        )
        self.y_val = np.random.randn(n_val)

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        from finance_ml.ml_workflow.analytics.eval import identify_optimal_complexity

        try:
            result = identify_optimal_complexity(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                model_type="RandomForest",
                complexity_param="max_depth",
                complexity_range=[3, 5, 10],
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_returns_expected_keys(self):
        """Test that result contains expected keys."""
        from finance_ml.ml_workflow.analytics.eval import identify_optimal_complexity

        try:
            result = identify_optimal_complexity(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                model_type="RandomForest",
                complexity_param="max_depth",
                complexity_range=[3, 5],
            )

            self.assertIn("optimal_value", result)
            self.assertIn("optimal_val_score", result)
            self.assertIn("complexity_param", result)
            self.assertIn("train_scores", result)
            self.assertIn("val_scores", result)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_with_gradient_boosting(self):
        """Test with GradientBoosting model."""
        from finance_ml.ml_workflow.analytics.eval import identify_optimal_complexity

        try:
            result = identify_optimal_complexity(
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                model_type="GradientBoosting",
                complexity_param="max_depth",
                complexity_range=[2, 3],
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises ValueError."""
        from finance_ml.ml_workflow.analytics.eval import identify_optimal_complexity

        try:
            with self.assertRaises(ValueError):
                identify_optimal_complexity(
                    self.X_train,
                    self.y_train,
                    self.X_val,
                    self.y_val,
                    model_type="InvalidModel",
                    complexity_param="max_depth",
                    complexity_range=[3, 5],
                )
        except ImportError:
            self.skipTest("sklearn not available")


class TestTimeSeriesCV(unittest.TestCase):
    """Tests for time-series cross-validation functions."""

    def test_create_expanding_window_cv(self):
        """Test creating expanding window CV."""
        from finance_ml.ml_workflow.analytics.eval import create_expanding_window_cv

        try:
            cv = create_expanding_window_cv(n_splits=5)

            # Should be a TimeSeriesSplit object
            from sklearn.model_selection import TimeSeriesSplit

            self.assertIsInstance(cv, TimeSeriesSplit)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_create_rolling_window_cv(self):
        """Test creating rolling window CV."""
        from finance_ml.ml_workflow.analytics.eval import create_rolling_window_cv

        try:
            cv = create_rolling_window_cv(n_splits=5, max_train_size=100)

            from sklearn.model_selection import TimeSeriesSplit

            self.assertIsInstance(cv, TimeSeriesSplit)
        except ImportError:
            self.skipTest("sklearn not available")


class TestLearningCurvesEdgeCases(unittest.TestCase):
    """Tests for edge cases in learning curve functions."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_small = pd.DataFrame(
            {"feature1": np.random.randn(20), "feature2": np.random.randn(20)}
        )
        self.y_small = np.random.randn(20)

    def test_learning_curve_small_dataset(self):
        """Test learning curve with small dataset."""
        from finance_ml.ml_workflow.analytics.eval import generate_learning_curve

        try:
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()

            result = generate_learning_curve(
                model, self.X_small, self.y_small, train_sizes=[0.5, 0.8], cv=2
            )

            self.assertIsInstance(result, dict)
        except ImportError:
            self.skipTest("sklearn not available")

    def test_diagnose_with_different_model_types(self):
        """Test diagnosis with different model types."""
        from finance_ml.ml_workflow.analytics.eval import diagnose_bias_variance

        try:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=5, random_state=42)
            model.fit(self.X_small[:15], self.y_small[:15])

            result = diagnose_bias_variance(
                model, self.X_small[:15], self.y_small[:15], self.X_small[15:], self.y_small[15:]
            )

            self.assertIsInstance(result, dict)
            self.assertIn("diagnosis", result)
        except ImportError:
            self.skipTest("sklearn not available")


if __name__ == "__main__":
    unittest.main()
