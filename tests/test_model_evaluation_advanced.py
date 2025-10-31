"""
Test suite for advanced model evaluation and error analysis features.

Tests cover:
- SHAP detailed analysis (waterfall, dependence, summary plots)
- LIME integration
- Model comparison framework
- Bias-variance diagnosis
- Learning curves and validation curves
- Time-series aware cross-validation
- Performance heatmaps (Sector × Region)
- Enhanced residual analysis
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from finance_ml import eval as eval_module


def _check_lime_available():
    """Check if LIME library is available."""
    try:
        import lime

        return True
    except ImportError:
        return False


class TestSHAPDetailedAnalysis(unittest.TestCase):
    """Test SHAP detailed analysis functions."""

    def setUp(self):
        """Create sample data for SHAP tests."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        self.y = pd.Series(np.random.randn(n_samples))

        # Create a simple model
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_compute_shap_values_returns_dict(self):
        """Test that compute_shap_values returns a dictionary."""
        result = eval_module.compute_shap_values(self.model, self.X, model_type="tree")
        self.assertIsInstance(result, dict)

    def test_compute_shap_values_has_required_keys(self):
        """Test that SHAP values dict has required keys."""
        result = eval_module.compute_shap_values(self.model, self.X, model_type="tree")
        self.assertIn("shap_values", result)
        self.assertIn("expected_value", result)
        self.assertIn("feature_names", result)

    def test_create_shap_summary_plot(self):
        """Test SHAP summary plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shap_summary.png"
            eval_module.create_shap_summary_plot(
                self.model, self.X, output_path=output_path, model_type="tree"
            )
            self.assertTrue(output_path.exists())

    def test_create_shap_waterfall_plot(self):
        """Test SHAP waterfall plot for individual predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shap_waterfall.png"
            eval_module.create_shap_waterfall_plot(
                self.model, self.X, sample_idx=0, output_path=output_path, model_type="tree"
            )
            self.assertTrue(output_path.exists())

    def test_create_shap_dependence_plot(self):
        """Test SHAP dependence plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shap_dependence.png"
            eval_module.create_shap_dependence_plot(
                self.model, self.X, feature="feature_0", output_path=output_path, model_type="tree"
            )
            self.assertTrue(output_path.exists())

    def test_analyze_shap_by_sector(self):
        """Test sector-specific SHAP analysis."""
        sectors = pd.Series(np.random.choice(["Tech", "Finance", "Energy"], size=len(self.X)))
        result = eval_module.analyze_shap_by_sector(self.model, self.X, sectors, model_type="tree")
        self.assertIsInstance(result, dict)
        self.assertIn("Tech", result)
        self.assertIn("Finance", result)


class TestLIMEIntegration(unittest.TestCase):
    """Test LIME integration for model explanations."""

    def setUp(self):
        """Create sample data for LIME tests."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        self.y = pd.Series(np.random.randn(100))

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    @unittest.skipIf(not _check_lime_available(), "LIME library not available")
    def test_explain_with_lime_returns_dict(self):
        """Test that LIME explanation returns a dictionary."""
        result = eval_module.explain_with_lime(self.model, self.X, sample_idx=0)
        self.assertIsInstance(result, dict)

    @unittest.skipIf(not _check_lime_available(), "LIME library not available")
    def test_explain_with_lime_has_feature_weights(self):
        """Test that LIME explanation contains feature weights."""
        result = eval_module.explain_with_lime(self.model, self.X, sample_idx=0)
        self.assertIn("feature_weights", result)
        self.assertIsInstance(result["feature_weights"], dict)

    @unittest.skipIf(not _check_lime_available(), "LIME library not available")
    def test_explain_with_lime_html_output(self):
        """Test LIME HTML explanation generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "lime_explanation.html"
            eval_module.explain_with_lime(self.model, self.X, sample_idx=0, output_path=output_path)
            self.assertTrue(output_path.exists())

    @unittest.skipIf(not _check_lime_available(), "LIME library not available")
    def test_compare_lime_shap_consistency(self):
        """Test consistency between LIME and SHAP explanations."""
        result = eval_module.compare_lime_shap_consistency(
            self.model, self.X, sample_idx=0, model_type="tree"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("lime_weights", result)
        self.assertIn("shap_values", result)
        self.assertIn("correlation", result)


class TestModelComparison(unittest.TestCase):
    """Test model comparison and selection framework."""

    def setUp(self):
        """Create sample models and data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5))
        self.y = pd.Series(np.random.randn(100))

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        self.model1 = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model2 = LinearRegression()

        self.model1.fit(self.X, self.y)
        self.model2.fit(self.X, self.y)

        self.models = {"RandomForest": self.model1, "LinearRegression": self.model2}

    def test_create_model_comparison_table(self):
        """Test model comparison table creation."""
        result = eval_module.create_model_comparison_table(self.models, self.X, self.y)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("mae", result.columns)
        self.assertIn("rmse", result.columns)
        self.assertIn("r2", result.columns)

    def test_statistical_model_comparison(self):
        """Test statistical significance testing between models."""
        y_pred1 = self.model1.predict(self.X)
        y_pred2 = self.model2.predict(self.X)

        result = eval_module.statistical_model_comparison(
            self.y, y_pred1, y_pred2, test_type="paired_ttest"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("statistic", result)
        self.assertIn("p_value", result)
        self.assertIn("significant", result)

    def test_automated_model_selection(self):
        """Test automated model selection based on metrics."""
        result = eval_module.automated_model_selection(
            self.models, self.X, self.y, metric="rmse", cross_validate=True
        )
        self.assertIsInstance(result, dict)
        self.assertIn("best_model_name", result)
        self.assertIn("best_score", result)
        self.assertIn("all_scores", result)


class TestLearningCurves(unittest.TestCase):
    """Test learning curves and validation curves."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(200, 5))
        self.y = pd.Series(np.random.randn(200))

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)

    def test_generate_learning_curve(self):
        """Test learning curve generation."""
        result = eval_module.generate_learning_curve(
            self.model, self.X, self.y, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0]
        )
        self.assertIsInstance(result, dict)
        self.assertIn("train_sizes", result)
        self.assertIn("train_scores", result)
        self.assertIn("val_scores", result)

    def test_plot_learning_curve(self):
        """Test learning curve plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "learning_curve.png"
            eval_module.plot_learning_curve(self.model, self.X, self.y, output_path=output_path)
            self.assertTrue(output_path.exists())

    def test_generate_validation_curve(self):
        """Test validation curve for hyperparameter tuning."""
        result = eval_module.generate_validation_curve(
            self.model, self.X, self.y, param_name="n_estimators", param_range=[5, 10, 20, 50]
        )
        self.assertIsInstance(result, dict)
        self.assertIn("param_range", result)
        self.assertIn("train_scores", result)
        self.assertIn("val_scores", result)

    def test_plot_validation_curve(self):
        """Test validation curve plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "validation_curve.png"
            eval_module.plot_validation_curve(
                self.model,
                self.X,
                self.y,
                param_name="n_estimators",
                param_range=[5, 10, 20],
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())


class TestBiasVarianceDiagnosis(unittest.TestCase):
    """Test bias-variance diagnosis functions."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(100, 5))
        self.y_train = pd.Series(np.random.randn(100))
        self.X_val = pd.DataFrame(np.random.randn(50, 5))
        self.y_val = pd.Series(np.random.randn(50))

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_diagnose_bias_variance(self):
        """Test bias-variance diagnosis."""
        result = eval_module.diagnose_bias_variance(
            self.model, self.X_train, self.y_train, self.X_val, self.y_val
        )
        self.assertIsInstance(result, dict)
        self.assertIn("train_score", result)
        self.assertIn("val_score", result)
        self.assertIn("diagnosis", result)

    def test_bias_variance_decomposition(self):
        """Test bias-variance decomposition calculation."""
        result = eval_module.bias_variance_decomposition(
            self.model, self.X_train, self.y_train, self.X_val, self.y_val, n_bootstraps=10
        )
        self.assertIsInstance(result, dict)
        self.assertIn("bias_squared", result)
        self.assertIn("variance", result)
        self.assertIn("mse", result)

    def test_plot_bias_variance(self):
        """Test bias-variance plot creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bias_variance.png"
            eval_module.plot_bias_variance(
                self.model,
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

    def test_identify_optimal_complexity(self):
        """Test identification of optimal model complexity."""
        result = eval_module.identify_optimal_complexity(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            model_type="RandomForest",
            complexity_param="max_depth",
            complexity_range=[3, 5, 10, 20],
        )
        self.assertIsInstance(result, dict)
        self.assertIn("optimal_value", result)
        self.assertIn("train_scores", result)
        self.assertIn("val_scores", result)


class TestTimeSeriesCrossValidation(unittest.TestCase):
    """Test time-series aware cross-validation."""

    def setUp(self):
        """Create sample time-series data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.X = pd.DataFrame(np.random.randn(100, 5))
        self.y = pd.Series(np.random.randn(100))
        self.dates = pd.Series(dates)

    def test_create_expanding_window_cv(self):
        """Test expanding window CV creation."""
        cv = eval_module.create_expanding_window_cv(n_splits=5)
        self.assertIsNotNone(cv)

        # Test that splits are generated
        splits = list(cv.split(self.X))
        self.assertEqual(len(splits), 5)

    def test_create_rolling_window_cv(self):
        """Test rolling window CV creation."""
        cv = eval_module.create_rolling_window_cv(n_splits=5)
        self.assertIsNotNone(cv)

        splits = list(cv.split(self.X))
        self.assertEqual(len(splits), 5)

    def test_evaluate_with_time_series_cv(self):
        """Test evaluation with time-series CV."""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        result = eval_module.evaluate_with_time_series_cv(
            model, self.X, self.y, cv_type="expanding", n_splits=3
        )
        self.assertIsInstance(result, dict)
        self.assertIn("cv_scores", result)
        self.assertIn("mean_score", result)

    def test_temporal_ordering_validation(self):
        """Test that time-series CV maintains temporal ordering."""
        cv = eval_module.create_expanding_window_cv(n_splits=3)

        for train_idx, test_idx in cv.split(self.X):
            # Ensure test indices are always after train indices
            self.assertGreater(min(test_idx), max(train_idx))


class TestPerformanceHeatmaps(unittest.TestCase):
    """Test performance heatmap creation (Sector × Region)."""

    def setUp(self):
        """Create sample data with sectors and regions."""
        np.random.seed(42)
        n_samples = 100

        self.df = pd.DataFrame(
            {
                "y_true": np.random.randn(n_samples) * 10 + 100,
                "y_pred": np.random.randn(n_samples) * 10 + 100,
                "sector": np.random.choice(["Tech", "Finance", "Energy"], n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
            }
        )

    def test_create_sector_region_performance_heatmap(self):
        """Test sector × region performance heatmap creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "performance_heatmap.png"
            eval_module.create_sector_region_performance_heatmap(
                self.df,
                y_true_col="y_true",
                y_pred_col="y_pred",
                sector_col="sector",
                region_col="region",
                metric="mae",
                output_path=output_path,
            )
            self.assertTrue(output_path.exists())

    def test_compute_sector_region_metrics(self):
        """Test computation of metrics by sector and region."""
        result = eval_module.compute_sector_region_metrics(
            self.df,
            y_true_col="y_true",
            y_pred_col="y_pred",
            sector_col="sector",
            region_col="region",
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("sector", result.columns)
        self.assertIn("region", result.columns)
        self.assertIn("mae", result.columns)

    def test_heatmap_with_different_metrics(self):
        """Test heatmap creation with different metrics."""
        for metric in ["mae", "rmse", "r2", "mape"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / f"heatmap_{metric}.png"
                eval_module.create_sector_region_performance_heatmap(
                    self.df,
                    y_true_col="y_true",
                    y_pred_col="y_pred",
                    sector_col="sector",
                    region_col="region",
                    metric=metric,
                    output_path=output_path,
                )
                self.assertTrue(output_path.exists())


class TestEnhancedResidualAnalysis(unittest.TestCase):
    """Test enhanced residual analysis functions."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.y_true = np.random.randn(100) * 10 + 100
        self.y_pred = self.y_true + np.random.randn(100) * 2

        self.df = pd.DataFrame(
            {
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )

    def test_plot_residuals_vs_features(self):
        """Test residuals vs features plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            eval_module.plot_residuals_vs_features(
                self.df,
                y_true_col="y_true",
                y_pred_col="y_pred",
                feature_cols=["feature1", "feature2"],
                output_dir=output_dir,
            )
            # Check that plots were created
            plots = list(output_dir.glob("*.png"))
            self.assertGreater(len(plots), 0)

    def test_identify_systematic_bias_patterns(self):
        """Test identification of systematic bias patterns."""
        result = eval_module.identify_systematic_bias_patterns(
            self.df, y_true_col="y_true", y_pred_col="y_pred", segment_cols=["feature1"]
        )
        self.assertIsInstance(result, dict)

    def test_analyze_residual_homoscedasticity(self):
        """Test homoscedasticity analysis."""
        result = eval_module.analyze_residual_homoscedasticity(self.y_true, self.y_pred)
        self.assertIsInstance(result, dict)
        self.assertIn("test_statistic", result)
        self.assertIn("p_value", result)
        self.assertIn("is_homoscedastic", result)


class TestFeatureImportanceRanking(unittest.TestCase):
    """Test feature importance ranking and stability."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        self.y = pd.Series(
            self.X["f0"] * 2 + self.X["f1"] - self.X["f2"] * 0.5 + np.random.randn(100) * 0.1
        )

        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_compute_permutation_importance(self):
        """Test permutation importance calculation."""
        result = eval_module.compute_permutation_importance(self.model, self.X, self.y, n_repeats=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("importance_mean", result.columns)
        self.assertIn("importance_std", result.columns)

    def test_rank_features_by_importance(self):
        """Test feature ranking by importance."""
        result = eval_module.rank_features_by_importance(self.model, self.X, self.y, method="all")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.X.columns))

    def test_feature_importance_stability_across_folds(self):
        """Test feature importance stability across CV folds."""
        result = eval_module.feature_importance_stability_across_folds(
            self.model, self.X, self.y, n_splits=3
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("stability_score", result.columns)


if __name__ == "__main__":
    unittest.main()
