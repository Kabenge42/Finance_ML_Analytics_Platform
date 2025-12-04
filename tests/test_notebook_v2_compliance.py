"""
Test notebook v2.0 compliance for ml_finance_model_main2_0.ipynb.

Validates that the notebook meets v2.0 requirements including:
- Classification models: XGBoost, LightGBM, CatBoost, SVM, Neural Network
- Regression models: Linear, Gradient Boosting, Tree, Neural Network, Ensemble
- Evaluation: SHAP, confusion matrices, learning curves, calibration
- Output schemas: Standardized predictions format
"""

import json
import os
import re
import unittest
from pathlib import Path


class TestClassificationModelsV2(unittest.TestCase):
    """Test that notebook includes all required classification models."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_xgboost_classifier_reference(self):
        """Test that notebook references XGBoost classifier."""
        has_model = (
            "xgboost" in self.all_source.lower()
            or "XGBoost" in self.all_source
            or "train_xgboost_classifier" in self.all_source
        )
        self.assertTrue(has_model, "Missing XGBoost classifier reference")

    def test_has_lightgbm_classifier_reference(self):
        """Test that notebook references LightGBM classifier."""
        has_model = (
            "lightgbm" in self.all_source.lower()
            or "LightGBM" in self.all_source
            or "train_lightgbm_classifier" in self.all_source
        )
        self.assertTrue(has_model, "Missing LightGBM classifier reference")

    def test_has_catboost_classifier_reference(self):
        """Test that notebook references CatBoost classifier."""
        has_model = (
            "catboost" in self.all_source.lower()
            or "CatBoost" in self.all_source
            or "train_catboost_classifier" in self.all_source
        )
        self.assertTrue(has_model, "Missing CatBoost classifier reference")

    def test_has_svm_classifier_reference(self):
        """Test that notebook references SVM classifier."""
        has_model = (
            "svm" in self.all_source.lower()
            or "SVM" in self.all_source
            or "SVC" in self.all_source
            or "train_svm_classifier" in self.all_source
        )
        self.assertTrue(has_model, "Missing SVM classifier reference")

    def test_has_neural_network_classifier_reference(self):
        """Test that notebook references Neural Network classifier."""
        has_model = (
            "neural" in self.all_source.lower()
            or "Neural" in self.all_source
            or "MLP" in self.all_source
            or "train_neural_network_classifier" in self.all_source
        )
        self.assertTrue(has_model, "Missing Neural Network classifier reference")


class TestRegressionModelsV2(unittest.TestCase):
    """Test that notebook includes all required regression model types."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_linear_models_reference(self):
        """Test that notebook references linear regression models."""
        has_model = (
            "ridge" in self.all_source.lower()
            or "lasso" in self.all_source.lower()
            or "elasticnet" in self.all_source.lower()
            or "Ridge" in self.all_source
            or "Lasso" in self.all_source
        )
        self.assertTrue(has_model, "Missing linear regression models reference")

    def test_has_gradient_boosting_regressors(self):
        """Test that notebook references gradient boosting regressors."""
        has_model = (
            "xgboost" in self.all_source.lower()
            or "lightgbm" in self.all_source.lower()
            or "catboost" in self.all_source.lower()
            or "histgradient" in self.all_source.lower()
        )
        self.assertTrue(has_model, "Missing gradient boosting regressors reference")

    def test_has_tree_models_reference(self):
        """Test that notebook references tree-based models."""
        has_model = (
            "randomforest" in self.all_source.lower()
            or "random_forest" in self.all_source.lower()
            or "extratrees" in self.all_source.lower()
            or "RandomForest" in self.all_source
        )
        self.assertTrue(has_model, "Missing tree-based models reference")

    def test_has_neural_network_regressor_reference(self):
        """Test that notebook references neural network regressor."""
        has_model = (
            "neural" in self.all_source.lower()
            or "tensorflow" in self.all_source.lower()
            or "keras" in self.all_source.lower()
            or "DNN" in self.all_source
        )
        self.assertTrue(has_model, "Missing neural network regressor reference")

    def test_has_ensemble_methods_reference(self):
        """Test that notebook references ensemble methods."""
        has_model = (
            "voting" in self.all_source.lower()
            or "stacking" in self.all_source.lower()
            or "ensemble" in self.all_source.lower()
        )
        self.assertTrue(has_model, "Missing ensemble methods reference")

    def test_has_compare_regressors_reference(self):
        """Test that notebook references model comparison."""
        has_compare = (
            "compare_regressors" in self.all_source
            or "compare_models" in self.all_source.lower()
            or "model comparison" in self.all_source.lower()
        )
        self.assertTrue(has_compare, "Missing compare_regressors reference")


class TestEvaluationCapabilitiesV2(unittest.TestCase):
    """Test that notebook includes comprehensive evaluation capabilities."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_shap_interpretation(self):
        """Test that notebook includes SHAP-based interpretation."""
        has_shap = "shap" in self.all_source.lower() or "SHAP" in self.all_source
        self.assertTrue(has_shap, "Missing SHAP interpretation")

    def test_has_feature_importance(self):
        """Test that notebook includes feature importance analysis."""
        has_importance = (
            "feature_importance" in self.all_source.lower()
            or "feature importance" in self.all_source.lower()
            or "importance" in self.all_source.lower()
        )
        self.assertTrue(has_importance, "Missing feature importance analysis")

    def test_has_confusion_matrix(self):
        """Test that notebook includes confusion matrix."""
        has_matrix = "confusion" in self.all_source.lower() or "Confusion" in self.all_source
        self.assertTrue(has_matrix, "Missing confusion matrix")

    def test_has_cross_validation(self):
        """Test that notebook includes cross-validation."""
        has_cv = (
            "cross_val" in self.all_source.lower()
            or "cv_folds" in self.all_source.lower()
            or "CV_FOLDS" in self.all_source
            or "kfold" in self.all_source.lower()
        )
        self.assertTrue(has_cv, "Missing cross-validation")

    def test_has_per_sector_evaluation(self):
        """Test that notebook includes per-sector evaluation."""
        has_sector_eval = "sector" in self.all_source.lower() and (
            "metric" in self.all_source.lower() or "eval" in self.all_source.lower()
        )
        self.assertTrue(has_sector_eval, "Missing per-sector evaluation")

    def test_has_calibration_analysis(self):
        """Test that notebook includes calibration analysis."""
        has_calibration = (
            "calibration" in self.all_source.lower()
            or "Calibration" in self.all_source
            or "conformal" in self.all_source.lower()
        )
        self.assertTrue(has_calibration, "Missing calibration analysis")


class TestOutputSchemasV2(unittest.TestCase):
    """Test that notebook produces standardized output schemas."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_predictions_output(self):
        """Test that notebook creates predictions output."""
        has_output = "predictions" in self.all_source.lower() or "y_pred" in self.all_source
        self.assertTrue(has_output, "Missing predictions output")

    def test_has_metrics_output(self):
        """Test that notebook creates metrics output."""
        has_output = (
            "metrics" in self.all_source.lower()
            or "mae" in self.all_source.lower()
            or "rmse" in self.all_source.lower()
        )
        self.assertTrue(has_output, "Missing metrics output")

    def test_has_model_persistence(self):
        """Test that notebook includes model persistence."""
        has_persistence = (
            "save" in self.all_source.lower()
            or "dump" in self.all_source.lower()
            or "pickle" in self.all_source.lower()
            or "joblib" in self.all_source.lower()
        )
        self.assertTrue(has_persistence, "Missing model persistence")

    def test_has_output_directory_reference(self):
        """Test that notebook references output directory."""
        has_output_dir = "output" in self.all_source.lower() or "OUTPUT_DIR" in self.all_source
        self.assertTrue(has_output_dir, "Missing output directory reference")


class TestQuantileRegressionV2(unittest.TestCase):
    """Test that notebook includes quantile regression for uncertainty bounds."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_quantile_references(self):
        """Test that notebook references quantile regression."""
        has_quantile = "quantile" in self.all_source.lower() or "QUANTILES" in self.all_source
        self.assertTrue(has_quantile, "Missing quantile regression reference")

    def test_has_prediction_intervals(self):
        """Test that notebook creates prediction intervals."""
        has_intervals = (
            "interval" in self.all_source.lower()
            or "p10" in self.all_source.lower()
            or "p90" in self.all_source.lower()
            or "lower" in self.all_source.lower()
            or "upper" in self.all_source.lower()
        )
        self.assertTrue(has_intervals, "Missing prediction intervals")

    def test_has_uncertainty_quantification(self):
        """Test that notebook includes uncertainty quantification."""
        has_uncertainty = (
            "uncertainty" in self.all_source.lower() or "confidence" in self.all_source.lower()
        )
        self.assertTrue(has_uncertainty, "Missing uncertainty quantification")


class TestPhase94ClassificationV2(unittest.TestCase):
    """Test Phase 9.4 Classification Workflow specifics."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_event_classification_labels(self):
        """Test that notebook creates event classification labels."""
        has_labels = (
            "event" in self.all_source.lower()
            or "label" in self.all_source.lower()
            or "class" in self.all_source.lower()
        )
        self.assertTrue(has_labels, "Missing event classification labels")

    def test_has_classification_metrics(self):
        """Test that notebook includes classification metrics."""
        has_metrics = (
            "accuracy" in self.all_source.lower()
            or "precision" in self.all_source.lower()
            or "recall" in self.all_source.lower()
            or "f1" in self.all_source.lower()
        )
        self.assertTrue(has_metrics, "Missing classification metrics")


class TestPhase95RegressionV2(unittest.TestCase):
    """Test Phase 9.5 Regression Workflow specifics."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_sector_optimized_reference(self):
        """Test that notebook references sector-optimized models."""
        has_sector = "sector" in self.all_source.lower() and (
            "model" in self.all_source.lower() or "regressor" in self.all_source.lower()
        )
        self.assertTrue(has_sector, "Missing sector-optimized model reference")

    def test_has_price_target_prediction(self):
        """Test that notebook predicts price targets."""
        has_prediction = (
            "price_target" in self.all_source.lower() or "TARGET_COL" in self.all_source
        )
        self.assertTrue(has_prediction, "Missing price target prediction")


class TestPhase96EvaluationV2(unittest.TestCase):
    """Test Phase 9.6 Model Evaluation specifics."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_performance_by_sector_region(self):
        """Test that notebook evaluates performance by sector/region."""
        has_eval = (
            "sector" in self.all_source.lower() or "region" in self.all_source.lower()
        ) and ("performance" in self.all_source.lower() or "metric" in self.all_source.lower())
        self.assertTrue(has_eval, "Missing performance by sector/region")


class TestPhase97AnalyticsV2(unittest.TestCase):
    """Test Phase 9.7 Analytics specifics."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_stock_ranking(self):
        """Test that notebook includes stock ranking."""
        has_ranking = "rank" in self.all_source.lower() or "top" in self.all_source.lower()
        self.assertTrue(has_ranking, "Missing stock ranking")

    def test_has_mispricing_calculation(self):
        """Test that notebook calculates mispricing scores."""
        has_mispricing = (
            "mispricing" in self.all_source.lower()
            or "undervalued" in self.all_source.lower()
            or "overvalued" in self.all_source.lower()
        )
        self.assertTrue(has_mispricing, "Missing mispricing calculation")


if __name__ == "__main__":
    unittest.main()
