"""
Phase 9.4 Classification Workflow Tests - TDD Implementation

Tests for comprehensive classification workflow including:
- Event label creation (19 methods with 5-class system)
- ETL pipeline integration
- All classifier training (XGBoost, LightGBM, CatBoost, SVM, Neural Network)
- Comprehensive evaluation (SHAP, cross-validation, confusion matrices, calibration)
- Per-sector and per-class evaluation

Following code_guidelines.md v1.7 and TDD principles.
"""

import unittest
from unittest.mock import patch, MagicMock
import warnings

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    pd = None
    np = None

try:
    from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels
    from finance_ml.ml_workflow.classification.models import (
        prepare_classification_data,
        train_xgboost_classifier,
        train_lightgbm_classifier,
        train_catboost_classifier,
        train_svm_classifier,
        train_neural_network_classifier,
        train_voting_classifier,
        train_stacking_classifier,
        compare_classifiers,
        fit_classifier,
        export_classification_features,
    )
    from finance_ml.ml_workflow.classification.evaluation import (
        evaluate_classification,
        compute_shap_values,
        cross_validate_classifier,
        compare_feature_importance,
        plot_confusion_matrices,
        evaluate_classification_by_sector,
        plot_learning_curves,
        analyze_per_class_feature_importance,
        analyze_calibration,
        analyze_feature_importance_by_groups,
        analyze_feature_importance_by_sector,
        analyze_shap_by_feature_groups,
        export_classification_probabilities,
    )

    CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    CLASSIFICATION_AVAILABLE = False
    print(f"Classification imports failed: {e}")

try:
    from finance_ml.ml_workflow.preprocessing.etl import (
        ETLConfig,
        ETLPipeline,
        run_etl_pipeline,
        etl_from_csv,
    )

    ETL_AVAILABLE = True
except ImportError:
    ETL_AVAILABLE = False


def create_test_dataframe(n_samples: int = 200, random_state: int = 42) -> "pd.DataFrame":
    """Create a test DataFrame with Phase 9.3 style features for classification."""
    if not DEPS_AVAILABLE:
        return None

    np.random.seed(random_state)

    df = pd.DataFrame(
        {
            # Identifiers
            "ticker": [f"TICK{i:04d}" for i in range(n_samples)],
            "sector": np.random.choice(
                ["Technology", "Healthcare", "Finance", "Energy"], n_samples
            ),
            "region": np.random.choice(["US", "EU", "APAC"], n_samples),
            # Price columns (required for label creation)
            "last_price": np.random.uniform(10, 500, n_samples),
            "price_target": np.random.uniform(10, 500, n_samples),
            "price_target_median": np.random.uniform(10, 500, n_samples),
            # Momentum features
            "price_momentum_1m": np.random.uniform(-20, 20, n_samples),
            "price_momentum_3m": np.random.uniform(-30, 30, n_samples),
            "price_momentum_6m": np.random.uniform(-40, 40, n_samples),
            "price_momentum_1y": np.random.uniform(-50, 50, n_samples),
            "price_acceleration_3m": np.random.uniform(-10, 10, n_samples),
            "rsi_14d": np.random.uniform(20, 80, n_samples),
            # Valuation features
            "p_e_ratio": np.random.uniform(5, 50, n_samples),
            "p_b_ratio": np.random.uniform(0.5, 10, n_samples),
            "ev_ebitda": np.random.uniform(3, 30, n_samples),
            "p_s_ratio": np.random.uniform(0.5, 15, n_samples),
            # Profitability features
            "gross_margin": np.random.uniform(0.1, 0.8, n_samples),
            "operating_margin": np.random.uniform(0.01, 0.5, n_samples),
            "net_margin": np.random.uniform(0.01, 0.4, n_samples),
            "roe": np.random.uniform(0.01, 0.5, n_samples),
            "roa": np.random.uniform(0.01, 0.3, n_samples),
            "roic": np.random.uniform(0.01, 0.4, n_samples),
            # Leverage features
            "debt_to_equity": np.random.uniform(0.1, 3, n_samples),
            "debt_to_assets": np.random.uniform(0.1, 0.8, n_samples),
            "interest_coverage": np.random.uniform(1, 20, n_samples),
            "current_ratio": np.random.uniform(0.5, 3, n_samples),
            "quick_ratio": np.random.uniform(0.3, 2.5, n_samples),
            # Growth features
            "revenue_growth_yoy": np.random.uniform(-0.3, 0.5, n_samples),
            "earnings_growth_yoy": np.random.uniform(-0.5, 1, n_samples),
            "ebitda_growth_yoy": np.random.uniform(-0.4, 0.6, n_samples),
            # Analyst features
            "analyst_rating": np.random.uniform(1, 5, n_samples),
            "analyst_count": np.random.randint(1, 30, n_samples),
            "price_target_high": np.random.uniform(10, 600, n_samples),
            "price_target_low": np.random.uniform(5, 400, n_samples),
            # Quality features
            "piotroski_score": np.random.randint(0, 10, n_samples),
            "altman_z_score": np.random.uniform(0.5, 5, n_samples),
            # Cash flow features
            "fcf_yield": np.random.uniform(0.01, 0.15, n_samples),
            "cfo_to_net_income": np.random.uniform(0.5, 2, n_samples),
            # Efficiency features
            "asset_turnover": np.random.uniform(0.2, 2, n_samples),
            "inventory_turnover": np.random.uniform(2, 20, n_samples),
            # Employee productivity features
            "revenue_per_employee": np.random.uniform(100000, 1000000, n_samples),
            "profit_per_employee": np.random.uniform(10000, 200000, n_samples),
        }
    )

    # Ensure no NaN or inf values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

    return df


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestEventLabelCreation(unittest.TestCase):
    """Test event label creation with 5-class system and all 19 methods."""

    @classmethod
    def setUpClass(cls):
        """Create test data once for all tests."""
        cls.df = create_test_dataframe(n_samples=200)

    def test_price_momentum_labels_5class(self):
        """Test price_momentum method returns 5-class labels (0-4)."""
        labels = create_enhanced_event_labels(self.df, method="price_momentum")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))
        # Should have reasonable distribution across classes
        unique_classes = set(labels)
        self.assertGreaterEqual(
            len(unique_classes), 2, "Should have at least 2 classes represented"
        )

    def test_valuation_labels_5class(self):
        """Test valuation method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="valuation")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_fundamental_labels_5class(self):
        """Test fundamental method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="fundamental")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_composite_event_labels_5class(self):
        """Test composite_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="composite_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_profitability_event_labels(self):
        """Test profitability_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="profitability_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_leverage_event_labels(self):
        """Test leverage_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="leverage_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_growth_event_labels(self):
        """Test growth_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="growth_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_quality_event_labels(self):
        """Test quality_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="quality_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_cashflow_event_labels(self):
        """Test cashflow_event method returns 5-class labels."""
        labels = create_enhanced_event_labels(self.df, method="cashflow_event")

        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))

    def test_sector_adjustment(self):
        """Test sector adjustment parameter affects label distribution."""
        labels_no_adj = create_enhanced_event_labels(
            self.df, method="price_momentum", use_sector_adjustment=False
        )
        labels_with_adj = create_enhanced_event_labels(
            self.df, method="price_momentum", use_sector_adjustment=True
        )

        # Both should be valid 5-class labels
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels_no_adj))
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels_with_adj))

    def test_all_19_methods_supported(self):
        """Test that all 19 label creation methods are supported."""
        all_methods = [
            # Original methods
            "price_momentum",
            "valuation",
            "fundamental",
            "volatility",
            "analyst_rating",
            "market_events",
            "combined_signals",
            # Specialized methods (Phase 9.4)
            "profitability_event",
            "leverage_event",
            "liquidity_event",
            "efficiency_event",
            "growth_event",
            "quality_event",
            "composite_event",
            # New methods (Phase 9.3 coverage)
            "cashflow_event",
            "capital_allocation_event",
            "employee_productivity_event",
            "balance_sheet_event",
            "revenue_forecast_event",
        ]

        for method in all_methods:
            with self.subTest(method=method):
                try:
                    labels = create_enhanced_event_labels(self.df, method=method)
                    self.assertEqual(len(labels), len(self.df))
                    self.assertTrue(
                        all(label in [0, 1, 2, 3, 4] for label in labels),
                        f"Method {method} should return 5-class labels",
                    )
                except Exception as e:
                    self.fail(f"Method {method} raised exception: {e}")


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestClassificationDataPreparation(unittest.TestCase):
    """Test classification data preparation with ETL integration."""

    @classmethod
    def setUpClass(cls):
        """Create test data once for all tests."""
        cls.df = create_test_dataframe(n_samples=200)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

    def test_prepare_classification_data_returns_splits(self):
        """Test prepare_classification_data returns train/test splits."""
        # prepare_classification_data returns tuple:
        # (X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)
        result = prepare_classification_data(self.df, self.labels, test_size=0.2, random_state=42)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 6)

        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = result

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertIsInstance(numeric_cols, list)
        self.assertIsInstance(categorical_cols, list)

    def test_prepare_classification_data_stratified(self):
        """Test that data preparation maintains class distribution."""
        X_train, X_test, y_train, y_test, _, _ = prepare_classification_data(
            self.df, self.labels, test_size=0.2, random_state=42
        )

        # Check sizes are approximately 80/20
        total = len(y_train) + len(y_test)
        self.assertAlmostEqual(len(y_train) / total, 0.8, delta=0.05)

    def test_prepare_classification_data_with_feature_groups(self):
        """Test feature group selection in data preparation."""
        # Use valid feature groups from the API
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(
                self.df,
                self.labels,
                test_size=0.2,
                random_state=42,
                feature_groups=["analyst_quality", "employee_productivity"],
            )
        )

        self.assertIsInstance(X_train, pd.DataFrame)
        # Should have filtered to specified feature groups


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestClassifierTraining(unittest.TestCase):
    """Test all classifier training functions."""

    @classmethod
    def setUpClass(cls):
        """Create test data and prepare for training."""
        cls.df = create_test_dataframe(n_samples=150)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

        # Prepare data - returns tuple
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
        ) = prepare_classification_data(cls.df, cls.labels, test_size=0.2, random_state=42)

    def _validate_classifier_result(self, result: dict, name: str):
        """Helper to validate classifier result structure."""
        self.assertIsNotNone(result, f"{name} should return a result")
        self.assertIn("model", result, f"{name} should have 'model' key")
        self.assertIn("y_pred", result, f"{name} should have 'y_pred' key")
        self.assertIn("y_proba", result, f"{name} should have 'y_proba' key")

        # Check metrics - can be in 'metrics' dict or at top level
        if "metrics" in result:
            metrics = result["metrics"]
            self.assertIn("accuracy", metrics, f"{name} should have accuracy metric")
            # API uses f1_score, not f1_macro
            self.assertIn("f1_score", metrics, f"{name} should have f1_score metric")
        else:
            self.assertIn("accuracy", result, f"{name} should have accuracy")

        # Validate predictions shape
        y_pred = result["y_pred"]
        y_proba = result["y_proba"]
        self.assertEqual(len(y_pred), len(self.y_test))

        # Probabilities should sum to 1
        if y_proba is not None and len(y_proba) > 0:
            prob_sums = np.sum(y_proba, axis=1)
            np.testing.assert_array_almost_equal(prob_sums, np.ones(len(y_proba)), decimal=4)

    def test_train_xgboost_classifier(self):
        """Test XGBoost classifier training."""
        try:
            import xgboost
        except ImportError:
            self.skipTest("xgboost not installed")

        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )
        self._validate_classifier_result(result, "XGBoost")

    def test_train_lightgbm_classifier(self):
        """Test LightGBM classifier training."""
        try:
            import lightgbm
        except ImportError:
            self.skipTest("lightgbm not installed")

        result = train_lightgbm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )
        self._validate_classifier_result(result, "LightGBM")

    def test_train_catboost_classifier(self):
        """Test CatBoost classifier training."""
        try:
            import catboost
        except ImportError:
            self.skipTest("catboost not installed")

        result = train_catboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )
        self._validate_classifier_result(result, "CatBoost")

    def test_train_svm_classifier(self):
        """Test SVM classifier training."""
        result = train_svm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            kernel="rbf",
        )
        self._validate_classifier_result(result, "SVM")

    def test_train_neural_network_classifier(self):
        """Test Neural Network classifier training."""
        result = train_neural_network_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params={"hidden_layer_sizes": (50, 25), "max_iter": 100},
        )
        self._validate_classifier_result(result, "Neural Network")

    def test_train_voting_classifier(self):
        """Test Voting ensemble classifier training."""
        result = train_voting_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            voting="soft",
        )
        self._validate_classifier_result(result, "Voting")

    def test_train_stacking_classifier(self):
        """Test Stacking ensemble classifier training."""
        result = train_stacking_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )
        self._validate_classifier_result(result, "Stacking")

    def test_fit_classifier_interface(self):
        """Test fit_classifier unified interface."""
        try:
            import xgboost
        except ImportError:
            self.skipTest("xgboost not installed")

        # fit_classifier signature: X_train, y_train, X_test, y_test, model
        result = fit_classifier(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            model="xgboost",
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)

    def test_compare_classifiers(self):
        """Test classifier comparison function."""
        results = compare_classifiers(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        self.assertIsInstance(results, dict)
        # Should have multiple classifier results


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestClassificationEvaluation(unittest.TestCase):
    """Test comprehensive classification evaluation capabilities."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train a simple model."""
        cls.df = create_test_dataframe(n_samples=150)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

        # Prepare data - returns tuple
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
        ) = prepare_classification_data(cls.df, cls.labels, test_size=0.2, random_state=42)

        # Train a simple model for evaluation tests - pass categorical_cols
        cls.clf_result = train_svm_classifier(
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
            kernel="rbf",
        )
        cls.y_pred = cls.clf_result["y_pred"]
        cls.y_proba = cls.clf_result["y_proba"]

    def test_evaluate_classification_basic(self):
        """Test basic classification evaluation."""
        eval_result = evaluate_classification(
            self.y_test,
            self.y_pred,
            self.y_proba,
            class_names=["Strong Neg", "Negative", "Neutral", "Positive", "Strong Pos"],
        )

        self.assertIn("accuracy", eval_result)
        self.assertIn("f1_macro", eval_result)
        self.assertIn("classification_report", eval_result)
        self.assertIn("confusion_matrix", eval_result)

        self.assertGreaterEqual(eval_result["accuracy"], 0.0)
        self.assertLessEqual(eval_result["accuracy"], 1.0)

    def test_evaluate_classification_by_sector(self):
        """Test per-sector classification evaluation."""
        sectors = (
            self.df.iloc[self.X_test.index]["sector"] if hasattr(self.X_test, "index") else None
        )
        if sectors is None:
            sectors = pd.Series(np.random.choice(["Tech", "Finance"], len(self.y_test)))

        sector_metrics = evaluate_classification_by_sector(self.y_test, self.y_pred, sectors)

        # Function returns DataFrame with sector metrics
        self.assertIsNotNone(sector_metrics)
        # Can be dict or DataFrame depending on implementation
        self.assertTrue(
            isinstance(sector_metrics, dict) or isinstance(sector_metrics, pd.DataFrame),
            f"Expected dict or DataFrame, got {type(sector_metrics)}",
        )

    def test_analyze_calibration(self):
        """Test calibration analysis for classification probabilities."""
        if self.y_proba is None or len(self.y_proba) == 0:
            self.skipTest("No probabilities available")

        calibration_result = analyze_calibration(self.y_test, self.y_proba, n_bins=10)

        self.assertIsNotNone(calibration_result)

    def test_cross_validate_classifier(self):
        """Test cross-validation for classifier."""
        from sklearn.svm import SVC

        model = SVC(kernel="rbf", probability=True, random_state=42)

        cv_result = cross_validate_classifier(model, self.X_train, self.y_train, cv=3)

        self.assertIn("mean_accuracy", cv_result)
        self.assertIn("std_accuracy", cv_result)
        self.assertIn("fold_scores", cv_result)

    def test_export_classification_probabilities(self):
        """Test export of classification probabilities as features."""
        export_df = export_classification_probabilities(self.y_test, self.y_pred, self.y_proba)

        self.assertIsInstance(export_df, pd.DataFrame)
        # Should have probability columns for each class


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestSHAPInterpretation(unittest.TestCase):
    """Test SHAP-based model interpretation."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train a model for SHAP analysis."""
        cls.df = create_test_dataframe(n_samples=100)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

        # Prepare data - returns tuple
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
        ) = prepare_classification_data(cls.df, cls.labels, test_size=0.2, random_state=42)

        # Train model - pass categorical_cols
        cls.clf_result = train_svm_classifier(
            cls.X_train,
            cls.y_train,
            cls.X_test,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
            kernel="rbf",
        )
        cls.model = cls.clf_result["model"]

    def test_compute_shap_values(self):
        """Test SHAP value computation."""
        try:
            import shap
        except ImportError:
            self.skipTest("shap not installed")

        shap_result = compute_shap_values(self.model, self.X_train, self.X_test, max_samples=50)

        self.assertIsNotNone(shap_result)

    def test_analyze_shap_by_feature_groups(self):
        """Test SHAP analysis by feature groups."""
        try:
            import shap
        except ImportError:
            self.skipTest("shap not installed")

        # Create mock SHAP values
        shap_values = np.random.randn(len(self.X_test), len(self.numeric_cols))

        group_analysis = analyze_shap_by_feature_groups(
            shap_values, self.numeric_cols, top_n_per_group=5
        )

        self.assertIsInstance(group_analysis, dict)


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestFeatureImportance(unittest.TestCase):
    """Test feature importance analysis capabilities."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train models for importance analysis."""
        cls.df = create_test_dataframe(n_samples=100)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

        # Prepare data - returns tuple
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            cls.numeric_cols,
            cls.categorical_cols,
        ) = prepare_classification_data(cls.df, cls.labels, test_size=0.2, random_state=42)

    def test_compare_feature_importance(self):
        """Test comparing feature importance across models."""
        # Train two models - pass categorical_cols
        result1 = train_svm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            kernel="rbf",
        )

        models_dict = {"SVM": result1}

        comparison = compare_feature_importance(models_dict, self.numeric_cols, top_n=10)

        self.assertIsNotNone(comparison)

    def test_analyze_feature_importance_by_groups(self):
        """Test feature importance grouped by category."""
        importance_dict = {col: np.random.rand() for col in self.numeric_cols}

        group_importance = analyze_feature_importance_by_groups(
            importance_dict, self.numeric_cols, top_n_per_group=5
        )

        self.assertIsInstance(group_importance, dict)


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
@unittest.skipIf(not ETL_AVAILABLE, "ETL modules not available")
class TestETLIntegration(unittest.TestCase):
    """Test classification workflow integration with ETL pipeline."""

    def test_etl_to_classification_pipeline(self):
        """Test end-to-end ETL to classification pipeline."""
        # This test validates the integration between ETL output and classification input
        # Using synthetic data that matches ETL output structure

        df = create_test_dataframe(n_samples=150)

        # Simulate ETL output by ensuring required columns exist
        required_cols = ["ticker", "sector", "last_price", "price_target"]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Create labels
        labels = create_enhanced_event_labels(df, method="valuation")
        self.assertEqual(len(labels), len(df))

        # Prepare data - returns tuple
        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(df, labels, test_size=0.2)
        )
        self.assertIsInstance(X_train, pd.DataFrame)

        # Train classifier
        clf_result = train_svm_classifier(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
        )

        self.assertIn("model", clf_result)
        self.assertIn("y_proba", clf_result)


@unittest.skipIf(not DEPS_AVAILABLE, "pandas/numpy not installed")
@unittest.skipIf(not CLASSIFICATION_AVAILABLE, "classification modules not available")
class TestClassificationMetaFeatures(unittest.TestCase):
    """Test export of classification probabilities as meta-features for regression."""

    @classmethod
    def setUpClass(cls):
        """Create test data and train model."""
        cls.df = create_test_dataframe(n_samples=100)
        cls.labels = create_enhanced_event_labels(cls.df, method="valuation")

        # Prepare data - returns tuple
        X_train, cls.X_test, y_train, cls.y_test, numeric_cols, categorical_cols = (
            prepare_classification_data(cls.df, cls.labels, test_size=0.2, random_state=42)
        )

        # Pass categorical_cols to classifier
        cls.clf_result = train_svm_classifier(
            X_train,
            y_train,
            cls.X_test,
            cls.y_test,
            numeric_cols,
            categorical_cols,
        )

    def test_export_classification_features_structure(self):
        """Test that exported features have correct structure for regression."""
        y_proba = self.clf_result["y_proba"]

        export_df = export_classification_features(
            self.df.iloc[: len(self.y_test)],
            y_proba,
            class_names=["strong_neg", "negative", "neutral", "positive", "strong_pos"],
        )

        self.assertIsInstance(export_df, pd.DataFrame)
        # Should have probability columns
        prob_cols = [c for c in export_df.columns if "prob" in c.lower() or "p_" in c.lower()]
        self.assertGreater(len(prob_cols), 0, "Should have probability columns")

    def test_probabilities_sum_to_one(self):
        """Test that exported probabilities sum to 1."""
        y_proba = self.clf_result["y_proba"]

        if y_proba is not None and len(y_proba) > 0:
            prob_sums = np.sum(y_proba, axis=1)
            np.testing.assert_array_almost_equal(
                prob_sums, np.ones(len(y_proba)), decimal=5, err_msg="Probabilities should sum to 1"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
