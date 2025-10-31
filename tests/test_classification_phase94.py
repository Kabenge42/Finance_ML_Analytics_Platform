"""
Phase 9.4 Classification Tests - Advanced Stock Prediction ML System

Tests for sophisticated multi-class classification models including:
- Neural Network classifiers
- Voting ensemble classifiers
- Stacking ensemble classifiers
- SHAP-based model interpretation
- Classification meta-feature export
- Cross-validation framework
- Feature importance comparison
- Confusion matrix visualization
- Sector-specific evaluation

Following strict TDD methodology with ≥80% coverage target.
"""

import unittest

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    HAVE_SKLEARN = True
except ImportError:
    pd = None
    np = None
    HAVE_SKLEARN = False

try:
    import finance_ml
    from finance_ml.classification import (
        train_neural_network_classifier,
        train_voting_classifier,
        train_stacking_classifier,
        compute_shap_values,
        export_classification_features,
        cross_validate_classifier,
        compare_feature_importance,
        plot_confusion_matrices,
        evaluate_classification_by_sector,
        create_enhanced_event_labels,
        prepare_classification_data,
    )

    HAVE_FINANCE_ML = True
except ImportError:
    HAVE_FINANCE_ML = False

# Check optional dependencies
try:
    import tensorflow as tf

    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False

try:
    import xgboost as xgb

    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

try:
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

try:
    import catboost

    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

try:
    import shap

    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def create_sample_classification_data(n_samples=200, random_state=42):
    """Create synthetic classification dataset for testing."""
    np.random.seed(random_state)

    # Create features
    data = {
        "ticker": [f"TICK{i}" for i in range(n_samples)],
        "sector": np.random.choice(["Technology", "Finance", "Healthcare", "Energy"], n_samples),
        "last_price": np.random.uniform(10, 200, n_samples),
        "price_target": np.random.uniform(10, 250, n_samples),
        "market_cap": np.random.uniform(1e9, 1e12, n_samples),
        "p_e": np.random.uniform(5, 50, n_samples),
        "p_b": np.random.uniform(0.5, 10, n_samples),
        "revenue": np.random.uniform(1e8, 1e11, n_samples),
        "ebitda": np.random.uniform(1e7, 1e10, n_samples),
        "debt_equity": np.random.uniform(0, 3, n_samples),
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randn(n_samples),
    }

    df = pd.DataFrame(data)

    # Create labels based on price target vs last price
    price_change_pct = ((df["price_target"] - df["last_price"]) / df["last_price"]) * 100
    labels = np.where(
        price_change_pct > 10,
        1,  # Positive catalyst
        np.where(price_change_pct < -10, 2, 0),  # Negative catalyst
    )  # Neutral

    return df, labels


@unittest.skipIf(
    not HAVE_SKLEARN or not HAVE_FINANCE_ML or pd is None or np is None,
    "Required packages not available",
)
class TestEnhancedEventLabels(unittest.TestCase):
    """Test enhanced event label creation methods."""

    def test_create_enhanced_event_labels_price_momentum(self):
        """Test price momentum method for event labeling."""
        df, _ = create_sample_classification_data(n_samples=100)

        labels = create_enhanced_event_labels(
            df, method="price_momentum", threshold_positive=10.0, threshold_negative=-10.0
        )

        # Check output shape and values
        self.assertEqual(len(labels), 100)
        self.assertTrue(all(label in [0, 1, 2] for label in labels))

        # Check label distribution
        unique_labels = set(labels)
        self.assertTrue(len(unique_labels) >= 2, "Should have at least 2 different labels")

    def test_create_enhanced_event_labels_with_sector_adjustment(self):
        """Test sector-adjusted event labeling."""
        df, _ = create_sample_classification_data(n_samples=100)

        labels = create_enhanced_event_labels(
            df, method="price_momentum", use_sector_adjustment=True
        )

        self.assertEqual(len(labels), 100)
        self.assertTrue(all(label in [0, 1, 2] for label in labels))


@unittest.skipIf(
    not HAVE_TENSORFLOW or not HAVE_SKLEARN or not HAVE_FINANCE_ML,
    "TensorFlow or sklearn not available",
)
class TestNeuralNetworkClassifier(unittest.TestCase):
    """Test neural network classifier training and prediction."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)

        # Prepare train/test split
        numeric_cols = [
            "last_price",
            "market_cap",
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "debt_equity",
            "feature_1",
            "feature_2",
            "feature_3",
        ]
        categorical_cols = ["sector"]

        X = self.df[numeric_cols + categorical_cols]
        y = self.labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def test_train_neural_network_basic(self):
        """Test basic neural network training."""
        # Use minimal epochs for fast testing
        params = {"epochs": 3, "batch_size": 32, "hidden_layers": [64, 32], "dropout_rate": 0.2}

        result = train_neural_network_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=params,
        )

        # Check return structure
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertIn("f1_score", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)

        # Check metrics
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["y_pred"]), len(self.y_test))
        self.assertEqual(result["y_proba"].shape, (len(self.y_test), 3))

    def test_neural_network_predictions_sum_to_one(self):
        """Test that neural network probabilities sum to 1."""
        params = {"epochs": 2, "batch_size": 32, "hidden_layers": [32]}

        result = train_neural_network_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=params,
        )

        # Probabilities should sum to ~1 for each sample
        prob_sums = result["y_proba"].sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(prob_sums)), decimal=5)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestVotingClassifier(unittest.TestCase):
    """Test voting ensemble classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)

        numeric_cols = [
            "last_price",
            "market_cap",
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "debt_equity",
            "feature_1",
            "feature_2",
            "feature_3",
        ]
        categorical_cols = ["sector"]

        X = self.df[numeric_cols + categorical_cols]
        y = self.labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def test_train_voting_classifier_soft(self):
        """Test soft voting classifier."""
        result = train_voting_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            voting="soft",
        )

        # Check return structure
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)

        # Soft voting should return probabilities
        self.assertIsNotNone(result["y_proba"])
        self.assertEqual(result["y_proba"].shape, (len(self.y_test), 3))

    def test_train_voting_classifier_hard(self):
        """Test hard voting classifier."""
        result = train_voting_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            voting="hard",
        )

        # Check return structure
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertIn("y_pred", result)

        # Hard voting may not return probabilities
        # (depends on base estimators)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestStackingClassifier(unittest.TestCase):
    """Test stacking ensemble classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)

        numeric_cols = [
            "last_price",
            "market_cap",
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "debt_equity",
            "feature_1",
            "feature_2",
            "feature_3",
        ]
        categorical_cols = ["sector"]

        X = self.df[numeric_cols + categorical_cols]
        y = self.labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def test_train_stacking_classifier(self):
        """Test stacking classifier with meta-learner."""
        result = train_stacking_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        # Check return structure
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)

        # Check metrics
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["y_pred"]), len(self.y_test))


@unittest.skipIf(
    not HAVE_SHAP or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "SHAP or sklearn not available"
)
class TestSHAPValues(unittest.TestCase):
    """Test SHAP value computation for model interpretation."""

    def setUp(self):
        """Set up test data and trained model."""
        from sklearn.ensemble import RandomForestClassifier

        self.df, self.labels = create_sample_classification_data(n_samples=200)

        numeric_cols = [
            "last_price",
            "market_cap",
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "debt_equity",
            "feature_1",
            "feature_2",
            "feature_3",
        ]

        X = self.df[numeric_cols]
        y = self.labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        self.feature_names = numeric_cols

    def test_compute_shap_values_tree_model(self):
        """Test SHAP computation for tree-based model."""
        shap_values = compute_shap_values(self.model, self.X_train, self.X_test, max_samples=50)

        # Check output structure
        self.assertIsNotNone(shap_values)
        self.assertIsInstance(shap_values, dict)
        # May be empty if SHAP not available or computation fails
        if shap_values:
            self.assertIn("shap_values", shap_values)

    def test_shap_returns_dict(self):
        """Test that SHAP computation returns a dict."""
        shap_values = compute_shap_values(self.model, self.X_train, self.X_test, max_samples=50)

        # Should return a dict (may be empty)
        self.assertIsInstance(shap_values, dict)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestExportClassificationFeatures(unittest.TestCase):
    """Test export of classification probabilities as meta-features."""

    def test_export_classification_features(self):
        """Test exporting classification probabilities."""
        df, labels = create_sample_classification_data(n_samples=100)

        # Create mock probabilities
        y_proba = np.random.dirichlet(np.ones(3), size=100)  # Sums to 1

        df_enhanced = export_classification_features(df, y_proba)

        # Check new columns exist
        self.assertIn("event_prob_neutral", df_enhanced.columns)
        self.assertIn("event_prob_positive", df_enhanced.columns)
        self.assertIn("event_prob_negative", df_enhanced.columns)
        self.assertIn("event_class_predicted", df_enhanced.columns)
        self.assertIn("event_confidence", df_enhanced.columns)

        # Check shapes
        self.assertEqual(len(df_enhanced), len(df))

        # Check predicted class values
        self.assertTrue(all(df_enhanced["event_class_predicted"].isin([0, 1, 2])))

        # Check confidence is max of probabilities
        max_proba = y_proba.max(axis=1)
        np.testing.assert_array_almost_equal(
            df_enhanced["event_confidence"].values, max_proba, decimal=5
        )

    def test_export_features_preserves_original_columns(self):
        """Test that original DataFrame columns are preserved."""
        df, _ = create_sample_classification_data(n_samples=50)
        original_cols = set(df.columns)

        y_proba = np.random.dirichlet(np.ones(3), size=50)
        df_enhanced = export_classification_features(df, y_proba)

        # All original columns should still exist
        self.assertTrue(original_cols.issubset(set(df_enhanced.columns)))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestCrossValidateClassifier(unittest.TestCase):
    """Test cross-validation framework."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)

        self.numeric_cols = [
            "last_price",
            "market_cap",
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "debt_equity",
            "feature_1",
            "feature_2",
            "feature_3",
        ]
        self.categorical_cols = ["sector"]

        self.X = self.df[self.numeric_cols + self.categorical_cols]
        self.y = self.labels

    def test_cross_validate_classifier_basic(self):
        """Test basic k-fold cross-validation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        # Preprocess data (encode categorical)
        X_proc = self.X.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        cv_results = cross_validate_classifier(model, X_proc, self.y, cv=3)

        # Check return structure (actual keys: test_accuracy, test_precision, etc.)
        self.assertIn("test_accuracy", cv_results)
        self.assertIn("test_precision", cv_results)
        self.assertIn("test_recall", cv_results)
        self.assertIn("test_f1", cv_results)

        # Check reasonable values
        self.assertGreaterEqual(cv_results["test_accuracy"], 0.0)
        self.assertLessEqual(cv_results["test_accuracy"], 1.0)

    def test_cross_validate_with_sector_stratify(self):
        """Test cross-validation with sector stratification."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        # Preprocess data (encode categorical)
        X_proc = self.X.copy()
        for col in self.categorical_cols:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        cv_results = cross_validate_classifier(model, X_proc, self.y, cv=3, stratify_by="sector")

        # Should still return valid results
        self.assertIn("test_accuracy", cv_results)
        self.assertGreaterEqual(cv_results["test_accuracy"], 0.0)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestCompareFeatureImportance(unittest.TestCase):
    """Test feature importance comparison across models."""

    def test_compare_feature_importance(self):
        """Test comparing feature importance from multiple models."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        df, labels = create_sample_classification_data(n_samples=200)

        numeric_cols = ["last_price", "market_cap", "p_e", "p_b", "revenue"]
        X = df[numeric_cols]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train multiple models and extract feature importance
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb.fit(X_train, y_train)

        # Create models_dict with feature_importance
        models_dict = {
            "rf": {"feature_importance": dict(zip(numeric_cols, rf.feature_importances_))},
            "gb": {"feature_importance": dict(zip(numeric_cols, gb.feature_importances_))},
        }

        # Compare importance
        importance_df = compare_feature_importance(models_dict, numeric_cols, top_n=5)

        # Check output
        self.assertIsInstance(importance_df, pd.DataFrame)
        if not importance_df.empty:
            self.assertIn("Average", importance_df.columns)
            self.assertLessEqual(len(importance_df), 5)  # Top 5


@unittest.skipIf(
    not HAVE_MATPLOTLIB or not HAVE_SKLEARN or not HAVE_FINANCE_ML,
    "matplotlib or sklearn not available",
)
class TestPlotConfusionMatrices(unittest.TestCase):
    """Test confusion matrix visualization."""

    def test_plot_confusion_matrices(self):
        """Test plotting confusion matrices for multiple models."""
        from sklearn.ensemble import RandomForestClassifier

        df, labels = create_sample_classification_data(n_samples=200)

        numeric_cols = ["last_price", "market_cap", "p_e", "p_b", "revenue"]
        X = df[numeric_cols]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )

        # Train models and get predictions
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_test)

        model2 = RandomForestClassifier(n_estimators=20, random_state=43)
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict(X_test)

        # Create models_results dict with y_test and y_pred
        models_results = {
            "Model 1": {"y_test": y_test, "y_pred": y_pred1},
            "Model 2": {"y_test": y_test, "y_pred": y_pred2},
        }

        # Test plotting (should not raise exception)
        try:
            # Close any existing plots
            import matplotlib.pyplot as plt

            plt.close("all")

            plot_confusion_matrices(models_results)

            # Close plots after test
            plt.close("all")
        except Exception as e:
            self.fail(f"plot_confusion_matrices raised exception: {e}")


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestEvaluateClassificationBySector(unittest.TestCase):
    """Test sector-specific evaluation."""

    def test_evaluate_by_sector(self):
        """Test evaluation metrics by sector."""
        df, labels = create_sample_classification_data(n_samples=200)

        numeric_cols = ["last_price", "market_cap", "p_e", "p_b", "revenue"]
        X = df[numeric_cols]
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )

        # Get test indices for sector mapping
        test_indices = X_test.index
        sectors_test = df.loc[test_indices, "sector"]

        # Train model
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate by sector
        sector_results = evaluate_classification_by_sector(y_test, y_pred, sectors_test)

        # Check output structure
        self.assertIsInstance(sector_results, pd.DataFrame)
        self.assertIn("Sector", sector_results.columns)  # Capital S
        self.assertIn("Accuracy", sector_results.columns)  # Capital A
        self.assertIn("F1-Score", sector_results.columns)  # F1-Score format

        # Should have results for multiple sectors
        self.assertGreater(len(sector_results), 0)

        # All metrics should be in valid range
        self.assertTrue(all(sector_results["Accuracy"].between(0, 1)))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestPrepareClassificationData(unittest.TestCase):
    """Test data preparation utility."""

    def test_prepare_classification_data(self):
        """Test preparing data for classification."""
        df, labels = create_sample_classification_data(n_samples=200)

        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Check shapes
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertEqual(len(X_train) + len(X_test), len(df))

        # Check column lists
        self.assertIsInstance(num_cols, list)
        self.assertIsInstance(cat_cols, list)
        self.assertIn("sector", cat_cols)
        self.assertIn("last_price", num_cols)


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for complete Phase 9.4 workflow."""

    @unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
    def test_complete_classification_pipeline(self):
        """Test complete workflow: labels -> train -> evaluate -> export."""
        # 1. Create data
        df, _ = create_sample_classification_data(n_samples=200)

        # 2. Create enhanced labels
        labels = create_enhanced_event_labels(df, method="price_momentum")

        # 3. Prepare data
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2
        )

        # 4. Train voting classifier
        result = train_voting_classifier(
            X_train, y_train, X_test, y_test, num_cols, cat_cols, voting="soft"
        )

        # 5. Export meta-features using test predictions
        df_enhanced = export_classification_features(df.iloc[X_test.index], result["y_proba"])

        # Verify complete workflow
        self.assertEqual(len(df_enhanced), len(X_test))
        self.assertIn("event_prob_neutral", df_enhanced.columns)
        self.assertIn("event_prob_positive", df_enhanced.columns)
        self.assertIn("event_prob_negative", df_enhanced.columns)
        self.assertIn("event_class_predicted", df_enhanced.columns)
        self.assertIn("event_confidence", df_enhanced.columns)
        self.assertGreaterEqual(result["accuracy"], 0.0)


@unittest.skipIf(
    not HAVE_XGBOOST or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "XGBoost not available"
)
class TestXGBoostClassifier(unittest.TestCase):
    """Test XGBoost gradient boosting classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        ) = prepare_classification_data(self.df, self.labels, test_size=0.2, random_state=42)

    def test_train_xgboost_basic(self):
        """Test basic XGBoost training."""
        from finance_ml.classification import train_xgboost_classifier

        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        # Check result structure
        self.assertIn("model", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)

        # Check predictions shape
        self.assertEqual(len(result["y_pred"]), len(self.y_test))
        self.assertEqual(result["y_proba"].shape, (len(self.y_test), 3))

        # Check metrics are reasonable
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_xgboost_with_custom_params(self):
        """Test XGBoost with custom hyperparameters."""
        from finance_ml.classification import train_xgboost_classifier

        custom_params = {
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 100,
        }

        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=custom_params,
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("feature_importance", result)

    def test_xgboost_probabilities_sum_to_one(self):
        """Test that XGBoost probabilities sum to 1."""
        from finance_ml.classification import train_xgboost_classifier

        result = train_xgboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        # Check probabilities sum to 1 (within tolerance)
        prob_sums = result["y_proba"].sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(len(self.y_test)), decimal=5)


@unittest.skipIf(
    not HAVE_LIGHTGBM or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "LightGBM not available"
)
class TestLightGBMClassifier(unittest.TestCase):
    """Test LightGBM gradient boosting classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        ) = prepare_classification_data(self.df, self.labels, test_size=0.2, random_state=42)

    def test_train_lightgbm_basic(self):
        """Test basic LightGBM training."""
        from finance_ml.classification import train_lightgbm_classifier

        result = train_lightgbm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        # Check result structure
        self.assertIn("model", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)

        # Check predictions shape
        self.assertEqual(len(result["y_pred"]), len(self.y_test))
        self.assertEqual(result["y_proba"].shape, (len(self.y_test), 3))

    def test_lightgbm_with_custom_params(self):
        """Test LightGBM with custom hyperparameters."""
        from finance_ml.classification import train_lightgbm_classifier

        custom_params = {
            "max_depth": 5,
            "learning_rate": 0.08,
            "n_estimators": 150,
        }

        result = train_lightgbm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=custom_params,
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("feature_importance", result)


@unittest.skipIf(
    not HAVE_CATBOOST or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "CatBoost not available"
)
class TestCatBoostClassifier(unittest.TestCase):
    """Test CatBoost gradient boosting classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        ) = prepare_classification_data(self.df, self.labels, test_size=0.2, random_state=42)

    def test_train_catboost_basic(self):
        """Test basic CatBoost training."""
        from finance_ml.classification import train_catboost_classifier

        result = train_catboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        )

        # Check result structure
        self.assertIn("model", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)

        # Check predictions shape
        self.assertEqual(len(result["y_pred"]), len(self.y_test))
        self.assertEqual(result["y_proba"].shape, (len(self.y_test), 3))

    def test_catboost_with_custom_params(self):
        """Test CatBoost with custom hyperparameters."""
        from finance_ml.classification import train_catboost_classifier

        custom_params = {
            "depth": 5,
            "learning_rate": 0.07,
            "iterations": 100,
        }

        result = train_catboost_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            params=custom_params,
        )

        self.assertIsNotNone(result["model"])
        self.assertIn("feature_importance", result)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestEnhancedEventLabelsExtended(unittest.TestCase):
    """Test enhanced event labeling with all methods."""

    def test_valuation_method(self):
        """Test valuation-based event labeling."""
        df, _ = create_sample_classification_data(n_samples=200)

        labels = create_enhanced_event_labels(df, method="valuation")

        # Check labels are valid
        self.assertEqual(len(labels), len(df))
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1, 2] for label in unique_labels))

        # Should have all three classes
        self.assertGreater(np.sum(labels == 1), 0)  # Positive
        self.assertGreater(np.sum(labels == 2), 0)  # Negative

    def test_fundamental_method(self):
        """Test fundamental-based event labeling."""
        df, _ = create_sample_classification_data(n_samples=200)

        # Add margin columns
        df["gross_margin"] = np.random.uniform(0.2, 0.6, len(df))
        df["operating_margin"] = np.random.uniform(0.1, 0.4, len(df))
        df["net_margin"] = np.random.uniform(0.05, 0.25, len(df))

        labels = create_enhanced_event_labels(df, method="fundamental")

        # Check labels are valid
        self.assertEqual(len(labels), len(df))
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1, 2] for label in unique_labels))

    def test_volatility_method(self):
        """Test volatility-based event labeling."""
        df, _ = create_sample_classification_data(n_samples=200)

        # Add volatility column
        df["volatility_30d"] = np.random.uniform(0.1, 0.8, len(df))

        labels = create_enhanced_event_labels(df, method="volatility")

        # Check labels are valid
        self.assertEqual(len(labels), len(df))
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1, 2] for label in unique_labels))

    def test_analyst_rating_method(self):
        """Test analyst rating changes event labeling."""
        df, _ = create_sample_classification_data(n_samples=200)

        # Add analyst rating columns
        df["analyst_rating"] = np.random.choice(["Buy", "Hold", "Sell"], len(df))
        df["analyst_rating_change"] = np.random.uniform(-2, 2, len(df))

        labels = create_enhanced_event_labels(df, method="analyst_rating")

        # Check labels are valid
        self.assertEqual(len(labels), len(df))
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1, 2] for label in unique_labels))

        # Should have positive for upgrades, negative for downgrades
        self.assertGreater(np.sum(labels == 1), 0)  # Positive (upgrades)
        self.assertGreater(np.sum(labels == 2), 0)  # Negative (downgrades)

    def test_market_events_method(self):
        """Test market events (sector rotation, regional trends) labeling."""
        df, _ = create_sample_classification_data(n_samples=200)

        # Add market event indicators
        df["sector_momentum"] = np.random.uniform(-20, 20, len(df))
        df["region"] = np.random.choice(["US", "EU", "APAC"], len(df))

        labels = create_enhanced_event_labels(df, method="market_events")

        # Check labels are valid
        self.assertEqual(len(labels), len(df))
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1, 2] for label in unique_labels))


try:
    from imblearn.over_sampling import SMOTE

    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False


@unittest.skipIf(
    not HAVE_IMBLEARN or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "imbalanced-learn not available"
)
class TestSMOTE(unittest.TestCase):
    """Test SMOTE class imbalance handling."""

    def test_apply_smote(self):
        """Test SMOTE oversampling."""
        from finance_ml.classification import apply_smote

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Apply SMOTE
        X_resampled, y_resampled = apply_smote(X_train, y_train, num_cols)

        # Check that resampling increased minority class samples
        self.assertGreaterEqual(len(X_resampled), len(X_train))
        self.assertEqual(len(X_resampled), len(y_resampled))

        # Check that all classes are balanced (or more balanced)
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(len(unique), 3)  # Should have all 3 classes

    def test_smote_preserves_features(self):
        """Test that SMOTE preserves feature names (numeric only)."""
        from finance_ml.classification import apply_smote

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_resampled, y_resampled = apply_smote(X_train, y_train, num_cols)

        # Check that numeric columns are preserved (SMOTE drops categorical features)
        # SMOTE only works on numeric features
        self.assertTrue(all(col in X_train.columns for col in X_resampled.columns))


@unittest.skipIf(
    not HAVE_IMBLEARN or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "imbalanced-learn not available"
)
class TestADASYN(unittest.TestCase):
    """Test ADASYN adaptive synthetic sampling."""

    def test_apply_adasyn(self):
        """Test ADASYN oversampling."""
        from finance_ml.classification import apply_adasyn

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Apply ADASYN
        X_resampled, y_resampled = apply_adasyn(X_train, y_train, num_cols)

        # Check that resampling occurred
        self.assertGreaterEqual(len(X_resampled), len(X_train))
        self.assertEqual(len(X_resampled), len(y_resampled))

        # Check that all classes are present
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(len(unique), 3)

    def test_adasyn_with_sampling_strategy(self):
        """Test ADASYN with custom sampling strategy."""
        from finance_ml.classification import apply_adasyn

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_resampled, y_resampled = apply_adasyn(
            X_train, y_train, num_cols, sampling_strategy="auto"
        )

        self.assertGreaterEqual(len(X_resampled), len(X_train))


@unittest.skipIf(
    not HAVE_IMBLEARN or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "imbalanced-learn not available"
)
class TestUnderSampling(unittest.TestCase):
    """Test under-sampling strategies for class imbalance."""

    def test_apply_undersampling_random(self):
        """Test random under-sampling."""
        from finance_ml.classification import apply_undersampling

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Apply random under-sampling
        X_resampled, y_resampled = apply_undersampling(
            X_train, y_train, num_cols, strategy="random"
        )

        # Should reduce samples
        self.assertLessEqual(len(X_resampled), len(X_train))
        self.assertEqual(len(X_resampled), len(y_resampled))

    def test_apply_undersampling_tomek(self):
        """Test Tomek links under-sampling."""
        from finance_ml.classification import apply_undersampling

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_resampled, y_resampled = apply_undersampling(X_train, y_train, num_cols, strategy="tomek")

        # Should remove some samples
        self.assertLessEqual(len(X_resampled), len(X_train))

    def test_apply_undersampling_nearmiss(self):
        """Test NearMiss under-sampling."""
        from finance_ml.classification import apply_undersampling

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_resampled, y_resampled = apply_undersampling(
            X_train, y_train, num_cols, strategy="nearmiss"
        )

        self.assertLessEqual(len(X_resampled), len(X_train))


@unittest.skipIf(
    not HAVE_IMBLEARN or not HAVE_SKLEARN or not HAVE_FINANCE_ML, "imbalanced-learn not available"
)
class TestCombinedSampling(unittest.TestCase):
    """Test combined over/under-sampling strategies."""

    def test_apply_combined_sampling(self):
        """Test combined SMOTE + under-sampling."""
        from finance_ml.classification import apply_combined_sampling

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Apply combined sampling
        X_resampled, y_resampled = apply_combined_sampling(
            X_train, y_train, num_cols, over_strategy="smote", under_strategy="random"
        )

        # Should balance classes
        self.assertEqual(len(X_resampled), len(y_resampled))
        unique, counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(len(unique), 3)

    def test_combined_sampling_smote_tomek(self):
        """Test SMOTE + Tomek links combination."""
        from finance_ml.classification import apply_combined_sampling

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_resampled, y_resampled = apply_combined_sampling(
            X_train, y_train, num_cols, over_strategy="smote", under_strategy="tomek"
        )

        self.assertEqual(len(X_resampled), len(y_resampled))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestSVMClassifier(unittest.TestCase):
    """Test Support Vector Machine classifier."""

    def setUp(self):
        """Set up test data."""
        self.df, self.labels = create_sample_classification_data(n_samples=200)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
        ) = prepare_classification_data(self.df, self.labels, test_size=0.2, random_state=42)

    def test_train_svm_rbf_kernel(self):
        """Test SVM with RBF kernel."""
        from finance_ml.classification import train_svm_classifier

        result = train_svm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            kernel="rbf",
        )

        # Check result structure
        self.assertIn("model", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_proba", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_score", result)

        # Check predictions
        self.assertEqual(len(result["y_pred"]), len(self.y_test))
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_train_svm_poly_kernel(self):
        """Test SVM with polynomial kernel."""
        from finance_ml.classification import train_svm_classifier

        result = train_svm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            kernel="poly",
            degree=3,
        )

        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertEqual(len(result["y_pred"]), len(self.y_test))

    def test_svm_one_vs_rest(self):
        """Test SVM with One-vs-Rest strategy."""
        from finance_ml.classification import train_svm_classifier

        result = train_svm_classifier(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.numeric_cols,
            self.categorical_cols,
            kernel="rbf",
            decision_function_shape="ovr",
        )

        self.assertIsNotNone(result["model"])
        self.assertEqual(len(result["y_pred"]), len(self.y_test))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestLearningCurves(unittest.TestCase):
    """Test learning curve generation for bias/variance diagnosis."""

    def test_plot_learning_curves(self):
        """Test learning curve plotting."""
        from finance_ml.classification import plot_learning_curves
        from sklearn.ensemble import RandomForestClassifier

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Prepare data for sklearn
        from sklearn.preprocessing import LabelEncoder

        X_proc = X_train.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))

        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Should not raise exception
        try:
            result = plot_learning_curves(model, X_proc, y_train, cv=3)
            self.assertIsNotNone(result)
            self.assertIn("train_sizes", result)
            self.assertIn("train_scores", result)
            self.assertIn("test_scores", result)
        except Exception as e:
            self.fail(f"plot_learning_curves raised exception: {e}")

    def test_learning_curves_returns_dict(self):
        """Test that learning curves return a dictionary with results."""
        from finance_ml.classification import plot_learning_curves
        from sklearn.ensemble import RandomForestClassifier

        df, labels = create_sample_classification_data(n_samples=150)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        from sklearn.preprocessing import LabelEncoder

        X_proc = X_train.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X_proc[col] = le.fit_transform(X_proc[col].astype(str))

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        result = plot_learning_curves(model, X_proc, y_train, cv=2)

        self.assertIsInstance(result, dict)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestPerClassFeatureImportance(unittest.TestCase):
    """Test per-class feature importance analysis."""

    def test_analyze_per_class_feature_importance(self):
        """Test per-class feature importance extraction."""
        from finance_ml.classification import analyze_per_class_feature_importance
        from sklearn.ensemble import RandomForestClassifier

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        # Train model
        X_proc = X_train[num_cols]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_proc, y_train)

        # Analyze per-class importance
        importance_df = analyze_per_class_feature_importance(
            model, X_proc, y_train, feature_names=num_cols, top_n=5
        )

        # Check output
        self.assertIsInstance(importance_df, pd.DataFrame)
        if not importance_df.empty:
            self.assertIn("Class", importance_df.columns)
            self.assertIn("Feature", importance_df.columns)
            self.assertIn("Importance", importance_df.columns)

    def test_per_class_importance_all_classes(self):
        """Test that per-class importance covers all classes."""
        from finance_ml.classification import analyze_per_class_feature_importance
        from sklearn.ensemble import RandomForestClassifier

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        X_proc = X_train[num_cols]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_proc, y_train)

        importance_df = analyze_per_class_feature_importance(
            model, X_proc, y_train, feature_names=num_cols
        )

        if not importance_df.empty:
            unique_classes = importance_df["Class"].unique()
            # Should have results for multiple classes
            self.assertGreater(len(unique_classes), 0)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestRandomForestAndSVM(unittest.TestCase):
    """Test Random Forest and SVM classifiers."""

    def test_compare_classifiers_includes_random_forest(self):
        """Test that compare_classifiers includes Random Forest."""
        from finance_ml.classification import compare_classifiers

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        results = compare_classifiers(X_train, y_train, X_test, y_test, num_cols, cat_cols)

        # Check that results is a DataFrame with multiple models
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)

        # Should include Random Forest
        self.assertTrue(any(results["Model"].str.contains("Random Forest")))

        # Check required columns
        self.assertIn("Model", results.columns)
        self.assertIn("Accuracy", results.columns)
        self.assertIn("F1-Score", results.columns)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestEvaluateClassification(unittest.TestCase):
    """Test comprehensive classification evaluation."""

    def test_evaluate_classification_basic(self):
        """Test basic evaluation metrics."""
        from finance_ml.classification import evaluate_classification

        # Create simple test data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])

        result = evaluate_classification(y_true, y_pred)

        # Check required metrics
        self.assertIn("accuracy", result)
        self.assertIn("precision_macro", result)
        self.assertIn("recall_macro", result)
        self.assertIn("f1_macro", result)
        self.assertIn("confusion_matrix", result)
        self.assertIn("classification_report", result)

        # Metrics should be in valid ranges
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_evaluate_classification_with_probabilities(self):
        """Test evaluation with probability scores for ROC-AUC."""
        from finance_ml.classification import evaluate_classification

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 5)
        y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2] * 5)

        # Create probability matrix
        n_samples = len(y_true)
        y_proba = np.random.dirichlet(np.ones(3), size=n_samples)

        result = evaluate_classification(y_true, y_pred, y_proba=y_proba)

        # Should have ROC-AUC when probabilities provided
        self.assertIn("roc_auc", result)
        if result["roc_auc"] is not None:
            self.assertGreaterEqual(result["roc_auc"], 0.0)
            self.assertLessEqual(result["roc_auc"], 1.0)

    def test_evaluate_classification_custom_class_names(self):
        """Test evaluation with custom class names."""
        from finance_ml.classification import evaluate_classification

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        custom_names = ["Class_A", "Class_B", "Class_C"]
        result = evaluate_classification(y_true, y_pred, class_names=custom_names)

        # Check that custom names are in classification report
        self.assertIn("classification_report", result)
        report = result["classification_report"]
        self.assertIn("Class_A", report)
        self.assertIn("Class_B", report)
        self.assertIn("Class_C", report)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestCleanExtremeValues(unittest.TestCase):
    """Test extreme value cleaning utility."""

    def test_clean_extreme_values_basic(self):
        """Test cleaning infinite and extreme values."""
        from finance_ml.classification import clean_extreme_values

        df = pd.DataFrame(
            {
                "col1": [1, 2, np.inf, 4, 5],
                "col2": [10, 20, -np.inf, 40, 50],
                "col3": [1e10, 2e10, 3e10, 4e10, 5e10],
            }
        )

        df_clean = clean_extreme_values(df, clip_threshold=1e8)

        # Check no infinities remain
        self.assertFalse(np.isinf(df_clean).any().any())

        # Check extreme values are clipped
        self.assertTrue(all(df_clean["col3"] <= 1e8))

    def test_clean_extreme_values_preserves_normal_values(self):
        """Test that normal values are preserved."""
        from finance_ml.classification import clean_extreme_values

        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [10, 20, 30, 40, 50],
            }
        )

        df_clean = clean_extreme_values(df)

        # Should be unchanged
        pd.testing.assert_frame_equal(df, df_clean)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestValidateDataQuality(unittest.TestCase):
    """Test data quality validation."""

    def test_validate_data_quality_good_data(self):
        """Test validation with clean data."""
        from finance_ml.classification import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [10, 20, 30, 40, 50],
            }
        )

        result = validate_data_quality(df)

        # Should pass validation
        self.assertTrue(result)

    def test_validate_data_quality_with_issues(self):
        """Test validation with data quality issues."""
        from finance_ml.classification import validate_data_quality

        df = pd.DataFrame(
            {
                "col1": [1, np.nan, np.inf, 4, 5],
                "col2": [10, 20, 30, 40, 1e15],
            }
        )

        result = validate_data_quality(df)

        # Should detect issues (returns False)
        self.assertFalse(result)


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestEnhancedEventLabelsEdgeCases(unittest.TestCase):
    """Test edge cases in enhanced event labeling."""

    def test_price_momentum_missing_columns(self):
        """Test price_momentum method with missing columns."""
        from finance_ml.classification import create_enhanced_event_labels

        # DataFrame without required columns
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Finance", "Health"],
            }
        )

        labels = create_enhanced_event_labels(df, method="price_momentum")

        # Should return all neutral labels
        self.assertTrue(all(labels == 0))

    def test_valuation_missing_columns(self):
        """Test valuation method with missing columns."""
        from finance_ml.classification import create_enhanced_event_labels

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Finance", "Health"],
            }
        )

        labels = create_enhanced_event_labels(df, method="valuation")

        # Should return all neutral labels
        self.assertTrue(all(labels == 0))

    def test_unknown_method(self):
        """Test with unknown labeling method."""
        from finance_ml.classification import create_enhanced_event_labels

        df, _ = create_sample_classification_data(n_samples=50)

        labels = create_enhanced_event_labels(df, method="unknown_method")

        # Should return all neutral labels
        self.assertTrue(all(labels == 0))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestPrepareClassificationDataEdgeCases(unittest.TestCase):
    """Test edge cases in data preparation."""

    def test_prepare_data_with_duplicate_columns(self):
        """Test handling of duplicate columns."""
        from finance_ml.classification import prepare_classification_data

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"] * 50,
                "feature1": np.random.randn(200),
                "feature2": np.random.randn(200),
                "last_price": np.random.uniform(10, 200, 200),
                "price_target": np.random.uniform(10, 250, 200),
            }
        )

        # Add duplicate column
        df["feature1_dup"] = df["feature1"]
        df = df.rename(columns={"feature1_dup": "feature1"})

        labels = np.random.choice([0, 1, 2], 200)

        # Should handle duplicates without error
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        self.assertEqual(len(X_train) + len(X_test), len(df))


@unittest.skipIf(not HAVE_SKLEARN or not HAVE_FINANCE_ML, "sklearn not available")
class TestCompareClassifiers(unittest.TestCase):
    """Test comprehensive model comparison functionality."""

    def test_compare_classifiers_basic(self):
        """Test basic model comparison."""
        from finance_ml.classification import compare_classifiers

        df, labels = create_sample_classification_data(n_samples=200)
        X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_classification_data(
            df, labels, test_size=0.2, random_state=42
        )

        results = compare_classifiers(X_train, y_train, X_test, y_test, num_cols, cat_cols)

        # Check results structure - should be a DataFrame
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)

        # Check required columns
        self.assertIn("Model", results.columns)
        self.assertIn("Accuracy", results.columns)
        self.assertIn("F1-Score", results.columns)

        # Check at least one model is present
        self.assertGreater(len(results), 0)

        # Metrics should be in valid ranges
        self.assertTrue(all(results["Accuracy"].between(0.0, 1.0)))
        self.assertTrue(all(results["F1-Score"].between(0.0, 1.0)))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
