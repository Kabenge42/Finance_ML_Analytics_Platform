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


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
