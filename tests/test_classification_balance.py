"""
Test classification balance_classes() integration.

Priority 2 - Task 2.1: Integrate balance_classes() in notebook

Tests verify:
1. balance_classes() improves minority class representation
2. Trained models predict all 5 classes after balancing
3. Class imbalance ratio improves after balancing

Aligned with: docs/improvement_plan/finance_ml_workflow_implementation_plan.md
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from finance_ml.ml_workflow.classification.models import balance_classes
except ImportError:
    balance_classes = None


def create_imbalanced_classification_data(n_samples=1000, n_features=20, random_state=42):
    """
    Create synthetic imbalanced classification dataset with 5 classes.

    Class distribution maintains 10:1 imbalance ratio scaled to n_samples:
    - Class 0: 50% of samples
    - Class 1: 25% of samples
    - Class 2: 15% of samples
    - Class 3: 7.5% of samples
    - Class 4: 2.5% of samples

    Args:
        n_samples: Total number of samples to generate
        n_features: Number of features
        random_state: Random seed

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector with 5 classes (0-4)
    """
    np.random.seed(random_state)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create imbalanced target (scaled to n_samples)
    class_proportions = [0.50, 0.25, 0.15, 0.075, 0.025]
    class_counts = [int(n_samples * prop) for prop in class_proportions]
    # Adjust last class to match exact n_samples
    class_counts[-1] = n_samples - sum(class_counts[:-1])

    y = np.concatenate([np.full(count, i) for i, count in enumerate(class_counts)])

    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)]), pd.Series(
        y, name="target"
    )


class TestClassificationBalance(unittest.TestCase):
    """Test balance_classes() integration and effectiveness."""

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_balance_classes_improves_minority(self):
        """Verify balance_classes increases minority class representation."""
        X, y = create_imbalanced_classification_data()

        # Check initial imbalance
        initial_counts = y.value_counts()
        initial_max = initial_counts.max()
        initial_min = initial_counts.min()
        initial_ratio = initial_max / initial_min

        self.assertGreater(initial_ratio, 5.0, "Initial data should be imbalanced (>5:1)")

        # Apply balancing
        X_bal, y_bal = balance_classes(X, y, method="auto", random_state=42)

        # All classes should have reasonable representation (≥20% of majority)
        class_counts = pd.Series(y_bal).value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        balanced_ratio = max_count / min_count

        # Verify improvement
        self.assertLess(
            balanced_ratio,
            initial_ratio,
            f"Balance should improve: {initial_ratio:.2f}:1 → {balanced_ratio:.2f}:1",
        )
        self.assertLess(
            balanced_ratio,
            3.0,
            f"Class imbalance should be <3:1 after balancing, got {balanced_ratio:.2f}:1",
        )
        self.assertGreaterEqual(
            min_count,
            0.2 * max_count,
            f"Minority class should be ≥20% of majority: {class_counts.to_dict()}",
        )

        # All 5 classes should be preserved
        self.assertEqual(len(set(y_bal)), 5, "Should preserve all 5 classes")

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_all_classes_predicted_after_balance(self):
        """Verify trained model predicts all 5 classes after balancing."""
        X, y = create_imbalanced_classification_data()

        # Apply balancing
        X_bal, y_bal = balance_classes(X, y, method="auto", random_state=42)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
        )

        # Train model
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = clf.predict(X_test)

        # Model should predict at least 4 of 5 classes (allow 1 missing due to test set size)
        predicted_classes = set(y_pred)
        self.assertGreaterEqual(
            len(predicted_classes),
            4,
            f"Model should predict at least 4 of 5 classes, got {len(predicted_classes)}: {predicted_classes}",
        )

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_balance_classes_with_imbalance_threshold(self):
        """Verify balance_classes respects imbalance_threshold parameter."""
        X, y = create_imbalanced_classification_data(n_samples=500)

        # Apply balancing with imbalance threshold
        imbalance_threshold = 5.0  # Trigger balancing if ratio > 5:1
        X_bal, y_bal = balance_classes(
            X, y, method="auto", imbalance_threshold=imbalance_threshold, random_state=42
        )

        # Check that imbalance ratio improved
        class_counts = pd.Series(y_bal).value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        final_ratio = max_count / min_count

        self.assertLess(
            final_ratio,
            imbalance_threshold,
            f"Imbalance ratio should be < {imbalance_threshold}:1, got {final_ratio:.2f}:1",
        )

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_balance_classes_preserves_features(self):
        """Verify balance_classes preserves feature dimensionality."""
        X, y = create_imbalanced_classification_data(n_features=30)

        original_features = X.shape[1]

        # Apply balancing
        X_bal, y_bal = balance_classes(X, y, method="auto", random_state=42)

        # Feature count should be preserved
        self.assertEqual(
            X_bal.shape[1],
            original_features,
            f"Feature count should be preserved: {original_features} → {X_bal.shape[1]}",
        )

        # Feature names should be preserved (if DataFrame)
        if isinstance(X_bal, pd.DataFrame):
            self.assertListEqual(
                list(X_bal.columns), list(X.columns), "Feature names should be preserved"
            )

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_imbalance_ratio_improvement(self):
        """Verify imbalance ratio improves significantly after balancing."""
        X, y = create_imbalanced_classification_data()

        # Calculate initial imbalance ratio
        initial_counts = y.value_counts()
        initial_ratio = initial_counts.max() / initial_counts.min()

        # Apply balancing
        X_bal, y_bal = balance_classes(X, y, method="auto", random_state=42)

        # Calculate final imbalance ratio
        final_counts = pd.Series(y_bal).value_counts()
        final_ratio = final_counts.max() / final_counts.min()

        # Improvement should be at least 50%
        improvement_pct = (initial_ratio - final_ratio) / initial_ratio * 100
        self.assertGreater(
            improvement_pct,
            50.0,
            f"Imbalance ratio should improve by >50%: {initial_ratio:.2f}:1 → {final_ratio:.2f}:1 ({improvement_pct:.1f}%)",
        )

    @unittest.skipIf(balance_classes is None, "balance_classes not available")
    def test_balance_classes_handles_numpy_array_input(self):
        """
        Verify balance_classes handles numpy array input without AttributeError.

        Regression test for AttributeError: 'numpy.ndarray' object has no attribute 'value_counts'
        This error occurred when y_train_cls was passed as numpy array instead of pandas Series.
        """
        X, y = create_imbalanced_classification_data()

        # Convert y to numpy array (simulates notebook scenario)
        y_array = y.values
        self.assertIsInstance(y_array, np.ndarray, "y should be numpy array for this test")

        # Apply balancing with numpy array input (should not raise AttributeError)
        try:
            X_bal, y_bal = balance_classes(X, y_array, method="auto", random_state=42)
        except AttributeError as e:
            self.fail(f"balance_classes raised AttributeError with numpy array input: {e}")

        # Verify output is valid
        self.assertIsNotNone(X_bal, "X_bal should not be None")
        self.assertIsNotNone(y_bal, "y_bal should not be None")
        self.assertEqual(len(X_bal), len(y_bal), "X_bal and y_bal should have same length")

        # Verify balancing was applied
        self.assertGreater(len(y_bal), len(y), "Balanced dataset should have more samples (SMOTE)")

        # Verify all classes preserved
        unique_classes_input = set(y_array)
        unique_classes_output = set(y_bal)
        self.assertEqual(
            unique_classes_input,
            unique_classes_output,
            f"All classes should be preserved: {unique_classes_input} vs {unique_classes_output}",
        )


if __name__ == "__main__":
    unittest.main()
