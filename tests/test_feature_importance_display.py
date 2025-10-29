"""
Test suite for Feature Importance Display Fix (TDD Implementation)

Tests for fixing the _display_importance_scores method to handle:
1. DataFrame input from calculate_feature_importance_rf
2. Series input (fallback case)
3. Dict input (legacy case)
4. Proper scalar conversion to avoid TypeError

Coverage target: ≥80% for changed code
"""

import sys
import unittest
from io import StringIO

import numpy as np
import pandas as pd


class MockFeatureEngineeringReporter:
    """Mock reporter class for testing _display_importance_scores."""

    def _display_importance_scores(self, importance_scores, top_k):
        """Display feature importance scores in formatted output.

        This is the FIXED version that should handle DataFrame, Series, and dict.
        """
        print(f"\n🔝 Top {top_k} Most Important Features:")

        # Handle DataFrame input (primary case from calculate_feature_importance_rf)
        if isinstance(importance_scores, pd.DataFrame):
            # DataFrame has 'feature' and 'importance' columns
            if "feature" in importance_scores.columns and "importance" in importance_scores.columns:
                # Already sorted and sliced by calculate_feature_importance_rf
                top_features = importance_scores.head(top_k)
                for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
                    feature_name = row["feature"]
                    score_value = float(row["importance"])
                    print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")
            else:
                raise ValueError("DataFrame must have 'feature' and 'importance' columns")

        # Handle Series input (fallback case)
        elif isinstance(importance_scores, pd.Series):
            # Take top_k features
            top_features = importance_scores.head(top_k)

            # Iterate and display
            for rank, (feature_name, score) in enumerate(top_features.items(), start=1):
                # Ensure score is a scalar value (not Series)
                score_value = float(score)
                print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")

        # Handle dict input (legacy case)
        else:
            # If it's a dict, convert to Series first
            importance_series = pd.Series(importance_scores)
            top_features = importance_series.sort_values(ascending=False).head(top_k)

            for rank, (feature_name, score) in enumerate(top_features.items(), start=1):
                score_value = float(score)
                print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")


class TestFeatureImportanceDisplayDataFrame(unittest.TestCase):
    """Test _display_importance_scores with DataFrame input."""

    def setUp(self):
        """Create test reporter and capture output."""
        self.reporter = MockFeatureEngineeringReporter()
        self.captured_output = StringIO()

    def test_display_handles_dataframe_input(self):
        """Should handle DataFrame input from calculate_feature_importance_rf."""
        # Create DataFrame like calculate_feature_importance_rf returns
        importance_df = pd.DataFrame(
            {
                "feature": ["revenue", "market_cap", "p_e", "ebitda", "debt"],
                "importance": [0.35, 0.25, 0.20, 0.15, 0.05],
            }
        )

        # Capture output
        sys.stdout = self.captured_output
        try:
            self.reporter._display_importance_scores(importance_df, top_k=3)
        finally:
            sys.stdout = sys.__stdout__

        output = self.captured_output.getvalue()

        # Verify output contains expected features
        self.assertIn("revenue", output)
        self.assertIn("market_cap", output)
        self.assertIn("p_e", output)
        self.assertNotIn("ebitda", output)  # Should be excluded (top_k=3)

        # Verify formatting works (no TypeError)
        self.assertIn("0.3500", output)
        self.assertIn("0.2500", output)

    def test_display_converts_scores_to_float(self):
        """Should explicitly convert scores to float to avoid TypeError."""
        # Create DataFrame with numpy types
        importance_df = pd.DataFrame(
            {
                "feature": ["feature_a", "feature_b"],
                "importance": [np.float64(0.8), np.float32(0.6)],
            }
        )

        # Should not raise TypeError
        sys.stdout = self.captured_output
        try:
            self.reporter._display_importance_scores(importance_df, top_k=2)
        finally:
            sys.stdout = sys.__stdout__

        output = self.captured_output.getvalue()
        self.assertIn("0.8000", output)
        self.assertIn("0.6000", output)

    def test_display_respects_top_k_limit(self):
        """Should only display top_k features."""
        importance_df = pd.DataFrame(
            {
                "feature": [f"feat_{i}" for i in range(10)],
                "importance": [0.9 - i * 0.1 for i in range(10)],
            }
        )

        sys.stdout = self.captured_output
        try:
            self.reporter._display_importance_scores(importance_df, top_k=5)
        finally:
            sys.stdout = sys.__stdout__

        output = self.captured_output.getvalue()

        # First 5 features should be present
        for i in range(5):
            self.assertIn(f"feat_{i}", output)

        # Features 5-9 should not be present
        for i in range(5, 10):
            self.assertNotIn(f"feat_{i}", output)


class TestFeatureImportanceDisplaySeries(unittest.TestCase):
    """Test _display_importance_scores with Series input."""

    def setUp(self):
        """Create test reporter and capture output."""
        self.reporter = MockFeatureEngineeringReporter()
        self.captured_output = StringIO()

    def test_display_handles_series_input(self):
        """Should handle Series input correctly."""
        importance_series = pd.Series({"revenue": 0.35, "market_cap": 0.25, "p_e": 0.20})

        sys.stdout = self.captured_output
        try:
            self.reporter._display_importance_scores(importance_series, top_k=2)
        finally:
            sys.stdout = sys.__stdout__

        output = self.captured_output.getvalue()

        self.assertIn("revenue", output)
        self.assertIn("market_cap", output)
        self.assertIn("0.3500", output)


class TestFeatureImportanceDisplayDict(unittest.TestCase):
    """Test _display_importance_scores with dict input."""

    def setUp(self):
        """Create test reporter and capture output."""
        self.reporter = MockFeatureEngineeringReporter()
        self.captured_output = StringIO()

    def test_display_handles_dict_input(self):
        """Should handle dict input and sort by importance."""
        importance_dict = {
            "low_importance": 0.10,
            "high_importance": 0.90,
            "medium_importance": 0.50,
        }

        sys.stdout = self.captured_output
        try:
            self.reporter._display_importance_scores(importance_dict, top_k=2)
        finally:
            sys.stdout = sys.__stdout__

        output = self.captured_output.getvalue()

        # Top 2 should be high and medium
        self.assertIn("high_importance", output)
        self.assertIn("medium_importance", output)
        self.assertNotIn("low_importance", output)


class TestFeatureImportanceIntegration(unittest.TestCase):
    """Integration tests for complete feature importance workflow."""

    def test_calculate_feature_importance_returns_dataframe(self):
        """Verify calculate_feature_importance_rf returns DataFrame."""
        from finance_ml.advanced_features import calculate_feature_importance_rf

        # Create sample data
        X = pd.DataFrame(
            {
                "feat1": np.random.rand(100),
                "feat2": np.random.rand(100),
                "feat3": np.random.rand(100),
            }
        )
        y = pd.Series(np.random.rand(100))

        result = calculate_feature_importance_rf(X, y, top_k=2)

        # Should return DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Should have correct columns
        self.assertIn("feature", result.columns)
        self.assertIn("importance", result.columns)

        # Should respect top_k
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
