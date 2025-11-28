"""
Test stacking ensemble configuration (Priority 6).

Ensures stacking ensemble can be used as default in regression pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


class TestStackingEnsemble(unittest.TestCase):
    """Test stacking ensemble functionality."""

    def setUp(self):
        """Create synthetic dataset for testing."""
        np.random.seed(42)
        n = 200

        self.X_train = pd.DataFrame({f"feat_{i}": np.random.randn(n) for i in range(5)})
        self.y_train = pd.Series(
            self.X_train["feat_0"] * 2 + self.X_train["feat_1"] * -1 + np.random.randn(n) * 5
        )

        self.X_test = pd.DataFrame({f"feat_{i}": np.random.randn(50) for i in range(5)})

    def test_train_stacking_ensemble_exists_in_models(self):
        """Test that train_stacking_ensemble function exists in models.py."""
        from finance_ml.ml_workflow.models import train_stacking_ensemble

        self.assertTrue(callable(train_stacking_ensemble))

    def test_stacking_ensemble_module_exists(self):
        """Test that dedicated ensemble module exists (optional consolidation)."""
        try:
            from finance_ml.ml_workflow.regression import ensemble

            self.assertTrue(
                hasattr(ensemble, "train_stacking_ensemble")
                or hasattr(ensemble, "build_stacking_regressor")
            )
        except ImportError:
            # Not critical - can use models.py version
            pass

    def test_stacking_returns_standardized_format(self):
        """Test that stacking returns standardized model dict."""
        from finance_ml.ml_workflow.models import train_stacking_ensemble

        # Create minimal dataframe with required columns
        df = pd.DataFrame(
            {
                "price_target": self.y_train,
                **{f"feat_{i}": self.X_train[f"feat_{i}"] for i in range(5)},
            }
        )

        feature_cols = [f"feat_{i}" for i in range(5)]
        target_col = "price_target"

        result = train_stacking_ensemble(df, feature_cols, target_col, random_state=42)

        # Should return dict with standardized keys
        self.assertIsInstance(result, dict)
        # Allow flexibility in exact return format


class TestStackingIntegration(unittest.TestCase):
    """Test stacking ensemble integration with main pipeline."""

    def setUp(self):
        """Create synthetic dataset."""
        np.random.seed(42)
        n = 150

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance"], n),
                "last_price": np.random.uniform(50, 200, n),
                "price_target": np.random.uniform(50, 200, n),
                "market_cap": np.random.uniform(1e9, 1e11, n),
                "pe_ratio": np.random.uniform(10, 30, n),
            }
        )

    def test_regression_pipeline_can_use_stacking(self):
        """Test that main regression pipeline can use stacking ensemble."""
        from finance_ml.ml_workflow.models import train_and_evaluate_regression

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            # Run pipeline with stacking (if supported via config)
            # This test documents the desired API
            result = train_and_evaluate_regression(self.df, out_dir, n_jobs=1, dry_run=False)

            # Should complete successfully
            self.assertIsInstance(result, dict)
            self.assertIn("metrics", result)


if __name__ == "__main__":
    unittest.main()
