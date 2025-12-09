"""
Test suite for Task 6: Stacking Ensemble Hyperparameter Tuning

Tests for automated hyperparameter tuning using Optuna for stacking base models,
base model selection, and meta-learner selection.

Phase 9.5 TDD Implementation Plan - Task 6 (Low Priority)
"""

import unittest
import time
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.regression.models import (
    tune_stacking_hyperparameters,
    select_stacking_base_models,
    select_meta_learner,
)


class TestStackingHyperparameterTuning(unittest.TestCase):
    """Test suite for stacking ensemble hyperparameter tuning."""

    def test_stacking_hyperparameter_search(self):
        """Hyperparameter search should complete within time budget."""
        # Given: Training data for regression
        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(200),
                "feature_2": np.random.randn(200),
                "feature_3": np.random.randn(200),
            }
        )
        y = X["feature_1"] * 2 + X["feature_2"] - 0.5 * X["feature_3"] + np.random.randn(200) * 0.1

        # When: Tune hyperparameters with time budget
        start_time = time.time()
        best_params, best_score = tune_stacking_hyperparameters(
            X, y, model_type="xgboost", n_trials=10, timeout=30  # 30 seconds max
        )
        elapsed = time.time() - start_time

        # Then: Completes within timeout
        self.assertLessEqual(elapsed, 35)  # 5 second buffer
        self.assertIsInstance(best_params, dict)
        self.assertIn("learning_rate", best_params)
        self.assertIsInstance(best_score, float)
        self.assertGreater(best_score, 0)  # Positive MAE score

    def test_stacking_base_model_selection(self):
        """Should select best base models for stacking."""
        # Given: Comparison results from multiple models
        comparison_results = {
            "xgboost": {"mae": 10.5, "rmse": 15.2, "r2": 0.85},
            "lightgbm": {"mae": 10.2, "rmse": 14.8, "r2": 0.87},
            "catboost": {"mae": 10.8, "rmse": 15.5, "r2": 0.83},
            "ridge": {"mae": 12.0, "rmse": 17.0, "r2": 0.75},
            "lasso": {"mae": 12.5, "rmse": 17.5, "r2": 0.73},
        }

        # When: Select top base models
        selected = select_stacking_base_models(comparison_results, metric="r2", top_k=3)

        # Then: Top 3 by R² selected
        self.assertEqual(len(selected), 3)
        self.assertIn("lightgbm", selected)
        self.assertIn("xgboost", selected)
        self.assertIn("catboost", selected)
        self.assertNotIn("ridge", selected)

    def test_stacking_meta_learner_selection(self):
        """Meta-learner should be selected via cross-validation."""
        # Given: Base model predictions
        X_base = pd.DataFrame(
            {
                "pred_xgb": np.random.uniform(50, 150, 100),
                "pred_lgb": np.random.uniform(50, 150, 100),
                "pred_cat": np.random.uniform(50, 150, 100),
            }
        )
        y = np.random.uniform(50, 150, 100)

        # When: Select best meta-learner
        best_meta, cv_scores = select_meta_learner(
            X_base, y, candidates=["ridge", "lasso", "huber"], cv=5
        )

        # Then: Best meta-learner selected
        self.assertIn(best_meta, ["ridge", "lasso", "huber"])
        self.assertIsInstance(cv_scores, dict)
        self.assertEqual(len(cv_scores), 3)

        # Best meta has highest score
        best_score = cv_scores[best_meta]
        for score in cv_scores.values():
            self.assertGreaterEqual(best_score, score)


if __name__ == "__main__":
    unittest.main()
