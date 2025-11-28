"""
P1: Stacking ensemble with classification meta-features and non-negativity.

Validates that:
- train_stacking_ensemble accepts meta-feature integration + interactions flags
- Predictions are non-negative (meta-learner wrapped)
- On a small synthetic dataset, stacking is not worse than the best base regressor
  by more than a small epsilon on a holdout split.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class TestStackingWithMetaFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n = 160
        # base features
        f0 = rng.normal(size=n)
        f1 = rng.normal(size=n)
        f2 = rng.normal(size=n)
        # valuation-like feature
        pe_ratio = rng.uniform(8, 30, size=n)

        # synthetic target with some noise
        y = 5 + 2.0 * f0 - 1.3 * f1 + 0.7 * f2 + 0.05 * pe_ratio + rng.normal(scale=0.5, size=n)
        # ensure non-negative target typical for prices
        y = np.maximum(y, 0.0)

        self.df = pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(n)],
                "sector": np.where(rng.random(n) < 0.5, "Tech", "Finance"),
                "snapshot_date": pd.to_datetime("2024-01-01")
                + pd.to_timedelta(np.arange(n), unit="D"),
                "feat_0": f0,
                "feat_1": f1,
                "feat_2": f2,
                "pe_ratio": pe_ratio,
                "price_target": y,
            }
        )

        # 5-class probabilities roughly correlated with signal on feat_0
        # create logits then softmax to get probabilities
        logits = np.stack(
            [
                -1.0 - f0,
                -0.5 - 0.5 * f0,
                0.0 * f0,
                0.5 + 0.5 * f0,
                1.0 + f0,
            ],
            axis=1,
        )
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        self.y_proba = exp / exp.sum(axis=1, keepdims=True)

    def test_stacking_with_meta_features_interactions(self):
        from finance_ml.ml_workflow.regression.models import train_stacking_regressor

        feature_cols = ["feat_0", "feat_1", "feat_2", "pe_ratio"]
        target_col = "price_target"

        # holdout split
        train_idx, test_idx = train_test_split(self.df.index, test_size=0.25, random_state=42)
        df_train = self.df.loc[train_idx]
        df_test = self.df.loc[test_idx]

        # We pass the full df_train but the function might expect just X and y.
        # The signature I'm planning:
        # train_stacking_regressor(X, y, ..., use_meta_features=True, classification_probabilities=..., ...)

        # Need to adjust input. The existing test passed df_train and feature_cols.
        # I should probably make train_stacking_regressor accept X, y like others,
        # but also allow extra args for meta features.

        result = train_stacking_regressor(
            df_train[feature_cols],
            df_train[target_col],
            random_state=42,
            use_meta_features=True,
            classification_probabilities=self.y_proba[train_idx],
            enable_interactions=True,
            interaction_valuation_cols=["pe_ratio"],
            cv_policy="time_series",
            date_col="snapshot_date",
            group_col="ticker",
            groups=df_train["ticker"],  # might need groups for CV
            dates=df_train["snapshot_date"],  # might need dates for CV
        )

        model = result["model"]

        # Prepare X_test with same meta-features and interactions
        # 1. Get probabilities for test set
        test_proba = self.y_proba[test_idx]

        # 2. Integrate
        from finance_ml.ml_workflow.regression.dataset import (
            integrate_classification_features,
            create_classification_interactions,
        )

        X_test_enhanced = df_test[feature_cols].copy()
        X_test_enhanced = integrate_classification_features(X_test_enhanced, test_proba)

        if True:  # enable_interactions=True
            interaction_valuation_cols = ["pe_ratio"]
            class_cols = [
                c
                for c in X_test_enhanced.columns
                if c.startswith("event_prob_") or c == "event_confidence"
            ]
            X_test_enhanced = create_classification_interactions(
                X_test_enhanced, class_cols, interaction_valuation_cols
            )

        # Predictions should be non-negative
        y_pred = model.predict(X_test_enhanced)
        self.assertTrue(np.all(y_pred >= 0.0))

        # Compare to best single base model (trained on same train set)
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        X_test = df_test[feature_cols]  # Base features for base models
        y_test = df_test[target_col]

        base_rf = RandomForestRegressor(n_estimators=60, max_depth=5, random_state=42).fit(
            X_train, y_train
        )
        base_gb = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42).fit(
            X_train, y_train
        )

        mse_rf = mean_squared_error(y_test, base_rf.predict(X_test))
        mse_gb = mean_squared_error(y_test, base_gb.predict(X_test))
        mse_stack = mean_squared_error(y_test, y_pred)

        best_base = min(mse_rf, mse_gb)
        # Allow small tolerance since data and CV differ
        self.assertLessEqual(mse_stack, best_base * 1.10)


if __name__ == "__main__":
    unittest.main()
