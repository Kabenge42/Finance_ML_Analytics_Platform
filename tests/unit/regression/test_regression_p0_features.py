import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from finance_ml.ml_workflow.regression.cv import get_regression_cv_splitter
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor,
    predict_quantile_regression,
)
from finance_ml.ml_workflow.regression.io import validate_predictions_schema
from finance_ml.ml_workflow.regression.dataset import extract_classification_features


class TestRegressionP0Features(unittest.TestCase):

    # P0.1: Time-series CV splitter
    def test_cv_splitter_timeseries(self):
        df = pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(
                    ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01"]
                ),
                "val": range(5),
            }
        )
        splitter = get_regression_cv_splitter(
            policy="time_series", n_splits=2, date_col="snapshot_date"
        )
        splits = list(splitter.split(df))
        self.assertEqual(len(splits), 2)

        # Check order: train indices must be strictly before test indices in time
        # But split returns integer indices (positional) mapped to df.index
        # Since df is already sorted by date here, positional index corresponds to time order
        for train_idx, test_idx in splits:
            train_dates = df.loc[train_idx, "snapshot_date"]
            test_dates = df.loc[test_idx, "snapshot_date"]
            self.assertLess(train_dates.max(), test_dates.min())

    # P0.2: Non-negative wrapper
    def test_non_negative_wrapper(self):
        # Create data that would naturally lead to negative predictions
        X = np.array([[1], [2], [3], [4]])
        y = np.array([-5, -4, -3, -2])  # Linear trend but negative

        base = Ridge()
        model = NonNegativeRegressionWrapper(base)
        model.fit(X, y)
        preds = model.predict(X)

        self.assertTrue(np.all(preds >= 0), "All predictions must be non-negative")
        self.assertTrue(np.any(preds == 0), "Some predictions should be clipped to 0")

    # P0.3: Quantile regression monotonicity and non-negativity
    def test_quantile_regression_monotonicity_and_nonnegativity(self):
        # Create noisy linear data
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(100, 2), columns=["f1", "f2"])
        y = 2 * X["f1"] + 3 * X["f2"] + np.random.normal(0, 1, 100)

        quantiles = [0.1, 0.5, 0.9]
        result = train_quantile_regressor(X, y, quantiles=quantiles)
        models = result["model"]

        # Predict on new data
        X_new = pd.DataFrame(np.random.rand(20, 2), columns=["f1", "f2"])
        preds_df = predict_quantile_regression(models, quantiles, X_new)

        # Check columns exist
        expected_cols = [f"pred_q{q}" for q in quantiles]
        for col in expected_cols:
            self.assertIn(col, preds_df.columns)

        # Check non-negativity (assuming enforce_nonnegative=True by default or we pass it)
        # The issue says "Enforce quantile monotonicity and non-negativity in predict_quantile_regression"
        # So we expect it to be enforced.
        self.assertTrue((preds_df.values >= 0).all(), "Quantile predictions must be non-negative")

        # Check monotonicity (row-wise)
        # q0.1 <= q0.5 <= q0.9
        q01 = preds_df[f"pred_q{quantiles[0]}"]
        q05 = preds_df[f"pred_q{quantiles[1]}"]
        q09 = preds_df[f"pred_q{quantiles[2]}"]

        self.assertTrue((q01 <= q05).all(), "q0.1 <= q0.5 violated")
        self.assertTrue((q05 <= q09).all(), "q0.5 <= q0.9 violated")

    # P0.4: Schema validator
    def test_predictions_schema_validator_strengthened(self):
        # Minimal valid dataframe
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "isin": ["US..."],
                "sector": ["Tech"],
                "region": ["US"],
                "last_price": [100.0],
                "y_true": [105.0],
                "y_pred": [104.0],
                "y_pred_calibrated": [104.0],
                "pred_p10": [100.0],
                "pred_p50": [104.0],
                "pred_p90": [110.0],
                "interval_width": [10.0],
                "abs_error": [1.0],
                "pct_error": [0.01],
                "model_version": ["v1"],
                "snapshot_date": ["2021-01-01"],
            }
        )

        # Should pass
        validate_predictions_schema(df)

        # Fail: negative predictions
        df_neg = df.copy()
        df_neg.loc[0, "y_pred"] = -1.0
        with self.assertRaises(ValueError):
            validate_predictions_schema(df_neg)

        # Fail: missing required column (y_pred is required, interval_width is auto-created)
        df_missing = df.drop(columns=["y_pred"])
        with self.assertRaises(ValueError):
            validate_predictions_schema(df_missing)

    # P0.5: Classification meta-features
    def test_classification_meta_features(self):
        # 5-class probabilities as required by dataset.py
        probs = np.array([[0.1, 0.1, 0.6, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
        features = extract_classification_features(probs)

        expected_cols = [
            "event_prob_strong_negative",
            "event_prob_negative",
            "event_prob_neutral",
            "event_prob_positive",
            "event_prob_strong_positive",
            "event_class_predicted",
            "event_confidence",
        ]
        for col in expected_cols:
            self.assertIn(col, features.columns)

        self.assertEqual(len(features), 2)


if __name__ == "__main__":
    unittest.main()
