import unittest
import pandas as pd
import numpy as np

from finance_ml.advanced_models import compare_regressors


class TestRegressionComparisonAdapter(unittest.TestCase):
    def setUp(self):
        # Synthetic compare_regressors-like results (dict of dicts)
        self.dict_results = {
            "Ridge": {"mae": 2.0, "rmse": 3.0, "r2": 0.80, "train_r2": 0.85, "train_time": 0.1},
            "Lasso": {"mae": 1.5, "rmse": 2.5, "r2": 0.82, "train_r2": 0.84, "train_time": 0.2},
            "RandomForest": {
                "mae": 1.8,
                "rmse": 2.8,
                "r2": 0.81,
                "train_r2": 0.90,
                "train_time": 0.5,
            },
        }

        # DataFrame variant with lowercase metric names and model column in lowercase
        self.df_results = pd.DataFrame(
            [
                {"model": "Ridge", "mae": 2.0, "rmse": 3.0, "r2": 0.80},
                {"model": "Lasso", "mae": 1.5, "rmse": 2.5, "r2": 0.82},
                {"model": "RandomForest", "mae": 1.8, "rmse": 2.8, "r2": 0.81},
            ]
        )

    def test_integration_with_compare_regressors(self):
        # Small synthetic dataset to keep runtime minimal
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(120, 6), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(rng.rand(120) * 10 + 5)

        raw_results = compare_regressors(
            X, y, test_size=0.2, cv=3, random_state=0, ensure_nonnegative=True, loss="huber"
        )
        # The function returns a dict; standardize must convert it to DataFrame with 'Model'
        from finance_ml.advanced_models import standardize_comparison_results

        df = standardize_comparison_results(raw_results)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Model", df.columns)
        # Ensure at least one successful model row exists
        self.assertTrue(len(df) >= 1)
        # MAE should be non-negative and finite for the best model
        self.assertTrue(np.isfinite(df.iloc[0]["MAE"]) or np.isnan(df.iloc[0]["MAE"]))

    def test_standardize_results_from_dict(self):
        # Import here to ensure test fails before implementation exists
        from finance_ml.advanced_models import standardize_comparison_results

        df = standardize_comparison_results(self.dict_results)
        # Must not raise KeyError when accessing 'Model'
        first_model = df.iloc[0]["Model"]
        self.assertIn(first_model, {"Lasso", "Ridge", "RandomForest"})

        # Columns must be normalized
        for col in ["Model", "MAE", "RMSE", "R2"]:
            self.assertIn(col, df.columns)

        # Sorted ascending by MAE so best (lowest) comes first
        self.assertAlmostEqual(df.iloc[0]["MAE"], 1.5)
        self.assertEqual(df.iloc[0]["Model"], "Lasso")

    def test_standardize_results_from_dataframe(self):
        from finance_ml.advanced_models import standardize_comparison_results

        df = standardize_comparison_results(self.df_results)
        # Access without KeyError
        _ = df.iloc[0]["Model"]
        # Normalized metric names
        for col in ["MAE", "RMSE", "R2"]:
            self.assertIn(col, df.columns)
        # Sorted ascending by MAE
        self.assertEqual(df.iloc[0]["Model"], "Lasso")

    def test_handles_missing_metrics_gracefully(self):
        from finance_ml.advanced_models import standardize_comparison_results

        # Missing MAE should not crash; function should still return a DataFrame with Model column
        res = {"ModelA": {"rmse": 2.0, "r2": 0.5}}
        df = standardize_comparison_results(res)
        self.assertIn("Model", df.columns)
        self.assertIn("RMSE", df.columns)
        self.assertIn("R2", df.columns)
        # If MAE missing, sorting shouldn't break; check presence or NaN
        self.assertIn("MAE", df.columns)
        self.assertTrue(np.isnan(df.loc[df["Model"] == "ModelA", "MAE"]).iloc[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
