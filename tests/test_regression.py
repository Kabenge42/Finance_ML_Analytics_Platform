import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestRegressionPerSectorRoot(unittest.TestCase):
    def small_df(self):
        rows = []
        for i in range(20):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "sector": "Tech",
                    "region": "US",
                    "feature_a": float(i),
                    "last_price": float(10 + i % 3),
                    "price_target": float(11 + (i % 3)),
                }
            )
        for i in range(20):
            rows.append(
                {
                    "ticker": f"E{i}",
                    "sector": "Energy",
                    "region": "EU",
                    "feature_a": float(i),
                    "last_price": float(8 + i % 2),
                    "price_target": float(9 + (i % 2)),
                }
            )
        return pd.DataFrame(rows)

    def test_regression_metrics_by_sector_are_saved(self):
        df = self.small_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            metrics = mod.train_and_evaluate_regression_by_sector(df, out_dir)
            self.assertIn("sector", metrics.columns)
            self.assertIn("mae", metrics.columns)
            self.assertIn("rmse", metrics.columns)
            self.assertIn("r2", metrics.columns)
            self.assertGreaterEqual(len(metrics), 2)
            csv_path = out_dir / "regression_metrics_by_sector.csv"
            self.assertTrue(csv_path.exists())


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestQuantileRegression(unittest.TestCase):
    def small_df(self):
        rows = []
        for i in range(50):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "sector": "Tech",
                    "region": "US",
                    "feature_a": float(i),
                    "feature_b": float(i * 2),
                    "last_price": float(10 + i % 5),
                    "price_target": float(11 + (i % 5)),
                }
            )
        return pd.DataFrame(rows)

    def test_train_quantile_regression_returns_model(self):
        """Test that quantile regression training returns a model object"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"
        quantiles = [0.1, 0.5, 0.9]

        model = mod.train_quantile_regression(df, feature_cols, target_col, quantiles=quantiles)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))

    def test_quantile_regression_predictions_have_multiple_quantiles(self):
        """Test that quantile regression produces predictions for all specified quantiles"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"
        quantiles = [0.1, 0.5, 0.9]

        model = mod.train_quantile_regression(df, feature_cols, target_col, quantiles=quantiles)
        predictions = mod.predict_quantile_regression(model, df[feature_cols], quantiles=quantiles)

        # Should return a DataFrame with columns for each quantile
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), len(df))
        self.assertIn("q_0.1", predictions.columns)
        self.assertIn("q_0.5", predictions.columns)
        self.assertIn("q_0.9", predictions.columns)

    def test_quantile_predictions_are_ordered(self):
        """Test that lower quantiles produce lower predictions than higher quantiles"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"
        quantiles = [0.1, 0.5, 0.9]

        model = mod.train_quantile_regression(df, feature_cols, target_col, quantiles=quantiles)
        predictions = mod.predict_quantile_regression(model, df[feature_cols], quantiles=quantiles)

        # q_0.1 should be <= q_0.5 <= q_0.9
        self.assertTrue((predictions["q_0.1"] <= predictions["q_0.5"]).all())
        self.assertTrue((predictions["q_0.5"] <= predictions["q_0.9"]).all())

    def test_quantile_regression_with_sector_split(self):
        """Test quantile regression trained per sector"""
        df = self.small_df()
        # Add Energy sector
        for i in range(50):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "ticker": f"E{i}",
                                "sector": "Energy",
                                "region": "EU",
                                "feature_a": float(i),
                                "feature_b": float(i * 2),
                                "last_price": float(8 + i % 3),
                                "price_target": float(9 + (i % 3)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"
        quantiles = [0.1, 0.5, 0.9]

        models = mod.train_quantile_regression_by_sector(
            df, feature_cols, target_col, quantiles=quantiles
        )
        self.assertIsInstance(models, dict)
        self.assertIn("Tech", models)
        self.assertIn("Energy", models)


@unittest.skipIf(pd is None or mod is None, "pandas/sklearn not installed")
class TestMetaLearnerStacking(unittest.TestCase):
    def small_df(self):
        rows = []
        for i in range(100):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "sector": "Tech",
                    "region": "US",
                    "feature_a": float(i),
                    "feature_b": float(i * 2),
                    "last_price": float(10 + i % 5),
                    "price_target": float(11 + (i % 5)),
                }
            )
        for i in range(100):
            rows.append(
                {
                    "ticker": f"E{i}",
                    "sector": "Energy",
                    "region": "EU",
                    "feature_a": float(i),
                    "feature_b": float(i * 2),
                    "last_price": float(8 + i % 3),
                    "price_target": float(9 + (i % 3)),
                }
            )
        return pd.DataFrame(rows)

    def test_train_stacking_ensemble_returns_model(self):
        """Test that stacking ensemble training returns a model object"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"

        stacking_model = mod.train_stacking_ensemble(df, feature_cols, target_col)
        self.assertIsNotNone(stacking_model)
        self.assertTrue(hasattr(stacking_model, "predict"))

    def test_stacking_predictions_match_data_length(self):
        """Test that stacking ensemble produces predictions for all rows"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"

        stacking_model = mod.train_stacking_ensemble(df, feature_cols, target_col)
        X_test = df[feature_cols]
        predictions = stacking_model.predict(X_test)

        self.assertEqual(len(predictions), len(df))
        self.assertTrue(all(~pd.isna(predictions)))

    def test_stacking_ensemble_has_base_models(self):
        """Test that stacking ensemble contains base models and meta-learner"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"

        stacking_model = mod.train_stacking_ensemble(df, feature_cols, target_col)

        # Check for base_models and meta_model attributes
        self.assertTrue(hasattr(stacking_model, "base_models"))
        self.assertTrue(hasattr(stacking_model, "meta_model"))
        self.assertGreater(len(stacking_model.base_models), 0)

    def test_stacking_with_sector_optimization(self):
        """Test that stacking uses sector-specific base models"""
        df = self.small_df()
        feature_cols = ["feature_a", "feature_b"]
        target_col = "price_target"

        stacking_model = mod.train_stacking_ensemble_by_sector(df, feature_cols, target_col)

        # Should have models for each sector
        self.assertIsInstance(stacking_model, dict)
        self.assertIn("Tech", stacking_model)
        self.assertIn("Energy", stacking_model)

        # Each sector model should be able to predict
        for sector, model in stacking_model.items():
            self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
