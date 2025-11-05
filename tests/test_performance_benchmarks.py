import os
import time
import unittest
import numpy as np
import pandas as pd


def _make_perf_df(n_rows: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "last_price": rng.uniform(10, 200, n_rows),
            "market_cap": rng.uniform(1e8, 1e12, n_rows),
            "total_revenue": rng.uniform(1e7, 1e11, n_rows),
            "ebitda": rng.uniform(1e6, 1e10, n_rows),
            "total_debt": rng.uniform(0, 1e11, n_rows),
            "volatility": rng.uniform(0.1, 0.6, n_rows),
        }
    )


@unittest.skipUnless(
    os.environ.get("FAST_BENCH", "0") == "1", "Set FAST_BENCH=1 to run performance benchmarks"
)
class TestPerformanceBenchmarks(unittest.TestCase):
    def test_feature_engineering_speed(self):
        from finance_ml import features

        df = _make_perf_df(15000)
        t0 = time.time()
        out = features.engineer_basic_ratios(df)
        dt = time.time() - t0
        self.assertEqual(len(out), len(df))
        # Soft threshold: 2 seconds on 15k rows for simple ratios
        self.assertLess(dt, float(os.environ.get("BENCH_RATIO_SECONDS", 2.5)))

    def test_classifier_training_speed(self):
        from finance_ml import classification

        if not getattr(classification, "HAVE_XGBOOST", False):
            self.skipTest("XGBoost not available")
        # 1500 rows for a quick benchmark
        n = 1500
        rng = np.random.default_rng(7)
        X = pd.DataFrame(
            {
                "last_price": rng.uniform(10, 200, n),
                "market_cap": rng.uniform(1e8, 1e12, n),
                "volatility": rng.uniform(0.1, 0.6, n),
            }
        )
        # 3 classes
        y = rng.integers(0, 3, size=n)
        t0 = time.time()
        res = classification.train_xgboost_classifier(
            X, y, X, y, ["last_price", "market_cap", "volatility"], []
        )
        dt = time.time() - t0
        self.assertIn("model", res)
        # Soft threshold: 10 seconds default
        self.assertLess(dt, float(os.environ.get("BENCH_XGB_SECONDS", 12.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
