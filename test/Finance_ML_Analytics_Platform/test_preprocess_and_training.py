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
class TestPreprocessAndTraining(unittest.TestCase):
    def test_preprocess_coerces_numeric_and_filters_required(self):
        # last_price as string should be coerced; row with missing sector should be dropped
        df_raw = pd.DataFrame(
            {
                "ticker": ["A", "B", None],
                "sector": ["Tech", None, "Energy"],
                "last_price": ["10.0", "20.0", "30.0"],
                "feature_strnum": ["1", "2", "foo"],  # only numeric-like should be coerced
            }
        )
        df = mod.preprocess(df_raw)
        # Expect rows with missing ticker/sector filtered out
        self.assertEqual(len(df), 1)
        # last_price should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df["last_price"]))
        # feature_strnum: with errors='coerce', numeric-like strings are converted to numbers
        # Row 0 has '1' which becomes 1.0 (float). Accept both string and numeric representations.
        self.assertIn(str(df.iloc[0]["feature_strnum"]), {"1", "1.0"})

    def test_infer_region_from_filename(self):
        # Should infer region tokens from filename
        cases = {
            "screening_us.csv": "US",
            "foo_EU_bar.csv": "EU",
            "apac_list.csv": "APAC",
            "data_rotw_2020.csv": "ROTW",
            "unknown.csv": None,
        }
        for name, expected in cases.items():
            got = mod.infer_region_from_filename(Path(name))
            self.assertEqual(got, expected)

    def test_train_and_evaluate_regression_dry_run_and_small_sample(self):
        # Build small dataset with target but too few rows so early exit occurs
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [10.0, 12.0, 11.0],
                "price_target": [11.0, 13.0, 12.0],
                "region": ["US", "US", "US"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            # Dry run should return None early without fitting
            res = mod.train_and_evaluate_regression(df, out_dir, n_jobs=1, dry_run=True)
            self.assertIsNone(res)
            # Even without dry_run, too few samples should cause early exit None
            res2 = mod.train_and_evaluate_regression(df, out_dir, n_jobs=1, dry_run=False)
            self.assertIsNone(res2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
