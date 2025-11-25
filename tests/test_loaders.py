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
class TestLoaders(unittest.TestCase):
    def test_load_from_csv_adds_region_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # Create only the US CSV without Region column
            us_csv = data_dir / "screening_us.csv"
            us_csv.write_text("Ticker,Last Price\nA,10.0\n", encoding="utf-8")

            df = mod.load_from_csv(data_dir)
            # load_from_csv normalizes columns, so check for 'region' not 'Region'
            self.assertIn("region", df.columns)
            # All rows should be tagged US
            self.assertTrue((df["region"] == "US").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
