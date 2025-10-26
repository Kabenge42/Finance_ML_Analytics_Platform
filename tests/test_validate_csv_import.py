import json
# Ensure project root on path
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestValidateCsvImport(unittest.TestCase):
    def setUp(self):
        # Temporary data and output directories
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp_dir.name) / "data"
        self.out_dir = Path(self.tmp_dir.name) / "out"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_missing_critical_column_triggers_error(self):
        # Create a CSV missing the 'Sector' column
        us_csv = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Last Price": [150.0, 300.0],
                # "Sector" missing on purpose
            }
        )
        df.to_csv(us_csv, index=False)

        import validate_csv_import as vci

        res = vci.validate_csv_file(us_csv, "US")
        self.assertTrue(res.exists)
        self.assertTrue(any("Critical column 'Sector' not found" in e for e in res.errors))

    def test_non_numeric_last_price_detected(self):
        # Create a CSV with non-numeric value in 'Last Price'
        eu_csv = self.data_dir / "screening_eu.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["SAP"],
                "Sector": ["Technology"],
                "Last Price": ["abc"],  # non-numeric
            }
        )
        df.to_csv(eu_csv, index=False)

        import validate_csv_import as vci

        res = vci.validate_csv_file(eu_csv, "EU")
        self.assertTrue(any("Last Price" in issue for issue in res.numeric_issues))

    def test_cli_writes_json_and_exit_codes(self):
        # US valid, EU invalid
        (self.data_dir / "screening_us.csv").write_text(
            "Ticker,Sector,Last Price\nAAPL,Technology,150\n", encoding="utf-8"
        )
        (self.data_dir / "screening_eu.csv").write_text(
            "Ticker,Last Price\nSAP,120\n", encoding="utf-8"  # missing Sector
        )

        import validate_csv_import as vci

        report_path = self.out_dir / "data_quality_import.json"

        with patch(
            "sys.argv",
            [
                "validate_csv_import.py",
                "--region",
                "all",
                "--data-dir",
                str(self.data_dir),
                "--out",
                str(self.out_dir),
            ],
        ):
            rc = vci.main()

        self.assertEqual(1, rc, msg="Expect non-zero exit when at least one region has errors")
        self.assertTrue(report_path.exists(), msg="JSON report should be created")

        data = json.loads(report_path.read_text(encoding="utf-8"))
        # Expect results list with per-region entries
        self.assertIn("results", data)
        # Map by region code
        by_region = {item["region"].upper(): item for item in data["results"]}
        self.assertIn("US", by_region)
        self.assertIn("EU", by_region)
        self.assertEqual(0, by_region["US"]["errors_count"])
        self.assertGreater(by_region["EU"]["errors_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
