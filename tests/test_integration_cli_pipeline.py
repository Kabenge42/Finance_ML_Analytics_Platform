import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
import unittest

import pandas as pd


def _make_synthetic_csvs(base_dir: Path, n_per_region: int = 200) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    regions = [
        ("us", "screening_us.csv"),
        ("eu", "screening_eu.csv"),
        ("apac", "screening_apac.csv"),
        ("rotw", "screening_rotw.csv"),
    ]
    sectors = ["Technology", "Healthcare", "Financials", "Energy"]
    rows = []
    for rname, fname in regions:
        rows.clear()
        for i in range(n_per_region):
            rows.append(
                {
                    "Ticker": f"{rname.upper()}T{i:04d}",
                    "Sector": sectors[i % len(sectors)],
                    # Use spaces in column name to match raw CSVs; normalize later
                    "Last Price": 10.0 + (i % 50) * 0.1,
                    "Price Target": 12.0 + (i % 50) * 0.1,
                }
            )
        df = pd.DataFrame(rows)
        (base_dir / fname).write_text("", encoding="utf-8")  # ensure file exists
        df.to_csv(base_dir / fname, index=False)


@unittest.skipUnless(
    os.environ.get("RUN_CLI_INTEGRATION", "0") == "1",
    "Set RUN_CLI_INTEGRATION=1 to enable CLI integration test",
)
class TestIntegrationCliPipeline(unittest.TestCase):
    def test_cli_runs_end_to_end_with_csv_and_dry_run(self):
        with tempfile.TemporaryDirectory() as td_data, tempfile.TemporaryDirectory() as td_out:
            data_dir = Path(td_data)
            out_dir = Path(td_out)
            _make_synthetic_csvs(data_dir, n_per_region=150)

            env = os.environ.copy()
            env["DATA_DIR"] = str(data_dir)
            env["FINANCE_ML_FAST_TEST"] = "1"

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "ml_finance_model_main.py"),
                "--data-source",
                "csv",
                "--limit",
                "500",
                "--out-dir",
                str(out_dir),
                "--dry-run",
            ]
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                text=True,
                check=False,
            )
            # Expect successful exit
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"CLI pipeline failed with code {proc.returncode}. Output:\n{proc.stdout}",
            )

            # Artifacts: eda_summary.json must exist; regression predictions should not (dry-run)
            eda_path = out_dir / "eda_summary.json"
            self.assertTrue(eda_path.exists(), msg=f"Missing {eda_path}\nOutput:\n{proc.stdout}")
            # Validate JSON structure minimally
            data = json.loads(eda_path.read_text(encoding="utf-8"))
            self.assertIn("row_count", data)
            self.assertIn("column_count", data)

            preds_path = out_dir / "regression_predictions.csv"
            self.assertFalse(
                preds_path.exists(),
                msg="regression_predictions.csv should not be created in dry-run",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
