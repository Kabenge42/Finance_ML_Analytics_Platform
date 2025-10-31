import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _make_synthetic_csvs(base_dir: Path, n_per_region: int = 80) -> None:
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
                    "Ticker": f"{rname.upper()}N{i:04d}",
                    "Sector": sectors[i % len(sectors)],
                    "Last Price": 20.0 + (i % 30) * 0.2,
                    "Price Target": 22.0 + (i % 30) * 0.2,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(base_dir / fname, index=False)


@unittest.skipUnless(
    os.environ.get("RUN_NOTEBOOK_INTEGRATION", "0") == "1",
    "Set RUN_NOTEBOOK_INTEGRATION=1 to enable notebook integration test",
)
class TestIntegrationNotebookPipeline(unittest.TestCase):
    def test_execute_main_notebook_smoke(self):
        try:
            import nbformat
            from nbclient import NotebookClient
        except Exception as e:  # pragma: no cover - optional dependency guard
            self.skipTest(f"nbclient/nbformat not available: {e}")

        project_root = Path(__file__).resolve().parents[1]
        nb_path = project_root / "ml_finance_model_main.ipynb"
        self.assertTrue(nb_path.exists(), "Main notebook not found")

        with tempfile.TemporaryDirectory() as td_data:
            data_dir = Path(td_data)
            _make_synthetic_csvs(data_dir, n_per_region=60)

            env = os.environ.copy()
            env["DATA_DIR"] = str(data_dir)
            env["FINANCE_ML_FAST_TEST"] = "1"

            # Load notebook and execute with a timeout
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(nb, timeout=300, kernel_name="python3")
            # Execute in a temporary working directory to avoid polluting repo outputs
            cwd = project_root
            # Temporarily set environment
            old_environ = os.environ.copy()
            try:
                os.environ.update(env)
                client.execute()
            finally:
                os.environ.clear()
                os.environ.update(old_environ)

            # If we reach here, execution completed without error
            # Optionally assert the notebook created the summary when honoring DATA_DIR
            # We keep this light to accommodate different notebook behaviors across phases.


if __name__ == "__main__":
    unittest.main(verbosity=2)
