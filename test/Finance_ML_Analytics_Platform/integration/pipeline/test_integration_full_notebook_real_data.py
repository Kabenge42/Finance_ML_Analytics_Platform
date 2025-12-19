import os
import unittest
from pathlib import Path


class TestIntegrationFullNotebookRealData(unittest.TestCase):
    """Runs ml_stock_prediction_model.ipynb end-to-end using real CSVs if available.

    This test is skipped by default. Enable by setting RUN_NOTEBOOK_FULL=1.
    It will look for CSV files under data/ and execute the notebook with a
    conservative row limit to keep runtime manageable.
    """

    @unittest.skipUnless(
        os.environ.get("RUN_NOTEBOOK_FULL", "0") == "1",
        "Set RUN_NOTEBOOK_FULL=1 to enable full notebook execution with real data",
    )
    def test_notebook_executes_with_real_csvs(self):
        try:
            import nbformat
            from nbclient import NotebookClient
        except Exception as e:
            self.skipTest(f"nbclient/nbformat not available: {e}")

        project_root = Path(__file__).resolve().parents[1]
        nb_path = project_root / "ml_stock_prediction_model.ipynb"
        self.assertTrue(nb_path.exists(), "Notebook file not found")

        data_dir = project_root / "data"
        # Basic heuristic: at least one CSV in data/
        csvs = list(data_dir.glob("*.csv")) if data_dir.exists() else []
        if not csvs:
            self.skipTest("No CSV files found under data/; provide real data to run this test")

        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=1200, kernel_name="python3")

        # Set environment for CSV source
        old_environ = os.environ.copy()
        try:
            os.environ["FINANCE_ML_FAST_TEST"] = "1"
            os.environ["DATA_SOURCE"] = "csv"
            os.environ["DATA_DIR"] = str(data_dir)
            # use a moderate limit to keep CI stable; override via env if needed
            os.environ.setdefault("DATA_LIMIT", "2000")
            client.execute()
        finally:
            os.environ.clear()
            os.environ.update(old_environ)

        # Optional artifact check
        outputs_dir = project_root / "outputs"
        results_csv = outputs_dir / "stock_prediction_results.csv"
        # Not all runs will save artifacts (depending on config); if file exists, assert size
        if results_csv.exists():
            self.assertGreater(results_csv.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
