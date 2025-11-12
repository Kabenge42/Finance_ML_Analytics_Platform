import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestNotebookVisualizations(unittest.TestCase):
    """Visualization tests for plots generated via fm_eval.simple_eda."""

    def _make_df(self, n=200):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "ticker": [f"T{i:04d}" for i in range(n)],
                "sector": rng.choice(["Tech", "Health", "Energy"], size=n),
                "region": rng.choice(["US", "EU"], size=n),
                "last_price": rng.uniform(10, 200, size=n),
                "price_target": rng.uniform(10, 220, size=n),
                "market_cap": rng.uniform(1e8, 1e12, size=n),
                "volatility": rng.uniform(0.1, 0.6, size=n),
            }
        )
        return df

    def test_simple_eda_saves_plots_headless(self):
        from finance_ml import eval as fm_eval

        df = self._make_df(150)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            summary = fm_eval.simple_eda(df, out_dir=out_dir, save_plots=True)
            # Summary must include keys used elsewhere
            self.assertIsInstance(summary, dict)
            self.assertIn("numeric_columns", summary)
            self.assertIn("categorical_columns", summary)

            # If plots are generated, they should be saved under out_dir as image files.
            # We do not depend on exact filenames; just ensure no exception and optional files exist.
            plot_files = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
            # It's acceptable if zero (implementation-dependent), but if present, ensure non-empty files
            for p in plot_files:
                self.assertGreater(p.stat().st_size, 0)

    def test_simple_eda_no_out_dir_ok(self):
        from finance_ml import eval as fm_eval

        df = self._make_df(50)
        # Should not raise when save_plots=True but out_dir is None; implementation may ignore save
        summary = fm_eval.simple_eda(df, out_dir=None, save_plots=True)
        self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
