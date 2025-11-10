"""
Phase 9.3 (Week 8) — Composite & Sector-Relative Interactions (TDD)

Covers:
- Composite quality score from distress/accounting
- Momentum score scaling from return_stability_score
- Sector-relative interactions (vs median and vs top quartile)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:  # pragma: no cover - import shim
    from finance_ml.ml_workflow.features.advanced import (
        engineer_composite_scores,
        engineer_sector_relative_interactions,
    )
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import (  # type: ignore
        engineer_composite_scores,
        engineer_sector_relative_interactions,
    )


class TestCompositeAndInteractions(unittest.TestCase):
    def test_composite_scores(self):
        df = pd.DataFrame(
            {
                "distress_risk_score": [80.0, 20.0],
                "accounting_quality_score": [60.0, 40.0],
                "return_stability_score": [1.0, 0.0],
            }
        )
        res = engineer_composite_scores(df)
        # composite_quality_score is mean of two components
        self.assertIn("composite_quality_score", res.columns)
        self.assertAlmostEqual(
            float(res.loc[0, "composite_quality_score"]), (80.0 + 60.0) / 2.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[1, "composite_quality_score"]), (20.0 + 40.0) / 2.0, places=6
        )
        # momentum score in [0,100]
        self.assertIn("momentum_score", res.columns)
        self.assertGreaterEqual(float(res.loc[0, "momentum_score"]), 0.0)
        self.assertLessEqual(float(res.loc[0, "momentum_score"]), 100.0)

    def test_sector_relative_interactions(self):
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Energy", "Energy"],
                "p_e_ratio": [10.0, 20.0, 15.0, 25.0],
                "roe": [5.0, 15.0, 10.0, 20.0],
                "net_margin_pct": [10.0, 20.0, 30.0, 40.0],
                "ev_ebitda_ratio": [8.0, 12.0, 7.0, 14.0],
            }
        )
        res = engineer_sector_relative_interactions(df, sector_col="sector")
        # Check columns exist
        for m in ("p_e_ratio", "roe", "net_margin_pct", "ev_ebitda_ratio"):
            self.assertIn(f"{m}_vs_sector_median", res.columns)
            self.assertIn(f"{m}_vs_sector_top_quartile", res.columns)
        # Spot-check for "Tech" sector metrics
        tech_mask = res["sector"] == "Tech"
        tech_pe = res.loc[tech_mask, "p_e_ratio"].to_numpy()
        tech_median = np.median(tech_pe)
        tech_vs_med = res.loc[tech_mask, "p_e_ratio_vs_sector_median"].to_numpy()
        np.testing.assert_allclose(tech_vs_med, tech_pe - tech_median)


if __name__ == "__main__":
    unittest.main(verbosity=2)
