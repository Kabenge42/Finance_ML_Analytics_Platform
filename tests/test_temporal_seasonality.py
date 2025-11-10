"""
Phase 9.3 (Week 8) — Temporal & Seasonality Features (TDD)

Covers:
- Seasonality vs 5Y averages (revenue, EBITDA)
- Earnings/reporting timing deltas (days)
- Basic date-derived features (quarter, month, year)
"""

from __future__ import annotations

import datetime as dt
import unittest

import numpy as np
import pandas as pd

try:  # pragma: no cover - import shim
    from finance_ml.ml_workflow.features.advanced import engineer_temporal_features
except Exception:  # pragma: no cover
    from finance_ml.advanced_features import engineer_temporal_features  # type: ignore


class TestTemporalSeasonality(unittest.TestCase):
    def test_temporal_and_seasonality_metrics(self):
        df = pd.DataFrame(
            {
                "next_earnings": [pd.Timestamp("2025-12-31")],
                "last_updated": [pd.Timestamp("2025-12-01")],
                "income_statement_report_date": [pd.Timestamp("2025-11-15")],
                # Seasonality inputs
                "total_revenues_ltm": [1000.0],
                "total_revenues_5yavg": [800.0],
                "ebitda_fq": [120.0],
                "ebitda_5yavgfq": [100.0],
            }
        )
        ref = pd.Timestamp("2025-12-10")
        res = engineer_temporal_features(df, date_col="next_earnings", reference_date=ref)
        # Date-derived
        self.assertIn("fiscal_quarter", res.columns)
        self.assertIn("month", res.columns)
        self.assertIn("year", res.columns)
        self.assertEqual(int(res.loc[0, "fiscal_quarter"]), 4)
        self.assertEqual(int(res.loc[0, "month"]), 12)
        self.assertEqual(int(res.loc[0, "year"]), 2025)
        # Timing deltas
        self.assertIn("days_to_earnings", res.columns)
        self.assertIn("earnings_report_recency", res.columns)
        self.assertIn("reporting_lag", res.columns)
        self.assertEqual(int(res.loc[0, "days_to_earnings"]), 30)
        self.assertEqual(
            int(res.loc[0, "earnings_report_recency"]), (ref - pd.Timestamp("2025-12-01")).days
        )
        self.assertEqual(
            int(res.loc[0, "reporting_lag"]),
            (pd.Timestamp("2025-12-01") - pd.Timestamp("2025-11-15")).days,
        )
        # Seasonality ratios
        self.assertIn("ltm_vs_5yavg_revenue", res.columns)
        self.assertIn("fq_vs_5yavg_ebitda", res.columns)
        self.assertAlmostEqual(
            float(res.loc[0, "ltm_vs_5yavg_revenue"]), (1000.0 - 800.0) / 800.0, places=6
        )
        self.assertAlmostEqual(
            float(res.loc[0, "fq_vs_5yavg_ebitda"]), (120.0 - 100.0) / 100.0, places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
