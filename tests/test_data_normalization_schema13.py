"""Tests for Schema 1.3 column normalization and safe numeric handling.

Phase 9.3 preprocessing slice:
- Ensure that new Schema 1.3 raw columns are mapped by
  finance_ml.ml_workflow.preprocessing.data.normalize_columns.
- Verify basic NaN / inf robustness via safe_divide.
"""

import math
import unittest

import numpy as np
import pandas as pd


class TestSchema13Normalization(unittest.TestCase):
    """Test normalize_columns for Schema 1.3 specific inputs."""

    def test_normalize_columns_maps_schema13_raw_columns(self):
        """Normalize a mini frame with Schema 1.3 raw columns and check mappings.

        This test focuses on a representative subset from the Phase 9.3 plan:
        - Revenue estimates (avg/median NTM/FY1E)
        - Valuation time-series (EV/Sales, EV/EBITDA)
        - Technical indicators (EMAs, 52W high/low, relative volume)
        - Employment metrics (Total Employees FY/FQ)
        """

        from finance_ml.ml_workflow.preprocessing.data import normalize_columns

        raw_df = pd.DataFrame(
            {
                "Revenues - Est Avg (NTM)": [1.0, np.nan],
                "Revenues - Est Med (FY1E)": [2.0, 3.0],
                "EV/Sales (LTM)": [4.0, 5.0],
                "EV/EBITDA (NTM)": [6.0, 7.0],
                "52W High/Adj": [10.0, 12.0],
                "52W Low/Adj": [5.0, 4.0],
                "EMA (20D)": [100.0, 110.0],
                "EMA (250D)": [90.0, 95.0],
                "Rel. Volume": [1.5, 0.8],
                "Total Employees (FY)": [1000, 1200],
                "Total Employees (FQ)": [950, 1150],
            }
        )

        normalized = normalize_columns(raw_df, preserve_schema=True)

        expected_columns = {
            "revenues_est_avg_ntm",
            "revenues_est_med_fy1e",
            "ev_sales_ltm",
            "ev_ebitda_ntm",
            "52w_high_adj",
            "52w_low_adj",
            "ema_20d",
            "ema_250d",
            "rel_volume",
            "total_employees_fy",
            "total_employees_fq",
        }

        self.assertTrue(
            expected_columns.issubset(set(normalized.columns)),
            msg=f"Missing expected normalized columns: {expected_columns - set(normalized.columns)}",
        )


class TestSafeDivideSchema13Interaction(unittest.TestCase):
    """Test safe_divide robustness on representative Schema 1.3-derived series."""

    def test_safe_divide_handles_zero_and_nan_in_schema13_like_series(self):
        """safe_divide should avoid inf and propagate NaNs predictably.

        We approximate a Schema 1.3 ratio, such as revenue growth implied by
        estimates, using simple numeric series with zeros and NaNs in the
        denominator to validate robustness.
        """

        from finance_ml.ml_workflow.preprocessing.data import safe_divide

        numerator = pd.Series([1.0, 2.0, np.nan, 4.0, -5.0])
        denominator = pd.Series([1.0, 0.0, 2.0, np.nan, 0.0])

        result = safe_divide(numerator, denominator)

        # Length must be preserved
        self.assertEqual(len(result), len(numerator))

        # No infinities should be present
        self.assertFalse(np.isinf(result).any())

        # Where denominator is zero, result should be NaN by default
        self.assertTrue(math.isnan(result.iloc[1]))
        self.assertTrue(math.isnan(result.iloc[4]))

        # Where numerator is NaN, result should be NaN
        self.assertTrue(math.isnan(result.iloc[2]))

        # Valid finite division where both inputs are finite and denominator non-zero
        self.assertAlmostEqual(result.iloc[0], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
