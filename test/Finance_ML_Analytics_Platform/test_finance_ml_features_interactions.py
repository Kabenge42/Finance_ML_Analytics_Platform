"""
P2: Compact interaction generator tests.

Validates deterministic naming and counts for build_prob_valuation_interactions
and ensures no additional NaNs are introduced beyond existing data.
"""

import unittest
import numpy as np
import pandas as pd


class TestProbValuationInteractions(unittest.TestCase):
    def test_deterministic_naming_and_count(self):
        from finance_ml.ml_workflow.regression.features import (
            build_prob_valuation_interactions,
        )

        df = pd.DataFrame(
            {
                "pe_ratio": [10.0, 20.0, 30.0],
                "pb_ratio": [1.1, 2.2, 3.3],
                "event_prob_strong_negative": [0.1, 0.2, 0.3],
                "event_prob_negative": [0.2, 0.2, 0.2],
                "event_prob_neutral": [0.3, 0.2, 0.1],
            }
        )

        out = build_prob_valuation_interactions(
            df,
            valuation_cols=["pe_ratio", "pb_ratio"],
            prob_cols=[
                "event_prob_strong_negative",
                "event_prob_negative",
                "event_prob_neutral",
            ],
        )

        expected_names = [
            "pe_ratio_x_event_prob_strong_negative",
            "pe_ratio_x_event_prob_negative",
            "pe_ratio_x_event_prob_neutral",
            "pb_ratio_x_event_prob_strong_negative",
            "pb_ratio_x_event_prob_negative",
            "pb_ratio_x_event_prob_neutral",
        ]

        for name in expected_names:
            self.assertIn(name, out.columns)

        # Count only the expected interaction columns, ignore any other columns
        actual_interactions = [c for c in out.columns if c in expected_names]
        self.assertEqual(len(expected_names), len(actual_interactions))

    def test_no_extra_nans_introduced(self):
        from finance_ml.ml_workflow.regression.features import (
            build_prob_valuation_interactions,
        )

        df = pd.DataFrame(
            {
                "pe_ratio": [10.0, np.nan, 30.0],
                "event_prob_negative": [0.2, 0.5, 0.3],
            }
        )

        out = build_prob_valuation_interactions(
            df, valuation_cols=["pe_ratio"], prob_cols=["event_prob_negative"]
        )
        inter_col = "pe_ratio_x_event_prob_negative"
        self.assertIn(inter_col, out.columns)
        # NaNs should only be where original pe_ratio was NaN
        self.assertTrue(np.isnan(out.loc[1, inter_col]))
        self.assertFalse(np.isnan(out.loc[0, inter_col]))
        self.assertFalse(np.isnan(out.loc[2, inter_col]))


if __name__ == "__main__":
    unittest.main()
