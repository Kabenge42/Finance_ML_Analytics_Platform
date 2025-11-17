"""Tests for regression.dataset.create_classification_interactions.

Phase 9.3/9.5 integration: verify that
``finance_ml.ml_workflow.regression.dataset.create_classification_interactions``
creates interaction features between classification probability
columns and valuation metrics, and that it degrades gracefully when the
requested valuation columns are missing from the dataframe.
"""

import unittest

import numpy as np
import pandas as pd


class TestCreateClassificationInteractions(unittest.TestCase):
    """Unit tests for create_classification_interactions."""

    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "event_prob_0": np.linspace(0.1, 0.9, 5),
                "event_prob_1": np.linspace(0.9, 0.1, 5),
                # Present valuation metric
                "p_e_ntm": np.array([5.0, 10.0, 15.0, 20.0, 25.0]),
            }
        )

    def test_interactions_created_for_existing_columns(self):
        """Interactions should be created for columns that exist in df."""

        from finance_ml.ml_workflow.regression.dataset import (
            create_classification_interactions,
        )

        result = create_classification_interactions(
            self.df,
            classification_cols=["event_prob_0", "event_prob_1"],
            valuation_cols=["p_e_ntm"],
        )

        expected_cols = {
            "event_prob_0_x_p_e_ntm",
            "event_prob_1_x_p_e_ntm",
        }
        self.assertTrue(expected_cols.issubset(result.columns))

        # Check one interaction numerically
        expected = self.df["event_prob_0"] * self.df["p_e_ntm"]
        pd.testing.assert_series_equal(
            result["event_prob_0_x_p_e_ntm"], expected, check_names=False
        )

    def test_missing_valuation_columns_are_ignored(self):
        """Missing valuation columns should be ignored without raising.

        This allows callers to pass a canonical Phase 9.3 valuation
        candidate list (including columns that may not exist in the
        current dataset) without pre-filtering it.
        """

        from finance_ml.ml_workflow.regression.dataset import (
            create_classification_interactions,
        )

        # p_b_ltm is not present in self.df and should be ignored
        result = create_classification_interactions(
            self.df,
            classification_cols=["event_prob_0"],
            valuation_cols=["p_e_ntm", "p_b_ltm"],
        )

        interaction_cols = [c for c in result.columns if "_x_" in c]

        # Only interactions for existing valuation columns should appear
        self.assertIn("event_prob_0_x_p_e_ntm", interaction_cols)
        self.assertNotIn("event_prob_0_x_p_b_ltm", interaction_cols)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
