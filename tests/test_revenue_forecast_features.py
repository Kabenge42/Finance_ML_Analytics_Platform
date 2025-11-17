"""Tests for revenue forecasting & analyst consensus features (Schema 1.3).

Focus on a subset of features implemented in ``engineer_revenue_forecast_features``:

- revenue_estimate_spread_ntm
- revenue_estimate_spread_fy1e
- revenue_growth_implied_ntm

The tests verify numeric correctness and NaN handling when estimates or
historical revenues are missing.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import assert_no_inf


class TestRevenueForecastCore(unittest.TestCase):
    def test_revenue_spreads_and_growth_basic(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_revenue_forecast_features,
        )

        df = pd.DataFrame(
            {
                "revenues_est_avg_ntm": [110.0, 200.0],
                "revenues_est_med_ntm": [100.0, 200.0],
                "revenues_est_avg_fy1e": [210.0, 400.0],
                "revenues_est_med_fy1e": [200.0, 380.0],
                "total_revenues_ltm": [100.0, 250.0],
            }
        )

        res = engineer_revenue_forecast_features(df)

        # Spread = (avg - med)/med
        np.testing.assert_allclose(
            res.loc[0, "revenue_estimate_spread_ntm"], (110.0 - 100.0) / 100.0
        )
        np.testing.assert_allclose(res.loc[1, "revenue_estimate_spread_ntm"], 0.0)

        np.testing.assert_allclose(
            res.loc[0, "revenue_estimate_spread_fy1e"], (210.0 - 200.0) / 200.0
        )
        np.testing.assert_allclose(
            res.loc[1, "revenue_estimate_spread_fy1e"], (400.0 - 380.0) / 380.0
        )

        # Growth implied NTM = (avg_ntm - revenue_ltm)/revenue_ltm
        np.testing.assert_allclose(
            res.loc[0, "revenue_growth_implied_ntm"], (110.0 - 100.0) / 100.0
        )
        np.testing.assert_allclose(
            res.loc[1, "revenue_growth_implied_ntm"], (200.0 - 250.0) / 250.0
        )

        assert_no_inf(res)

    def test_revenue_features_missing_inputs(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_revenue_forecast_features,
        )

        # Missing median NTM and LTM revenues for the second row
        df = pd.DataFrame(
            {
                "revenues_est_avg_ntm": [110.0, 200.0],
                "revenues_est_med_ntm": [np.nan, np.nan],
                "total_revenues_ltm": [100.0, np.nan],
            }
        )

        res = engineer_revenue_forecast_features(df)

        # Where denominators are missing, we expect NaN rather than crashes/inf
        self.assertTrue(np.isnan(res.loc[0, "revenue_estimate_spread_ntm"]))
        self.assertTrue(np.isnan(res.loc[1, "revenue_estimate_spread_ntm"]))
        self.assertTrue(np.isnan(res.loc[1, "revenue_growth_implied_ntm"]))
        assert_no_inf(res)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
