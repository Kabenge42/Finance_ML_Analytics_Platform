"""Tests for valuation multiples time-series features (Phase 9.3 Schema 1.3).

Covers core indicators implemented in ``engineer_valuation_timeseries_features``:

- ev_sales_trend_1y
- ev_sales_vs_3y_avg
- ev_ebitda_vs_3y_avg
- p_e_forward_discount

The tests use small deterministic samples and validate both numeric values
and numerical hygiene (no infinities, reasonable handling of missing data).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import assert_no_inf


class TestValuationTimeseriesCore(unittest.TestCase):
    def test_ev_sales_and_ev_ebitda_trends_basic(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_valuation_timeseries_features,
        )

        df = pd.DataFrame(
            {
                "ev_sales_ltm": [2.0, 1.0],
                "ev_sales_1fyltm": [1.0, 2.0],
                "ev_sales_3yavgltm": [1.5, 1.5],
                "ev_ebitda_ltm": [8.0, 10.0],
                "ev_ebitda_3yavgltm": [10.0, 10.0],
            }
        )

        res = engineer_valuation_timeseries_features(df)

        # ev_sales_trend_1y = (ltm - 1fyltm) / 1fyltm
        np.testing.assert_allclose(res.loc[0, "ev_sales_trend_1y"], (2.0 - 1.0) / 1.0)
        np.testing.assert_allclose(res.loc[1, "ev_sales_trend_1y"], (1.0 - 2.0) / 2.0)

        # ev_sales_vs_3y_avg = (ltm - 3yavg) / 3yavg
        np.testing.assert_allclose(res.loc[0, "ev_sales_vs_3y_avg"], (2.0 - 1.5) / 1.5)
        np.testing.assert_allclose(res.loc[1, "ev_sales_vs_3y_avg"], (1.0 - 1.5) / 1.5)

        # ev_ebitda_vs_3y_avg = (ltm - 3yavg)/3yavg
        np.testing.assert_allclose(res.loc[0, "ev_ebitda_vs_3y_avg"], (8.0 - 10.0) / 10.0)
        np.testing.assert_allclose(res.loc[1, "ev_ebitda_vs_3y_avg"], 0.0)

        assert_no_inf(res)

    def test_p_e_forward_discount_with_missing(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_valuation_timeseries_features,
        )

        df = pd.DataFrame(
            {
                "p_e_ltm": [15.0, np.nan],
                "p_e_est_fy1": [12.0, 20.0],
            }
        )

        res = engineer_valuation_timeseries_features(df)

        # (est - ltm) / ltm for the first row
        expected = (12.0 - 15.0) / 15.0
        np.testing.assert_allclose(res.loc[0, "p_e_forward_discount"], expected)
        # second row has missing base, should yield NaN rather than crash/inf
        self.assertTrue(np.isnan(res.loc[1, "p_e_forward_discount"]))
        assert_no_inf(res)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
