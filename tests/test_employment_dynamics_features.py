"""Tests for employment dynamics & growth signals (Schema 1.3).

We validate a compact subset of features from ``engineer_employment_dynamics_features``:

- revenue_per_employee_fy
- revenue_per_employee_ltm
- employee_base_scale_flag

The tests emphasize correctness of simple ratios and edge-case handling.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.utils.feature_test_helpers import assert_no_inf


class TestEmploymentDynamicsCore(unittest.TestCase):
    def test_revenue_per_employee_and_scale_flag(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_employment_dynamics_features,
        )

        df = pd.DataFrame(
            {
                "total_revenues_fy": [1_000.0, 2_000.0],
                "total_revenues_ltm": [900.0, 2_100.0],
                "total_employees_fy": [100, 20_000],
                "avg_employees_ltm": [80, 19_000],
            }
        )

        res = engineer_employment_dynamics_features(df)

        # revenue_per_employee_fy = total_revenues_fy / total_employees_fy
        np.testing.assert_allclose(res.loc[0, "revenue_per_employee_fy"], 1_000.0 / 100.0)
        np.testing.assert_allclose(res.loc[1, "revenue_per_employee_fy"], 2_000.0 / 20_000.0)

        # revenue_per_employee_ltm = total_revenues_ltm / avg_employees_ltm
        np.testing.assert_allclose(res.loc[0, "revenue_per_employee_ltm"], 900.0 / 80.0)
        np.testing.assert_allclose(res.loc[1, "revenue_per_employee_ltm"], 2_100.0 / 19_000.0)

        # employee_base_scale_flag = 1 if employees_fy >= 10_000 else 0
        self.assertEqual(int(res.loc[0, "employee_base_scale_flag"]), 0)
        self.assertEqual(int(res.loc[1, "employee_base_scale_flag"]), 1)

        assert_no_inf(res)

    def test_employment_features_with_missing_values(self):
        from finance_ml.ml_workflow.features.advanced import (
            engineer_employment_dynamics_features,
        )

        df = pd.DataFrame(
            {
                "total_revenues_fy": [np.nan],
                "total_employees_fy": [0],
                "avg_employees_ltm": [np.nan],
            }
        )

        res = engineer_employment_dynamics_features(df)
        # No infinities even when denominators are zero/missing
        assert_no_inf(res)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
