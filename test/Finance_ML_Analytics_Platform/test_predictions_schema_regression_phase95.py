import unittest
import pandas as pd
import numpy as np


REQUIRED = {
    "ticker",
    "isin",
    "sector",
    "region",
    "last_price",
    "y_true",
    "y_pred",
    "y_pred_calibrated",
    "pred_p10",
    "pred_p50",
    "pred_p90",
    "interval_width",
    "abs_error",
    "pct_error",
    "model_version",
    "snapshot_date",
}


class TestPredictionsSchemaPhase95(unittest.TestCase):
    def test_required_columns_and_invariants(self):
        df = pd.DataFrame([{c: 1 for c in REQUIRED}])
        df.loc[0, ["y_pred", "pred_p10", "pred_p50", "pred_p90"]] = [5.0, 4.0, 5.0, 6.0]
        df.loc[0, "interval_width"] = df.loc[0, "pred_p90"] - df.loc[0, "pred_p10"]

        # required columns
        self.assertTrue(REQUIRED.issubset(set(df.columns)))

        # Use validator to enforce invariants
        from finance_ml.ml_workflow.regression.io import validate_predictions_schema

        out = validate_predictions_schema(df)

        # non-negativity and monotonicity
        self.assertGreaterEqual(out.loc[0, "y_pred"], 0)
        self.assertTrue(out.loc[0, "pred_p10"] <= out.loc[0, "pred_p50"] <= out.loc[0, "pred_p90"])
        self.assertGreaterEqual(out.loc[0, "interval_width"], 0)


if __name__ == "__main__":
    unittest.main()
