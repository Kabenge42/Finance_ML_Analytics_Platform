import unittest
import numpy as np
import pandas as pd


class TestEvalAndDataEdgeCases(unittest.TestCase):
    def test_calculate_mispricing_score_missing_columns(self):
        from finance_ml import eval as fm_eval

        df = pd.DataFrame({"last_price": [100, 120]})  # missing predicted column
        with self.assertRaises(ValueError):
            fm_eval.calculate_mispricing_score(df)

    def test_compare_prediction_vs_analyst_targets_empty_df(self):
        from finance_ml import eval as fm_eval

        df = pd.DataFrame(columns=["predicted_price_target", "price_target", "last_price"])  # empty
        result = fm_eval.compare_prediction_vs_analyst_targets(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("total_count"), 0)
        self.assertEqual(result.get("agreement_rate"), 0)

    def test_directional_accuracy_empty_df(self):
        from finance_ml import eval as fm_eval

        df = pd.DataFrame(columns=["predicted_price_target", "price_target", "last_price"])  # empty
        acc = fm_eval.calculate_directional_accuracy(df)
        self.assertEqual(acc, 0.0)

    def test_data_quality_handles_inf_nan_extremes(self):
        from finance_ml import data as fm_data

        df = pd.DataFrame(
            {
                "a": [1, 2, np.inf, -np.inf, np.nan, 1e12],
                "b": [np.nan, 0, 1, 2, 3, -1e11],
            }
        )
        report = fm_data.validate_financial_data_quality(df, region="US")
        self.assertIsInstance(report, dict)
        # Should detect some issues
        self.assertGreater(report.get("infinity_values", 0), 0)
        self.assertGreaterEqual(report.get("extreme_outliers", 0), 0)
        self.assertGreaterEqual(report.get("null_values", 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
