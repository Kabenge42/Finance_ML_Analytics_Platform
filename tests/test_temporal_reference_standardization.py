import unittest

import pandas as pd

from finance_ml.features.advanced import engineer_temporal_features


class TestTemporalReferenceDateStandardization(unittest.TestCase):
    def test_reference_date_is_propagated_and_used(self):
        df = pd.DataFrame(
            {
                "last_updated": [pd.Timestamp("2025-01-10")],
                "next_earnings": [pd.Timestamp("2025-01-20")],
                "income_statement_report_date": [pd.Timestamp("2024-12-31")],
            }
        )
        reference_date = pd.Timestamp("2025-01-15")

        result = engineer_temporal_features(
            df, date_col="last_updated", reference_date=reference_date
        )

        self.assertIn("_reference_date", result.columns)
        self.assertEqual(result.loc[0, "_reference_date"], reference_date)
        self.assertEqual(result.loc[0, "days_to_earnings"], 5)
        self.assertEqual(result.loc[0, "earnings_report_recency"], 15)
        self.assertEqual(result.loc[0, "reporting_lag"], 20)


if __name__ == "__main__":
    unittest.main()
