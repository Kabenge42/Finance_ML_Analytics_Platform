import unittest
import pandas as pd

from finance_ml.ml_workflow.regression.cv import get_regression_cv_splitter


class TestRegressionTimeSeriesCVPolicy(unittest.TestCase):
    def test_time_order_preserved(self):
        df = pd.DataFrame(
            {
                "ticker": ["A"] * 6,
                "snapshot_date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-06",
                    ]
                ),
            }
        )
        cv = get_regression_cv_splitter(policy="time_series", date_col="snapshot_date")
        splits = list(cv.split(df))
        # basic check: train indices earlier than test indices
        for tr, te in splits:
            self.assertLess(df.loc[tr, "snapshot_date"].max(), df.loc[te, "snapshot_date"].min())


if __name__ == "__main__":
    unittest.main()
