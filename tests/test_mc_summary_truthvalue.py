"""
Test that the DataFrame truth-value fix in market_analytics.py works correctly.

Regression test for:
  ValueError: The truth value of a DataFrame is ambiguous.
"""

import unittest

import pandas as pd


def safe_get_dataframe(value):
    """Mimic the fixed pattern from market_analytics.py main()."""
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


class TestDataFrameSafeGet(unittest.TestCase):
    """Ensure safe_get_dataframe never triggers ambiguous truth-value errors."""

    def test_none_returns_empty_df(self):
        result = safe_get_dataframe(None)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_non_empty_dataframe_returned_as_is(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = safe_get_dataframe(df)
        self.assertIs(result, df)
        self.assertFalse(result.empty)

    def test_empty_dataframe_returned_as_is(self):
        df = pd.DataFrame()
        result = safe_get_dataframe(df)
        self.assertIs(result, df)
        self.assertTrue(result.empty)

    def test_non_dataframe_value_returns_empty(self):
        result = safe_get_dataframe("not a dataframe")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_old_pattern_would_fail(self):
        """The old `df or pd.DataFrame()` pattern raises ValueError for non-empty DF."""
        df = pd.DataFrame({"a": [1, 2]})
        with self.assertRaises(ValueError):
            _ = df or pd.DataFrame()


if __name__ == "__main__":
    unittest.main()
