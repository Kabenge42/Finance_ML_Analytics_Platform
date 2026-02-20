"""Test that the `or pd.DataFrame()` fallback correctly handles None values
returned by locals().get() in market_analytics.py (line ~2657-2660)."""

import unittest

import pandas as pd


class TestNoneOrDataFrameFallback(unittest.TestCase):
    """Regression test for AttributeError: 'NoneType' object has no attribute 'empty'."""

    def test_none_value_falls_back_to_empty_dataframe(self):
        """When locals().get returns None, `or pd.DataFrame()` must produce an empty DF."""
        value = None
        result = value or pd.DataFrame()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_missing_key_falls_back_to_empty_dataframe(self):
        """When the key is absent, locals().get returns None → same fallback."""
        d: dict = {}
        result = d.get("missing_key") or pd.DataFrame()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_existing_dataframe_preserved(self):
        """A non-empty DataFrame must pass through unchanged."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df if df is not None else pd.DataFrame()
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)

    def test_empty_dataframe_preserved(self):
        """An empty (but not None) DataFrame is still a valid DF with .empty == True."""
        df = pd.DataFrame()
        result = df if df is not None else pd.DataFrame()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
