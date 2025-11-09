"""
Unit tests for notebook outlier detection function calls.

Tests verify that outlier detection functions are called with correct parameter names
per the code_guidelines.md Appendix C migration notes.
"""

import unittest
import pandas as pd
import numpy as np
from finance_ml import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
)


class TestNotebookOutlierDetection(unittest.TestCase):
    """Test cases for notebook outlier detection calls."""

    def setUp(self):
        """Create sample financial data for testing."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "ticker": [f"STOCK{i:03d}" for i in range(100)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], 100),
                "last_price": np.random.uniform(10, 500, 100),
                "market_cap": np.random.lognormal(20, 2, 100),
                "p_e": np.random.uniform(5, 50, 100),
                "profit_margin": np.random.uniform(-0.2, 0.4, 100),
            }
        )
        # Add some outliers
        self.df.loc[0, "market_cap"] = 1e15  # Extreme outlier
        self.df.loc[1, "p_e"] = 500  # Outlier

    def test_detect_outliers_iqr_with_column_singular(self):
        """Test detect_outliers_iqr works with singular 'column' parameter."""
        # This should work with OLD data.py signature: column (singular), multiplier
        try:
            result = detect_outliers_iqr(self.df, column="market_cap", multiplier=1.5)
            self.assertIsNotNone(result)
            # Result should be a list of indices or similar
            self.assertTrue(isinstance(result, (list, pd.Series, pd.DataFrame)))
        except TypeError as e:
            if "unexpected keyword argument 'column'" in str(e):
                self.fail(f"detect_outliers_iqr should accept 'column' parameter: {e}")
            else:
                raise

    def test_detect_outliers_iqr_with_columns_plural_fails(self):
        """Test that detect_outliers_iqr with 'columns' (plural) raises TypeError with OLD version."""
        # This tests that the WRONG parameter name (from original broken notebook) fails
        # The OLD data.py version expects 'column' (singular), not 'columns' (plural)
        with self.assertRaises(TypeError) as cm:
            detect_outliers_iqr(
                self.df, columns=["market_cap"], multiplier=1.5  # Wrong: plural 'columns'
            )
        # Should get error about unexpected keyword argument
        self.assertIn("unexpected keyword argument", str(cm.exception).lower())

    def test_detect_outliers_zscore_signature(self):
        """Test detect_outliers_zscore function signature."""
        # Check if it accepts 'columns' (plural) - both OLD and NEW use plural
        try:
            result = detect_outliers_zscore(self.df, columns=["p_e"], threshold=3.0)
            self.assertIsNotNone(result)
        except TypeError as e:
            self.fail(f"detect_outliers_zscore should work: {e}")

    def test_detect_outliers_isolation_forest_signature(self):
        """Test detect_outliers_isolation_forest function signature."""
        try:
            result = detect_outliers_isolation_forest(
                self.df, columns=["market_cap", "p_e"], contamination=0.1, random_state=42
            )
            self.assertIsNotNone(result)
        except TypeError as e:
            self.fail(f"detect_outliers_isolation_forest should work: {e}")

    def test_winsorize_by_sector_with_old_parameters(self):
        """Test winsorize_by_sector with OLD parameter names (currently imported version)."""
        # OLD data.py version uses: columns, lower, upper, sector_column
        try:
            result = winsorize_by_sector(
                self.df,
                columns=["market_cap", "p_e"],
                lower=0.01,
                upper=0.99,
                sector_column="sector",
            )
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(self.df))
        except TypeError as e:
            self.fail(f"winsorize_by_sector with OLD parameters should work: {e}")

    def test_winsorize_by_sector_with_new_parameters_fails(self):
        """Test that winsorize_by_sector with NEW parameter names fails when using OLD version."""
        # NEW version would use: lower_percentile, upper_percentile, by_sector
        # But OLD version (currently imported) doesn't accept these
        with self.assertRaises(TypeError) as cm:
            winsorize_by_sector(
                self.df,
                columns=["market_cap"],
                lower_percentile=0.01,  # Wrong for OLD: should be 'lower'
                upper_percentile=0.99,  # Wrong for OLD: should be 'upper'
                by_sector=True,  # Wrong for OLD: should be 'sector_column'
            )
        self.assertIn("unexpected keyword argument", str(cm.exception).lower())

    def test_notebook_pattern_loop_through_columns(self):
        """Test the CORRECT pattern: loop through columns individually for OLD version."""
        # This is the solution recommended in code_guidelines.md Appendix C
        financial_metrics = ["market_cap", "p_e", "profit_margin"]

        outliers_iqr = {}
        for col in financial_metrics:
            try:
                outliers_iqr[col] = detect_outliers_iqr(
                    self.df, column=col, multiplier=1.5  # Singular 'column'
                )
            except TypeError as e:
                self.fail(f"Loop pattern with singular 'column' should work: {e}")

        # Should have results for all columns
        self.assertEqual(len(outliers_iqr), len(financial_metrics))
        for col, result in outliers_iqr.items():
            self.assertIsNotNone(result)


class TestNotebookOutlierDetectionIntegration(unittest.TestCase):
    """Integration tests simulating actual notebook usage."""

    def setUp(self):
        """Create realistic financial dataset."""
        np.random.seed(42)
        n = 200
        self.all_stocks = pd.DataFrame(
            {
                "ticker": [f"STOCK{i:03d}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], n),
                "region": np.random.choice(["US", "EU", "APAC"], n),
                "last_price": np.random.uniform(10, 500, n),
                "market_cap": np.random.lognormal(20, 2, n),
                "p_e": np.random.uniform(5, 50, n),
                "p_b": np.random.uniform(0.5, 10, n),
                "ev_ebitda": np.random.uniform(2, 30, n),
                "operating_margin": np.random.uniform(-0.3, 0.5, n),
                "roe": np.random.uniform(-0.2, 0.4, n),
            }
        )

    def test_notebook_section_outlier_detection_corrected(self):
        """Test the CORRECTED notebook outlier detection section."""
        numeric_cols = self.all_stocks.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        # CORRECTED: Loop through columns individually
        outliers_iqr = {}
        for col in financial_metrics[:5]:  # Test first 5
            outliers_iqr[col] = detect_outliers_iqr(
                self.all_stocks, column=col, multiplier=1.5  # Singular 'column'
            )

        outliers_zscore = {}
        for col in financial_metrics[:5]:
            outliers_zscore[col] = detect_outliers_zscore(
                self.all_stocks, columns=[col], threshold=3.0  # Plural 'columns' works for zscore
            )

        outliers_iforest = {}
        for col in financial_metrics[:5]:
            outliers_iforest[col] = detect_outliers_isolation_forest(
                self.all_stocks,
                columns=[col],  # Plural 'columns'
                contamination=0.1,
                random_state=42,
            )

        # Verify we got results
        self.assertEqual(len(outliers_iqr), 5)
        self.assertEqual(len(outliers_zscore), 5)
        self.assertEqual(len(outliers_iforest), 5)

    def test_notebook_section_winsorization_corrected(self):
        """Test the CORRECTED notebook winsorization section."""
        numeric_cols = self.all_stocks.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        # CORRECTED: Use OLD parameter names (matches current notebook code)
        result = winsorize_by_sector(
            self.all_stocks,
            columns=financial_metrics[:5],
            lower=0.01,  # OLD version: lower (not lower_percentile)
            upper=0.99,  # OLD version: upper (not upper_percentile)
            sector_column="sector",  # OLD version: sector_column (not by_sector)
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.all_stocks))


if __name__ == "__main__":
    unittest.main()
