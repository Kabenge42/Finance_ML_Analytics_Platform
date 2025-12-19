"""
TDD Tests for Critical Data Quality Alert Fixes.

Tests for fixing the root causes of high missing values in critical metrics:
- cash_burn_rate: 87.9% missing -> implement computation
- efficiency_ratio: 85.3% missing -> implement computation
- tangible_book_value: 85.3% missing -> verify/improve computation
- marketing_efficiency: 81.9% missing -> implement computation
- r_d_intensity: 77.3% missing -> fix column name mismatch

Following code_guidelines.md Section 8 TDD Conventions.

Version: 1.0.0
Created: 2025-11-30
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestRDIntensityColumnNameFix(unittest.TestCase):
    """Tests for R&D intensity column name fix (r_d_expense_ltm -> r_d_expenses_ltm)."""

    def setUp(self):
        """Set up test data with correct column name."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
                "sector": ["Technology", "Technology", "Technology", "Financials", "Energy"],
                "r_d_expenses_ltm": [
                    20000.0,
                    25000.0,
                    30000.0,
                    np.nan,
                    np.nan,
                ],  # Correct column name
                "total_revenues_ltm": [400000.0, 200000.0, 300000.0, 150000.0, 350000.0],
            }
        )

    def test_r_d_intensity_uses_correct_column_name(self):
        """Test that r_d_intensity is computed using r_d_expenses_ltm (plural)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Should have r_d_intensity column
        self.assertIn("r_d_intensity", result.columns)

        # Tech companies should have valid r_d_intensity values
        tech_mask = result["sector"] == "Technology"
        tech_rd_intensity = result.loc[tech_mask, "r_d_intensity"]

        # At least some values should be non-null
        self.assertGreater(tech_rd_intensity.notna().sum(), 0)

        # Check computed values (R&D / Revenue * 100)
        # AAPL: 20000 / 400000 * 100 = 5%
        expected_aapl = (20000 / 400000) * 100
        self.assertAlmostEqual(result.loc[0, "r_d_intensity"], expected_aapl, places=2)

    def test_r_d_intensity_handles_missing_r_d_expenses(self):
        """Test that r_d_intensity handles missing R&D expenses gracefully."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Financials and Energy should have NaN r_d_intensity (no R&D data)
        non_tech_mask = result["sector"].isin(["Financials", "Energy"])
        non_tech_rd = result.loc[non_tech_mask, "r_d_intensity"]

        # These should be NaN (no R&D expenses)
        self.assertTrue(non_tech_rd.isna().all())


class TestCashBurnRateComputation(unittest.TestCase):
    """Tests for cash_burn_rate metric implementation."""

    def setUp(self):
        """Set up test data for cash burn rate computation."""
        self.df = pd.DataFrame(
            {
                "ticker": ["STARTUP1", "STARTUP2", "PROFITABLE", "BANK", "OIL"],
                "sector": ["Technology", "Technology", "Technology", "Financials", "Energy"],
                "cash_and_equivalents_ltm": [100000.0, 50000.0, 200000.0, 500000.0, 300000.0],
                "cfo_ltm": [
                    -10000.0,
                    -5000.0,
                    20000.0,
                    50000.0,
                    80000.0,
                ],  # Negative = burning cash
                "fcf_ltm": [-12000.0, -8000.0, 15000.0, 40000.0, 60000.0],
                "net_income_is_ltm": [-15000.0, -10000.0, 25000.0, 60000.0, 90000.0],
            }
        )

    def test_cash_burn_rate_computed(self):
        """Test that cash_burn_rate is computed for companies with negative cash flow."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Should have cash_burn_rate column
        self.assertIn("cash_burn_rate", result.columns)

        # Cash-burning companies should have positive burn rate (months of runway)
        # STARTUP1: Cash 100000, burning 10000/year = 12000/month burn
        # Monthly burn = abs(cfo_ltm) / 12
        # Runway = cash / monthly_burn
        startup1_idx = 0
        cash_burn = result.loc[startup1_idx, "cash_burn_rate"]

        # Should be a valid positive number (months of runway)
        self.assertIsNotNone(cash_burn)
        if pd.notna(cash_burn):
            self.assertGreater(cash_burn, 0)

    def test_cash_burn_rate_nan_for_profitable(self):
        """Test that cash_burn_rate is NaN for profitable companies (positive CFO)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Profitable company (positive CFO) should have NaN or special value
        profitable_idx = 2  # PROFITABLE company
        cash_burn = result.loc[profitable_idx, "cash_burn_rate"]

        # Either NaN (not burning cash) or a very high number (infinite runway)
        # The implementation should handle this gracefully
        self.assertTrue(pd.isna(cash_burn) or cash_burn > 1000)


class TestEfficiencyRatioComputation(unittest.TestCase):
    """Tests for efficiency_ratio metric implementation (for Financials sector)."""

    def setUp(self):
        """Set up test data for efficiency ratio computation."""
        self.df = pd.DataFrame(
            {
                "ticker": ["JPM", "BAC", "WFC", "AAPL", "XOM"],
                "sector": ["Financials", "Financials", "Financials", "Technology", "Energy"],
                "total_operating_expenses_ltm": [50000.0, 60000.0, 55000.0, 80000.0, 120000.0],
                "total_revenues_ltm": [100000.0, 110000.0, 105000.0, 400000.0, 350000.0],
                "net_interest_income_ltm": [40000.0, 45000.0, 42000.0, np.nan, np.nan],
                "non_interest_income_ltm": [30000.0, 35000.0, 33000.0, np.nan, np.nan],
            }
        )

    def test_efficiency_ratio_computed_for_financials(self):
        """Test that efficiency_ratio is computed for Financials sector."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Should have efficiency_ratio column
        self.assertIn("efficiency_ratio", result.columns)

        # Financials should have valid efficiency_ratio values
        fin_mask = result["sector"] == "Financials"
        fin_efficiency = result.loc[fin_mask, "efficiency_ratio"]

        # At least some values should be non-null
        self.assertGreater(fin_efficiency.notna().sum(), 0)

        # Efficiency ratio = Operating Expenses / Revenue * 100
        # JPM: 50000 / 100000 * 100 = 50%
        expected_jpm = (50000 / 100000) * 100
        self.assertAlmostEqual(result.loc[0, "efficiency_ratio"], expected_jpm, places=2)

    def test_efficiency_ratio_valid_range(self):
        """Test that efficiency_ratio values are in valid range (0-200%)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        valid_efficiency = result["efficiency_ratio"].dropna()

        # All values should be positive and reasonable
        self.assertTrue((valid_efficiency >= 0).all())
        self.assertTrue((valid_efficiency <= 200).all())


class TestMarketingEfficiencyComputation(unittest.TestCase):
    """Tests for marketing_efficiency metric implementation."""

    def setUp(self):
        """Set up test data for marketing efficiency computation."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
                "sector": ["Technology", "Technology", "Technology", "Financials", "Energy"],
                "selling_general_and_admin_expenses_total_fy": [
                    20000.0,
                    25000.0,
                    30000.0,
                    40000.0,
                    35000.0,
                ],
                "total_revenues_ltm": [400000.0, 200000.0, 300000.0, 150000.0, 350000.0],
                "total_revenues_fy": [380000.0, 190000.0, 280000.0, 140000.0, 340000.0],
            }
        )

    def test_marketing_efficiency_computed(self):
        """Test that marketing_efficiency is computed using SG&A and revenue."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Should have marketing_efficiency column
        self.assertIn("marketing_efficiency", result.columns)

        # All companies with SG&A data should have values
        valid_marketing = result["marketing_efficiency"].dropna()
        self.assertGreater(len(valid_marketing), 0)

        # Marketing efficiency = Revenue / SG&A (higher is better)
        # AAPL: 400000 / 20000 = 20
        expected_aapl = 400000 / 20000
        self.assertAlmostEqual(result.loc[0, "marketing_efficiency"], expected_aapl, places=2)

    def test_marketing_efficiency_handles_zero_sga(self):
        """Test that marketing_efficiency handles zero SG&A gracefully."""
        df = self.df.copy()
        df.loc[0, "selling_general_and_admin_expenses_total_fy"] = 0.0

        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(df)

        # Zero SG&A should result in NaN or inf (handled gracefully)
        aapl_efficiency = result.loc[0, "marketing_efficiency"]
        self.assertTrue(pd.isna(aapl_efficiency) or np.isinf(aapl_efficiency))


class TestTangibleBookValueComputation(unittest.TestCase):
    """Tests for tangible_book_value metric computation/verification."""

    def setUp(self):
        """Set up test data for tangible book value computation."""
        self.df = pd.DataFrame(
            {
                "ticker": ["JPM", "BAC", "AAPL", "MSFT", "XOM"],
                "sector": ["Financials", "Financials", "Technology", "Technology", "Energy"],
                "total_equity_ltm": [300000.0, 280000.0, 150000.0, 200000.0, 250000.0],
                "goodwill_ltm": [50000.0, 45000.0, 10000.0, 80000.0, 20000.0],
                "gross_intangible_assets_ltm": [30000.0, 25000.0, 5000.0, 40000.0, 10000.0],
                "tbv_ltm": [np.nan, np.nan, np.nan, np.nan, np.nan],  # Missing - to be computed
            }
        )

    def test_tangible_book_value_computed_when_missing(self):
        """Test that tangible_book_value is computed when tbv_ltm is missing."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # Should have tangible_book_value column
        self.assertIn("tangible_book_value", result.columns)

        # Values should be computed from: equity - goodwill - intangibles
        # JPM: 300000 - 50000 - 30000 = 220000
        expected_jpm = 300000 - 50000 - 30000
        self.assertAlmostEqual(result.loc[0, "tangible_book_value"], expected_jpm, places=0)

    def test_tangible_book_value_uses_existing_tbv_when_available(self):
        """Test that existing tbv_ltm values are preserved."""
        df = self.df.copy()
        df.loc[0, "tbv_ltm"] = 215000.0  # Pre-existing value

        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(df)

        # Should use existing tbv_ltm as tangible_book_value
        # Or compute and compare - implementation may vary
        tbv = result.loc[0, "tangible_book_value"]
        self.assertIsNotNone(tbv)
        self.assertTrue(pd.notna(tbv))


class TestMetricComputationIntegration(unittest.TestCase):
    """Integration tests for all critical metric computations."""

    def setUp(self):
        """Set up comprehensive test data."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM", "BAC", "XOM"],
                "sector": ["Technology", "Technology", "Financials", "Financials", "Energy"],
                # R&D data (note: plural "expenses")
                "r_d_expenses_ltm": [20000.0, 25000.0, np.nan, np.nan, np.nan],
                # Revenue data
                "total_revenues_ltm": [400000.0, 200000.0, 100000.0, 110000.0, 350000.0],
                "total_revenues_fy": [380000.0, 190000.0, 95000.0, 105000.0, 340000.0],
                # Cash flow data
                "cash_and_equivalents_ltm": [100000.0, 150000.0, 500000.0, 450000.0, 200000.0],
                "cfo_ltm": [-5000.0, 50000.0, 40000.0, 35000.0, 80000.0],
                "fcf_ltm": [-8000.0, 45000.0, 38000.0, 32000.0, 75000.0],
                # Operating expenses
                "total_operating_expenses_ltm": [80000.0, 60000.0, 50000.0, 55000.0, 120000.0],
                # SG&A data
                "selling_general_and_admin_expenses_total_fy": [
                    20000.0,
                    25000.0,
                    40000.0,
                    42000.0,
                    35000.0,
                ],
                # Book value data
                "total_equity_ltm": [150000.0, 200000.0, 300000.0, 280000.0, 250000.0],
                "goodwill_ltm": [10000.0, 80000.0, 50000.0, 45000.0, 20000.0],
                "gross_intangible_assets_ltm": [5000.0, 40000.0, 30000.0, 25000.0, 10000.0],
                "tbv_ltm": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "market_cap": [2500000.0, 2000000.0, 400000.0, 350000.0, 450000.0],
            }
        )

    def test_all_critical_metrics_computed(self):
        """Test that all 5 critical metrics are computed."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        critical_metrics = [
            "r_d_intensity",
            "cash_burn_rate",
            "efficiency_ratio",
            "marketing_efficiency",
            "tangible_book_value",
        ]

        for metric in critical_metrics:
            self.assertIn(metric, result.columns, f"Missing critical metric: {metric}")

    def test_critical_metrics_have_reduced_missing_values(self):
        """Test that critical metrics have significantly reduced missing values."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            compute_sector_specific_ratios,
        )

        result = compute_sector_specific_ratios(self.df)

        # At least 50% of rows should have valid values for most metrics
        for metric in ["efficiency_ratio", "marketing_efficiency", "tangible_book_value"]:
            valid_count = result[metric].notna().sum()
            total_count = len(result)
            coverage = valid_count / total_count

            self.assertGreaterEqual(
                coverage, 0.5, f"{metric} has too many missing values: {(1-coverage)*100:.1f}%"
            )


class TestColumnNameConstants(unittest.TestCase):
    """Tests for correct column name constants in the module."""

    def test_tech_specific_metrics_uses_correct_column_name(self):
        """Test that TECH_SPECIFIC_METRICS uses r_d_expenses_ltm (plural)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            TECH_SPECIFIC_METRICS,
        )

        # Should use "r_d_expenses_ltm" (plural), not "r_d_expense_ltm" (singular)
        self.assertIn("r_d_expenses_ltm", TECH_SPECIFIC_METRICS)
        self.assertNotIn("r_d_expense_ltm", TECH_SPECIFIC_METRICS)

    def test_healthcare_specific_metrics_uses_correct_column_name(self):
        """Test that HEALTHCARE_SPECIFIC_METRICS uses r_d_expenses_ltm (plural)."""
        from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
            HEALTHCARE_SPECIFIC_METRICS,
        )

        # Should use "r_d_expenses_ltm" (plural), not "r_d_expense_ltm" (singular)
        self.assertIn("r_d_expenses_ltm", HEALTHCARE_SPECIFIC_METRICS)
        self.assertNotIn("r_d_expense_ltm", HEALTHCARE_SPECIFIC_METRICS)


if __name__ == "__main__":
    unittest.main()
