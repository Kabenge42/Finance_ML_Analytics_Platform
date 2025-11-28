import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.analytics.eval import filter_stocks_by_criteria


def create_sample_portfolio_data() -> pd.DataFrame:
    """Small helper to build a deterministic sample universe for tests."""

    data = {
        "ticker": ["A", "B", "C", "D", "E"],
        "sector": [
            "Technology",
            "Technology",
            "Healthcare",
            "Finance",
            "Energy",
        ],
        "region": ["US", "EU", "US", "EU", "APAC"],
        # Market caps expressed in absolute dollars
        "market_cap": [60e9, 120e9, 15e9, 800e6, 5e9],
        # Mispricing in percentage terms
        "mispricing_pct": [10.0, 3.0, 8.0, -2.0, 6.0],
        "valuation_category": [
            "Undervalued",
            "Fair Value",
            "Undervalued",
            "Overvalued",
            "Undervalued",
        ],
    }
    return pd.DataFrame(data)


class TestFilterStocksWithCapUnits(unittest.TestCase):
    """Phase 1.1.1 – tests for cap_unit support in filter_stocks_by_criteria.

    These tests are derived from the TDD examples in
    docs/improvement_plan/portfolio_optimization_enhancement_plan.md
    and ensure that market-cap thresholds can be expressed in
    billions (B), millions (M), or thousands (K) while the underlying
    data remains in absolute currency units.
    """

    def test_filter_stocks_with_market_cap_units(self):
        """Filter with market cap thresholds expressed in B and M units.

        The sample data uses absolute market_cap values in dollars. When the
        thresholds are provided in billions (B) or millions (M), the function
        should internally scale these values before applying the filter.
        """

        df = create_sample_portfolio_data()

        # Billion‑dollar range: 50B–500B
        filtered_b = filter_stocks_by_criteria(
            df,
            min_market_cap=50,
            max_market_cap=500,
            cap_unit="B",
        )

        self.assertFalse(filtered_b.empty)
        self.assertTrue((filtered_b["market_cap"] >= 50e9).all())
        self.assertTrue((filtered_b["market_cap"] <= 500e9).all())

        # Million‑dollar range: 100M–1,000M
        filtered_m = filter_stocks_by_criteria(
            df,
            min_market_cap=100,
            max_market_cap=1000,
            cap_unit="M",
        )

        self.assertFalse(filtered_m.empty)
        self.assertTrue((filtered_m["market_cap"] >= 100e6).all())
        self.assertTrue((filtered_m["market_cap"] <= 1000e6).all())


class TestFilterByMultipleCriteria(unittest.TestCase):
    """Phase 1.1.2 – combined sector/region/mispricing filters."""

    def test_filter_by_multiple_criteria(self):
        """Filter by sector, region, market cap and mispricing jointly.

        This mirrors the example in the enhancement plan and verifies that
        all criteria are applied conjunctively (logical AND).
        """

        df = create_sample_portfolio_data()

        filtered = filter_stocks_by_criteria(
            df,
            sectors=["Technology", "Healthcare"],
            regions=["US", "EU"],
            min_market_cap=10,  # Interpreted as 10B when cap_unit="B"
            min_mispricing=5.0,
            valuation_categories=["Undervalued", "Fair Value"],
            cap_unit="B",
        )

        # Sector and region should be within requested sets
        self.assertTrue(set(filtered["sector"]).issubset({"Technology", "Healthcare"}))
        self.assertTrue(set(filtered["region"]).issubset({"US", "EU"}))

        # Market cap and mispricing thresholds should hold
        self.assertTrue((filtered["market_cap"] >= 10e9).all())
        self.assertTrue((filtered["mispricing_pct"] >= 5.0).all())


if __name__ == "__main__":  # pragma: no cover - direct execution helper
    unittest.main()
