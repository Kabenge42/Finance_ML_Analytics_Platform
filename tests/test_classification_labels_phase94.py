"""
Phase 9.4 - Classification Labels Enhancement Tests

Tests for new event label creation methods based on Phase 9.3 advanced features.
Follows strict TDD methodology (Red-Green-Refactor).

New methods tested:
1. profitability_event - Based on ROE, ROA, ROIC
2. leverage_event - Based on debt ratios, net_debt/EBITDA
3. liquidity_event - Based on current_ratio, quick_ratio
4. efficiency_event - Based on asset_turnover, inventory_turnover
5. growth_event - Based on revenue_growth, earnings_growth
6. quality_event - Based on accounting quality, analyst quality
7. composite_event - Based on Piotroski F-Score, Altman Z-Score
"""

import unittest
import numpy as np
import pandas as pd

from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels


class TestProfitabilityEventLabels(unittest.TestCase):
    """Test profitability_event label creation method."""

    def setUp(self):
        """Create sample data with profitability ratios."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "sector": [
                    "Tech",
                    "Tech",
                    "Finance",
                    "Finance",
                    "Energy",
                    "Energy",
                    "Health",
                    "Health",
                ],
                "roe": [0.25, 0.18, 0.08, 0.02, 0.15, -0.05, 0.20, 0.12],  # Return on Equity
                "roa": [0.15, 0.10, 0.05, 0.01, 0.08, -0.03, 0.12, 0.07],  # Return on Assets
                "roic": [
                    0.20,
                    0.14,
                    0.06,
                    0.015,
                    0.10,
                    -0.04,
                    0.16,
                    0.09,
                ],  # Return on Invested Capital
            }
        )

    def test_profitability_event_basic(self):
        """Test profitability_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="profitability_event")

        # Should return numpy array
        self.assertIsInstance(labels, np.ndarray)
        # Should have same length as input
        self.assertEqual(len(labels), len(self.df))
        # Labels should be 0, 1, 2, 3, or 4 (5-class)
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_profitability_event_high_profitability_positive(self):
        """Test high profitability stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="profitability_event")

        # Stock A has highest profitability (ROE=0.25, ROA=0.15, ROIC=0.20)
        # Should be labeled as positive (3) or strong positive (4)
        self.assertIn(labels[0], [3, 4])

    def test_profitability_event_low_profitability_negative(self):
        """Test low/negative profitability stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="profitability_event")

        # Stock D has low profitability (ROE=0.02, ROA=0.01, ROIC=0.015)
        # Stock F has negative profitability (ROE=-0.05, ROA=-0.03, ROIC=-0.04)
        # At least one should be labeled as negative (0 or 1)
        self.assertTrue(labels[3] in [0, 1] or labels[5] in [0, 1])

    def test_profitability_event_missing_columns(self):
        """Test profitability_event handles missing columns gracefully."""
        df_incomplete = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Finance", "Energy"],
            }
        )

        labels = create_enhanced_event_labels(df_incomplete, method="profitability_event")

        # Should return all neutral (0) when columns missing
        self.assertTrue(np.all(labels == 0))

    def test_profitability_event_with_nan_values(self):
        """Test profitability_event handles NaN values."""
        df_with_nan = self.df.copy()
        df_with_nan.loc[2, "roe"] = np.nan
        df_with_nan.loc[3, "roa"] = np.nan

        labels = create_enhanced_event_labels(df_with_nan, method="profitability_event")

        # Should still return valid labels (5-class)
        self.assertEqual(len(labels), len(df_with_nan))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))


class TestLeverageEventLabels(unittest.TestCase):
    """Test leverage_event label creation method."""

    def setUp(self):
        """Create sample data with leverage ratios."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
                "sector": [
                    "Tech",
                    "Tech",
                    "Finance",
                    "Finance",
                    "Energy",
                    "Energy",
                    "Health",
                    "Health",
                ],
                "debt_to_equity": [0.3, 0.8, 2.5, 4.0, 1.2, 0.5, 1.8, 0.7],
                "net_debt_to_ebitda": [0.5, 1.5, 4.0, 6.0, 2.0, 0.8, 3.0, 1.2],
                "debt_to_assets": [0.2, 0.4, 0.65, 0.75, 0.5, 0.3, 0.6, 0.35],
            }
        )

    def test_leverage_event_basic(self):
        """Test leverage_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="leverage_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_leverage_event_low_leverage_positive(self):
        """Test low leverage stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="leverage_event")

        # Stock A has low leverage (debt_to_equity=0.3, net_debt_to_ebitda=0.5)
        # Should be labeled as positive (3) or strong positive (4)
        self.assertIn(labels[0], [3, 4])

    def test_leverage_event_high_leverage_negative(self):
        """Test high leverage stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="leverage_event")

        # Stock D has high leverage (debt_to_equity=4.0, net_debt_to_ebitda=6.0)
        # Should be labeled as negative (1) or strong negative (0)
        self.assertIn(labels[3], [0, 1])

    def test_leverage_event_missing_columns(self):
        """Test leverage_event handles missing columns."""
        df_incomplete = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "sector": ["Tech", "Finance"],
            }
        )

        labels = create_enhanced_event_labels(df_incomplete, method="leverage_event")
        self.assertTrue(np.all(labels == 0))


class TestLiquidityEventLabels(unittest.TestCase):
    """Test liquidity_event label creation method."""

    def setUp(self):
        """Create sample data with liquidity ratios."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Finance", "Energy", "Health", "Retail", "Manufacturing"],
                "current_ratio": [3.5, 2.0, 1.2, 0.8, 1.8, 1.5],
                "quick_ratio": [2.8, 1.6, 0.9, 0.5, 1.4, 1.1],
                "cash_ratio": [1.5, 0.8, 0.4, 0.2, 0.7, 0.5],
            }
        )

    def test_liquidity_event_basic(self):
        """Test liquidity_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="liquidity_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_liquidity_event_high_liquidity_positive(self):
        """Test high liquidity stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="liquidity_event")

        # Stock A has high liquidity (current_ratio=3.5, quick_ratio=2.8)
        self.assertIn(labels[0], [3, 4])

    def test_liquidity_event_low_liquidity_negative(self):
        """Test low liquidity stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="liquidity_event")

        # Stock D has low liquidity (current_ratio=0.8, quick_ratio=0.5)
        self.assertIn(labels[3], [0, 1])


class TestEfficiencyEventLabels(unittest.TestCase):
    """Test efficiency_event label creation method."""

    def setUp(self):
        """Create sample data with efficiency ratios."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Retail", "Manufacturing", "Energy", "Finance", "Health"],
                "asset_turnover": [1.8, 2.5, 1.2, 0.6, 0.3, 1.0],
                "inventory_turnover": [
                    8.0,
                    12.0,
                    6.0,
                    3.0,
                    np.nan,
                    7.0,
                ],  # Finance has no inventory
                "receivables_turnover": [10.0, 15.0, 8.0, 5.0, 12.0, 9.0],
            }
        )

    def test_efficiency_event_basic(self):
        """Test efficiency_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="efficiency_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_efficiency_event_high_efficiency_positive(self):
        """Test high efficiency stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="efficiency_event")

        # Stock B has high efficiency (asset_turnover=2.5, inventory_turnover=12.0)
        self.assertIn(labels[1], [3, 4])

    def test_efficiency_event_low_efficiency_negative(self):
        """Test low efficiency stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="efficiency_event")

        # Stock D has low efficiency (asset_turnover=0.6, inventory_turnover=3.0)
        self.assertIn(labels[3], [0, 1])


class TestGrowthEventLabels(unittest.TestCase):
    """Test growth_event label creation method."""

    def setUp(self):
        """Create sample data with growth metrics."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Tech", "Finance", "Energy", "Health", "Retail"],
                "revenue_growth": [0.25, 0.15, 0.05, -0.10, 0.18, 0.02],
                "earnings_growth": [0.30, 0.20, 0.08, -0.15, 0.22, 0.03],
                "ebitda_growth": [0.28, 0.18, 0.06, -0.12, 0.20, 0.04],
            }
        )

    def test_growth_event_basic(self):
        """Test growth_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="growth_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_growth_event_high_growth_positive(self):
        """Test high growth stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="growth_event")

        # Stock A has high growth (revenue=0.25, earnings=0.30, ebitda=0.28)
        self.assertIn(labels[0], [3, 4])

    def test_growth_event_negative_growth_negative(self):
        """Test negative growth stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="growth_event")

        # Stock D has negative growth (revenue=-0.10, earnings=-0.15)
        self.assertIn(labels[3], [0, 1])


class TestQualityEventLabels(unittest.TestCase):
    """Test quality_event label creation method."""

    def setUp(self):
        """Create sample data with quality metrics (Phase 9.3 columns)."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Finance", "Energy", "Health", "Retail", "Manufacturing"],
                # Phase 9.3 generated quality columns
                "accounting_quality_score": [95, 80, 60, 20, 70, 65],  # Higher is better (0-100)
                "analyst_coverage_quality": [
                    0.85,
                    0.70,
                    0.55,
                    0.30,
                    0.65,
                    0.60,
                ],  # Higher is better
                "exceptional_items_to_ebitda": [
                    0.01,
                    0.05,
                    0.15,
                    0.35,
                    0.10,
                    0.12,
                ],  # Lower is better
                "has_goodwill_impairment": [0, 0, 0, 1, 0, 0],  # Binary flag, lower is better
                "has_asset_writedown": [0, 0, 1, 1, 0, 0],  # Binary flag, lower is better
            }
        )

    def test_quality_event_basic(self):
        """Test quality_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="quality_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_quality_event_high_quality_positive(self):
        """Test high quality stocks get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="quality_event")

        # Stock A has high quality (low accruals, low DSO, high analyst scores)
        self.assertIn(labels[0], [3, 4])

    def test_quality_event_low_quality_negative(self):
        """Test low quality stocks get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="quality_event")

        # Stock D has low quality (high accruals, high DSO, low analyst scores)
        self.assertIn(labels[3], [0, 1])


class TestDividendEventLabels(unittest.TestCase):
    """Test dividend_event label creation method (Phase 9.3 dividend features).

    This suite verifies that the new dividend_event method in
    create_enhanced_event_labels correctly interprets the Phase 9.3
    dividend reliability features engineered by
    engineer_dividend_reliability_features:

    - dividend_consistency_score (0-100, higher is better)
    - income_stock_flag (1 for reliable income stocks)
    - dividend_payout_ratio (very high payout is a risk)
    """

    def setUp(self):
        """Create sample data with dividend reliability features."""

        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D"],
                # High consistency, income flag, reasonable payout
                "dividend_consistency_score": [85.0, 20.0, 60.0, 40.0],
                "income_stock_flag": [1, 0, 1, 0],
                # Payout ratios: moderate, very high, low, negative (edge)
                "dividend_payout_ratio": [0.6, 1.8, 0.3, -0.1],
            }
        )

    def test_dividend_event_basic(self):
        """Test dividend_event method returns valid 5-class labels."""

        labels = create_enhanced_event_labels(self.df, method="dividend_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_dividend_event_high_reliability_positive(self):
        """High consistency & income flag should map to positive label (3 or 4)."""

        labels = create_enhanced_event_labels(self.df, method="dividend_event")

        # Stock A: high consistency, income flag, moderate payout
        self.assertIn(labels[0], [3, 4])

    def test_dividend_event_very_high_payout_negative(self):
        """Very high payout ratio should act as a negative signal."""

        labels = create_enhanced_event_labels(self.df, method="dividend_event")

        # Stock B: low consistency, no income flag, very high payout
        self.assertIn(labels[1], [0, 1])

    def test_dividend_event_missing_columns(self):
        """Method should return all neutral when dividend features are absent."""

        df_incomplete = pd.DataFrame({"ticker": ["X", "Y"]})

        labels = create_enhanced_event_labels(df_incomplete, method="dividend_event")
        self.assertTrue(np.all(labels == 0))


class TestValuationEventPhase93SchemaFallback(unittest.TestCase):
    """Phase 9.3: valuation_event should use Schema 1.3 columns when ratios missing.

    These tests ensure that create_enhanced_event_labels can fall back to
    preprocessed valuation columns such as p_e_ntm, p_b_ltm, and
    ev_ebitda_ltm when Phase 9.3 engineered ratios like p_e_ratio or
    ev_ebitda_ratio are not present.
    """

    def test_valuation_uses_p_e_ntm_when_ratios_absent(self):
        """With only p_e_ntm available, valuation labels should not be all neutral.

        This approximates the preprocessed_stocks_metadata.json structure,
        where raw P/E timelines (p_e_ntm, p_e_ltm, etc.) are present even if
        p_e_ratio has not yet been engineered.
        """

        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech"] * 5,
                # Monotonic P/E values to induce a clear ordering
                "p_e_ntm": [5.0, 10.0, 15.0, 20.0, 25.0],
            }
        )

        labels = create_enhanced_event_labels(df, method="valuation")

        # Basic invariants
        self.assertEqual(len(labels), len(df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

        # If p_e_ntm was ignored, the implementation would log a warning and
        # return all-neutral (0). We assert that at least one label is
        # non-zero to confirm the fallback column was used.
        self.assertTrue(np.any(labels != 0), msg="valuation_event produced all neutral labels")


class TestCompositeEventLabels(unittest.TestCase):
    """Test composite_event label creation method."""

    def setUp(self):
        """Create sample data with composite scores."""
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E", "F"],
                "sector": ["Tech", "Finance", "Energy", "Health", "Retail", "Manufacturing"],
                "piotroski_f_score": [8, 7, 5, 2, 6, 4],  # 0-9 scale, higher is better
                "altman_z_score": [4.5, 3.2, 2.0, 0.8, 2.8, 1.5],  # >2.99 safe, <1.81 distress
                "beneish_m_score": [
                    -2.5,
                    -1.8,
                    -1.2,
                    0.5,
                    -1.5,
                    -0.8,
                ],  # <-1.78 unlikely manipulator
            }
        )

    def test_composite_event_basic(self):
        """Test composite_event method returns valid labels."""
        labels = create_enhanced_event_labels(self.df, method="composite_event")

        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), len(self.df))
        self.assertTrue(np.all(np.isin(labels, [0, 1, 2, 3, 4])))

    def test_composite_event_high_scores_positive(self):
        """Test high composite scores get positive label (3 or 4)."""
        labels = create_enhanced_event_labels(self.df, method="composite_event")

        # Stock A has high scores (Piotroski=8, Altman=4.5, Beneish=-2.5)
        self.assertIn(labels[0], [3, 4])

    def test_composite_event_low_scores_negative(self):
        """Test low composite scores get negative label (0 or 1)."""
        labels = create_enhanced_event_labels(self.df, method="composite_event")

        # Stock D has low scores (Piotroski=2, Altman=0.8, Beneish=0.5)
        self.assertIn(labels[3], [0, 1])

    def test_composite_event_missing_columns(self):
        """Test composite_event handles missing columns."""
        df_incomplete = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "sector": ["Tech", "Finance"],
            }
        )

        labels = create_enhanced_event_labels(df_incomplete, method="composite_event")
        self.assertTrue(np.all(labels == 0))


class TestEventLabelsEdgeCases(unittest.TestCase):
    """Test edge cases for all new event label methods."""

    def test_empty_dataframe(self):
        """Test all methods handle empty DataFrames."""
        df_empty = pd.DataFrame()

        methods = [
            "profitability_event",
            "leverage_event",
            "liquidity_event",
            "efficiency_event",
            "growth_event",
            "quality_event",
            "composite_event",
            "dividend_event",
        ]

        for method in methods:
            with self.subTest(method=method):
                labels = create_enhanced_event_labels(df_empty, method=method)
                self.assertEqual(len(labels), 0)

    def test_single_row_dataframe(self):
        """Test all methods handle single-row DataFrames."""
        df_single = pd.DataFrame(
            {
                "ticker": ["A"],
                "sector": ["Tech"],
                "roe": [0.15],
                "current_ratio": [2.0],
                "revenue_growth": [0.10],
            }
        )

        methods = [
            "profitability_event",
            "leverage_event",
            "liquidity_event",
            "efficiency_event",
            "growth_event",
            "quality_event",
            "composite_event",
            "dividend_event",
        ]

        for method in methods:
            with self.subTest(method=method):
                labels = create_enhanced_event_labels(df_single, method=method)
                self.assertEqual(len(labels), 1)
                self.assertIn(labels[0], [0, 1, 2, 3, 4])

    def test_all_nan_columns(self):
        """Test methods handle all-NaN feature columns."""
        df_nan = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Finance", "Energy"],
                "roe": [np.nan, np.nan, np.nan],
                "current_ratio": [np.nan, np.nan, np.nan],
            }
        )

        methods = [
            "profitability_event",
            "leverage_event",
            "liquidity_event",
            "efficiency_event",
            "growth_event",
            "quality_event",
            "composite_event",
            "dividend_event",
        ]

        for method in methods:
            with self.subTest(method=method):
                labels = create_enhanced_event_labels(df_nan, method=method)
                # Should return all neutral when no valid data
                self.assertTrue(np.all(labels == 0))


class TestBackwardCompatibility(unittest.TestCase):
    """Test that existing methods still work after adding new methods."""

    def test_existing_methods_still_work(self):
        """Test all 6 original methods still function correctly."""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "sector": ["Tech", "Finance", "Energy"],
                "price_target": [120, 80, 50],
                "last_price": [100, 100, 100],
                "p_e": [15, 25, 35],
            }
        )

        existing_methods = [
            "price_momentum",
            "valuation",
            "fundamental",
            "volatility",
            "analyst_rating",
            "market_events",
        ]

        for method in existing_methods:
            with self.subTest(method=method):
                labels = create_enhanced_event_labels(df, method=method)
                self.assertIsInstance(labels, np.ndarray)
                self.assertEqual(len(labels), 3)


if __name__ == "__main__":
    unittest.main()
