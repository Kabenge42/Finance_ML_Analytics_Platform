"""Quick validation tests for visualization module refactoring."""
import unittest
import pandas as pd
import numpy as np


class TestSharedAliases(unittest.TestCase):
    def test_new_aliases_resolve(self):
        from finance_ml.analytics.visualizations._shared import resolve_column

        df = pd.DataFrame({
            "growth_ebitda_growth_yoy": [1],
            "valuation_dividend_yield": [2],
            "accounting_quality_score_comp": [3],
            "rnd_intensity_ltm": [4],
            "earnings_quality_composite": [5],
            "total_asset_turnover": [6],
        })
        self.assertEqual(resolve_column(df, "ebitda_growth_yoy"), "growth_ebitda_growth_yoy")
        self.assertEqual(resolve_column(df, "dividend_yield"), "valuation_dividend_yield")
        self.assertEqual(resolve_column(df, "accounting_quality_score"), "accounting_quality_score_comp")
        self.assertEqual(resolve_column(df, "rnd_intensity"), "rnd_intensity_ltm")
        self.assertEqual(resolve_column(df, "earnings_quality_composite"), "earnings_quality_composite")
        self.assertEqual(resolve_column(df, "asset_turnover"), "total_asset_turnover")

    def test_direct_column_preferred(self):
        from finance_ml.analytics.visualizations._shared import resolve_column

        df = pd.DataFrame({"asset_turnover": [1], "total_asset_turnover": [2]})
        self.assertEqual(resolve_column(df, "asset_turnover"), "asset_turnover")


class TestExpectedReturnsViz(unittest.TestCase):
    def test_mc_return_distribution_empty(self):
        from finance_ml.analytics.visualizations.expected_returns_viz import create_mc_return_distribution
        fig = create_mc_return_distribution(pd.DataFrame())
        self.assertIn("no data", fig.layout.annotations[0].text.lower())

    def test_var_analysis_empty(self):
        from finance_ml.analytics.visualizations.expected_returns_viz import create_var_analysis
        fig = create_var_analysis(pd.DataFrame())
        self.assertIn("no data", fig.layout.annotations[0].text.lower())

    def test_sector_heatmap_uses_industry(self):
        from finance_ml.analytics.visualizations.expected_returns_viz import create_sector_return_analytics_heatmap
        sa = pd.DataFrame({"industry": ["Tech", "Health"], "mc_mean": [10, 20], "mc_median": [8, 18]})
        fig = create_sector_return_analytics_heatmap(sa)
        self.assertTrue(len(fig.data) > 0)

    def test_sector_heatmap_fallback_sector(self):
        from finance_ml.analytics.visualizations.expected_returns_viz import create_sector_return_analytics_heatmap
        sa = pd.DataFrame({"sector": ["Tech"], "mc_mean": [10]})
        fig = create_sector_return_analytics_heatmap(sa)
        self.assertTrue(len(fig.data) > 0)

    def test_model_dispersion_agreement_score_3(self):
        """Verify consensus uses ==3 not ==4."""
        from finance_ml.analytics.visualizations.expected_returns_viz import create_model_dispersion_dashboard
        summary = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(20)],
            "industry": ["Tech"] * 20,
            "expected_upside_pct": np.random.randn(20) * 10,
            "filtered_upside": np.random.randn(20) * 10,
            "expected_return_prob_weighted": np.random.randn(20) * 10,
            "agreement_score": [3] * 10 + [1] * 10,
            "weighted_agreement": np.random.rand(20),
        })
        fig = create_model_dispersion_dashboard(summary)
        # Should have traces (not empty)
        self.assertTrue(len(fig.data) > 0)


class TestCategoryCharts(unittest.TestCase):
    def test_accounting_quality_uses_correct_column(self):
        from finance_ml.analytics.visualizations.category_charts import create_accounting_quality_breakdown
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "accounting_quality_score": [75.0],
            "non_operating_income_share": [5.0],
            "gaap_adj_eps_gap_pct": [2.0],
            "asset_sale_boost": [1.0],
            "exceptional_items_frequency": [0.5],
        })
        fig = create_accounting_quality_breakdown(df, "AAPL")
        # Should produce a radar chart with data
        self.assertTrue(len(fig.data) > 0)

    def test_balance_sheet_uses_schema_columns(self):
        from finance_ml.analytics.visualizations.category_charts import create_balance_sheet_composition_chart
        df = pd.DataFrame({
            "industry": ["Tech", "Health"],
            "assets_fq": [100, 200],
            "debt_fq": [50, 80],
            "wc_fq": [30, 40],
        })
        fig = create_balance_sheet_composition_chart(df)
        self.assertTrue(len(fig.data) > 0)

    def test_growth_correlation_resolves_aliases(self):
        from finance_ml.analytics.visualizations.category_charts import create_growth_correlation_heatmap
        df = pd.DataFrame({
            "revenue_growth_yoy": np.random.randn(50),
            "ebitda_growth_yoy": np.random.randn(50),
            "eps_yoy_growth": np.random.randn(50),
        })
        fig = create_growth_correlation_heatmap(df)
        self.assertTrue(len(fig.data) > 0)


class TestGrowthAnalysis(unittest.TestCase):
    def test_default_param_changed(self):
        from finance_ml.analytics.visualizations.growth_analysis import create_growth_vs_profitability_quadrant
        import inspect
        sig = inspect.signature(create_growth_vs_profitability_quadrant)
        self.assertEqual(sig.parameters["growth_metric"].default, "revenue_growth_yoy")


class TestExpectedReturnsV3(unittest.TestCase):
    def test_compute_sector_return_analytics_uses_group_col(self):
        import sys
        sys.path.insert(0, ".")
        from expected_returns_v3 import compute_sector_return_analytics
        summary = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "industry": ["Tech", "Tech", "Health"],
            "expected_upside_pct": [10, 20, 30],
            "agreement_score": [3, 2, 3],
        })
        result = compute_sector_return_analytics(summary)
        self.assertIn("industry", result.columns)
        self.assertNotIn("sector", result.columns)

    def test_pct_full_consensus_uses_3(self):
        import sys
        sys.path.insert(0, ".")
        from expected_returns_v3 import compute_sector_return_analytics
        summary = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(10)],
            "industry": ["Tech"] * 10,
            "expected_upside_pct": [10] * 10,
            "agreement_score": [3] * 5 + [2] * 5,
        })
        result = compute_sector_return_analytics(summary)
        self.assertAlmostEqual(result["pct_full_consensus"].iloc[0], 50.0)


if __name__ == "__main__":
    unittest.main()
