import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import plotly.graph_objects as go


class TestViewAnalyticsEnhancements(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "isin": ["ISIN1", "ISIN2"],
                "ticker": ["TICK1", "TICK2"],
                "name": ["Name1", "Name2"],
                "industry": ["Ind1", "Ind2"],
                "sector": ["Sec1", "Sec2"],
                "p_e_ratio": [15.0, 20.0],
                "distress_risk_score": [20.0, 40.0],
            }
        )

    @patch("finance_ml.analytics.data_utils.create_engine")
    @patch("pandas.read_sql")
    def test_load_all_feature_views(self, mock_read_sql, mock_create_engine):
        from finance_ml.analytics.data_utils import load_all_feature_views

        mock_read_sql.return_value = self.sample_df
        os.environ["DB_URL"] = "postgresql://user:pass@host/db"

        # Test returning dict
        result_dict = load_all_feature_views(views=["vw_features_momentum"], return_dict=True)
        self.assertIn("vw_features_momentum", result_dict)
        self.assertEqual(len(result_dict["vw_features_momentum"]), 2)

        # Test merged dataframe
        df_merged = load_all_feature_views(views=["vw_features_momentum"], return_dict=False)
        self.assertIsInstance(df_merged, pd.DataFrame)
        self.assertIn("p_e_ratio", df_merged.columns)

    def test_get_view_category_mapping(self):
        from finance_ml.analytics.data_utils import (
            get_view_category_mapping,
            get_view_category_labels,
            get_view_feature_cols,
        )

        mapping = get_view_category_mapping()
        self.assertEqual(mapping["vw_features_momentum"]["category"], "Momentum")
        self.assertIsInstance(mapping["vw_features_momentum"]["feature_cols"], list)
        self.assertEqual(len(mapping), 17)

        # Backward-compatible flat mapping
        labels = get_view_category_labels()
        self.assertEqual(labels["vw_features_momentum"], "Momentum")
        self.assertEqual(len(labels), 17)

        # Feature cols helper
        cols = get_view_feature_cols("vw_features_momentum")
        self.assertIn("price_momentum_1m", cols)

    @patch("finance_ml.analytics.statistical_analysis.bayesian_category_analysis")
    @patch("finance_ml.analytics.statistical_analysis.fit_distributions_by_category")
    @patch("finance_ml.analytics.statistical_analysis.calculate_conditional_probabilities")
    def test_run_category_probability_analytics(self, mock_cond, mock_fit, mock_bayesian):
        from finance_ml.analytics.statistical_analysis import run_category_probability_analytics

        mock_bayesian.return_value = {"posterior": "data"}
        mock_fit.return_value = {"fit": "data"}
        mock_cond.return_value = {"cond": "data"}

        results = run_category_probability_analytics(
            self.sample_df, "Valuation Ratios", ["p_e_ratio"]
        )

        self.assertEqual(results["category"], "Valuation Ratios")
        self.assertIn("bayesian_results", results)
        self.assertIn("summary_statistics", results)
        self.assertIn("p_e_ratio", results["summary_statistics"])

    def test_category_probability_analyzer(self):
        from finance_ml.analytics.probability_analytics import CategoryProbabilityAnalyzer

        analyzer = CategoryProbabilityAnalyzer("Valuation")
        results = analyzer.analyze_view(self.sample_df, ["p_e_ratio"])

        self.assertFalse(results.empty)
        self.assertIn("percentile", results.columns)
        self.assertIn("z_score", results.columns)

    def test_create_view_probability_dashboard(self):
        from finance_ml.analytics.probability_analytics import create_view_probability_dashboard

        fig = create_view_probability_dashboard(
            self.sample_df, "vw_features_valuation", "Valuation"
        )
        self.assertIsInstance(fig, go.Figure)

    def test_visualization_functions(self):
        from finance_ml.analytics.visualizations.category_charts import (
            create_balance_sheet_composition_chart,
            create_cost_structure_breakdown,
            create_unusual_items_heatmap,
        )

        bs_df = pd.DataFrame(
            {
                "industry": ["Tech", "Tech", "Finance"],
                "total_assets": [100, 110, 200],
                "total_liabilities": [50, 60, 150],
                "total_equity": [50, 50, 50],
            }
        )
        fig1 = create_balance_sheet_composition_chart(bs_df)
        self.assertIsNotNone(fig1)

        cost_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "cogs_pct": [60],
                "sga_pct": [10],
                "rnd_pct": [20],
                "other_opex_pct": [10],
            }
        )
        fig2 = create_cost_structure_breakdown(cost_df, "AAPL")
        self.assertIsNotNone(fig2)

        unusual_df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "unusual_item_flag": [0, 1]})
        fig3 = create_unusual_items_heatmap(unusual_df)
        self.assertIsNotNone(fig3)


if __name__ == "__main__":
    unittest.main()
