import os
import sys
import unittest

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from finance_ml.features.advanced.earnings import (
    engineer_estimated_vs_actual_analytics,
)
from finance_ml.features.advanced.growth import engineer_growth_metrics
from finance_ml.features.advanced.dividends import engineer_dividend_reliability_features
from finance_ml.features.advanced.profitability import (
    engineer_profitability_ratios,
)
from finance_ml.features.advanced.quality import (
    engineer_accounting_quality_features,
)
from finance_ml.features.advanced.revenue import engineer_revenue_forecast_features
from finance_ml.features.advanced.momentum import engineer_momentum_features
from finance_ml.features.advanced.sector import engineer_sector_specific_features


class TestNewFeatures(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with necessary columns
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Communication Services"],
                # Earnings
                "eps_est_avg_rev_pct_fy1e_1w": [0.5, -0.2, 0.1],
                "eps_est_avg_rev_pct_fy1e_1y": [10.0, 5.0, -2.0],
                "eps_gaap_est_avg_rev_pct_fy1e_1m": [0.4, -0.1, 0.0],
                "eps_est_avg_rev_pct_fy1e_1m": [0.6, -0.3, 0.1],  # Normalized revision
                "revenues_est_med_ntm": [100.0, 200.0, 150.0],
                "revenues_est_avg_ntm": [102.0, 205.0, 148.0],
                # Growth
                "revenues_est_yoy_pct_fy1e": [15.0, 12.0, 10.0],
                "total_revenues_cagr_5y_fy": [14.0, 11.0, 9.0],
                "revenue_growth_yoy": [16.0, 13.0, 8.0],
                # Dividends
                "buyback_yield_ltm": [2.0, 1.5, 3.0],
                "div_yield_ltm": [0.5, 1.0, 0.0],
                "div_yield_ntm": [0.6, 1.1, 0.0],
                # Profitability
                "randd_expenses_ltm": [10.0, 20.0, 15.0],
                "total_revenues_ltm": [100.0, 200.0, 150.0],
                "marketing_expenses_fy": [5.0, 10.0, 8.0],
                "total_revenues": [100.0, 200.0, 150.0],  # Alias for total_revenues_ltm often used
                "selling_general_and_admin_expenses_total_fy": [8.0, 12.0, 10.0],
                "net_income": [20.0, 40.0, 30.0],
                "total_assets": [500.0, 1000.0, 800.0],
                "total_equity": [200.0, 500.0, 400.0],
                "net_profit_margin_pct": [20.0, 20.0, 20.0],  # Already computed usually
                "asset_turnover": [0.2, 0.2, 0.1875],
                # Quality
                "merger_and_restructuring_charges_ltm": [1.0, 0.0, 0.5],
                "market_cap": [1000.0, 2000.0, 1500.0],
                "interest_income_on_investments_ltm": [2.0, 5.0, 1.0],
                "net_income_ltm": [20.0, 40.0, 30.0],
                "gain_loss_on_sale_of_assets_ltm": [0.5, -0.2, 0.0],
                # Revenue Forecast
                # 'revenues_est_avg_ntm' defined above
                # 'total_revenues_ltm' defined above
                # Momentum
                "beta_1y": [1.2, 1.0, 1.1],
                "beta_5y": [1.0, 0.9, 1.0],
                "volatility_1m": [20.0, 15.0, 18.0],
                "volatility_1y": [15.0, 12.0, 16.0],
                # Sector
                "tbv_ltm": [150.0, 300.0, 250.0],
                "tbv_fy": [140.0, 280.0, 240.0],
                "market_cap_country_r": [1, 2, 3],  # Rank
                "shares_outstanding": [100.0, 200.0, 100.0],
                "last_price": [10.0, 10.0, 15.0],
            }
        )

    def test_earnings_enhancements(self):
        print("\nTesting Earnings Enhancements...")
        df = engineer_estimated_vs_actual_analytics(self.df)

        # 1. Short & Long-Term Revision Momentum (Updated surprise_momentum_score)
        # Note: We need to check if the score implementation changed.
        # Currently it sums 1M, 3M, 6M. New plan adds 1W and 1Y? Or replaces?
        # Plan says: "Include the 1-week revision trend... and the 1-year trend... in surprise_momentum_score"

        # 2. GAAP vs Non-GAAP Revisions
        # New Feature: gaap_revision_divergence
        if "gaap_revision_divergence" in df.columns:
            print("  [Pass] gaap_revision_divergence column found.")
            print(f"  Values: {df['gaap_revision_divergence'].tolist()}")
        else:
            print("  [Fail] gaap_revision_divergence column NOT found.")

        # 3. Revenue Forecast Skew
        # New Feature: revenue_forecast_skew
        if "revenue_forecast_skew" in df.columns:
            print("  [Pass] revenue_forecast_skew column found.")
            print(f"  Values: {df['revenue_forecast_skew'].tolist()}")
        else:
            print("  [Fail] revenue_forecast_skew column NOT found.")

    def test_growth_enhancements(self):
        print("\nTesting Growth Enhancements...")
        df = engineer_growth_metrics(self.df)

        if "forward_revenue_growth" in df.columns:
            print("  [Pass] forward_revenue_growth column found.")
        else:
            print("  [Fail] forward_revenue_growth column NOT found.")

        if "revenue_cagr_5y" in df.columns:
            print("  [Pass] revenue_cagr_5y column found.")
        else:
            print("  [Fail] revenue_cagr_5y column NOT found.")

        if "growth_persistence_score" in df.columns:
            print("  [Pass] growth_persistence_score column found.")
        else:
            print("  [Fail] growth_persistence_score column NOT found.")

    def test_dividends_enhancements(self):
        print("\nTesting Dividends Enhancements...")
        df = engineer_dividend_reliability_features(self.df)

        if "buyback_yield" in df.columns:
            print("  [Pass] buyback_yield column found.")
        else:
            print("  [Fail] buyback_yield column NOT found.")

        if "total_shareholder_yield" in df.columns:
            print("  [Pass] total_shareholder_yield column found.")
        else:
            print("  [Fail] total_shareholder_yield column NOT found.")

        if "dividend_growth_expectation" in df.columns:
            print("  [Pass] dividend_growth_expectation column found.")
        else:
            print("  [Fail] dividend_growth_expectation column NOT found.")

    def test_profitability_enhancements(self):
        print("\nTesting Profitability Enhancements...")
        df = engineer_profitability_ratios(self.df)

        if "rnd_intensity" in df.columns:
            print("  [Pass] rnd_intensity column found.")
        else:
            print("  [Fail] rnd_intensity column NOT found.")

        if "marketing_efficiency" in df.columns:
            print("  [Pass] marketing_efficiency column found.")
        else:
            print("  [Fail] marketing_efficiency column NOT found.")

        if "sga_ratio" in df.columns:
            print("  [Pass] sga_ratio column found.")
        else:
            print("  [Fail] sga_ratio column NOT found.")

        # Dupont decomposition update
        # Need to check if logic is updated, maybe check if columns involved are used or if a new decomposed column exists?
        # The plan says: "Refine roe decomposition into net_profit_margin * asset_turnover * equity_multiplier explicitly"
        # It doesn't explicitly ask for a new column, but "to allow for factor-based screening".
        # It might mean creating 'equity_multiplier' if not exists, or `dupont_roe`?
        # I'll check for 'equity_multiplier'.
        if "equity_multiplier" in df.columns:
            print("  [Pass] equity_multiplier column found.")
        else:
            print("  [Fail] equity_multiplier column NOT found.")

    def test_quality_enhancements(self):
        print("\nTesting Quality Enhancements...")
        df = engineer_accounting_quality_features(self.df)

        if "merger_impact_ratio" in df.columns:
            print("  [Pass] merger_impact_ratio column found.")
        else:
            print("  [Fail] merger_impact_ratio column NOT found.")

        if "non_operating_income_share" in df.columns:
            print("  [Pass] non_operating_income_share column found.")
        else:
            print("  [Fail] non_operating_income_share column NOT found.")

        if "asset_sale_boost" in df.columns:
            print("  [Pass] asset_sale_boost column found.")
        else:
            print("  [Fail] asset_sale_boost column NOT found.")

    def test_revenue_enhancements(self):
        print("\nTesting Revenue Enhancements...")
        df = engineer_revenue_forecast_features(self.df)

        if "revenue_estimate_momentum" in df.columns:
            print("  [Pass] revenue_estimate_momentum column found.")
        else:
            print("  [Fail] revenue_estimate_momentum column NOT found.")

        if "revenue_surprise_volatility" in df.columns:
            print("  [Pass] revenue_surprise_volatility column found.")
        else:
            print("  [Fail] revenue_surprise_volatility column NOT found.")

    def test_momentum_enhancements(self):
        print("\nTesting Momentum Enhancements...")
        df = engineer_momentum_features(self.df)

        if "beta_momentum" in df.columns:
            print("  [Pass] beta_momentum column found.")
        else:
            print("  [Fail] beta_momentum column NOT found.")

        if "volatility_term_structure" in df.columns:
            print("  [Pass] volatility_term_structure column found.")
        else:
            print("  [Fail] volatility_term_structure column NOT found.")

    def test_sector_enhancements(self):
        print("\nTesting Sector Enhancements...")
        df = engineer_sector_specific_features(self.df)

        # Check if TBV was used (hard to check logic without mock spying, but we can assume if it runs without error)
        # Check size_factor_percentile
        if "size_factor_percentile" in df.columns:
            print("  [Pass] size_factor_percentile column found.")
        else:
            print("  [Fail] size_factor_percentile column NOT found.")


if __name__ == "__main__":
    unittest.main()
