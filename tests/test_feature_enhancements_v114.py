"""TDD for Phase 9.3 v1.14 feature enhancements (schema + generators)."""

import unittest

import pandas as pd

from finance_ml.core.schema import COLUMN_SCHEMA, PHASE93_FEATURE_CATEGORIES
from finance_ml.features.advanced import (
    FEATURE_REGISTRY,
    __all__ as ADVANCED_ALL,
    engineer_cashflow_temporal_features,
    engineer_dividend_timing_features,
    engineer_eps_trajectory_features,
    engineer_fiscal_calendar_features,
    engineer_price_target_dynamics,
    get_total_feature_count,
)


class TestSchemaAlignmentV114(unittest.TestCase):
    """Ensure all new v1.14 features are defined in the canonical schema."""

    def test_column_schema_contains_new_features(self):
        expected_meta = {
            # Price Target Dynamics (Analyst Sentiment)
            "pt_momentum_1w": ("Float64", "feature"),
            "pt_momentum_1m": ("Float64", "feature"),
            "pt_momentum_3m": ("Float64", "feature"),
            "pt_momentum_6m": ("Float64", "feature"),
            "pt_momentum_1y": ("Float64", "feature"),
            "pt_acceleration_short": ("Float64", "feature"),
            "pt_acceleration_long": ("Float64", "feature"),
            "pt_consensus_convergence": ("Float64", "feature"),
            "analyst_coverage_change_1m": ("Float64", "feature"),
            "analyst_coverage_change_3m": ("Float64", "feature"),
            "pt_vs_price_momentum": ("Float64", "feature"),
            "pt_qtd_momentum": ("Float64", "feature"),
            "pt_ytd_momentum": ("Float64", "feature"),
            "pt_skew_trend": ("Float64", "feature"),
            "pt_high_low_spread_trend": ("Float64", "feature"),
            # Cash Flow Temporal
            "fcf_quarterly_trend": ("Float64", "feature"),
            "fcf_quarterly_volatility": ("Float64", "feature"),
            "fcf_positive_ratio": ("Float64", "feature"),
            "cfo_quarterly_trend": ("Float64", "feature"),
            "cfo_yoy_quarterly": ("Float64", "percentage"),
            "investment_intensity_trend": ("Float64", "feature"),
            "cfo_5y_trend": ("Float64", "feature"),
            "cfo_5y_stability": ("Float64", "feature"),
            "cfo_margin_current": ("Float64", "percentage"),
            "cfo_margin_trend": ("Float64", "percentage"),
            "acquisition_activity_trend": ("Float64", "feature"),
            "acquisition_quarters_active": ("Int64", "feature"),
            # EPS Trajectory (Earnings Quality)
            "eps_quarterly_trend": ("Float64", "feature"),
            "eps_quarterly_volatility": ("Float64", "feature"),
            "eps_yoy_quarterly_growth": ("Float64", "percentage"),
            "eps_qoq_growth": ("Float64", "percentage"),
            "eps_positive_streak": ("Int64", "feature"),
            "eps_cagr_5y": ("Float64", "percentage"),
            "eps_cagr_3y": ("Float64", "percentage"),
            "eps_annual_trend": ("Float64", "feature"),
            "eps_vs_5y_avg": ("Float64", "feature"),
            "eps_growth_acceleration": ("Float64", "feature"),
            # Fiscal Calendar (Temporal Patterns)
            "fiscal_year_progress": ("Float64", "feature"),
            "days_to_quarter_end": ("Int64", "feature"),
            "fiscal_half": ("Int64", "feature"),
            "reporting_lag_zscore": ("Float64", "feature"),
            "late_reporter_flag": ("boolean", "feature"),
            "days_since_fy_end": ("Int64", "feature"),
            "days_to_next_fy_end": ("Int64", "feature"),
            "earnings_imminent": ("boolean", "feature"),
            "pre_earnings_window": ("boolean", "feature"),
            # Dividend Timing (Dividend Reliability)
            "days_to_dividend_ex_date": ("Float64", "feature"),
            "days_to_dividend_record_date": ("Float64", "feature"),
            "days_to_dividend_payable_date": ("Float64", "feature"),
            "approaching_ex_date": ("boolean", "feature"),
            "recently_ex_dividend": ("boolean", "feature"),
            "dividend_cycle_days": ("Int64", "feature"),
            "dividend_cycle_position": ("Float64", "feature"),
            "dividend_announcement_recency": ("Float64", "feature"),
        }

        for feature, (dtype, role) in expected_meta.items():
            with self.subTest(feature=feature):
                self.assertIn(feature, COLUMN_SCHEMA)
                self.assertEqual(COLUMN_SCHEMA[feature]["dtype"], dtype)
                self.assertEqual(COLUMN_SCHEMA[feature]["role"], role)


class TestCategoryAndRegistryAlignment(unittest.TestCase):
    """Validate Phase 9.3 category counts and registry integration for v1.14."""

    def test_phase93_category_counts_and_members(self):
        expected_counts = {
            "Analyst Sentiment": 25,
            "Cash Flow": 17,
            "Temporal Patterns": 26,
            "Earnings Quality": 43,
            "Dividend Reliability": 20,
        }

        expected_new_members = {
            "Analyst Sentiment": {
                "pt_momentum_1w",
                "pt_momentum_1m",
                "pt_momentum_3m",
                "pt_momentum_6m",
                "pt_momentum_1y",
                "pt_acceleration_short",
                "pt_acceleration_long",
                "pt_consensus_convergence",
                "analyst_coverage_change_1m",
                "analyst_coverage_change_3m",
                "pt_vs_price_momentum",
                "pt_qtd_momentum",
                "pt_ytd_momentum",
                "pt_skew_trend",
                "pt_high_low_spread_trend",
            },
            "Cash Flow": {
                "fcf_quarterly_trend",
                "fcf_quarterly_volatility",
                "fcf_positive_ratio",
                "cfo_quarterly_trend",
                "cfo_yoy_quarterly",
                "investment_intensity_trend",
                "cfo_5y_trend",
                "cfo_5y_stability",
                "cfo_margin_current",
                "cfo_margin_trend",
                "acquisition_activity_trend",
                "acquisition_quarters_active",
            },
            "Temporal Patterns": {
                "fiscal_year_progress",
                "days_to_quarter_end",
                "fiscal_half",
                "reporting_lag_zscore",
                "late_reporter_flag",
                "days_since_fy_end",
                "days_to_next_fy_end",
                "earnings_imminent",
                "pre_earnings_window",
            },
            "Earnings Quality": {
                "eps_quarterly_trend",
                "eps_quarterly_volatility",
                "eps_yoy_quarterly_growth",
                "eps_qoq_growth",
                "eps_positive_streak",
                "eps_cagr_5y",
                "eps_cagr_3y",
                "eps_annual_trend",
                "eps_vs_5y_avg",
                "eps_growth_acceleration",
            },
            "Dividend Reliability": {
                "days_to_dividend_ex_date",
                "days_to_dividend_record_date",
                "days_to_dividend_payable_date",
                "approaching_ex_date",
                "recently_ex_dividend",
                "dividend_cycle_days",
                "dividend_cycle_position",
                "dividend_announcement_recency",
            },
        }

        for category, expected_count in expected_counts.items():
            with self.subTest(category=category):
                self.assertIn(category, PHASE93_FEATURE_CATEGORIES)
                features = set(PHASE93_FEATURE_CATEGORIES[category])
                self.assertEqual(
                    len(features), expected_count, f"{category} count should be {expected_count}"
                )
                self.assertTrue(
                    expected_new_members[category].issubset(features),
                    f"{category} missing new v1.14 features",
                )

    def test_feature_registry_and_exports_updated(self):
        expected_registry_entries = {
            "price_target_dynamics": ("Analyst Sentiment", 15),
            "cashflow_temporal": ("Cash Flow", 12),
            "eps_trajectory": ("Earnings Quality", 10),
            "fiscal_calendar": ("Temporal Patterns", 9),
            "dividend_timing": ("Dividend Reliability", 8),
        }

        for key, (category, feature_count) in expected_registry_entries.items():
            with self.subTest(entry=key):
                self.assertIn(key, FEATURE_REGISTRY)
                entry = FEATURE_REGISTRY[key]
                self.assertEqual(entry.get("category"), category)
                self.assertEqual(entry.get("feature_count"), feature_count)

        self.assertGreaterEqual(get_total_feature_count(), 350)

        for export in {
            "engineer_price_target_dynamics",
            "engineer_cashflow_temporal_features",
            "engineer_eps_trajectory_features",
            "engineer_fiscal_calendar_features",
            "engineer_dividend_timing_features",
        }:
            with self.subTest(export=export):
                self.assertIn(export, ADVANCED_ALL)


class TestFeatureFunctionOutputs(unittest.TestCase):
    """Validate key outputs from newly added generator functions."""

    def test_price_target_dynamics_computations(self):
        df = pd.DataFrame(
            {
                "price_target": [120.0],
                "price_target_1w_ago": [110.0],
                "price_target_1m_ago": [100.0],
                "price_target_3m_ago": [90.0],
                "price_target_6m_ago": [80.0],
                "price_target_1y_ago": [60.0],
                "price_target_qtd_ago": [115.0],
                "price_target_ytd_ago": [118.0],
                "price_target_high": [130.0],
                "price_target_low": [110.0],
                "price_target_median": [120.0],
                "price_target_high_3m_ago": [125.0],
                "price_target_low_3m_ago": [95.0],
                "price_target_median_3m_ago": [110.0],
                "price_target_count": [10],
                "price_target_count_1m_ago": [8],
                "price_target_count_3m_ago": [7],
                "last_price": [100.0],
                "price_3m_ago": [85.0],
            }
        )

        result = engineer_price_target_dynamics(df)

        self.assertAlmostEqual(result.loc[0, "pt_momentum_1m"], 0.2, places=4)
        self.assertAlmostEqual(result.loc[0, "pt_acceleration_short"], -0.1333, places=3)
        self.assertAlmostEqual(result.loc[0, "pt_consensus_convergence"], 0.106, places=3)
        self.assertEqual(result.loc[0, "analyst_coverage_change_1m"], 2)
        self.assertAlmostEqual(result.loc[0, "pt_vs_price_momentum"], 0.133, places=3)

    def test_fiscal_calendar_and_dividend_timing(self):
        base_date = pd.Timestamp("2025-01-20")
        df = pd.DataFrame(
            {
                "fiscal_month": [2],
                "reporting_lag": [45],
                "fy_end_date": [pd.Timestamp("2024-12-31")],
                "next_fy_end_date": [pd.Timestamp("2025-12-31")],
                "next_earnings": [pd.Timestamp("2025-02-20")],
                "dividend_record_ex_date": [pd.Timestamp("2025-02-01")],
                "dividend_record_record_date": [pd.Timestamp("2025-02-05")],
                "dividend_record_payable_date": [pd.Timestamp("2025-02-15")],
                "dividend_record_frequency": ["Quarterly"],
                "dividend_record_announce_date": [pd.Timestamp("2025-01-01")],
            }
        )

        fiscal = engineer_fiscal_calendar_features(df, reference_date=base_date)
        dividend = engineer_dividend_timing_features(df, reference_date=base_date)

        self.assertAlmostEqual(fiscal.loc[0, "fiscal_year_progress"], 2 / 12.0, places=4)
        self.assertEqual(fiscal.loc[0, "days_to_quarter_end"], 60)
        self.assertEqual(fiscal.loc[0, "fiscal_half"], 1)
        self.assertFalse(bool(fiscal.loc[0, "late_reporter_flag"]))
        self.assertEqual(fiscal.loc[0, "days_since_fy_end"], 20)
        self.assertEqual(fiscal.loc[0, "days_to_next_fy_end"], 345)
        self.assertFalse(bool(fiscal.loc[0, "earnings_imminent"]))
        self.assertFalse(bool(fiscal.loc[0, "pre_earnings_window"]))

        self.assertAlmostEqual(dividend.loc[0, "days_to_dividend_ex_date"], 12.0)
        self.assertAlmostEqual(dividend.loc[0, "days_to_dividend_record_date"], 16.0)
        self.assertAlmostEqual(dividend.loc[0, "days_to_dividend_payable_date"], 26.0)
        self.assertFalse(bool(dividend.loc[0, "approaching_ex_date"]))
        self.assertFalse(bool(dividend.loc[0, "recently_ex_dividend"]))
        self.assertEqual(dividend.loc[0, "dividend_cycle_days"], 90)
        self.assertAlmostEqual(dividend.loc[0, "dividend_cycle_position"], (90 - 12) / 90)
        self.assertEqual(dividend.loc[0, "dividend_announcement_recency"], 19)

    def test_eps_trajectory_outputs(self):
        df = pd.DataFrame(
            {
                "net_eps_basic_fq": [2.0],
                "net_eps_basic_1fqfq": [1.0],
                "net_eps_basic_2fqfq": [0.5],
                "net_eps_basic_3fqfq": [0.0],
                "net_eps_basic_4fqfq": [-1.0],
                "net_eps_basic_fy": [8.0],
                "net_eps_basic_1fy": [6.0],
                "net_eps_basic_2fy": [4.0],
                "net_eps_basic_3fy": [2.0],
                "net_eps_basic_4fy": [1.0],
                "net_eps_basic_5fy": [-1.0],
            }
        )

        result = engineer_eps_trajectory_features(df)

        self.assertAlmostEqual(result.loc[0, "eps_qoq_growth"], 1.0)
        self.assertAlmostEqual(result.loc[0, "eps_yoy_quarterly_growth"], 3.0)
        self.assertEqual(result.loc[0, "eps_positive_streak"], 3)
        self.assertIn("eps_growth_acceleration", result.columns)

    def test_cashflow_temporal_outputs(self):
        df = pd.DataFrame(
            {
                "fcf_fq": [100.0],
                "fcf_1fqfq": [80.0],
                "fcf_2fqfq": [60.0],
                "fcf_3fqfq": [40.0],
                "fcf_4fqfq": [20.0],
                "cfo_fq": [200.0],
                "cfo_1fqfq": [180.0],
                "cfo_4fqfq": [160.0],
                "cfi_fq": [-50.0],
                "cfi_1fqfq": [-40.0],
                "cfi_2fqfq": [-30.0],
                "cfi_3fqfq": [-20.0],
                "cfi_4fqfq": [-10.0],
                "cfo_fy": [220.0],
                "cfo_1fy": [210.0],
                "cfo_2fy": [200.0],
                "cfo_3fy": [190.0],
                "cfo_4fy": [180.0],
                "cfo_ltm": [240.0],
                "total_revenues_ltm": [400.0],
                "total_revenues_1fy": [380.0],
                "cash_acquisitions_fq": [5.0],
                "cash_acquisitions_1fqfq": [4.0],
                "cash_acquisitions_2fqfq": [3.0],
                "cash_acquisitions_3fqfq": [2.0],
                "cash_acquisitions_4fqfq": [1.0],
            }
        )

        result = engineer_cashflow_temporal_features(df)

        self.assertAlmostEqual(result.loc[0, "fcf_positive_ratio"], 1.0)
        self.assertAlmostEqual(result.loc[0, "cfo_yoy_quarterly"], 0.25, places=2)
        self.assertAlmostEqual(result.loc[0, "cfo_margin_current"], 0.6)
        self.assertIn("acquisition_quarters_active", result.columns)


if __name__ == "__main__":
    unittest.main()
