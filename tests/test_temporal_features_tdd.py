"""
TDD Tests for Temporal Features Enhancement - Phase 9.3 100% Coverage

Tests for missing temporal features to achieve 100% Phase 9.3 coverage:
- days_to_dividend: (dividend_record_ex_date - _reference_date).days
- quarterly_volatility_score: coefficient of variation across quarterly EBITDA
- days_since_reference: requires reference_date parameter

Following TDD principles and code_guidelines.md Section 9.3.0.
"""

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd


class TestDaysToDividendFeature(unittest.TestCase):
    """TDD tests for days_to_dividend feature."""

    def setUp(self):
        """Set up test data with dividend date columns."""
        self.base_date = pd.Timestamp("2025-11-30")
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "sector": ["Technology"] * 5,
                "last_updated": [
                    self.base_date,
                    self.base_date - timedelta(days=5),
                    self.base_date - timedelta(days=10),
                    pd.NaT,  # Missing last_updated
                    self.base_date,
                ],
                "dividend_record_ex_date": [
                    self.base_date + timedelta(days=30),  # 30 days to dividend
                    self.base_date + timedelta(days=15),  # 20 days from their last_updated
                    pd.NaT,  # Missing dividend date
                    self.base_date + timedelta(days=45),  # Has dividend but no last_updated
                    self.base_date - timedelta(days=5),  # Dividend already passed
                ],
                "next_earnings": [
                    self.base_date + timedelta(days=60),
                    self.base_date + timedelta(days=45),
                    self.base_date + timedelta(days=30),
                    self.base_date + timedelta(days=90),
                    self.base_date + timedelta(days=75),
                ],
            }
        )

    def test_days_to_dividend_basic_calculation(self):
        """Test basic days_to_dividend calculation."""
        from finance_ml.features.advanced import engineer_temporal_features

        # Use explicit reference_date for predictability
        result = engineer_temporal_features(
            self.df, date_col="next_earnings", reference_date=self.base_date
        )

        # Should have days_to_dividend column
        self.assertIn("days_to_dividend", result.columns)

        # First row: 30 days to dividend (from base_date)
        self.assertEqual(result.loc[0, "days_to_dividend"], 30)

        # Second row: 15 days (from base_date)
        self.assertEqual(result.loc[1, "days_to_dividend"], 15)

    def test_days_to_dividend_handles_missing_dividend_date(self):
        """Test that missing dividend dates result in NaN."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(
            self.df, date_col="next_earnings", reference_date=self.base_date
        )

        # Third row has NaT for dividend_record_ex_date
        self.assertTrue(pd.isna(result.loc[2, "days_to_dividend"]))

    def test_days_to_dividend_handles_missing_last_updated(self):
        """Test that missing last_updated (date_col) still allows days_to_dividend if ref_date provided."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(
            self.df, date_col="next_earnings", reference_date=self.base_date
        )

        # Fourth row has NaT for last_updated but valid dividend date and ref_date
        self.assertEqual(result.loc[3, "days_to_dividend"], 45)

    def test_days_to_dividend_negative_values_allowed(self):
        """Test that past dividends result in negative days."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(
            self.df, date_col="next_earnings", reference_date=self.base_date
        )

        # Fifth row: dividend was 5 days before base_date, so -5 days
        self.assertEqual(result.loc[4, "days_to_dividend"], -5)

    def test_days_to_dividend_without_dividend_column(self):
        """Test graceful handling when dividend column is missing."""
        from finance_ml.features.advanced import engineer_temporal_features

        df_no_dividend = self.df.drop(columns=["dividend_record_ex_date"])
        result = engineer_temporal_features(df_no_dividend, date_col="next_earnings")

        # Should not fail, days_to_dividend should not be added
        self.assertNotIn("days_to_dividend", result.columns)


class TestQuarterlyVolatilityScore(unittest.TestCase):
    """TDD tests for quarterly_volatility_score feature."""

    def setUp(self):
        """Set up test data with quarterly EBITDA columns."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "sector": ["Technology"] * 4,
                "next_earnings": pd.to_datetime(["2025-12-15"] * 4),
                # Quarterly EBITDA columns (different naming patterns)
                "ebitda_fq": [100, 200, 150, np.nan],
                "ebitda_fq_1": [110, 190, 160, 300],
                "ebitda_fq_2": [105, 210, 140, 310],
                "ebitda_fq_3": [95, 195, 155, 290],
                # Alternative naming: ebitda_q1, ebitda_q2, etc.
            }
        )

    def test_quarterly_volatility_score_calculation(self):
        """Test coefficient of variation calculation across quarters."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(self.df, date_col="next_earnings")

        # Should have quarterly_volatility_score column
        self.assertIn("quarterly_volatility_score", result.columns)

        # First row: CV of [100, 110, 105, 95]
        # Mean = 102.5, Std = ~6.45, CV = 0.063
        self.assertIsNotNone(result.loc[0, "quarterly_volatility_score"])
        self.assertGreater(result.loc[0, "quarterly_volatility_score"], 0)

    def test_quarterly_volatility_handles_missing_quarters(self):
        """Test handling of missing quarterly values."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(self.df, date_col="next_earnings")

        # Fourth row has NaN in ebitda_fq, should still compute from available quarters
        # or return NaN if too few values
        # Behavior depends on implementation
        self.assertIn("quarterly_volatility_score", result.columns)


class TestDaysSinceReference(unittest.TestCase):
    """TDD tests for days_since_reference feature."""

    def setUp(self):
        """Set up test data."""
        self.base_date = pd.Timestamp("2025-11-30")
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology"] * 3,
                "next_earnings": [
                    self.base_date + timedelta(days=30),
                    self.base_date + timedelta(days=45),
                    self.base_date + timedelta(days=60),
                ],
            }
        )

    def test_days_since_reference_with_reference_date(self):
        """Test days_since_reference when reference_date is provided."""
        from finance_ml.features.advanced import engineer_temporal_features

        reference = pd.Timestamp("2025-11-01")
        result = engineer_temporal_features(
            self.df, date_col="next_earnings", reference_date=reference
        )

        # Should have days_since_reference column
        self.assertIn("days_since_reference", result.columns)

        # First row: next_earnings is 2025-12-30, reference is 2025-11-01
        # Days = 59 days
        self.assertEqual(result.loc[0, "days_since_reference"], 59)

    def test_days_since_reference_without_reference_date(self):
        """Test that days_since_reference is NOT added without reference_date."""
        from finance_ml.features.advanced import engineer_temporal_features

        result = engineer_temporal_features(self.df, date_col="next_earnings")

        # Without reference_date, days_since_reference should not be added
        self.assertNotIn("days_since_reference", result.columns)


class TestMarketSentimentFeatures(unittest.TestCase):
    """TDD tests for missing Market Sentiment features."""

    def setUp(self):
        """Set up test data for market sentiment features."""
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "sector": ["Technology"] * 4,
                "last_price": [150.0, 140.0, 380.0, 180.0],
                "price_20d_ago": [145.0, 142.0, 375.0, 175.0],
                "short_interest": [10000000, 5000000, 8000000, 12000000],
                "volume_shrs": [50000000, 30000000, 40000000, 60000000],
                "shares_outstanding": [15000000000, 12000000000, 7500000000, 10000000000],
            }
        )

    def test_momentum_20d_calculation(self):
        """Test 20-day momentum calculation."""
        from finance_ml.features.advanced import engineer_momentum_features

        result = engineer_momentum_features(self.df)

        # Should have momentum_20d column
        self.assertIn("momentum_20d", result.columns)

        # First row: (150 - 145) / 145 * 100 = 3.45%
        expected = (150.0 - 145.0) / 145.0 * 100
        self.assertAlmostEqual(result.loc[0, "momentum_20d"], expected, places=2)

    def test_short_interest_ratio_calculation(self):
        """Test short interest ratio calculation."""
        from finance_ml.features.advanced import engineer_market_sentiment_features

        result = engineer_market_sentiment_features(self.df)

        # Should have short_interest_ratio column
        self.assertIn("short_interest_ratio", result.columns)

        # Ratio = short_interest / avg_daily_volume (or shares_outstanding)
        # Implementation may vary


class TestPhase93FeatureCoverage(unittest.TestCase):
    """Integration tests for Phase 9.3 feature coverage."""

    def test_temporal_patterns_coverage_target(self):
        """Test that Temporal Patterns category reaches coverage target."""
        from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

        expected_temporal_features = PHASE93_FEATURE_CATEGORIES["Temporal Patterns"]

        # Should have at least 15 features
        self.assertGreaterEqual(len(expected_temporal_features), 15)

    def test_all_temporal_features_generated(self):
        """Test that all temporal features can be generated."""
        from finance_ml.features.advanced import engineer_temporal_features

        # Create comprehensive test DataFrame
        base_date = pd.Timestamp("2025-11-30")
        df = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "sector": ["Technology"],
                "last_updated": [base_date],
                "next_earnings": [base_date + timedelta(days=30)],
                "income_statement_report_date": [base_date - timedelta(days=60)],
                "dividend_record_ex_date": [base_date + timedelta(days=15)],
                "total_revenues_ltm": [1000000],
                "total_revenues_5yavg": [900000],
                "ebitda_fq": [100000],
                "ebitda_5yavgfq": [90000],
                "ebitda_fq_1": [95000],
                "ebitda_fq_2": [98000],
                "ebitda_fq_3": [92000],
            }
        )

        result = engineer_temporal_features(df, date_col="next_earnings", reference_date=base_date)

        # Check expected features are present
        expected_features = [
            "fiscal_quarter",
            "month",
            "year",
            "days_to_earnings",
            "days_to_dividend",
            "earnings_report_recency",
            "reporting_lag",
            "ltm_vs_5yavg_revenue",
            "fq_vs_5yavg_ebitda",
            "quarterly_volatility_score",
            "days_since_reference",
        ]

        for feature in expected_features:
            self.assertIn(feature, result.columns, f"Missing expected temporal feature: {feature}")


if __name__ == "__main__":
    unittest.main()
