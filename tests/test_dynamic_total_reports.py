"""Tests for dynamic total_reports and forward estimate integration."""

import unittest

import numpy as np
import pandas as pd

from finance_ml.analytics.probability_analytics import (
    BeatProbabilityResult,
    EarningsBeatProbabilityModel,
    EPSStreakAnalyzer,
    ForwardEstimateSignals,
    ReportedEPSHistory,
)


class TestReportedEPSHistoryProperties(unittest.TestCase):
    """Test new count properties on ReportedEPSHistory."""

    def test_total_reports_count(self):
        h = ReportedEPSHistory(
            eps_basic_fq=1.0,
            eps_basic_1fqfq=0.9,
            eps_basic_fy=2.0,
            eps_basic_1fy=1.8,
        )
        self.assertEqual(h.total_reports_count, 4)

    def test_total_reports_count_all_none(self):
        h = ReportedEPSHistory()
        self.assertEqual(h.total_reports_count, 0)

    def test_quarterly_reports_count(self):
        h = ReportedEPSHistory(eps_basic_fq=1.0, eps_basic_2fqfq=0.5)
        self.assertEqual(h.quarterly_reports_count, 2)

    def test_annual_reports_count(self):
        h = ReportedEPSHistory(eps_basic_fy=2.0, eps_basic_1fy=1.8, eps_basic_3fy=1.0)
        self.assertEqual(h.annual_reports_count, 3)


class TestBeatProbabilityResultNewFields(unittest.TestCase):
    """Test new optional fields on BeatProbabilityResult."""

    def test_new_fields_default_none(self):
        r = BeatProbabilityResult(
            ticker="X",
            name="X",
            sector="",
            industry="",
            country="",
            exchange="",
            prior_alpha=2,
            prior_beta=2,
            posterior_alpha=3,
            posterior_beta=3,
            prior_mean=0.5,
            posterior_mean=0.5,
            posterior_std=0.1,
            credible_interval_90=(0.3, 0.7),
            credible_interval_95=(0.2, 0.8),
            beat_probability=0.5,
            confidence_score=0.7,
            historical_beat_rate=0.5,
            n_observations=5,
        )
        self.assertIsNone(r.dynamic_total_reports)
        self.assertIsNone(r.analyst_count)
        self.assertIsNone(r.eps_norm_est_ntm)

    def test_new_fields_set(self):
        r = BeatProbabilityResult(
            ticker="X",
            name="X",
            sector="",
            industry="",
            country="",
            exchange="",
            prior_alpha=2,
            prior_beta=2,
            posterior_alpha=3,
            posterior_beta=3,
            prior_mean=0.5,
            posterior_mean=0.5,
            posterior_std=0.1,
            credible_interval_90=(0.3, 0.7),
            credible_interval_95=(0.2, 0.8),
            beat_probability=0.5,
            confidence_score=0.7,
            historical_beat_rate=0.5,
            n_observations=5,
            dynamic_total_reports=10,
            analyst_count=5,
        )
        self.assertEqual(r.dynamic_total_reports, 10)
        self.assertEqual(r.analyst_count, 5)


class TestForwardAdjustedDynamicTotal(unittest.TestCase):
    """Test compute_forward_adjusted_beat_probability uses dynamic counts."""

    def test_fallback_uses_quarterly_reports_count(self):
        model = EarningsBeatProbabilityModel()
        # No annual pairs → should fallback to quarterly_reports_count
        h = ReportedEPSHistory(eps_basic_fq=1.0, eps_basic_1fqfq=0.8, eps_basic_2fqfq=0.5)
        fs = ForwardEstimateSignals(eps_norm_fy1e=2.0, revision_1m=5.0)
        result = model.compute_forward_adjusted_beat_probability(h, fs)
        self.assertIn("posterior_mean", result)
        self.assertGreater(result["posterior_mean"], 0)


class TestAnalyzeDataframeEnhanced(unittest.TestCase):
    """Test analyze_dataframe_enhanced outputs dynamic_total_reports."""

    def test_enhanced_path_has_dynamic_total(self):
        model = EarningsBeatProbabilityModel()
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "name": ["TestA"],
                "sector": ["Tech"],
                "net_eps_basic_fq": [1.0],
                "net_eps_basic_1fqfq": [0.9],
                "net_eps_basic_fy": [2.0],
                "net_eps_basic_1fy": [1.5],
                "eps_norm_est_avg_fy1e": [2.5],
                "eps_est_avg_rev_pct_fy1e_1m": [3.0],
            }
        )
        res = model.analyze_dataframe_enhanced(df)
        self.assertIn("dynamic_total_reports", res.columns)
        self.assertGreater(res.iloc[0]["dynamic_total_reports"], 0)
        self.assertIn("next_earnings_status", res.columns)
        self.assertIn("analyst_count", res.columns)

    def test_trajectory_fallback_uses_dynamic_total(self):
        model = EarningsBeatProbabilityModel()
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "name": ["TestA"],
                "sector": ["Tech"],
                "eps_trajectory_score": [80.0],
                "net_eps_basic_fq": [1.0],
                "net_eps_basic_1fqfq": [0.9],
                "net_eps_basic_fy": [2.0],
                "net_eps_basic_1fy": [1.5],
            }
        )
        res = model.analyze_dataframe_enhanced(df)
        self.assertIn("dynamic_total_reports", res.columns)
        self.assertGreater(res.iloc[0]["dynamic_total_reports"], 0)
        # total_reports should use dynamic count, not hardcoded 5
        self.assertGreater(res.iloc[0]["total_reports"], 0)


class TestEPSStreakAnalyzerEnhanced(unittest.TestCase):
    """Test EPSStreakAnalyzer with forward signals and dynamic totals."""

    def test_analyze_dataframe_outputs_new_columns(self):
        analyzer = EPSStreakAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["B"],
                "name": ["TestB"],
                "sector": ["Fin"],
                "industry": ["Bank"],
                "country": ["US"],
                "exchange": ["NYSE"],
                "eps_trajectory_score": [75.0],
                "eps_positive_streak": [3],
                "eps_improvement_count": [2],
                "net_eps_basic_fq": [1.5],
                "net_eps_basic_1fqfq": [1.2],
                "eps_norm_est_avg_fy1e": [2.0],
                "eps_est_avg_rev_pct_fy1e_1m": [3.0],
            }
        )
        res = analyzer.analyze_dataframe(df)
        self.assertIn("dynamic_total_reports", res.columns)
        self.assertIn("historical_beat_rate", res.columns)
        self.assertIn("gaap_revision_momentum", res.columns)
        self.assertIn("next_earnings_status", res.columns)

    def test_compute_streak_with_forward_signals_adjusts_prob(self):
        analyzer = EPSStreakAnalyzer()
        h = ReportedEPSHistory(eps_basic_fq=1.0, eps_basic_1fqfq=0.9, eps_basic_2fqfq=0.8)
        fs_positive = ForwardEstimateSignals(
            eps_norm_fy1e=2.0,
            revision_1m=10.0,
            revision_1w=15.0,
        )
        fs_negative = ForwardEstimateSignals(
            eps_norm_fy1e=2.0,
            revision_1m=-10.0,
            revision_1w=-15.0,
        )
        result_pos = analyzer.compute_streak_from_trajectory(
            eps_trajectory_score=80,
            reported_history=h,
            forward_signals=fs_positive,
        )
        result_neg = analyzer.compute_streak_from_trajectory(
            eps_trajectory_score=80,
            reported_history=h,
            forward_signals=fs_negative,
        )
        # Positive momentum should yield higher continuation for beat streak
        self.assertGreater(
            result_pos.streak_continuation_prob,
            result_neg.streak_continuation_prob,
        )

    def test_no_history_cols_still_works(self):
        analyzer = EPSStreakAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["C"],
                "name": ["TestC"],
                "sector": ["Tech"],
                "industry": ["SW"],
                "country": ["US"],
                "exchange": ["NASDAQ"],
                "eps_trajectory_score": [60.0],
            }
        )
        res = analyzer.analyze_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["dynamic_total_reports"], 0)


if __name__ == "__main__":
    unittest.main()
