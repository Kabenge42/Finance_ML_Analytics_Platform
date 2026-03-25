"""
Tests for Enhanced Earnings Beat Probability Model.

Covers:
1. True beat counting from actual Net EPS - Basic quarterly/annual series
2. Three-layer evidence fusion (historical + revision momentum + GAAP quality)
3. Forward-looking revision signal (14 revision columns → momentum indicator)
4. Accounting quality guard (GAAP-vs-Norm divergence penalty)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_ml.analytics.probability_analytics import (
    EarningsBeatProbabilityModel,
    ForwardEstimateSignals,
    ReportedEPSHistory,
    PriorParameters,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def model() -> EarningsBeatProbabilityModel:
    """Default model with symmetric weak prior."""
    return EarningsBeatProbabilityModel(prior_alpha=2.0, prior_beta=2.0)


@pytest.fixture
def tech_model() -> EarningsBeatProbabilityModel:
    """Model with a strong tech-sector prior (historically ~70% beat rate)."""
    return EarningsBeatProbabilityModel(prior_alpha=2.0, prior_beta=2.0, sector_priors={
        "Information Technology": PriorParameters(3.5, 1.5),
    })


@pytest.fixture
def perfect_beat_history() -> ReportedEPSHistory:
    """5 consecutive years of YoY EPS improvement, all positive quarters."""
    return ReportedEPSHistory(
        eps_basic_fq=2.50,
        eps_basic_1fqfq=2.30,
        eps_basic_2fqfq=2.10,
        eps_basic_3fqfq=1.90,
        eps_basic_4fqfq=1.70,
        eps_basic_fy=9.60,
        eps_basic_1fy=8.00,
        eps_basic_2fy=6.50,
        eps_basic_3fy=5.00,
        eps_basic_4fy=4.00,
        eps_basic_5fy=3.00,
        eps_adj_fy=10.00,
        eps_adj_1fy=8.20,
        eps_adj_ltm=10.50,
        eps_adj_fq=2.60,
        eps_adj_1fqfq=2.40,
        eps_adj_2fqfq=2.20,
        eps_adj_3fqfq=2.00,
        eps_adj_4fqfq=1.80,
        eps_cont_fq=2.45,
        eps_cont_1fqfq=2.25,
        eps_cont_2fqfq=2.05,
        eps_cont_3fqfq=1.85,
        eps_cont_4fqfq=1.65,
    )


@pytest.fixture
def deteriorating_history() -> ReportedEPSHistory:
    """Declining EPS trajectory with recent miss."""
    return ReportedEPSHistory(
        eps_basic_fq=0.80,
        eps_basic_1fqfq=1.10,
        eps_basic_2fqfq=1.30,
        eps_basic_3fqfq=1.50,
        eps_basic_4fqfq=1.70,
        eps_basic_fy=4.70,
        eps_basic_1fy=6.20,
        eps_basic_2fy=7.50,
        eps_basic_3fy=8.00,
        eps_basic_4fy=7.80,
        eps_basic_5fy=7.00,
    )


@pytest.fixture
def mixed_history() -> ReportedEPSHistory:
    """Mixed EPS trajectory: some improvements, some declines."""
    return ReportedEPSHistory(
        eps_basic_fq=1.50,
        eps_basic_1fqfq=1.60,
        eps_basic_2fqfq=1.40,
        eps_basic_3fqfq=1.55,
        eps_basic_4fqfq=1.30,
        eps_basic_fy=6.00,
        eps_basic_1fy=6.50,
        eps_basic_2fy=5.80,
        eps_basic_3fy=6.00,
        eps_basic_4fy=5.50,
        eps_basic_5fy=5.00,
    )


@pytest.fixture
def sparse_history() -> ReportedEPSHistory:
    """Only 2 quarters of data, rest is None."""
    return ReportedEPSHistory(
        eps_basic_fq=1.20,
        eps_basic_1fqfq=1.00,
    )


@pytest.fixture
def strong_upgrade_signals() -> ForwardEstimateSignals:
    """Strong and consistent upward revisions across all horizons."""
    return ForwardEstimateSignals(
        eps_norm_ntm=5.20,
        eps_norm_fy1e=5.00,
        eps_gaap_ntm=4.90,
        eps_gaap_fy1e=4.80,
        revision_1w=3.5,
        revision_1m=2.8,
        revision_3m=1.5,
        revision_6m=0.8,
        revision_1y=-0.5,
        gaap_revision_1m=2.5,
        gaap_revision_3m=1.2,
        gaap_revision_6m=0.5,
        gaap_revision_1y=-0.8,
        analyst_count=18,
    )


@pytest.fixture
def downgrade_signals() -> ForwardEstimateSignals:
    """Consistent downward revisions across all horizons."""
    return ForwardEstimateSignals(
        eps_norm_ntm=2.10,
        eps_norm_fy1e=2.30,
        eps_gaap_ntm=1.80,
        eps_gaap_fy1e=2.00,
        revision_1w=-4.0,
        revision_1m=-3.5,
        revision_3m=-2.0,
        revision_6m=-1.0,
        revision_1y=0.5,
        gaap_revision_1m=-4.0,
        gaap_revision_3m=-2.5,
        gaap_revision_6m=-1.5,
        gaap_revision_1y=0.2,
        analyst_count=12,
    )


@pytest.fixture
def neutral_signals() -> ForwardEstimateSignals:
    """Flat revisions — no strong consensus direction."""
    return ForwardEstimateSignals(
        eps_norm_ntm=3.50,
        eps_norm_fy1e=3.50,
        eps_gaap_ntm=3.40,
        eps_gaap_fy1e=3.40,
        revision_1w=0.1,
        revision_1m=-0.1,
        revision_3m=0.2,
        revision_6m=-0.2,
        revision_1y=0.0,
        gaap_revision_1m=0.0,
        gaap_revision_3m=0.1,
        gaap_revision_6m=-0.1,
        gaap_revision_1y=0.0,
        analyst_count=8,
    )


@pytest.fixture
def high_gaap_divergence_signals() -> ForwardEstimateSignals:
    """Large GAAP-Norm divergence indicating accounting quality concerns."""
    return ForwardEstimateSignals(
        eps_norm_ntm=4.00,
        eps_norm_fy1e=3.80,
        eps_gaap_ntm=2.50,
        eps_gaap_fy1e=2.40,
        revision_1w=2.0,
        revision_1m=1.5,
        revision_3m=1.0,
        revision_6m=0.5,
        revision_1y=0.0,
        gaap_revision_1m=-1.0,
        gaap_revision_3m=-2.0,
        gaap_revision_6m=-3.0,
        gaap_revision_1y=-4.0,
        analyst_count=15,
    )


@pytest.fixture
def minimal_forward_signals() -> ForwardEstimateSignals:
    """Only FY1E estimate available — no revisions."""
    return ForwardEstimateSignals(
        eps_norm_fy1e=3.00,
    )


@pytest.fixture
def sample_enhanced_df() -> pd.DataFrame:
    """DataFrame simulating the full equities column set."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "XOM", "JPM", "TSLA"],
            "name": [
                "Apple Inc.",
                "Microsoft Corp.",
                "Exxon Mobil",
                "JPMorgan Chase",
                "Tesla Inc.",
            ],
            "sector": [
                "Information Technology",
                "Information Technology",
                "Energy",
                "Financials",
                "Consumer Discretionary",
            ],
            "eps_norm_est_avg_ntm": [7.20, 13.50, 9.80, 16.00, 3.20],
            "eps_norm_est_avg_fy1e": [7.00, 13.00, 9.50, 15.50, 3.00],
            "eps_gaap_est_avg_ntm": [6.90, 12.80, 9.30, 15.20, 2.50],
            "eps_gaap_est_avg_fy1e": [6.80, 12.60, 9.10, 15.00, 2.20],
            "eps_est_avg_rev_pct_fy1e_1w": [1.5, 2.0, -0.5, 0.8, -3.0],
            "eps_est_avg_rev_pct_fy1e_1m": [1.2, 1.8, -0.8, 0.5, -2.5],
            "eps_est_avg_rev_pct_fy1e_3m": [0.8, 1.0, -1.2, 0.3, -1.0],
            "eps_est_avg_rev_pct_fy1e_6m": [0.5, 0.5, -1.8, 0.0, 0.5],
            "eps_est_avg_rev_pct_fy1e_1y": [0.2, 0.0, -2.5, -0.5, 2.0],
            "eps_gaap_est_avg_rev_pct_fy1e_1m": [1.0, 1.5, -1.0, 0.3, -3.0],
            "eps_gaap_est_avg_rev_pct_fy1e_3m": [0.6, 0.8, -1.5, 0.1, -1.5],
            "eps_gaap_est_avg_rev_pct_fy1e_6m": [0.3, 0.3, -2.0, -0.1, 0.0],
            "eps_gaap_est_avg_rev_pct_fy1e_1y": [0.1, -0.1, -2.8, -0.6, 1.5],
            "eps_norm_est_num_fy1e": [35, 40, 22, 20, 30],
            "net_eps_basic_fq": [1.65, 3.00, 2.50, 4.10, 0.85],
            "net_eps_basic_1fqfq": [1.55, 2.85, 2.40, 3.90, 0.75],
            "net_eps_basic_2fqfq": [1.45, 2.70, 2.30, 3.80, 0.65],
            "net_eps_basic_3fqfq": [1.35, 2.55, 2.20, 3.70, 0.55],
            "net_eps_basic_4fqfq": [1.25, 2.40, 2.45, 3.60, 0.45],
            "net_eps_basic_fy": [6.50, 11.00, 9.50, 15.00, 2.50],
            "net_eps_basic_1fy": [5.80, 9.50, 10.20, 13.50, 2.00],
            "net_eps_basic_2fy": [5.10, 8.00, 8.00, 12.00, 1.20],
            "net_eps_basic_3fy": [4.50, 6.50, 7.00, 11.00, 0.50],
            "net_eps_basic_4fy": [3.80, 5.50, 6.50, 10.00, -0.30],
            "net_eps_basic_5fy": [3.00, 4.50, 7.50, 9.00, -1.00],
            "eps_adj_ltm": [6.80, 11.50, 9.80, 15.50, 2.80],
            "eps_adj_fy": [6.60, 11.10, 9.60, 15.10, 2.60],
            "eps_adj_1fy": [5.90, 9.60, 10.30, 13.60, 2.10],
            "eps_adj_fq": [1.70, 3.10, 2.55, 4.20, 0.90],
            "eps_adj_1fqfq": [1.60, 2.90, 2.45, 4.00, 0.80],
            "eps_adj_2fqfq": [1.50, 2.75, 2.35, 3.85, 0.70],
            "eps_adj_3fqfq": [1.40, 2.60, 2.25, 3.75, 0.60],
            "eps_adj_4fqfq": [1.30, 2.45, 2.50, 3.65, 0.50],
            "eps_cont_fq": [1.60, 2.95, 2.48, 4.05, 0.82],
            "eps_cont_1fqfq": [1.50, 2.80, 2.38, 3.85, 0.72],
            "eps_cont_2fqfq": [1.42, 2.65, 2.28, 3.75, 0.62],
            "eps_cont_3fqfq": [1.32, 2.50, 2.18, 3.65, 0.52],
            "eps_cont_4fqfq": [1.22, 2.35, 2.42, 3.55, 0.42],
            "eps_trajectory_score": [80.0, 85.0, 40.0, 75.0, 60.0],
        }
    )


# =============================================================================
# 1. TRUE BEAT COUNTING
# =============================================================================


class TestTrueBeatCounting:
    """Verify beat/miss counting from actual EPS quarterly & annual data."""

    def test_perfect_yoy_improvements(self, perfect_beat_history):
        n_beats, n_total = perfect_beat_history.count_yoy_improvements()
        assert n_beats == 5
        assert n_total == 5

    def test_deteriorating_yoy_improvements(self, deteriorating_history):
        n_beats, n_total = deteriorating_history.count_yoy_improvements()
        assert n_total == 5
        assert n_beats == 2

    def test_mixed_yoy_improvements(self, mixed_history):
        n_beats, n_total = mixed_history.count_yoy_improvements()
        assert n_total == 5
        assert n_beats == 3

    def test_sparse_history_yoy(self, sparse_history):
        n_beats, n_total = sparse_history.count_yoy_improvements()
        assert n_total == 0
        assert n_beats == 0

    def test_quarterly_beat_streak_perfect(self, perfect_beat_history):
        assert perfect_beat_history.quarterly_beat_streak() == 5

    def test_quarterly_beat_streak_deteriorating(self, deteriorating_history):
        assert deteriorating_history.quarterly_beat_streak() == 5

    def test_quarterly_beat_streak_with_negative(self):
        history = ReportedEPSHistory(
            eps_basic_fq=-0.10,
            eps_basic_1fqfq=1.00,
            eps_basic_2fqfq=1.20,
        )
        assert history.quarterly_beat_streak() == 0

    def test_quarterly_beat_streak_broken_in_middle(self):
        history = ReportedEPSHistory(
            eps_basic_fq=1.50,
            eps_basic_1fqfq=1.30,
            eps_basic_2fqfq=-0.20,
            eps_basic_3fqfq=1.10,
            eps_basic_4fqfq=1.00,
        )
        assert history.quarterly_beat_streak() == 2

    def test_quarterly_series_ordering(self, perfect_beat_history):
        series = perfect_beat_history.quarterly_series
        assert series[0] == 2.50
        assert series[4] == 1.70
        assert len(series) == 5

    def test_annual_series_ordering(self, perfect_beat_history):
        series = perfect_beat_history.annual_series
        assert series[0] == 9.60
        assert series[5] == 3.00
        assert len(series) == 6

    def test_count_beats_vs_estimate(self, perfect_beat_history):
        n_beats, n_total = perfect_beat_history.count_quarterly_beats_vs_estimate(2.00)
        assert n_total == 5
        assert n_beats == 3

    def test_count_beats_vs_none_estimate(self, perfect_beat_history):
        n_beats, n_total = perfect_beat_history.count_quarterly_beats_vs_estimate(None)
        assert n_beats == 0
        assert n_total == 0

    def test_all_none_history_gives_zero_streak(self):
        empty = ReportedEPSHistory()
        assert empty.quarterly_beat_streak() == 0
        assert empty.count_yoy_improvements() == (0, 0)

    def test_analyze_enhanced_uses_real_beats_not_proxy(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        assert len(result) == 5
        assert "data_source" in result.columns
        enhanced_rows = result[result["data_source"] == "forward_enhanced"]
        assert len(enhanced_rows) == 5
        aapl = result[result["ticker"] == "AAPL"].iloc[0]
        assert aapl["historical_beats"] == 5
        assert aapl["historical_beat_rate"] == pytest.approx(1.0)
        xom = result[result["ticker"] == "XOM"].iloc[0]
        assert xom["historical_beats"] == 3


# =============================================================================
# 2. THREE-LAYER EVIDENCE FUSION
# =============================================================================


class TestThreeLayerEvidenceFusion:
    def test_strong_history_plus_upgrades_yields_high_posterior(
        self, model, perfect_beat_history, strong_upgrade_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
            sector="Information Technology",
        )
        assert result["posterior_mean"] > 0.75
        assert result["confidence_score"] > 0.5
        assert result["classification_confidence"] in ("High", "Medium")

    def test_weak_history_plus_downgrades_yields_low_posterior(
        self, model, deteriorating_history, downgrade_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=deteriorating_history,
            forward_signals=downgrade_signals,
            sector="Energy",
        )
        assert result["posterior_mean"] < 0.45

    def test_mixed_history_with_neutral_signals_near_prior(
        self, model, mixed_history, neutral_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=neutral_signals,
        )
        assert 0.40 <= result["posterior_mean"] <= 0.60

    def test_revision_momentum_shifts_posterior_upward(
        self, model, mixed_history, strong_upgrade_signals, neutral_signals
    ):
        result_upgrade = model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=strong_upgrade_signals,
        )
        result_neutral = model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=neutral_signals,
        )
        assert result_upgrade["posterior_mean"] > result_neutral["posterior_mean"]

    def test_revision_momentum_shifts_posterior_downward(
        self, model, mixed_history, downgrade_signals, neutral_signals
    ):
        result_downgrade = model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=downgrade_signals,
        )
        result_neutral = model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=neutral_signals,
        )
        assert result_downgrade["posterior_mean"] < result_neutral["posterior_mean"]

    def test_all_three_layers_contribute(self, model, perfect_beat_history, strong_upgrade_signals):
        full_result = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
        )
        n_beats, n_total = perfect_beat_history.count_yoy_improvements()
        history_only = model.compute_beat_probability(n_beats, n_total)
        assert full_result["posterior_alpha"] != pytest.approx(
            history_only["posterior_alpha"], abs=0.01
        )

    def test_prior_influence_decreases_with_more_data(self, model):
        short_history = ReportedEPSHistory(
            eps_basic_fy=5.00,
            eps_basic_1fy=4.50,
            eps_basic_2fy=4.00,
        )
        signals = ForwardEstimateSignals(eps_norm_fy1e=5.50, revision_1m=1.0, revision_3m=0.5)
        result_short = model.compute_forward_adjusted_beat_probability(short_history, signals)
        long_history = ReportedEPSHistory(
            eps_basic_fy=5.00,
            eps_basic_1fy=4.50,
            eps_basic_2fy=4.00,
            eps_basic_3fy=3.50,
            eps_basic_4fy=3.00,
            eps_basic_5fy=2.50,
        )
        result_long = model.compute_forward_adjusted_beat_probability(long_history, signals)
        assert result_long["prior_influence_pct"] < result_short["prior_influence_pct"]

    def test_effective_sample_size_reflects_data_volume(
        self, model, perfect_beat_history, strong_upgrade_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
        )
        assert result["effective_sample_size"] > 0

    def test_credible_intervals_contain_posterior_mean(
        self, model, perfect_beat_history, strong_upgrade_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
        )
        mean = result["posterior_mean"]
        ci_90 = result["credible_interval_90"]
        ci_95 = result["credible_interval_95"]
        assert ci_90[0] <= mean <= ci_90[1]
        assert ci_95[0] <= mean <= ci_95[1]
        assert (ci_95[1] - ci_95[0]) >= (ci_90[1] - ci_90[0])

    def test_sector_prior_influences_outcome(self, tech_model, mixed_history, neutral_signals):
        result_tech = tech_model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=neutral_signals,
            sector="Information Technology",
        )
        result_default = tech_model.compute_forward_adjusted_beat_probability(
            reported_history=mixed_history,
            forward_signals=neutral_signals,
            sector="Unknown Sector",
        )
        assert result_tech["posterior_mean"] > result_default["posterior_mean"]

    def test_sparse_history_falls_back_to_streak(
        self, model, sparse_history, strong_upgrade_signals
    ):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=sparse_history,
            forward_signals=strong_upgrade_signals,
        )
        assert 0.0 < result["posterior_mean"] < 1.0
        assert result["confidence_score"] >= 0.0


# =============================================================================
# 3. FORWARD-LOOKING SIGNAL
# =============================================================================


class TestForwardEstimateSignals:
    def test_revision_momentum_all_positive(self, strong_upgrade_signals):
        score = strong_upgrade_signals.gaap_revision_momentum
        assert score > 75.0

    def test_revision_momentum_all_negative(self, downgrade_signals):
        score = downgrade_signals.gaap_revision_momentum
        assert score < 30.0

    def test_revision_momentum_neutral(self, neutral_signals):
        score = neutral_signals.gaap_revision_momentum
        assert 30.0 <= score <= 70.0

    def test_revision_momentum_empty(self):
        empty = ForwardEstimateSignals()
        assert empty.gaap_revision_momentum == 50.0

    def test_revision_momentum_partial_data(self):
        partial = ForwardEstimateSignals(revision_1w=2.0, revision_1m=1.5)
        score = partial.gaap_revision_momentum
        assert score == 100.0

    def test_revision_momentum_weighting_favors_recent(self):
        recent_positive = ForwardEstimateSignals(
            revision_1w=5.0,
            revision_1m=3.0,
            revision_3m=-1.0,
            revision_6m=-2.0,
            revision_1y=-3.0,
        )
        recent_negative = ForwardEstimateSignals(
            revision_1w=-5.0,
            revision_1m=-3.0,
            revision_3m=1.0,
            revision_6m=2.0,
            revision_1y=3.0,
        )
        assert recent_positive.gaap_revision_momentum > recent_negative.gaap_revision_momentum

    def test_short_term_revision_trend(self, strong_upgrade_signals):
        trend = strong_upgrade_signals.revision_trend_short
        assert trend == pytest.approx(0.7)

    def test_medium_term_revision_trend(self, strong_upgrade_signals):
        trend = strong_upgrade_signals.revision_trend_medium
        assert trend == pytest.approx(1.3)

    def test_revision_trend_none_when_missing(self, minimal_forward_signals):
        assert minimal_forward_signals.revision_trend_short is None
        assert minimal_forward_signals.revision_trend_medium is None

    def test_has_sufficient_data_with_full_signals(self, strong_upgrade_signals):
        assert strong_upgrade_signals.has_sufficient_data is True

    def test_has_sufficient_data_with_minimal_signals(self):
        minimal = ForwardEstimateSignals(eps_norm_fy1e=3.00)
        assert minimal.has_sufficient_data is False

    def test_has_sufficient_data_boundary(self):
        boundary = ForwardEstimateSignals(eps_norm_fy1e=3.00, revision_1m=0.5)
        assert boundary.has_sufficient_data is True

    def test_analyst_count_stored(self, strong_upgrade_signals):
        assert strong_upgrade_signals.analyst_count == 18

    def test_analyze_enhanced_df_carries_forward_signals(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        expected_columns = [
            "gaap_revision_momentum",
            "gaap_norm_spread",
            "revision_trend_short",
            "revision_trend_medium",
            "eps_norm_est_fy1e",
            "quarterly_beat_streak",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing enrichment column: {col}"

    def test_gaap_revision_momentum_in_output_range(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        scores = result["gaap_revision_momentum"].dropna()
        assert (scores >= 0).all()
        assert (scores <= 100).all()

    def test_upgrade_stock_has_higher_posterior_than_downgrade(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        aapl = result[result["ticker"] == "AAPL"]["posterior_beat_prob"].iloc[0]
        tsla = result[result["ticker"] == "TSLA"]["posterior_beat_prob"].iloc[0]
        assert aapl > tsla


# =============================================================================
# 4. ACCOUNTING QUALITY GUARD
# =============================================================================


class TestAccountingQualityGuard:
    def test_gaap_norm_spread_clean_alignment(self, strong_upgrade_signals):
        spread = strong_upgrade_signals.gaap_norm_spread
        assert spread == pytest.approx(-4.0, abs=0.1)

    def test_gaap_norm_spread_high_divergence(self, high_gaap_divergence_signals):
        spread = high_gaap_divergence_signals.gaap_norm_spread
        assert spread < -30.0

    def test_gaap_norm_spread_none_when_missing(self):
        empty = ForwardEstimateSignals()
        assert empty.gaap_norm_spread is None

    def test_high_divergence_reduces_posterior_confidence(
        self, model, perfect_beat_history, strong_upgrade_signals, high_gaap_divergence_signals
    ):
        result_clean = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
        )
        result_noisy = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=high_gaap_divergence_signals,
        )
        clean_ci_width = (
            result_clean["credible_interval_95"][1] - result_clean["credible_interval_95"][0]
        )
        noisy_ci_width = (
            result_noisy["credible_interval_95"][1] - result_noisy["credible_interval_95"][0]
        )
        noisy_less_extreme = abs(result_noisy["posterior_mean"] - 0.5) < abs(
            result_clean["posterior_mean"] - 0.5
        )
        noisy_wider_ci = noisy_ci_width > clean_ci_width
        assert noisy_less_extreme or noisy_wider_ci

    def test_small_divergence_no_penalty(self, model, perfect_beat_history, strong_upgrade_signals):
        result = model.compute_forward_adjusted_beat_probability(
            reported_history=perfect_beat_history,
            forward_signals=strong_upgrade_signals,
        )
        assert result["posterior_mean"] > 0.70

    def test_gaap_divergence_penalty_is_proportional(self, model, perfect_beat_history):
        moderate = ForwardEstimateSignals(
            eps_norm_fy1e=4.00,
            eps_gaap_fy1e=3.00,
            revision_1m=1.0,
            revision_3m=0.5,
        )
        extreme = ForwardEstimateSignals(
            eps_norm_fy1e=4.00,
            eps_gaap_fy1e=2.00,
            revision_1m=1.0,
            revision_3m=0.5,
        )
        result_moderate = model.compute_forward_adjusted_beat_probability(
            perfect_beat_history, moderate
        )
        result_extreme = model.compute_forward_adjusted_beat_probability(
            perfect_beat_history, extreme
        )
        assert abs(result_extreme["posterior_mean"] - 0.5) <= abs(
            result_moderate["posterior_mean"] - 0.5
        )

    def test_gaap_norm_spread_in_enhanced_output(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        assert "gaap_norm_spread" in result.columns
        assert result["gaap_norm_spread"].notna().sum() > 0

    def test_tsla_high_divergence_penalized_in_df(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        tsla = result[result["ticker"] == "TSLA"].iloc[0]
        assert tsla["gaap_norm_spread"] is not None
        assert abs(tsla["gaap_norm_spread"]) > 20.0

    def test_divergent_gaap_revisions_vs_norm_revisions(self, model, perfect_beat_history):
        aligned = ForwardEstimateSignals(
            eps_norm_fy1e=5.00,
            eps_gaap_fy1e=4.80,
            revision_1m=2.0,
            revision_3m=1.0,
            gaap_revision_1m=1.8,
            gaap_revision_3m=0.8,
        )
        divergent = ForwardEstimateSignals(
            eps_norm_fy1e=5.00,
            eps_gaap_fy1e=3.50,
            revision_1m=2.0,
            revision_3m=1.0,
            gaap_revision_1m=-1.0,
            gaap_revision_3m=-2.0,
        )
        result_aligned = model.compute_forward_adjusted_beat_probability(
            perfect_beat_history, aligned
        )
        result_divergent = model.compute_forward_adjusted_beat_probability(
            perfect_beat_history, divergent
        )
        assert result_divergent["posterior_mean"] <= result_aligned["posterior_mean"]


# =============================================================================
# INTEGRATION: Full DataFrame Pipeline
# =============================================================================


class TestEnhancedPipelineIntegration:
    def test_output_schema_complete(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        required_columns = [
            "ticker",
            "name",
            "sector",
            "historical_beats",
            "total_reports",
            "historical_beat_rate",
            "posterior_beat_prob",
            "posterior_std",
            "ci_90_lower",
            "ci_90_upper",
            "ci_95_lower",
            "ci_95_upper",
            "confidence_score",
            "prior_influence_pct",
            "effective_sample_size",
            "classification_confidence",
            "beat_classification",
            "gaap_revision_momentum",
            "gaap_norm_spread",
            "revision_trend_short",
            "revision_trend_medium",
            "eps_norm_est_fy1e",
            "quarterly_beat_streak",
            "data_source",
        ]
        for col in required_columns:
            assert col in result.columns, f"Missing output column: {col}"

    def test_no_nans_in_core_probability_columns(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        core_cols = [
            "posterior_beat_prob",
            "posterior_std",
            "confidence_score",
            "ci_90_lower",
            "ci_90_upper",
            "ci_95_lower",
            "ci_95_upper",
        ]
        for col in core_cols:
            assert result[col].notna().all(), f"NaN found in {col}"

    def test_posterior_probabilities_in_valid_range(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        assert (result["posterior_beat_prob"] > 0).all()
        assert (result["posterior_beat_prob"] < 1).all()

    def test_credible_interval_ordering(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        assert (result["ci_90_lower"] < result["ci_90_upper"]).all()
        assert (result["ci_95_lower"] < result["ci_95_upper"]).all()
        assert (result["ci_95_lower"] <= result["ci_90_lower"]).all()
        assert (result["ci_95_upper"] >= result["ci_90_upper"]).all()

    def test_beat_classification_values(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        valid_values = {"likely_beat", "uncertain"}
        assert set(result["beat_classification"].unique()).issubset(valid_values)

    def test_classification_confidence_values(self, model, sample_enhanced_df):
        result = model.analyze_dataframe_enhanced(sample_enhanced_df)
        valid_values = {"High", "Medium", "Low"}
        assert set(result["classification_confidence"].unique()).issubset(valid_values)

    def test_fallback_to_proxy_when_no_forward_data(self, model):
        df_no_forward = pd.DataFrame(
            {
                "ticker": ["TEST"],
                "name": ["Test Corp"],
                "sector": ["Industrials"],
                "eps_trajectory_score": [70.0],
            }
        )
        result = model.analyze_dataframe_enhanced(df_no_forward)
        assert len(result) == 1
        assert result.iloc[0]["data_source"] == "trajectory_proxy"

    def test_skip_rows_with_no_data(self, model):
        df_empty = pd.DataFrame(
            {
                "ticker": ["GHOST"],
                "name": ["Ghost Corp"],
                "sector": ["Unknown"],
            }
        )
        result = model.analyze_dataframe_enhanced(df_empty)
        assert len(result) == 0

    def test_mixed_data_availability(self, model, sample_enhanced_df):
        extra_row = pd.DataFrame(
            {
                "ticker": ["PRXY"],
                "name": ["Proxy Corp"],
                "sector": ["Materials"],
                "eps_trajectory_score": [60.0],
            }
        )
        df_mixed = pd.concat([sample_enhanced_df, extra_row], ignore_index=True)
        result = model.analyze_dataframe_enhanced(df_mixed)
        assert len(result) == 6
        proxy_row = result[result["ticker"] == "PRXY"]
        assert len(proxy_row) == 1
        assert proxy_row.iloc[0]["data_source"] == "trajectory_proxy"
        enhanced_rows = result[result["data_source"] == "forward_enhanced"]
        assert len(enhanced_rows) == 5

    def test_ordering_by_sector_prior_matters(self, tech_model, sample_enhanced_df):
        result = tech_model.analyze_dataframe_enhanced(sample_enhanced_df)
        tech_mean = result[result["sector"] == "Information Technology"][
            "posterior_beat_prob"
        ].mean()
        non_tech_mean = result[result["sector"] != "Information Technology"][
            "posterior_beat_prob"
        ].mean()
        assert tech_mean > non_tech_mean
