"""
Tests for market_analytics.py enhancements (A-H).

TDD tests covering:
- Enhancement A: Optimization status + cached data loading imports
- Enhancement B: optimized_ops + advanced statistical_analysis integrations
- Enhancement C: analyze_dataframe_enhanced + full probability export
- Enhancement D: Remaining screening functions
- Enhancement E: Step numbering fix + variable safety
- Enhancement F: export_probability_view_results in Step 3
- Enhancement G: Enriched export (composite scores, Kalman, MC)
- Enhancement H: Updated summary text
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def comprehensive_stock_df() -> pd.DataFrame:
    """DataFrame with all columns needed for enhanced workflow."""
    np.random.seed(42)
    n = 100

    industries = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

    df = pd.DataFrame(
        {
            # Identifiers
            "ticker": [f"TICK{i:03d}" for i in range(n)],
            "name": [f"Company {i}" for i in range(n)],
            "isin": [f"US000{i:04d}" for i in range(n)],
            "industry": np.random.choice(industries, n),
            "sector": np.random.choice(industries, n),
            # Price data
            "last_price": np.random.uniform(10, 500, n).round(2),
            "price_target_low": np.random.uniform(8, 400, n).round(2),
            "price_target_median": np.random.uniform(15, 550, n).round(2),
            "price_target_high": np.random.uniform(20, 700, n).round(2),
            "market_cap": np.random.uniform(1e9, 1e12, n),
            # Valuation Ratios
            "p_e_ratio": np.random.uniform(5, 60, n).round(2),
            "p_b_ratio": np.random.uniform(0.5, 10, n).round(2),
            "ev_ebitda_ratio": np.random.uniform(3, 30, n).round(2),
            "ev_sales_ratio": np.random.uniform(0.5, 15, n).round(2),
            "dividend_yield": np.random.uniform(0, 6, n).round(2),
            "peg_ratio": np.random.uniform(0.5, 5, n).round(2),
            "price_to_tangible_book": np.random.uniform(0.5, 8, n).round(2),
            "tangible_book_value_ltm": np.random.uniform(1e6, 1e10, n),
            # Momentum & Technical
            "price_momentum_1m": np.random.uniform(-30, 50, n).round(2),
            "price_momentum_3m": np.random.uniform(-40, 60, n).round(2),
            "price_momentum_6m": np.random.uniform(-50, 80, n).round(2),
            "price_momentum_1y": np.random.uniform(-60, 100, n).round(2),
            "range_52w_position": np.random.uniform(0, 1, n).round(3),
            # Profitability
            "roe": np.random.uniform(-20, 40, n).round(2),
            "roa": np.random.uniform(-10, 25, n).round(2),
            "gross_margin_pct": np.random.uniform(10, 80, n).round(2),
            "operating_margin_pct": np.random.uniform(-10, 40, n).round(2),
            "net_margin_pct": np.random.uniform(-15, 30, n).round(2),
            "ebitda_margin_pct": np.random.uniform(5, 50, n).round(2),
            "roic": np.random.uniform(-5, 30, n).round(2),
            # Quality & Risk
            "piotroski_f_score": np.random.randint(0, 10, n),
            "distress_risk_score": np.random.uniform(10, 95, n).round(1),
            "altman_z_score": np.random.uniform(-2, 10, n).round(2),
            "accounting_quality_score": np.random.uniform(20, 90, n).round(1),
            "earnings_quality_composite": np.random.uniform(20, 90, n).round(1),
            "cash_flow_quality_score": np.random.uniform(20, 90, n).round(1),
            # Leverage & Liquidity
            "debt_to_equity": np.random.uniform(0, 3, n).round(2),
            "current_ratio": np.random.uniform(0.5, 4, n).round(2),
            "quick_ratio": np.random.uniform(0.3, 3, n).round(2),
            "interest_coverage_ratio": np.random.uniform(0, 20, n).round(2),
            # Analyst Sentiment
            "upside_potential": np.random.uniform(-30, 80, n).round(2),
            # Earnings Quality
            "eps_trajectory_score": np.random.uniform(20, 90, n).round(1),
            "earnings_quality_score": np.random.uniform(20, 90, n).round(1),
            # Growth Metrics
            "revenue_growth_yoy": np.random.uniform(-30, 60, n).round(2),
            "eps_yoy_growth": np.random.uniform(-50, 100, n).round(2),
            # Cash Flow
            "fcf_positive_years": np.random.randint(0, 6, n),
            "fcf_margin": np.random.uniform(-20, 30, n).round(2),
            "fcf_yield": np.random.uniform(-10, 15, n).round(2),
            # Dividend Features
            "dividend_streak": np.random.randint(0, 20, n),
            "dividend_yield_ltm": np.random.uniform(0, 8, n).round(2),
            "dividend_payout_ratio": np.random.uniform(0, 120, n).round(1),
            "fcf_dividend_coverage": np.random.uniform(0, 5, n).round(2),
            # Statistical analysis columns
            "cash_burn_rate": np.random.uniform(-1e8, 1e8, n),
            "volatility": np.random.uniform(0.1, 0.8, n).round(3),
        }
    )

    return df


@pytest.fixture
def feature_categories() -> dict:
    """Minimal feature categories for testing."""
    return {
        "Valuation Ratios": ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio"],
        "Profitability": ["roe", "roa", "gross_margin_pct"],
        "Quality & Risk": ["piotroski_f_score", "distress_risk_score"],
    }


def _read_market_analytics_source() -> str:
    """Read the source code of market_analytics.py."""
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "market_analytics.py"
    return path.read_text(encoding="utf-8")


# =============================================================================
# Enhancement A — Optimization status + cached data loading
# =============================================================================


class TestEnhancementA_OptimizedImports:
    """Test that optimized_ops functions are imported in market_analytics."""

    def test_imports_get_optimization_status(self):
        source = _read_market_analytics_source()
        assert (
            "get_optimization_status" in source
        ), "market_analytics.py must import get_optimization_status from optimized_ops"

    def test_imports_load_feature_data_from_db_cached(self):
        source = _read_market_analytics_source()
        assert (
            "load_feature_data_from_db_cached" in source
        ), "market_analytics.py must import load_feature_data_from_db_cached"

    def test_imports_dataframe_hash(self):
        source = _read_market_analytics_source()
        assert (
            "dataframe_hash" in source
        ), "market_analytics.py must import dataframe_hash from optimized_ops"

    def test_optimization_status_called_in_main(self):
        """Step 0: Optimization status should be reported."""
        source = _read_market_analytics_source()
        assert (
            "get_optimization_status()" in source
        ), "main() must call get_optimization_status() to report optimization status"

    def test_cached_loader_used(self):
        """Data loading should use the cached variant."""
        source = _read_market_analytics_source()
        assert (
            "load_feature_data_from_db_cached()" in source
        ), "main() must use load_feature_data_from_db_cached() for data loading"

    def test_dataframe_hash_used(self):
        """DataFrame hash should be computed after loading."""
        source = _read_market_analytics_source()
        assert "dataframe_hash(df)" in source, "main() must call dataframe_hash(df) after loading"

    def test_get_optimization_status_works(self):
        """Verify the optimized_ops function returns expected structure."""
        from finance_ml.analytics.optimized_ops import get_optimization_status

        status = get_optimization_status()
        assert isinstance(status, dict)
        assert "numba_available" in status
        assert "db_cache_size" in status


# =============================================================================
# Enhancement B — Fast MC, ruin probability, z-scores, Kalman, copula, MCMC
# =============================================================================


class TestEnhancementB_StatisticalIntegrations:
    """Test integration of optimized_ops and advanced statistical_analysis."""

    def test_imports_fast_monte_carlo_simulation(self):
        source = _read_market_analytics_source()
        assert "fast_monte_carlo_simulation" in source

    def test_imports_fast_ruin_probability(self):
        source = _read_market_analytics_source()
        assert "fast_ruin_probability" in source

    def test_imports_vectorized_zscore(self):
        source = _read_market_analytics_source()
        assert "vectorized_zscore" in source

    def test_imports_vectorized_percentile_rank(self):
        source = _read_market_analytics_source()
        assert "vectorized_percentile_rank" in source

    def test_imports_kalman_filter_price_target(self):
        source = _read_market_analytics_source()
        assert "kalman_filter_price_target" in source

    def test_imports_kalman_momentum_filter(self):
        source = _read_market_analytics_source()
        assert "kalman_momentum_filter" in source

    def test_imports_fit_gaussian_copula(self):
        source = _read_market_analytics_source()
        assert "fit_gaussian_copula" in source

    def test_imports_parallel_mcmc_chains(self):
        source = _read_market_analytics_source()
        assert "parallel_mcmc_chains" in source

    def test_imports_hierarchical_mcmc_by_sector(self):
        source = _read_market_analytics_source()
        assert "hierarchical_mcmc_by_sector" in source

    def test_imports_fit_distributions_by_category(self):
        source = _read_market_analytics_source()
        assert "fit_distributions_by_category" in source

    def test_imports_calculate_conditional_probabilities(self):
        source = _read_market_analytics_source()
        assert "calculate_conditional_probabilities" in source

    def test_imports_metropolis_hastings_sampler(self):
        source = _read_market_analytics_source()
        assert "metropolis_hastings_sampler" in source

    def test_imports_mcmc_student_t(self):
        source = _read_market_analytics_source()
        assert "mcmc_student_t" in source

    def test_imports_run_category_probability_analytics(self):
        source = _read_market_analytics_source()
        assert "run_category_probability_analytics" in source

    def test_fast_ruin_used_in_step4(self):
        """Step 4 should use fast_ruin_probability instead of only calculate_ruin_probability."""
        source = _read_market_analytics_source()
        assert (
            "fast_ruin_probability(" in source
        ), "Step 4 must call fast_ruin_probability() from optimized_ops"

    def test_fast_monte_carlo_used_in_step4(self):
        source = _read_market_analytics_source()
        assert "fast_monte_carlo_simulation(" in source

    def test_vectorized_zscore_used(self):
        source = _read_market_analytics_source()
        assert "vectorized_zscore(df" in source or "vectorized_zscore(" in source

    def test_kalman_filter_price_target_used(self):
        source = _read_market_analytics_source()
        assert "kalman_filter_price_target(" in source

    def test_parallel_mcmc_chains_used(self):
        source = _read_market_analytics_source()
        assert "parallel_mcmc_chains(" in source

    def test_fit_gaussian_copula_used(self):
        source = _read_market_analytics_source()
        assert "fit_gaussian_copula(" in source

    def test_hierarchical_mcmc_used(self):
        source = _read_market_analytics_source()
        assert "hierarchical_mcmc_by_sector(" in source

    def test_fit_distributions_used(self):
        source = _read_market_analytics_source()
        assert "fit_distributions_by_category(" in source

    def test_conditional_probabilities_used(self):
        source = _read_market_analytics_source()
        assert "calculate_conditional_probabilities(" in source

    def test_category_probability_analytics_used(self):
        source = _read_market_analytics_source()
        assert "run_category_probability_analytics(" in source

    # Functional tests for the integrated functions

    def test_fast_monte_carlo_returns_dataframe(self, comprehensive_stock_df):
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation

        result = fast_monte_carlo_simulation(comprehensive_stock_df, n_simulations=100)
        assert isinstance(result, pd.DataFrame)
        assert "expected_upside" in result.columns
        assert "risk_reward_ratio" in result.columns

    def test_fast_ruin_probability_returns_dataframe(self, comprehensive_stock_df):
        from finance_ml.analytics.optimized_ops import fast_ruin_probability

        result = fast_ruin_probability(comprehensive_stock_df, n_simulations=50, n_days=30)
        assert isinstance(result, pd.DataFrame)
        assert "ruin_probability" in result.columns

    def test_vectorized_zscore_adds_columns(self, comprehensive_stock_df):
        from finance_ml.analytics.optimized_ops import vectorized_zscore

        result = vectorized_zscore(
            comprehensive_stock_df, ["roe", "p_e_ratio"], group_col="industry"
        )
        assert "roe_zscore" in result.columns
        assert "p_e_ratio_zscore" in result.columns

    def test_vectorized_percentile_rank_adds_columns(self, comprehensive_stock_df):
        from finance_ml.analytics.optimized_ops import vectorized_percentile_rank

        result = vectorized_percentile_rank(comprehensive_stock_df, ["roe"], group_col="industry")
        assert "roe_pctile" in result.columns

    def test_kalman_filter_price_target_works(self, comprehensive_stock_df):
        from finance_ml.analytics.statistical_analysis import kalman_filter_price_target

        result = kalman_filter_price_target(
            comprehensive_stock_df,
            observation_col="last_price",
            target_col="price_target_median",
        )
        assert isinstance(result, pd.DataFrame)

    def test_parallel_mcmc_chains_works(self):
        from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

        np.random.seed(42)
        data = np.random.normal(10, 2, 200)
        result = parallel_mcmc_chains(data, n_chains=2, n_samples=500)
        assert "r_hat" in result
        assert "converged" in result
        assert "posterior_mean" in result

    def test_fit_gaussian_copula_works(self, comprehensive_stock_df):
        from finance_ml.analytics.statistical_analysis import fit_gaussian_copula

        result = fit_gaussian_copula(comprehensive_stock_df, ["roe", "p_e_ratio"], n_simulations=100)
        assert "features" in result
        assert "n_observations" in result

    def test_hierarchical_mcmc_works(self):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_by_sector

        np.random.seed(42)
        n = 200
        df_large = pd.DataFrame(
            {
                "roe": np.random.uniform(-20, 40, n),
                "industry": np.random.choice(["Tech", "Health", "Finance"], n),
            }
        )
        result = hierarchical_mcmc_by_sector(df_large, "roe", sector_col="industry", n_samples=200)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_fit_distributions_by_category_works(self, comprehensive_stock_df):
        from finance_ml.analytics.statistical_analysis import fit_distributions_by_category

        result = fit_distributions_by_category(
            comprehensive_stock_df, "Profitability", ["roe", "roa"], n_simulations=100
        )
        assert isinstance(result, dict)

    def test_calculate_conditional_probabilities_works(self, comprehensive_stock_df):
        from finance_ml.analytics.statistical_analysis import calculate_conditional_probabilities

        cats = {"Profitability": ["roe", "roa"]}
        result = calculate_conditional_probabilities(comprehensive_stock_df, cats)
        assert isinstance(result, pd.DataFrame)

    def test_run_category_probability_analytics_works(self, comprehensive_stock_df):
        from finance_ml.analytics.statistical_analysis import run_category_probability_analytics

        result = run_category_probability_analytics(
            comprehensive_stock_df, "Profitability", ["roe", "roa"], n_simulations=100
        )
        assert isinstance(result, dict)
        assert "features_analyzed" in result


# =============================================================================
# Enhancement C — analyze_dataframe_enhanced + full probability export
# =============================================================================


class TestEnhancementC_ProbabilityEnhancements:
    """Test analyze_dataframe_enhanced usage and full probability export."""

    def test_analyze_dataframe_enhanced_called(self):
        """Should try enhanced analysis before fallback."""
        source = _read_market_analytics_source()
        assert (
            "analyze_dataframe_enhanced" in source
        ), "main() must call analyze_dataframe_enhanced for three-layer evidence fusion"

    def test_export_includes_credit_dividend_pt(self):
        """export_probability_analytics_results must receive credit/dividend/PT args."""
        source = _read_market_analytics_source()
        assert (
            "credit_risk_df=" in source
        ), "export_probability_analytics_results must pass credit_risk_df"
        assert (
            "dividend_safety_df=" in source
        ), "export_probability_analytics_results must pass dividend_safety_df"
        assert (
            "price_target_df=" in source
        ), "export_probability_analytics_results must pass price_target_df"

    def test_credit_results_variable_defined(self):
        """credit_results must be initialized."""
        source = _read_market_analytics_source()
        assert "credit_results = None" in source or "credit_results = credit_model" in source

    def test_dividend_results_variable_defined(self):
        source = _read_market_analytics_source()
        assert "dividend_results = None" in source or "dividend_results = dividend_model" in source

    def test_pt_results_variable_defined(self):
        source = _read_market_analytics_source()
        assert "pt_results = None" in source or "pt_results = pt_model" in source


# =============================================================================
# Enhancement D — Remaining screening functions
# =============================================================================


class TestEnhancementD_ScreeningIntegrations:
    """Test integration of remaining screening functions."""

    def test_imports_screen_earnings_quality(self):
        source = _read_market_analytics_source()
        assert "screen_earnings_quality" in source

    def test_imports_screen_dividend_quality(self):
        source = _read_market_analytics_source()
        assert "screen_dividend_quality" in source

    def test_imports_rank_stocks_by_composite_score(self):
        source = _read_market_analytics_source()
        assert "rank_stocks_by_composite_score" in source

    def test_imports_create_sector_relative_ranking(self):
        source = _read_market_analytics_source()
        assert "create_sector_relative_ranking" in source

    def test_screen_earnings_quality_called(self):
        source = _read_market_analytics_source()
        assert "screen_earnings_quality(" in source

    def test_screen_dividend_quality_called(self):
        source = _read_market_analytics_source()
        assert "screen_dividend_quality(" in source

    def test_rank_stocks_called(self):
        source = _read_market_analytics_source()
        assert "rank_stocks_by_composite_score(" in source

    def test_sector_relative_ranking_called(self):
        source = _read_market_analytics_source()
        assert "create_sector_relative_ranking(" in source

    # Functional tests for screening functions

    def test_screen_earnings_quality_returns_df(self, comprehensive_stock_df):
        from finance_ml.analytics.screening import screen_earnings_quality

        result = screen_earnings_quality(comprehensive_stock_df, min_quality_score=30)
        assert isinstance(result, pd.DataFrame)

    def test_screen_dividend_quality_returns_df(self, comprehensive_stock_df):
        from finance_ml.analytics.screening import screen_dividend_quality

        result = screen_dividend_quality(
            comprehensive_stock_df, min_dividend_yield=1.0, min_dividend_streak=1
        )
        assert isinstance(result, pd.DataFrame)

    def test_rank_stocks_returns_df(self, comprehensive_stock_df):
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        result = rank_stocks_by_composite_score(comprehensive_stock_df)
        assert isinstance(result, pd.DataFrame)
        assert "composite_score" in result.columns

    def test_sector_relative_ranking_adds_column(self, comprehensive_stock_df):
        from finance_ml.analytics.screening import create_sector_relative_ranking

        result = create_sector_relative_ranking(
            comprehensive_stock_df, "roe", sector_col="industry"
        )
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Enhancement E — Step numbering fix + variable safety
# =============================================================================


class TestEnhancementE_StepNumberingAndVariableSafety:
    """Test step numbering is fixed and probability variables are safely initialized."""

    def test_no_duplicate_step_5(self):
        """Step 5 should not appear twice."""
        source = _read_market_analytics_source()
        step_5_matches = re.findall(r"Step 5:", source)
        assert (
            len(step_5_matches) <= 1
        ), f"Step 5 appears {len(step_5_matches)} times — should appear at most once"

    def test_visualizations_step_is_7(self):
        """Visualizations should be Step 7, not Step 5."""
        source = _read_market_analytics_source()
        assert (
            "Step 7: Generating visualizations" in source
        ), "Visualizations should be labeled as Step 7"

    def test_export_step_is_8(self):
        """Export should be Step 8."""
        source = _read_market_analytics_source()
        assert "Step 8: Exporting results" in source, "Export should be labeled as Step 8"

    def test_step_numbers_sequential(self):
        """Steps should be numbered sequentially: 1-8."""
        source = _read_market_analytics_source()
        for i in range(1, 9):
            assert f"Step {i}:" in source, f"Step {i} not found in market_analytics.py"

    def test_probability_variables_defined_in_else_block(self):
        """When PROBABILITY_ANALYTICS_AVAILABLE is False, variables must still be defined."""
        source = _read_market_analytics_source()
        # After the probability block's else, variables should be set to None
        # Check that there's an else block defining these variables
        else_pattern = re.search(
            r"else:.*?probability_results\s*=\s*None.*?streak_results\s*=\s*None",
            source,
            re.DOTALL,
        )
        assert else_pattern is not None, (
            "Must have an else block setting probability_results and streak_results to None "
            "when PROBABILITY_ANALYTICS_AVAILABLE is False"
        )

    def test_credit_results_safe_in_else(self):
        source = _read_market_analytics_source()
        else_blocks = source.split("else:")
        found_credit_none = False
        for block in else_blocks:
            if "credit_results = None" in block[:500]:
                found_credit_none = True
                break
        assert found_credit_none, "credit_results must be set to None in else block"

    def test_dividend_results_safe_in_else(self):
        source = _read_market_analytics_source()
        else_blocks = source.split("else:")
        found_div_none = False
        for block in else_blocks:
            if "dividend_results = None" in block[:500]:
                found_div_none = True
                break
        assert found_div_none, "dividend_results must be set to None in else block"

    def test_pt_results_safe_in_else(self):
        source = _read_market_analytics_source()
        else_blocks = source.split("else:")
        found_pt_none = False
        for block in else_blocks:
            if "pt_results = None" in block[:500]:
                found_pt_none = True
                break
        assert found_pt_none, "pt_results must be set to None in else block"


# =============================================================================
# Enhancement F — export_probability_view_results in Step 3
# =============================================================================


class TestEnhancementF_ProbabilityViewExport:
    """Test integration of export_probability_view_results in view analytics."""

    def test_imports_export_probability_view_results(self):
        source = _read_market_analytics_source()
        assert (
            "export_probability_view_results" in source
        ), "market_analytics.py must import export_probability_view_results"

    def test_export_probability_view_results_called(self):
        """Per-feature probability export should be called in the view analytics loop."""
        source = _read_market_analytics_source()
        assert (
            "export_probability_view_results(" in source
        ), "main() must call export_probability_view_results() in Step 3"


# =============================================================================
# Enhancement G — Enriched export (composite, Kalman, MC)
# =============================================================================


class TestEnhancementG_EnrichedExport:
    """Test that Step 8 exports composite scores, Kalman targets, and MC results."""

    def test_export_composite_scores(self):
        """Composite quality scores should be exported."""
        source = _read_market_analytics_source()
        assert "composite_quality_scores" in source, "Step 8 must export composite_quality_scores"

    def test_export_kalman_filtered_targets(self):
        """Kalman-filtered price targets should be exported."""
        source = _read_market_analytics_source()
        assert (
            "kalman_filtered_price_targets" in source
        ), "Step 8 must export kalman_filtered_price_targets"

    def test_export_monte_carlo_results(self):
        """Monte Carlo simulation results should be exported."""
        source = _read_market_analytics_source()
        assert (
            "monte_carlo_simulation" in source
        ), "Step 8 must export monte_carlo_simulation results"


# =============================================================================
# Enhancement H — Updated summary
# =============================================================================


class TestEnhancementH_UpdatedSummary:
    """Test that the summary section reflects all integrations."""

    def test_summary_mentions_optimized_ops(self):
        source = _read_market_analytics_source()
        assert "optimized_ops" in source, "Summary must mention optimized_ops module"

    def test_summary_mentions_kalman(self):
        source = _read_market_analytics_source()
        # The summary line should mention Kalman
        assert "Kalman" in source, "Summary must mention Kalman"

    def test_summary_mentions_copula(self):
        source = _read_market_analytics_source()
        assert "Copula" in source or "copula" in source

    def test_summary_mentions_12_screeners(self):
        source = _read_market_analytics_source()
        assert "12 screeners" in source, "Summary must mention 12 screeners"

    def test_summary_reports_credit_results(self):
        source = _read_market_analytics_source()
        assert "credit_results" in source, "Summary must report credit risk analysis results"

    def test_summary_reports_dividend_results(self):
        source = _read_market_analytics_source()
        assert "dividend_results" in source, "Summary must report dividend safety analysis results"

    def test_summary_reports_pt_results(self):
        source = _read_market_analytics_source()
        assert "pt_results" in source, "Summary must report price target analysis results"


# =============================================================================
# Cross-Enhancement Integration Tests
# =============================================================================


class TestCrossEnhancementIntegration:
    """Test that multiple enhancements work together."""

    def test_fast_ruin_then_export(self, comprehensive_stock_df):
        """Fast ruin probability results should be exportable."""
        from finance_ml.analytics.optimized_ops import fast_ruin_probability
        from finance_ml.analytics.data_utils import compute_metric_statistics

        ruin_df = fast_ruin_probability(comprehensive_stock_df, n_simulations=50, n_days=30)
        assert len(ruin_df) > 0
        stats = compute_metric_statistics(ruin_df["ruin_probability"])
        assert stats is not None

    def test_monte_carlo_then_ranking(self, comprehensive_stock_df):
        """MC simulation + composite ranking should work together."""
        from finance_ml.analytics.optimized_ops import fast_monte_carlo_simulation
        from finance_ml.analytics.screening import rank_stocks_by_composite_score

        mc_results = fast_monte_carlo_simulation(comprehensive_stock_df, n_simulations=100)
        ranked = rank_stocks_by_composite_score(comprehensive_stock_df)
        assert len(mc_results) > 0
        assert "composite_score" in ranked.columns

    def test_zscore_then_screening(self, comprehensive_stock_df):
        """Vectorized z-scores then screening should work."""
        from finance_ml.analytics.optimized_ops import vectorized_zscore
        from finance_ml.analytics.screening import screen_earnings_quality

        df = vectorized_zscore(comprehensive_stock_df, ["roe"], group_col="industry")
        result = screen_earnings_quality(df, min_quality_score=30)
        assert isinstance(result, pd.DataFrame)

    def test_kalman_then_copula(self, comprehensive_stock_df):
        """Kalman momentum filter then copula should work."""
        from finance_ml.analytics.statistical_analysis import (
            kalman_momentum_filter,
            fit_gaussian_copula,
        )

        df = kalman_momentum_filter(
            comprehensive_stock_df,
            momentum_cols=["price_momentum_1m", "price_momentum_3m"],
        )
        copula = fit_gaussian_copula(df, ["roe", "p_e_ratio"], n_simulations=100)
        assert "features" in copula

    def test_full_step4_pipeline(self, comprehensive_stock_df):
        """Full Step 4 pipeline: fast ruin + MC + zscore + Kalman + copula + MCMC."""
        from finance_ml.analytics.optimized_ops import (
            fast_ruin_probability,
            fast_monte_carlo_simulation,
            vectorized_zscore,
            vectorized_percentile_rank,
        )
        from finance_ml.analytics.statistical_analysis import (
            kalman_filter_price_target,
            parallel_mcmc_chains,
            fit_gaussian_copula,
        )

        # Fast ruin
        ruin_df = fast_ruin_probability(comprehensive_stock_df, n_simulations=50, n_days=30)
        assert len(ruin_df) > 0

        # Fast MC
        mc_results = fast_monte_carlo_simulation(comprehensive_stock_df, n_simulations=100)
        assert len(mc_results) > 0

        # Z-scores
        df = vectorized_zscore(comprehensive_stock_df, ["roe", "p_e_ratio"], group_col="industry")
        df = vectorized_percentile_rank(df, ["roe", "p_e_ratio"], group_col="industry")
        assert "roe_zscore" in df.columns
        assert "roe_pctile" in df.columns

        # Kalman
        kalman_pt = kalman_filter_price_target(
            df, observation_col="last_price", target_col="price_target_median"
        )
        assert isinstance(kalman_pt, pd.DataFrame)

        # Parallel MCMC
        roe_data = df["roe"].dropna().values
        mcmc_result = parallel_mcmc_chains(roe_data, n_chains=2, n_samples=200)
        assert mcmc_result["r_hat"] > 0

        # Copula
        copula = fit_gaussian_copula(df, ["roe", "p_e_ratio"], n_simulations=100)
        assert len(copula["features"]) == 2
