"""
Expected Returns Analytics Module (v3.0)

Automated pipeline for expected returns analysis using the v3.0+ analytics platform:
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions
- **Price Target Achievement** — Probability-weighted expected returns by sector
- **Kalman Filtered Targets** — Noise-reduced price target signals
- **Earnings Beat Analysis** — Three-layer Bayesian earnings beat probability
- **Cross-Model Comparison** — MC vs Kalman vs Achievement model alignment
- **Quad-Model Agreement** — MC + Kalman + Achievement + Earnings Beat
- **Statistical Analysis** — Bayesian category analysis, copula dependency, MCMC
- **Probability Analytics** — Category-level probability distributions, credit risk
- **Stock Screening** — Quality, value, growth, dividend, GARP, health filters

Data sources (v3.0 — Materialized Views):
    - public.mv_expected_returns      (pre-joined expected returns feature set, 17 categories)
    - public.mv_all_stock_features    (full stock features superset for screening/enrichment)
    - analytics.*                     (model output tables)

Migration from v2.5:
    - Replaced 17× vw_features_* LEFT JOIN loading with single MV reads
    - Eliminated _backfill_market_data_columns (market data columns included in MVs)
    - Added screening integration (quality, value, growth, GARP, dividend, health)
    - Added per-category Bayesian probability analytics
    - Added credit risk & dividend safety models
    - Added Gaussian copula cross-model dependency analysis

Usage:
    python expected_returns_v3.py
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

# --- Data utilities ---
from finance_ml.analytics.data_utils import (
    ExportConfig,
    aggregate_probability_results,
    backfill_feature_columns,
    compute_metric_statistics,
    export_to_csv,
    export_to_db,
    export_to_json,
    get_identifier_cols_set,
    load_identifier_columns,
    reorder_with_identifiers,
    validate_feature_alignment,
)

# --- Optimised operations ---
from finance_ml.analytics.optimized_ops import (
    fast_ruin_probability,
    get_optimization_status,
    vectorized_percentile_rank,
    vectorized_zscore,
)

# --- Probability models ---
from finance_ml.analytics.probability_analytics import (
    CategoryProbabilityAnalyzer,
    CreditRiskProbabilityModel,
    DividendCutProbabilityModel,
    EarningsBeatProbabilityModel,
    EPSStreakAnalyzer,
    PriceTargetAchievementModel,
    ResampledBeatProbabilityModel,
    create_earnings_probability_dashboard,
    export_probability_analytics_results,
)

# --- Screening (quality filtering of results) ---
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    create_sector_relative_ranking,
    rank_stocks_by_composite_score,
    screen_dividend_quality,
    screen_earnings_quality,
    screen_financial_health,
    screen_garp_opportunities,
    screen_growth_momentum,
    screen_high_yield_safe_dividends,
    screen_integrity_filtered_growth,
    screen_valuation_reversion_candidates,
    screen_value_opportunities,
)

# --- Statistical analysis ---
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    bayesian_earnings_beat_model,
    calculate_conditional_probabilities,
    calculate_ruin_probability,
    detect_accounting_anomalies,
    fit_distributions_by_category,
    fit_gaussian_copula,
    kalman_filter_price_target,
    monte_carlo_price_target_simulation,
    run_category_probability_analytics,
)

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from finance_ml.analytics.inference_schema import (
        ARVIZ_AVAILABLE,
        EquityCoordinates,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        build_monte_carlo_inference_data,
        summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False
    EquityCoordinates = None  # type: ignore[assignment,misc]

# --- Probabilistic visualizations (ArviZ-backed) ---
# --- Other visualizations ---
from finance_ml.analytics.visualizations._shared import COLORS, PLOTLY_TEMPLATE
from finance_ml.analytics.visualizations.probability_viz import (
    create_bayesian_category_ridge,
    create_beat_probability_posterior,
    create_posterior_return_forest,
    create_ruin_probability_diagnostic,
    create_tri_model_posterior_comparison,
)

# --- Quality & risk visualizations ---
from finance_ml.analytics.visualizations.quality_risk import (
    create_distress_early_warning_dashboard,
    create_quality_risk_quadrant,
)

px.defaults.template = PLOTLY_TEMPLATE

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Categories (aligned with mv_expected_returns 17-category schema)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_CATEGORIES = {
    "Valuation Ratios": [
        "p_e_ratio",
        "p_b_ratio",
        "ev_ebitda_ratio",
        "peg_ratio",
        "price_to_tangible_book",
        "p_e_vs_3y_avg",
    ],
    "Analyst Sentiment": [
        "analyst_bullish_pct",
        "upside_potential",
        "eps_revision_momentum",
        "analyst_conviction",
        "pt_consensus_convergence",
        "analyst_rating_normalized",
        "analyst_coverage_trend",
        "pt_momentum_1m",
        "pt_acceleration_short",
        "price_target_spread_pct",
        "analyst_count",
    ],
    "Price Target Dynamics": [
        "price_target",
        "price_target_high",
        "price_target_low",
        "price_target_median",
        "pt_momentum_1m",
        "pt_momentum_3m",
        "pt_acceleration_short",
        "pt_consensus_convergence",
        # Historical price targets (consensus)
        "price_target_1w_ago",
        "price_target_1m_ago",
        "price_target_3m_ago",
        "price_target_6m_ago",
        "price_target_mtd_ago",
        "price_target_qtd_ago",
        "price_target_1y_ago",
        # Historical high targets
        "price_target_high_1w_ago",
        "price_target_high_1m_ago",
        "price_target_high_6m_ago",
        "price_target_high_mtd_ago",
        "price_target_high_3m_ago",
        "price_target_high_qtd_ago",
        "price_target_high_1y_ago",
        "price_target_high_ytd_ago",
        # Historical low targets
        "price_target_low_1w_ago",
        "price_target_low_1m_ago",
        "price_target_low_3m_ago",
        "price_target_low_6m_ago",
        "price_target_low_mtd_ago",
        "price_target_low_qtd_ago",
        "price_target_low_ytd_ago",
        "price_target_low_1y_ago",
        # Historical median targets
        "price_target_median_1w_ago",
        "price_target_median_1m_ago",
        "price_target_median_3m_ago",
        "price_target_median_6m_ago",
        "price_target_median_mtd_ago",
        "price_target_median_qtd_ago",
        "price_target_median_ytd_ago",
        "price_target_median_1y_ago",
    ],
    "Momentum & Technical": [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_5d",
        "price_momentum_3y",
        "price_momentum_5y",
        "range_52w_position",
        "ema_crossover_20_50",
        "long_term_trend_score",
        # Historical prices
        "price_5d_ago",
        "price_1w_ago",
        "price_1m_ago",
        "price_3m_ago",
        "price_6m_ago",
        "price_1y_ago",
        "price_3y_ago",
        "price_5y_ago",
        "price_qtd_ago",
    ],
    "Earnings Quality": [
        "eps_surprise_pct",
        "eps_trajectory_score",
        "eps_positive_streak",
        "eps_improvement_count",
        "eps_yoy_growth",
        "eps_qoq_growth",
        "earnings_quality_composite",
        "eps_adjustment_ratio",
        "accounting_quality_score",
        "net_income_positive_years",
    ],
    "Profitability": [
        "roe",
        "roa",
        "roic",
        "gross_margin_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "ebitda_margin_pct",
    ],
    "Growth Metrics": [
        "revenue_yoy_growth",
        "ebitda_growth_yoy",
        "net_income_growth_yoy",
        "eps_yoy_growth",
        "revenue_cagr_3y",
        "revenue_cagr_5y",
        "eps_cagr_3y",
        "fcf_growth_yoy",
    ],
    "Quality & Risk": [
        "piotroski_f_score",
        "distress_risk_score",
        "altman_z_score",
        "beta_stability_score",
        "combined_distress_score",
        "quality_momentum_score",
    ],
    "Leverage & Liquidity": [
        "debt_to_equity",
        "current_ratio",
        "quick_ratio",
        "interest_coverage",
        "cash_runway_months",
        "wc_deteriorating_flag",
    ],
    "Cash Flow": [
        "fcf_positive_years",
        "fcf_margin",
        "fcf_yield",
        "fcf_dividend_coverage",
        "cash_flow_quality_score",
        "operating_cf_to_net_income",
    ],
    "Dividends": [
        "dividend_yield_ltm",
        "dividend_streak",
        "dividend_payout_ratio",
        "dividend_growth_expectation",
        "sustainable_dividend_flag",
        "dividend_consistency",
        "fcf_dividend_coverage",
    ],
    "Technical Analysis": [
        "ema_slope_20d",
        "ema_trend_consistency",
        "breakout_signal",
        "volatility_compression",
        "volume_momentum_score",
    ],
    "Composite Scores": [
        "quality_momentum_score",
        "combined_distress_score",
        "earnings_quality_composite",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Per-Model Detailed Statistics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_model_detailed_statistics(
    df: pd.DataFrame,
    model_name: str,
    key_columns: list[str],
    group_col: str = "industry",
) -> dict:
    """
    Compute granular statistics for a model's output DataFrame.

    Uses ``compute_metric_statistics`` for each key column and adds
    distribution shape metrics (skewness, kurtosis), inter-model
    consistency indicators, and sector-level breakdowns.
    """
    if df.empty:
        logger.warning(
            "compute_model_detailed_statistics: %s — empty DataFrame", model_name
        )
        return {}

    results = {}
    for col in key_columns:
        if col not in df.columns:
            continue

        base_stats = compute_metric_statistics(df[col])
        if base_stats is None:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        shape_stats = {}
        if len(series) > 3:
            shape_stats["skewness"] = float(series.skew())
            shape_stats["kurtosis"] = float(series.kurtosis())
            shape_stats["iqr"] = float(base_stats["q75"] - base_stats["q25"])
            shape_stats["coefficient_of_variation"] = (
                float(series.std() / series.mean()) if series.mean() != 0 else None
            )
            mean, std = series.mean(), series.std()
            if std > 0:
                shape_stats["pct_beyond_2std"] = float(
                    ((series < mean - 2 * std) | (series > mean + 2 * std)).sum()
                    / len(series)
                    * 100
                )

        sector_breakdown = {}
        if group_col in df.columns:
            for sector, group in df.groupby(group_col):
                sector_stats = compute_metric_statistics(group[col])
                if sector_stats:
                    sector_breakdown[str(sector)] = {
                        "count": sector_stats["count"],
                        "mean": sector_stats["mean"],
                        "median": sector_stats["median"],
                        "std": sector_stats["std"],
                    }

        results[col] = {
            "global": base_stats,
            "distribution_shape": shape_stats,
            "sector_breakdown": sector_breakdown,
        }

    logger.info(
        "%s: computed detailed statistics for %d / %d columns",
        model_name,
        len(results),
        len(key_columns),
    )
    return results


def print_model_statistics(
    stats: dict,
    model_name: str,
    show_sectors: bool = False,
    top_n_sectors: int = 5,
) -> None:
    """Pretty-print the detailed statistics from compute_model_detailed_statistics."""
    if not stats:
        return

    print(f"\n  📊 {model_name} — Detailed Statistics:")
    for col, info in stats.items():
        g = info["global"]
        s = info.get("distribution_shape", {})
        print(f"    ▸ {col}:")
        print(
            f"        Count: {g['count']:,}  |  Mean: {g['mean']:.2f}  |  "
            f"Median: {g['median']:.2f}  |  Std: {g['std']:.2f}"
        )
        print(
            f"        Min: {g['min']:.2f}  |  Max: {g['max']:.2f}  |  "
            f"IQR: [{g['q25']:.2f}, {g['q75']:.2f}]"
        )
        print(
            f"        Positive: {g['positive_pct']:.1f}%  |  "
            f"Missing: {g['missing_pct']:.1f}%"
        )
        if s:
            print(
                f"        Skew: {s.get('skewness', 0):.3f}  |  "
                f"Kurtosis: {s.get('kurtosis', 0):.3f}  |  "
                f"CV: {s.get('coefficient_of_variation', 0):.3f}"
            )
            if "pct_beyond_2std" in s:
                print(f"        Outliers (>2σ): {s['pct_beyond_2std']:.1f}%")

        if show_sectors and info.get("sector_breakdown"):
            sorted_sectors = sorted(
                info["sector_breakdown"].items(),
                key=lambda x: x[1]["mean"],
                reverse=True,
            )[:top_n_sectors]
            print(f"        Top {top_n_sectors} sectors by mean:")
            for sector, sinfo in sorted_sectors:
                print(
                    f"          {sector}: mean={sinfo['mean']:.2f}, "
                    f"median={sinfo['median']:.2f}, n={sinfo['count']}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading — Materialized View Backend (v3.0)
# ═══════════════════════════════════════════════════════════════════════════════

# v3.0: Single-MV data loading replaces 17× vw_features_* LEFT JOIN merge
_MV_EXPECTED_RETURNS = "mv_expected_returns"
_MV_ALL_STOCK_FEATURES = "mv_all_stock_features"


def _load_materialized_view(
    mv_name: str,
    db_url: Optional[str] = None,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Load a materialized view from PostgreSQL.

    Parameters
    ----------
    mv_name : str
        Materialized view name (e.g. ``mv_expected_returns``).
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the materialized view.

    Returns
    -------
    pd.DataFrame
        Materialized view contents, or empty DataFrame on failure.
    """
    from sqlalchemy import create_engine

    url = db_url or os.environ.get("DB_URL")
    if not url:
        logger.warning("DB_URL not configured — cannot load %s.%s", schema, mv_name)
        return pd.DataFrame()

    try:
        engine = create_engine(url)
        query = f"SELECT * FROM {schema}.{mv_name}"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        logger.info(
            "Loaded %d rows × %d cols from %s.%s",
            len(df),
            len(df.columns),
            schema,
            mv_name,
        )
        return df
    except Exception as e:
        logger.warning("Failed to load %s.%s: %s", schema, mv_name, e)
        return pd.DataFrame()


def load_expected_returns_data(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Load expected returns feature data from ``mv_expected_returns``.

    v3.0 migration: replaces the 17-view merge + backfill approach with a
    single materialized view read. The MV includes all 17 feature categories
    (identifier, temporal, market data, valuation, analyst sentiment,
    price target dynamics, momentum, earnings, profitability, growth,
    quality & risk, composite scores, leverage & liquidity, cash flow,
    dividends, technical analysis, temporal features).

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the materialized view.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with identifier + all feature columns.
    """
    df = _load_materialized_view(_MV_EXPECTED_RETURNS, db_url=db_url, schema=schema)

    if df is not None and not df.empty:
        # Backfill computed columns (derived features not in the MV)
        df = backfill_feature_columns(df)

        # Validate feature coverage against expected categories
        validation = validate_feature_alignment(df, FEATURE_CATEGORIES)
        low_coverage = {k: v for k, v in validation.items() if v["coverage_pct"] < 50}
        if low_coverage:
            logger.warning(
                "Low feature coverage in %d categories: %s",
                len(low_coverage),
                {k: f"{v['coverage_pct']:.0f}%" for k, v in low_coverage.items()},
            )

        logger.info(
            "Loaded expected returns data: %d stocks × %d features",
            len(df),
            len(df.columns),
        )
    else:
        logger.warning("No data loaded from %s", _MV_EXPECTED_RETURNS)
        df = pd.DataFrame()
    return df


def load_all_stock_features(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Load the full stock features superset from ``mv_all_stock_features``.

    Used for screening, enrichment, and cross-category analytics that
    require columns beyond the expected returns feature set.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the materialized view.

    Returns
    -------
    pd.DataFrame
        Full feature DataFrame.
    """
    df = _load_materialized_view(_MV_ALL_STOCK_FEATURES, db_url=db_url, schema=schema)

    if df is not None and not df.empty:
        df = backfill_feature_columns(df)
        logger.info(
            "Loaded all stock features: %d stocks × %d features",
            len(df),
            len(df.columns),
        )
    else:
        logger.warning("No data loaded from %s", _MV_ALL_STOCK_FEATURES)
        df = pd.DataFrame()
    return df


def load_analytics_table(
    table_name: str,
    db_url: Optional[str] = None,
    schema: str = "analytics",
) -> pd.DataFrame:
    """
    Load a pre-computed analytics table from the ``analytics`` schema.

    Parameters
    ----------
    table_name : str
        Table name (e.g. ``monte_carlo_simulation``).
    db_url : str, optional
        Database URL. Falls back to DB_URL env var.
    schema : str, default "analytics"
        Schema containing the analytics tables.

    Returns
    -------
    pd.DataFrame
        Table contents, or empty DataFrame on failure.
    """
    from sqlalchemy import create_engine

    url = db_url or os.environ.get("DB_URL")
    if not url:
        logger.warning("DB_URL not configured — cannot load %s.%s", schema, table_name)
        return pd.DataFrame()

    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            df = pd.read_sql(f"SELECT * FROM {schema}.{table_name}", conn)
        logger.info("Loaded %d rows from %s.%s", len(df), schema, table_name)
        return df
    except Exception as e:
        logger.warning("Failed to load %s.%s: %s", schema, table_name, e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Runners
# ═══════════════════════════════════════════════════════════════════════════════

_MC_REQUIRED_COLS = [
    "price_target",
    "price_target_high",
    "price_target_low",
    "last_price",
]
_KALMAN_REQUIRED_COLS = ["last_price", "price_target"]

# Historical price columns (actual traded prices at prior points in time)
_HISTORICAL_PRICE_COLS = [
    "price_5d_ago",
    "price_1w_ago",
    "price_1m_ago",
    "price_3m_ago",
    "price_6m_ago",
    "price_1y_ago",
    "price_3y_ago",
    "price_5y_ago",
    "price_qtd_ago",
]

# Historical consensus price target columns (analyst targets at prior points in time)
_HISTORICAL_PRICE_TARGET_COLS = [
    "price_target_1w_ago",
    "price_target_1m_ago",
    "price_target_3m_ago",
    "price_target_6m_ago",
    "price_target_mtd_ago",
    "price_target_qtd_ago",
    "price_target_1y_ago",
]

# Historical high price target columns
_HISTORICAL_PRICE_TARGET_HIGH_COLS = [
    "price_target_high_1w_ago",
    "price_target_high_1m_ago",
    "price_target_high_6m_ago",
    "price_target_high_mtd_ago",
    "price_target_high_3m_ago",
    "price_target_high_qtd_ago",
    "price_target_high_1y_ago",
    "price_target_high_ytd_ago",
]

# Historical low price target columns
_HISTORICAL_PRICE_TARGET_LOW_COLS = [
    "price_target_low_1w_ago",
    "price_target_low_1m_ago",
    "price_target_low_3m_ago",
    "price_target_low_6m_ago",
    "price_target_low_mtd_ago",
    "price_target_low_qtd_ago",
    "price_target_low_ytd_ago",
    "price_target_low_1y_ago",
]

# Historical median price target columns
_HISTORICAL_PRICE_TARGET_MEDIAN_COLS = [
    "price_target_median_1w_ago",
    "price_target_median_1m_ago",
    "price_target_median_3m_ago",
    "price_target_median_6m_ago",
    "price_target_median_mtd_ago",
    "price_target_median_qtd_ago",
    "price_target_median_ytd_ago",
    "price_target_median_1y_ago",
]

# All historical columns combined (for validation / feature coverage checks)
ALL_HISTORICAL_PRICE_TARGET_COLS = (
    _HISTORICAL_PRICE_COLS
    + _HISTORICAL_PRICE_TARGET_COLS
    + _HISTORICAL_PRICE_TARGET_HIGH_COLS
    + _HISTORICAL_PRICE_TARGET_LOW_COLS
    + _HISTORICAL_PRICE_TARGET_MEDIAN_COLS
)


def _resolve_available_historical_cols(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Identify which historical price/target columns are present in the DataFrame.

    Returns a dict keyed by category name with lists of available column names.
    """
    return {
        "historical_prices": [c for c in _HISTORICAL_PRICE_COLS if c in df.columns],
        "historical_targets": [
            c for c in _HISTORICAL_PRICE_TARGET_COLS if c in df.columns
        ],
        "historical_targets_high": [
            c for c in _HISTORICAL_PRICE_TARGET_HIGH_COLS if c in df.columns
        ],
        "historical_targets_low": [
            c for c in _HISTORICAL_PRICE_TARGET_LOW_COLS if c in df.columns
        ],
        "historical_targets_median": [
            c for c in _HISTORICAL_PRICE_TARGET_MEDIAN_COLS if c in df.columns
        ],
    }


def _log_historical_coverage(available: dict[str, list[str]]) -> None:
    """Log how many historical price/target columns were found."""
    total_found = sum(len(v) for v in available.values())
    total_possible = len(ALL_HISTORICAL_PRICE_TARGET_COLS)
    logger.info(
        "Historical price/target coverage: %d / %d columns available (%s)",
        total_found,
        total_possible,
        ", ".join(f"{k}={len(v)}" for k, v in available.items()),
    )


def run_monte_carlo_analysis(
    df: pd.DataFrame,
    n_simulations: int = 25_000,
    max_stocks: int = 10_000,
    use_historical_targets: bool = True,
) -> pd.DataFrame:
    """
    Run Monte Carlo price target simulation on the feature DataFrame.

    v3.0: Increased default n_simulations to 25,000 for tighter confidence
    intervals on the triangular distribution sampling.

    v3.1: When ``use_historical_targets=True`` and historical price target
    columns are present, the simulation is enriched with:
    - Historical price target drift (consensus movement over time)
    - Historical price target spread evolution (high/low band changes)
    - Historical median target convergence signals

    These are passed as auxiliary columns so the downstream
    ``monte_carlo_price_target_simulation`` can optionally use them
    for informed drift and volatility priors.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with price target columns.
    n_simulations : int
        Number of triangular distribution samples per stock.
    max_stocks : int
        Cap on number of stocks to simulate.
    use_historical_targets : bool, default True
        Whether to compute and attach historical target drift columns
        to the simulation input.

    Returns
    -------
    pd.DataFrame
        Monte Carlo results with ``expected_upside_pct``, ``var_5_pct``,
        ``prob_positive_upside``, ``risk_reward_ratio``, etc.
        When historical targets are used, also includes
        ``pt_drift_1m``, ``pt_drift_3m``, ``pt_spread_change_1m``,
        ``historical_price_anchor``, and ``pt_median_drift_1m``.
    """
    missing = [c for c in _MC_REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning("MC simulation skipped — missing columns: %s", missing)
        return pd.DataFrame()

    sim_df = df.copy()

    # Enrich with historical target drift metrics when columns are available
    hist_available = _resolve_available_historical_cols(sim_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        sim_df = _enrich_with_historical_target_drift(sim_df, hist_available)

    mc = monte_carlo_price_target_simulation(
        sim_df,
        n_simulations=n_simulations,
        max_stocks=max_stocks,
    )
    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
) -> pd.DataFrame:
    """
    Estimate probability of reaching consensus price targets.

    Uses ``PriceTargetAchievementModel`` from probability_analytics.

    v3.1: When ``use_historical_targets=True``, historical price target
    columns are used to compute target drift and spread evolution,
    which refine the achievement probability via momentum-adjusted
    base probabilities and analyst conviction signals.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with price target and analyst sentiment columns.
    use_historical_targets : bool, default True
        Whether to enrich input with historical target drift columns.

    Returns
    -------
    pd.DataFrame
        Price target achievement results with ``achievement_probability``,
        ``expected_return_prob_weighted``, ``confidence_level``, etc.
        When historical targets are used, also includes
        ``pt_drift_1m``, ``pt_drift_3m``, ``pt_spread_change_1m``,
        ``historical_price_anchor``, and ``pt_median_drift_1m``.
    """
    pt_df = df.copy()

    hist_available = _resolve_available_historical_cols(pt_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        pt_df = _enrich_with_historical_target_drift(pt_df, hist_available)

    model = PriceTargetAchievementModel()
    pt = model.analyze_dataframe(pt_df)
    logger.info("Price target achievement: %d stocks processed", len(pt))
    return pt


def run_kalman_filter(
    df: pd.DataFrame,
    use_historical_targets: bool = True,
) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth noisy analyst price targets.

    Delegates to ``statistical_analysis.kalman_filter_price_target``.

    v3.1: When ``use_historical_targets=True``, historical price and
    price target columns are used to initialise the Kalman state with
    a more informed prior (anchored to recent historical price levels
    and target drift trajectories), reducing filter warm-up artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with ``last_price`` and ``price_target``.
    use_historical_targets : bool, default True
        Whether to enrich input with historical target drift columns.

    Returns
    -------
    pd.DataFrame
        Kalman-filtered results with ``filtered_upside``,
        ``kalman_estimate``, ``kalman_variance``, etc.
        When historical targets are used, also includes
        ``pt_drift_1m``, ``pt_drift_3m``, ``pt_spread_change_1m``,
        ``historical_price_anchor``, and ``pt_median_drift_1m``.
    """
    missing = [c for c in _KALMAN_REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning("Kalman filter skipped — missing columns: %s", missing)
        return pd.DataFrame()

    kal_df = df.copy()

    hist_available = _resolve_available_historical_cols(kal_df)
    if use_historical_targets:
        _log_historical_coverage(hist_available)
        kal_df = _enrich_with_historical_target_drift(kal_df, hist_available)

    kal = kalman_filter_price_target(kal_df)
    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


# ═══════════════════════════════════════════════════════════════════════════════
# Historical Price Target Drift Enrichment
# ═══════════════════════════════════════════════════════════════════════════════


def _enrich_with_historical_target_drift(
    df: pd.DataFrame,
    hist_available: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Compute derived drift and spread-evolution columns from historical
    price and price target data.

    Adds the following columns when the requisite inputs exist:

    - ``pt_drift_1m``   — % change in consensus target vs 1 month ago
    - ``pt_drift_3m``   — % change in consensus target vs 3 months ago
    - ``pt_drift_6m``   — % change in consensus target vs 6 months ago
    - ``pt_drift_1y``   — % change in consensus target vs 1 year ago
    - ``pt_spread_change_1m``  — change in (high − low) target spread vs 1m ago
    - ``pt_spread_change_3m``  — change in (high − low) target spread vs 3m ago
    - ``historical_price_anchor`` — best available recent historical price
                                    (5d → 1w → 1m fallback chain)
    - ``pt_median_drift_1m``  — % change in median target vs 1 month ago
    - ``pt_median_drift_3m``  — % change in median target vs 3 months ago
    - ``price_vs_historical_1m`` — % change in last_price vs price_1m_ago
    - ``price_vs_historical_3m`` — % change in last_price vs price_3m_ago
    - ``target_vs_price_convergence_1m`` — whether target drift and price
      movement are converging (positive) or diverging (negative)

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (mutated in place for performance; caller
        should pass a ``.copy()`` if the original must be preserved).
    hist_available : dict[str, list[str]]
        Output of ``_resolve_available_historical_cols``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional derived columns appended.
    """
    targets = hist_available["historical_targets"]
    targets_high = hist_available["historical_targets_high"]
    targets_low = hist_available["historical_targets_low"]
    targets_median = hist_available["historical_targets_median"]
    prices = hist_available["historical_prices"]

    if (
        not targets
        and not prices
        and not targets_high
        and not targets_low
        and not targets_median
    ):
        logger.debug(
            "No historical price/target columns found — skipping drift enrichment"
        )
        return df

    # --- Consensus target drift ---
    current_target = df.get("price_target")
    if current_target is not None:
        for horizon, col in [
            ("1m", "price_target_1m_ago"),
            ("3m", "price_target_3m_ago"),
            ("6m", "price_target_6m_ago"),
            ("1y", "price_target_1y_ago"),
        ]:
            if col in df.columns:
                prev = pd.to_numeric(df[col], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    drift = ((current_target - prev) / prev.abs()) * 100.0
                df[f"pt_drift_{horizon}"] = drift.replace([np.inf, -np.inf], np.nan)

    # --- Spread evolution (high − low band width change) ---
    current_high = df.get("price_target_high")
    current_low = df.get("price_target_low")
    if current_high is not None and current_low is not None:
        current_spread = pd.to_numeric(current_high, errors="coerce") - pd.to_numeric(
            current_low, errors="coerce"
        )
        for horizon, high_col, low_col in [
            ("1m", "price_target_high_1m_ago", "price_target_low_1m_ago"),
            ("3m", "price_target_high_3m_ago", "price_target_low_3m_ago"),
        ]:
            if high_col in df.columns and low_col in df.columns:
                prev_spread = pd.to_numeric(
                    df[high_col], errors="coerce"
                ) - pd.to_numeric(df[low_col], errors="coerce")
                df[f"pt_spread_change_{horizon}"] = current_spread - prev_spread

    # --- Median target drift ---
    current_median = df.get("price_target_median")
    if current_median is not None:
        for horizon, col in [
            ("1m", "price_target_median_1m_ago"),
            ("3m", "price_target_median_3m_ago"),
        ]:
            if col in df.columns:
                prev = pd.to_numeric(df[col], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    drift = ((current_median - prev) / prev.abs()) * 100.0
                df[f"pt_median_drift_{horizon}"] = drift.replace(
                    [np.inf, -np.inf], np.nan
                )

    # --- Historical price anchor (best-available recent price) ---
    anchor_chain = ["price_5d_ago", "price_1w_ago", "price_1m_ago"]
    anchor = pd.Series(np.nan, index=df.index, dtype=float)
    for col in anchor_chain:
        if col in df.columns:
            fill_mask = anchor.isna()
            if fill_mask.any():
                anchor = anchor.fillna(pd.to_numeric(df[col], errors="coerce"))
    if anchor.notna().any():
        df["historical_price_anchor"] = anchor

    # --- Price momentum vs historical levels ---
    last_price = df.get("last_price")
    if last_price is not None:
        last_price_num = pd.to_numeric(last_price, errors="coerce")
        for horizon, col in [("1m", "price_1m_ago"), ("3m", "price_3m_ago")]:
            if col in df.columns:
                prev = pd.to_numeric(df[col], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_change = ((last_price_num - prev) / prev.abs()) * 100.0
                df[f"price_vs_historical_{horizon}"] = pct_change.replace(
                    [np.inf, -np.inf], np.nan
                )

    # --- Target-vs-price convergence signal ---
    if "pt_drift_1m" in df.columns and "price_vs_historical_1m" in df.columns:
        # Positive = target rising faster than price (expanding upside)
        # Negative = price rising faster than target (converging/narrowing upside)
        df["target_vs_price_convergence_1m"] = (
            df["pt_drift_1m"] - df["price_vs_historical_1m"]
        )

    n_derived = sum(
        1
        for c in df.columns
        if c.startswith(
            (
                "pt_drift_",
                "pt_spread_change_",
                "pt_median_drift_",
                "historical_price_anchor",
                "price_vs_historical_",
                "target_vs_price_convergence_",
            )
        )
    )
    logger.info(
        "Historical target drift enrichment: %d derived columns added", n_derived
    )

    return df


def run_earnings_beat_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run enhanced three-layer Bayesian earnings beat probability model.

    Uses ``EarningsBeatProbabilityModel.analyze_dataframe_enhanced()``
    which fuses historical EPS, revision momentum, and GAAP quality layers.
    Enriches results with EPS streak analysis via ``EPSStreakAnalyzer``,
    resampled technical priors via ``ResampledBeatProbabilityModel``,
    and classical Bayesian beat estimates via ``bayesian_earnings_beat_model``.
    """
    model = EarningsBeatProbabilityModel()
    sector_col = "sector" if "sector" in df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(df, sector_col=sector_col)
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))

    # --- EPS streak analysis (Markov-chain continuation probabilities) ---
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        if not streak_df.empty and "ticker" in streak_df.columns:
            streak_cols = [c for c in streak_df.columns if c != "ticker" and c not in beat.columns]
            if streak_cols:
                beat = beat.merge(
                    streak_df[["ticker"] + streak_cols],
                    on="ticker",
                    how="left",
                )
                logger.info("EPS streak enrichment: %d columns added", len(streak_cols))
    except Exception as e:
        logger.warning("EPS streak analysis failed: %s", e)

    # --- Resampled technical priors ---
    try:
        resampled_model = ResampledBeatProbabilityModel(base_model=model)
        resampled_df = resampled_model.analyze_dataframe(df)
        if not resampled_df.empty and "ticker" in resampled_df.columns:
            resamp_cols = [c for c in resampled_df.columns if c != "ticker" and c not in beat.columns]
            if resamp_cols:
                beat = beat.merge(
                    resampled_df[["ticker"] + resamp_cols],
                    on="ticker",
                    how="left",
                )
                logger.info("Resampled beat enrichment: %d columns added", len(resamp_cols))
    except Exception as e:
        logger.warning("Resampled beat probability failed: %s", e)

    # --- Classical Bayesian earnings beat model ---
    try:
        bayesian_beat = bayesian_earnings_beat_model(df)
        if not bayesian_beat.empty and "ticker" in bayesian_beat.columns:
            bay_cols = [c for c in bayesian_beat.columns if c != "ticker" and c not in beat.columns]
            if bay_cols:
                beat = beat.merge(
                    bayesian_beat[["ticker"] + bay_cols],
                    on="ticker",
                    how="left",
                )
                logger.info("Bayesian beat enrichment: %d columns added", len(bay_cols))
    except Exception as e:
        logger.warning("Bayesian earnings beat model failed: %s", e)

    return beat


def run_credit_risk_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run credit risk and ruin probability analysis.

    Uses ``CreditRiskProbabilityModel`` for Bayesian distress estimation,
    ``fast_ruin_probability`` for Monte Carlo ruin simulation,
    ``calculate_ruin_probability`` for analytical ruin estimates,
    and ``detect_accounting_anomalies`` for Beneish-style red flags.
    """
    credit_model = CreditRiskProbabilityModel()
    credit = credit_model.analyze_dataframe(df)

    ruin = fast_ruin_probability(df, n_simulations=2000, n_days=252)
    if not ruin.empty and not credit.empty and "ticker" in ruin.columns:
        credit = credit.merge(
            ruin[["ticker", "ruin_probability", "survival_probability", "risk_tier"]],
            on="ticker",
            how="left",
            suffixes=("", "_mc"),
        )

    # --- Analytical ruin probability (Gambler's Ruin) ---
    try:
        ruin_analytical = calculate_ruin_probability(df)
        if not ruin_analytical.empty and "ticker" in ruin_analytical.columns:
            ruin_cols = [c for c in ruin_analytical.columns if c != "ticker" and c not in credit.columns]
            if ruin_cols:
                credit = credit.merge(
                    ruin_analytical[["ticker"] + ruin_cols],
                    on="ticker",
                    how="left",
                )
                logger.info("Analytical ruin enrichment: %d columns added", len(ruin_cols))
    except Exception as e:
        logger.warning("Analytical ruin probability failed: %s", e)

    # --- Accounting anomaly detection (Beneish M-Score style) ---
    try:
        anomalies = detect_accounting_anomalies(df)
        if not anomalies.empty and "ticker" in anomalies.columns:
            anom_cols = [c for c in anomalies.columns if c != "ticker" and c not in credit.columns]
            if anom_cols:
                credit = credit.merge(
                    anomalies[["ticker"] + anom_cols],
                    on="ticker",
                    how="left",
                )
                logger.info("Accounting anomaly enrichment: %d columns added", len(anom_cols))
    except Exception as e:
        logger.warning("Accounting anomaly detection failed: %s", e)

    logger.info("Credit risk analysis: %d stocks processed", len(credit))
    return credit


def run_dividend_safety_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run dividend cut probability analysis.

    Uses ``DividendCutProbabilityModel`` to estimate probability of
    dividend reduction based on FCF coverage, payout ratio, and streak.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with dividend columns.

    Returns
    -------
    pd.DataFrame
        Dividend safety results with ``dividend_cut_probability``,
        ``safety_score``, ``risk_category``.
    """
    model = DividendCutProbabilityModel()
    div_safety = model.analyze_dataframe(df)
    logger.info("Dividend safety analysis: %d stocks processed", len(div_safety))
    return div_safety


def run_category_probability_analysis(
    df: pd.DataFrame,
    categories: Optional[dict[str, list[str]]] = None,
) -> dict[str, dict]:
    """
    Run per-category Bayesian probability analytics.

    Computes Bayesian posterior estimation, distribution fitting,
    and conditional probability analysis for each feature category.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    categories : dict, optional
        Feature categories to analyze. Defaults to FEATURE_CATEGORIES.

    Returns
    -------
    dict[str, dict]
        Per-category analytics results.
    """
    cats = categories or FEATURE_CATEGORIES
    results = {}

    for cat_name, features in cats.items():
        available = [f for f in features if f in df.columns]
        if len(available) < 2:
            continue

        try:
            cat_results = run_category_probability_analytics(
                df,
                cat_name,
                available,
                n_simulations=10_000,
            )

            # --- CategoryProbabilityAnalyzer: Bayesian view-level analysis ---
            try:
                analyzer = CategoryProbabilityAnalyzer(category_name=cat_name)
                view_result = analyzer.analyze_view(df, feature_cols=available)
                if view_result is not None:
                    cat_results["category_probability_analysis"] = view_result
            except Exception as e:
                logger.debug("CategoryProbabilityAnalyzer skipped for %s: %s", cat_name, e)

            # --- Distribution fitting per category ---
            try:
                dist_results = fit_distributions_by_category(
                    df, cat_name, available,
                )
                if dist_results:
                    cat_results["distribution_fits"] = dist_results
            except Exception as e:
                logger.debug("Distribution fitting skipped for %s: %s", cat_name, e)

            # --- Conditional probability analysis ---
            try:
                cond_probs = calculate_conditional_probabilities(
                    df, {cat_name: available},
                )
                if cond_probs is not None and not (
                    isinstance(cond_probs, pd.DataFrame) and cond_probs.empty
                ):
                    cat_results["conditional_probabilities"] = cond_probs
            except Exception as e:
                logger.debug("Conditional probabilities skipped for %s: %s", cat_name, e)

            results[cat_name] = cat_results
            logger.info(
                "Category analytics: %s — %d features analyzed",
                cat_name,
                cat_results.get("features_analyzed", 0),
            )
        except Exception as e:
            logger.warning("Category analytics failed for %s: %s", cat_name, e)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. Screening Runners (v3.0 — NEW)
# ═══════════════════════════════════════════════════════════════════════════════


def run_stock_screening(
    df_all: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Run all stock screening strategies on the full feature set.

    Uses ``mv_all_stock_features`` as input (broader feature coverage
    than ``mv_expected_returns``).

    Parameters
    ----------
    df_all : pd.DataFrame
        Full feature DataFrame from ``mv_all_stock_features``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Screening results keyed by strategy name.
    """
    screens: dict[str, pd.DataFrame] = {}

    # Quality screening
    try:
        screens["quality"] = create_enhanced_screener(
            df_all,
            min_fscore=6,
            min_quality_momentum=40,
            min_fcf_positive_years=3,
        )
        logger.info("Quality screen: %d stocks", len(screens["quality"]))
    except Exception as e:
        logger.warning("Quality screening failed: %s", e)

    # Earnings quality
    try:
        screens["earnings_quality"] = screen_earnings_quality(
            df_all,
            min_quality_score=60,
            min_positive_years=3,
        )
        logger.info(
            "Earnings quality screen: %d stocks", len(screens["earnings_quality"])
        )
    except Exception as e:
        logger.warning("Earnings quality screening failed: %s", e)

    # Value opportunities
    try:
        screens["value"] = screen_value_opportunities(
            df_all,
            max_pe_ratio=25,
            min_upside_potential=20,
        )
        logger.info("Value screen: %d stocks", len(screens["value"]))
    except Exception as e:
        logger.warning("Value screening failed: %s", e)

    # Growth momentum
    try:
        screens["growth"] = screen_growth_momentum(
            df_all,
            min_revenue_growth=10,
            min_eps_growth=10,
        )
        logger.info("Growth screen: %d stocks", len(screens["growth"]))
    except Exception as e:
        logger.warning("Growth screening failed: %s", e)

    # GARP (Growth at a Reasonable Price)
    try:
        screens["garp"] = screen_garp_opportunities(
            df_all,
            max_peg_ratio=1.5,
            min_eps_growth=10,
        )
        logger.info("GARP screen: %d stocks", len(screens["garp"]))
    except Exception as e:
        logger.warning("GARP screening failed: %s", e)

    # Dividend quality
    try:
        screens["dividend"] = screen_dividend_quality(
            df_all,
            min_dividend_yield=2.0,
            min_dividend_streak=3,
        )
        logger.info("Dividend screen: %d stocks", len(screens["dividend"]))
    except Exception as e:
        logger.warning("Dividend screening failed: %s", e)

    # Financial health
    try:
        screens["healthy"] = screen_financial_health(
            df_all,
            min_distress_score=70,
            min_current_ratio=1.2,
        )
        logger.info("Financial health screen: %d stocks", len(screens["healthy"]))
    except Exception as e:
        logger.warning("Financial health screening failed: %s", e)

    # Valuation reversion candidates
    try:
        screens["valuation_reversion"] = screen_valuation_reversion_candidates(
            df_all,
            min_discount_pct=20.0,
            min_quality_score=50.0,
        )
        logger.info(
            "Valuation reversion screen: %d stocks", len(screens["valuation_reversion"])
        )
    except Exception as e:
        logger.warning("Valuation reversion screening failed: %s", e)

    # Integrity-filtered growth
    try:
        screens["integrity_growth"] = screen_integrity_filtered_growth(
            df_all,
            min_revenue_growth=15.0,
            min_accounting_quality=60.0,
        )
        logger.info(
            "Integrity growth screen: %d stocks", len(screens["integrity_growth"])
        )
    except Exception as e:
        logger.warning("Integrity growth screening failed: %s", e)

    # High-yield safe dividends
    try:
        screens["high_yield_safe"] = screen_high_yield_safe_dividends(
            df_all,
            min_yield=3.0,
            max_payout=70.0,
            min_distress_score=70.0,
        )
        logger.info(
            "High-yield safe dividend screen: %d stocks",
            len(screens["high_yield_safe"]),
        )
    except Exception as e:
        logger.warning("High-yield safe dividend screening failed: %s", e)

    # Sector-relative ranking (composite score)
    try:
        screens["sector_relative"] = create_sector_relative_ranking(
            df_all,
            metric="composite_score" if "composite_score" in df_all.columns else "upside_potential",
        )
        logger.info(
            "Sector-relative ranking: %d stocks",
            len(screens["sector_relative"]),
        )
    except Exception as e:
        logger.warning("Sector-relative ranking failed: %s", e)

    return screens


def filter_quality_stocks(
    summary: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply quality screening to the expected returns summary.

    Enriches the summary with a ``quality_tier`` from composite scoring
    and flags financially healthy stocks.
    """
    if summary.empty or source_df.empty:
        return summary

    ranked = rank_stocks_by_composite_score(source_df)
    if "composite_score" in ranked.columns and "ticker" in ranked.columns:
        score_map = ranked.set_index("ticker")["composite_score"]
        summary["composite_score"] = summary["ticker"].map(score_map)

        summary["quality_tier"] = pd.cut(
            summary["composite_score"],
            bins=[0, 30, 50, 70, 100],
            labels=["Low", "Below Avg", "Above Avg", "High"],
        )
        logger.info(
            "Quality scoring: %d High, %d Above Avg",
            (summary["quality_tier"] == "High").sum(),
            (summary["quality_tier"] == "Above Avg").sum(),
        )

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tri-Model & Quad-Model Alignment
# ═══════════════════════════════════════════════════════════════════════════════

_SIGNAL_LABELS = {
    0: "Strong Bearish (0/3)",
    1: "Bearish (1/3)",
    2: "Bullish (2/3)",
    3: "Strong Bullish (3/3)",
}

_SIGNAL_LABELS_4 = {
    0: "Strong Bearish (0/4)",
    1: "Bearish (1/4)",
    2: "Neutral (2/4)",
    3: "Bullish (3/4)",
    4: "Strong Bullish (4/4)",
}


def build_tri_model_alignment(
    mc: pd.DataFrame,
    kal: pd.DataFrame,
    pt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Monte Carlo, Kalman, and Price Target Achievement into a
    tri-model alignment DataFrame with direction agreement scores.
    """
    if mc.empty or kal.empty or pt.empty:
        logger.warning("Tri-model alignment skipped — one or more inputs empty")
        return pd.DataFrame()

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]
    mc_select = list(
        set(
            mc_id_cols
            + [
                "ticker",
                "expected_upside_pct",
                "price_target_mc",
                "prob_positive_upside",
                "var_5_pct",
                "risk_reward_ratio",
            ]
        )
    )

    tri = (
        mc[mc_select]
        .copy()
        .merge(
            kal[["ticker", "filtered_upside", "kalman_estimate", "kalman_variance"]],
            on="ticker",
            how="inner",
        )
        .merge(
            pt[
                [
                    "ticker",
                    "expected_return_prob_weighted",
                    "achievement_probability",
                    "price_target_prob_weighted",
                    "confidence_level",
                    "analyst_conviction",
                    "eps_revision_momentum",
                    "analyst_rating_normalized",
                ]
            ],
            on="ticker",
            how="inner",
        )
    )

    tri["mc_bullish"] = tri["expected_upside_pct"] > 0
    tri["kal_bullish"] = tri["filtered_upside"] > 0
    tri["pt_bullish"] = tri["expected_return_prob_weighted"] > 0
    tri["agreement_score"] = (
        tri["mc_bullish"].astype(int)
        + tri["kal_bullish"].astype(int)
        + tri["pt_bullish"].astype(int)
    )
    tri["signal"] = tri["agreement_score"].map(_SIGNAL_LABELS)

    logger.info(
        "Tri-model alignment: %d stocks, %d strong bullish",
        len(tri),
        (tri["agreement_score"] == 3).sum(),
    )
    return tri


def build_quad_model_alignment(
    tri: pd.DataFrame,
    beat: pd.DataFrame,
    beat_threshold: float = 0.6,
) -> pd.DataFrame:
    """Extend tri-model alignment with earnings beat probability for 4-model scoring."""
    if tri.empty or beat.empty:
        logger.warning("Quad-model alignment skipped — insufficient data")
        return pd.DataFrame()

    if "posterior_beat_prob" not in beat.columns:
        logger.warning("Quad-model skipped — beat results missing posterior_beat_prob")
        return pd.DataFrame()

    beat_slim = beat[["ticker", "posterior_beat_prob"]].rename(
        columns={"posterior_beat_prob": "beat_prob"}
    )
    quad = tri.merge(beat_slim, on="ticker", how="inner")
    if quad.empty:
        return quad

    quad["beat_bullish"] = (quad["beat_prob"] >= beat_threshold).astype(int)
    quad["quad_agreement"] = (
        quad["mc_bullish"].astype(int)
        + quad["kal_bullish"].astype(int)
        + quad["pt_bullish"].astype(int)
        + quad["beat_bullish"]
    )

    logger.info(
        "Quad-model alignment: %d stocks, full consensus (4/4): %d",
        len(quad),
        (quad["quad_agreement"] == 4).sum(),
    )
    return quad


def build_expected_returns_summary(
    mc: pd.DataFrame,
    kal: pd.DataFrame,
    pt: pd.DataFrame,
    earn: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge four expected-return model results into a unified summary DataFrame.

    v3.0: ``source_df`` is loaded from ``mv_all_stock_features`` (full
    superset) so that all identifier and market-data columns are available
    for enrichment without needing a backfill step.
    """
    if mc.empty or kal.empty or pt.empty or earn.empty:
        logger.warning(
            "Expected returns summary skipped — one or more inputs empty "
            "(mc=%d, kal=%d, pt=%d, earn=%d)",
            len(mc),
            len(kal),
            len(pt),
            len(earn),
        )
        return pd.DataFrame()

    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]

    market_data_cols = [
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "price_target_high",
        "price_target_low",
        "price_target_median",
        "volume_shrs",
        "shares_outstanding",
    ]
    available_market = [c for c in market_data_cols if c in mc.columns]

    mc_select = list(
        set(
            mc_id_cols
            + [
                "ticker",
                "expected_upside_pct",
                "price_target_mc",
                "prob_positive_upside",
                "var_5_pct",
                "risk_reward_ratio",
            ]
            + available_market
        )
    )

    summary = (
        mc[mc_select]
        .copy()
        .merge(
            kal[["ticker", "filtered_upside", "kalman_estimate"]],
            on="ticker",
            how="inner",
        )
        .merge(
            pt[
                [
                    "ticker",
                    "expected_return_prob_weighted",
                    "price_target_prob_weighted",
                    "achievement_probability",
                    "confidence_level",
                    "analyst_conviction",
                    "eps_revision_momentum",
                    "analyst_rating_normalized",
                ]
            ],
            on="ticker",
            how="inner",
        )
        .merge(
            earn[
                [
                    "ticker",
                    "posterior_beat_prob",
                    "confidence_score",
                    "beat_classification",
                ]
            ],
            on="ticker",
            how="inner",
        )
    )

    if summary.empty:
        logger.warning(
            "Expected returns summary: no overlapping tickers across all 4 models"
        )
        return summary

    # Enrich from source_df (mv_all_stock_features)
    if source_df is not None and "ticker" in source_df.columns:
        id_cols_ordered = load_identifier_columns()
        desired_cols = id_cols_ordered + market_data_cols
        missing_cols = [
            c
            for c in desired_cols
            if c in source_df.columns and c not in summary.columns
        ]
        if missing_cols:
            source_subset = source_df[["ticker"] + missing_cols].drop_duplicates(
                subset="ticker"
            )
            summary = summary.merge(source_subset, on="ticker", how="left")
            logger.info(
                "Enriched expected_returns_summary with %d columns from mv_all_stock_features",
                len(missing_cols),
            )

    # Direction flags
    summary["mc_bullish"] = summary["expected_upside_pct"] > 0
    summary["kal_bullish"] = summary["filtered_upside"] > 0
    summary["pt_bullish"] = summary["expected_return_prob_weighted"] > 0
    summary["earn_bullish"] = summary["posterior_beat_prob"] > 0.5

    # Agreement score: 0–4
    summary["agreement_score"] = (
        summary["mc_bullish"].astype(int)
        + summary["kal_bullish"].astype(int)
        + summary["pt_bullish"].astype(int)
        + summary["earn_bullish"].astype(int)
    )
    summary["signal"] = summary["agreement_score"].map(_SIGNAL_LABELS_4)

    # Confidence-weighted agreement (continuous 0–4 scale)
    mc_weight = summary["prob_positive_upside"].clip(0, 100) / 100.0
    kal_weight = 0.5
    pt_weight = (
        summary["confidence_level"]
        .map({"High": 0.9, "Medium": 0.6, "Low": 0.3})
        .fillna(0.5)
    )
    earn_weight = summary["confidence_score"].clip(0, 1)

    summary["weighted_agreement"] = (
        summary["mc_bullish"].astype(float) * mc_weight
        + summary["kal_bullish"].astype(float) * kal_weight
        + summary["pt_bullish"].astype(float) * pt_weight
        + summary["earn_bullish"].astype(float) * earn_weight
    )

    logger.info(
        "Expected returns summary: %d stocks, %d strong bullish (4/4)",
        len(summary),
        (summary["agreement_score"] == 4).sum(),
    )
    return summary


def extract_strong_consensus(
    tri: pd.DataFrame,
    min_prob_positive: float = 55.0,
    min_achievement: float = 0.6,
    top_n: int = 50,
) -> pd.DataFrame:
    """Filter strong consensus picks — all 3 models bullish with high confidence."""
    if tri.empty:
        return pd.DataFrame()

    strong = tri[
        (tri["agreement_score"] == 3)
        & (tri["prob_positive_upside"] >= min_prob_positive)
        & (tri["achievement_probability"] >= min_achievement)
    ].nlargest(top_n, "expected_upside_pct")

    logger.info("Strong consensus picks: %d stocks", len(strong))
    return strong


def compute_price_target_prob_weighted(
    pt: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "expected_return_prob_weighted",
    output_col: str = "price_target_prob_weighted",
) -> pd.DataFrame:
    """
    Calculate the expected price target from the probability-weighted return.

    ``price_target_prob_weighted = last_price * (1 + expected_return_prob_weighted / 100)``
    """
    if pt.empty:
        logger.warning("compute_price_target_prob_weighted: empty input — skipping")
        return pt

    result = pt.copy()

    if price_col not in result.columns:
        if "ticker" not in source_df.columns or price_col not in source_df.columns:
            logger.warning(
                "Cannot compute %s — '%s' or 'ticker' missing from source_df",
                output_col,
                price_col,
            )
            result[output_col] = np.nan
            return result

        price_map = (
            source_df[["ticker", price_col]]
            .drop_duplicates(subset="ticker")
            .set_index("ticker")[price_col]
        )
        result[price_col] = result["ticker"].map(price_map)

    with np.errstate(invalid="ignore"):
        result[output_col] = result[price_col] * (1 + result[return_col] / 100.0)

    result[output_col] = result[output_col].replace([np.inf, -np.inf], np.nan)

    valid_count = result[output_col].notna().sum()
    logger.info(
        "Computed %s for %d / %d stocks (mean=%.2f)",
        output_col,
        valid_count,
        len(result),
        result[output_col].mean() if valid_count > 0 else 0.0,
    )
    return result


def compute_price_target_mc(
    pt: pd.DataFrame,
    source_df: pd.DataFrame,
    price_col: str = "last_price",
    return_col: str = "expected_upside_pct",
    output_col: str = "price_target_mc",
) -> pd.DataFrame:
    """
    Calculate the expected price target from the monte carlo simulation.

    ``price_target_mc = last_price * (1 + expected_upside_pct / 100)``
    """
    if pt.empty:
        logger.warning("compute_price_target_mc: empty input — skipping")
        return pt

    result = pt.copy()

    if price_col not in result.columns:
        if "ticker" not in source_df.columns or price_col not in source_df.columns:
            logger.warning(
                "Cannot compute %s — '%s' or 'ticker' missing from source_df",
                output_col,
                price_col,
            )
            result[output_col] = np.nan
            return result

        price_map = (
            source_df[["ticker", price_col]]
            .drop_duplicates(subset="ticker")
            .set_index("ticker")[price_col]
        )
        result[price_col] = result["ticker"].map(price_map)

    with np.errstate(invalid="ignore"):
        result[output_col] = result[price_col] * (1 + result[return_col] / 100.0)

    result[output_col] = result[output_col].replace([np.inf, -np.inf], np.nan)

    valid_count = result[output_col].notna().sum()
    logger.info(
        "Computed %s for %d / %d stocks (mean=%.2f)",
        output_col,
        valid_count,
        len(result),
        result[output_col].mean() if valid_count > 0 else 0.0,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytical Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sector_expected_returns(tri: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expected return metrics by industry sector across all models."""
    if tri.empty or "industry" not in tri.columns:
        return pd.DataFrame()

    return (
        tri.groupby("industry")
        .agg(
            mc_mean=("expected_upside_pct", "mean"),
            mc_median=("expected_upside_pct", "median"),
            kalman_mean=("filtered_upside", "mean"),
            kalman_median=("filtered_upside", "median"),
            pt_mean=("expected_return_prob_weighted", "mean"),
            pt_median=("expected_return_prob_weighted", "median"),
            pct_bullish=("agreement_score", lambda x: (x == 3).mean() * 100),
            count=("ticker", "count"),
        )
        .reset_index()
        .sort_values("mc_mean", ascending=False)
    )


def compute_sector_return_analytics(
    summary: pd.DataFrame,
    group_col: str = "industry",
) -> pd.DataFrame:
    """
    Extended sector-level analytics with confidence intervals,
    distribution shape, and hit-rate diagnostics.
    """
    if summary.empty or group_col not in summary.columns:
        return pd.DataFrame()

    results = []
    for sector, group in summary.groupby(group_col):
        n = len(group)
        row = {"sector": sector, "count": n}

        for col, prefix in [
            ("expected_upside_pct", "mc"),
            ("filtered_upside", "kalman"),
            ("expected_return_prob_weighted", "pt"),
        ]:
            if col in group.columns:
                s = group[col].dropna()
                row[f"{prefix}_mean"] = float(s.mean()) if len(s) > 0 else None
                row[f"{prefix}_median"] = float(s.median()) if len(s) > 0 else None
                row[f"{prefix}_std"] = float(s.std()) if len(s) > 1 else None
                if len(s) > 1:
                    se = s.std() / np.sqrt(len(s))
                    row[f"{prefix}_ci_low"] = float(s.mean() - 1.96 * se)
                    row[f"{prefix}_ci_high"] = float(s.mean() + 1.96 * se)
                if len(s) > 3:
                    row[f"{prefix}_skew"] = float(s.skew())
                    row[f"{prefix}_kurtosis"] = float(s.kurtosis())

        if "agreement_score" in group.columns:
            row["pct_bullish_3plus"] = float(
                (group["agreement_score"] >= 3).mean() * 100
            )
            row["pct_full_consensus"] = float(
                (group["agreement_score"] == 4).mean() * 100
            )
        if "weighted_agreement" in group.columns:
            row["mean_weighted_agreement"] = float(group["weighted_agreement"].mean())

        if "expected_upside_pct" in group.columns:
            mc_mean = group["expected_upside_pct"].mean()
            mc_std = group["expected_upside_pct"].std()
            row["risk_adjusted_return"] = (
                float(mc_mean / mc_std) if mc_std > 0 else None
            )

        if "posterior_beat_prob" in group.columns:
            row["mean_beat_prob"] = float(group["posterior_beat_prob"].mean())

        results.append(row)

    return (
        pd.DataFrame(results)
        .sort_values("mc_mean", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def compute_return_zscore_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add industry-relative z-scores and percentile ranks for key return metrics."""
    if summary.empty:
        return summary

    return_cols = [
        c
        for c in [
            "expected_upside_pct",
            "filtered_upside",
            "expected_return_prob_weighted",
        ]
        if c in summary.columns
    ]
    group_col = "industry" if "industry" in summary.columns else None

    if return_cols:
        summary = vectorized_zscore(summary, return_cols, group_col=group_col)
        summary = vectorized_percentile_rank(summary, return_cols, group_col=group_col)
        logger.info(
            "Added z-scores and percentile ranks for %d return metrics",
            len(return_cols),
        )

    return summary


def compute_cross_model_correlation(mc: pd.DataFrame, kal: pd.DataFrame) -> dict:
    """Compute correlation and copula dependency between MC and Kalman returns."""
    if mc.empty or kal.empty:
        return {"correlation": None, "n_stocks": 0}

    mc_cols = {"ticker", "expected_upside_pct"}
    kal_cols = {"ticker", "filtered_upside"}
    if not mc_cols.issubset(mc.columns) or not kal_cols.issubset(kal.columns):
        return {"correlation": None, "n_stocks": 0}

    merged = mc[["ticker", "expected_upside_pct"]].merge(
        kal[["ticker", "filtered_upside"]],
        on="ticker",
        how="inner",
    )
    if len(merged) < 10:
        return {"correlation": None, "n_stocks": len(merged)}

    corr = merged[["expected_upside_pct", "filtered_upside"]].corr().iloc[0, 1]
    result: dict = {"correlation": float(corr), "n_stocks": len(merged)}

    if len(merged) > 50:
        try:
            copula = fit_gaussian_copula(
                merged,
                features=["expected_upside_pct", "filtered_upside"],
            )
            if copula:
                result["tail_dependence"] = copula.get("tail_dependence")
        except Exception as e:
            logger.debug("Copula fit skipped: %s", e)

    return result


def compute_cross_model_diagnostics(summary: pd.DataFrame) -> dict:
    """Comprehensive cross-model dispersion and convergence diagnostics."""
    if summary.empty:
        return {}

    return_cols = [
        "expected_upside_pct",
        "filtered_upside",
        "expected_return_prob_weighted",
    ]
    available_return_cols = [c for c in return_cols if c in summary.columns]
    if len(available_return_cols) < 2:
        return {}

    returns_df = summary[available_return_cols].dropna()

    pearson_corr = returns_df.corr(method="pearson").to_dict()
    spearman_corr = returns_df.corr(method="spearman").to_dict()

    row_means = returns_df.mean(axis=1)
    mad_per_stock = returns_df.sub(row_means, axis=0).abs().mean(axis=1)
    summary_copy = summary.loc[returns_df.index].copy()
    summary_copy["model_dispersion"] = mad_per_stock

    direction_agreement = (returns_df > 0).nunique(axis=1) == 1
    tail_agreement_pct = float(direction_agreement.mean() * 100)

    high_disp = pd.DataFrame()
    if "ticker" in summary.columns:
        summary_copy["ticker"] = summary.loc[returns_df.index, "ticker"].values
        high_disp = summary_copy.nlargest(20, "model_dispersion")[
            ["ticker", "model_dispersion"] + available_return_cols
        ]

    model_bias = {col: float(returns_df[col].mean()) for col in available_return_cols}

    try:
        from scipy.stats import kendalltau

        concordance_pairs = {}
        for i, c1 in enumerate(available_return_cols):
            for c2 in available_return_cols[i + 1 :]:
                tau, p = kendalltau(returns_df[c1], returns_df[c2])
                concordance_pairs[f"{c1} ↔ {c2}"] = {
                    "kendall_tau": float(tau),
                    "p_value": float(p),
                }
    except Exception:
        concordance_pairs = {}

    result = {
        "pairwise_pearson": pearson_corr,
        "pairwise_spearman": spearman_corr,
        "kendall_concordance": concordance_pairs,
        "mean_dispersion": float(mad_per_stock.mean()),
        "median_dispersion": float(mad_per_stock.median()),
        "tail_agreement_pct": tail_agreement_pct,
        "high_dispersion_tickers": high_disp,
        "model_bias": model_bias,
        "n_stocks": len(returns_df),
    }

    logger.info(
        "Cross-model diagnostics: tail agreement=%.1f%%, mean dispersion=%.2f",
        tail_agreement_pct,
        result["mean_dispersion"],
    )
    return result


def compute_return_distribution_analytics(
    mc: pd.DataFrame,
    summary: pd.DataFrame | None = None,
) -> dict:
    """Fit parametric distributions to MC simulation returns and compute risk metrics."""
    result = {}
    if mc.empty or "expected_upside_pct" not in mc.columns:
        return result

    upside = mc["expected_upside_pct"].dropna().values

    best_dist = None
    best_aic = np.inf
    candidates = [sp_stats.norm, sp_stats.t, sp_stats.skewnorm, sp_stats.laplace]

    for dist in candidates:
        try:
            params = dist.fit(upside)
            log_lik = dist.logpdf(upside, *params).sum()
            k = len(params)
            aic = 2 * k - 2 * log_lik
            if aic < best_aic:
                best_aic = aic
                best_dist = {
                    "name": dist.name,
                    "params": params,
                    "aic": float(aic),
                    "ks_statistic": float(
                        sp_stats.kstest(upside, dist.cdf, args=params).statistic
                    ),
                    "ks_pvalue": float(
                        sp_stats.kstest(upside, dist.cdf, args=params).pvalue
                    ),
                }
        except Exception:
            continue

    result["mc_distribution"] = best_dist

    var_1 = float(np.percentile(upside, 1))
    var_5 = float(np.percentile(upside, 5))
    cvar_5 = float(upside[upside <= var_5].mean()) if (upside <= var_5).any() else var_5
    downside = upside[upside < 0]
    downside_deviation = (
        float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 0.0
    )

    result["risk_metrics"] = {
        "var_1_pct": var_1,
        "var_5_pct": var_5,
        "cvar_5_pct": cvar_5,
        "downside_deviation": downside_deviation,
        "upside_capture": float((upside > 0).mean() * 100),
        "mean_positive_return": float(upside[upside > 0].mean())
        if (upside > 0).any()
        else 0.0,
        "mean_negative_return": float(downside.mean()) if len(downside) > 0 else 0.0,
        "gain_loss_ratio": (
            float(upside[upside > 0].mean() / abs(downside.mean()))
            if len(downside) > 0 and downside.mean() != 0
            else None
        ),
    }

    result["opportunity_tiers"] = {
        "high_conviction": (
            int(((upside > 20) & (mc["prob_positive_upside"].values > 70)).sum())
            if "prob_positive_upside" in mc.columns
            else 0
        ),
        "moderate": int(((upside > 0) & (upside <= 20)).sum()),
        "speculative": (
            int(
                (
                    (upside > 0)
                    & (
                        mc.get("prob_positive_upside", pd.Series(dtype=float)).values
                        < 50
                    )
                ).sum()
            )
            if "prob_positive_upside" in mc.columns
            else 0
        ),
        "avoid": int((upside <= 0).sum()),
    }

    if summary is not None and not summary.empty:
        ensemble_cols = [
            "expected_upside_pct",
            "filtered_upside",
            "expected_return_prob_weighted",
        ]
        available = [c for c in ensemble_cols if c in summary.columns]
        if available:
            ensemble_return = summary[available].mean(axis=1).dropna()
            result["ensemble_distribution"] = compute_metric_statistics(ensemble_return)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_mc_return_distribution(mc: pd.DataFrame) -> go.Figure:
    """Two-panel figure: expected upside histogram + P(positive) bar chart."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Expected Upside Distribution",
            "Probability of Positive Return",
        ),
        vertical_spacing=0.12,
    )

    upside = mc["expected_upside_pct"].clip(-100, 300)
    fig.add_trace(
        go.Histogram(
            x=upside,
            nbinsx=80,
            marker_color=COLORS[0],
            opacity=0.75,
            name="Expected Upside %",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    median_val = mc["expected_upside_pct"].median()
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_val:.1f}%",
        row=1,
        col=1,
    )

    prob_bins = pd.cut(
        mc["prob_positive_upside"],
        bins=[0, 25, 50, 75, 100],
        labels=["0–25%", "25–50%", "50–75%", "75–100%"],
    )
    prob_counts = prob_bins.value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=prob_counts.index.astype(str),
            y=prob_counts.values,
            marker_color=[COLORS[3], COLORS[1], COLORS[0], COLORS[2]],
            name="Stock Count",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Monte Carlo Simulation: Return Distribution Overview",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Expected Upside (%)", row=1, col=1)
    fig.update_xaxes(title_text="P(Positive Return) Bucket", row=2, col=1)
    fig.update_yaxes(title_text="Number of Stocks", row=1, col=1)
    fig.update_yaxes(title_text="Number of Stocks", row=2, col=1)
    return fig


def create_sector_risk_reward_scatter(mc: pd.DataFrame) -> go.Figure:
    """Sector-level bubble scatter: VaR 5% vs expected upside."""
    sector = (
        mc.groupby("industry")
        .agg(
            mean_upside=("expected_upside_pct", "mean"),
            mean_var5=("var_5_pct", "mean"),
            mean_prob=("prob_positive_upside", "mean"),
            count=("ticker", "count"),
        )
        .reset_index()
    )
    fig = px.scatter(
        sector,
        x="mean_var5",
        y="mean_upside",
        size="count",
        color="industry",
        hover_name="industry",
        hover_data={"mean_prob": ":.1f", "count": True},
        title="Industry Risk-Reward: Expected Upside vs VaR 5%",
        labels={
            "mean_var5": "Mean VaR 5% (%)",
            "mean_upside": "Mean Expected Upside (%)",
        },
        template=PLOTLY_TEMPLATE,
        height=550,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


def create_kalman_vs_raw_scatter(kal: pd.DataFrame) -> go.Figure:
    """Scatter of Kalman-filtered vs raw analyst upside (log-transformed)."""
    plot_df = kal.copy()
    if "raw_upside" not in plot_df.columns:
        plot_df["raw_upside"] = (
            (plot_df["original_target"] - plot_df["original_price"])
            / plot_df["original_price"]
            * 100
        )

    sample = plot_df.sample(min(2000, len(plot_df)), random_state=42).copy()
    sample["filtered_log"] = np.sign(sample["filtered_upside"]) * np.log1p(
        np.abs(sample["filtered_upside"])
    )
    sample["raw_log"] = np.sign(sample["raw_upside"]) * np.log1p(
        np.abs(sample["raw_upside"])
    )

    fig = px.scatter(
        sample,
        x="filtered_log",
        y="raw_log",
        color="industry" if "industry" in sample.columns else None,
        hover_name="ticker" if "ticker" in sample.columns else None,
        title="Kalman-Filtered vs Raw Analyst Upside (Log-Transformed)",
        labels={
            "filtered_log": "Kalman Filtered — sign(x)·log₁ₚ(|x|)",
            "raw_log": "Raw Analyst — sign(x)·log₁ₚ(|x|)",
        },
        template=PLOTLY_TEMPLATE,
        height=550,
        opacity=0.6,
    )
    log_max = max(
        sample["filtered_log"].abs().quantile(0.99),
        sample["raw_log"].abs().quantile(0.99),
    )
    fig.add_shape(
        type="line",
        x0=-log_max,
        y0=-log_max,
        x1=log_max,
        y1=log_max,
        line=dict(color="gray", dash="dash", width=1),
    )
    return fig


def create_tri_model_agreement_histogram(tri: pd.DataFrame) -> go.Figure:
    """Histogram of tri-model signal agreement (0/3 → 3/3)."""
    color_map = {
        v: [COLORS[3], COLORS[1], COLORS[0], COLORS[2]][k]
        for k, v in _SIGNAL_LABELS.items()
    }
    fig = px.histogram(
        tri,
        x="signal",
        color="signal",
        title="Tri-Model Signal Agreement (MC + Kalman + Achievement)",
        labels={"signal": "Model Agreement"},
        color_discrete_map=color_map,
        category_orders={"signal": list(_SIGNAL_LABELS.values())},
        template=PLOTLY_TEMPLATE,
        height=420,
    )
    fig.update_layout(showlegend=False)
    return fig


def create_strong_consensus_bar(strong: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: MC / Kalman / Achievement returns for top picks."""
    if strong.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No strong consensus picks",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["expected_upside_pct"],
            name="MC Expected Upside",
            marker_color=COLORS[0],
        )
    )
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["filtered_upside"],
            name="Kalman Filtered Upside",
            marker_color=COLORS[1],
        )
    )
    fig.add_trace(
        go.Bar(
            x=strong["ticker"],
            y=strong["expected_return_prob_weighted"],
            name="Prob-Weighted Return",
            marker_color=COLORS[2],
        )
    )
    fig.update_layout(
        title=f"Top {len(strong)} Strong Consensus Picks (All 3 Models Bullish)",
        yaxis_title="Expected Return (%)",
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=500,
        xaxis_tickangle=-45,
    )
    return fig


def create_sector_heatmap(tri: pd.DataFrame) -> go.Figure:
    """Sector expected returns heatmap across all models."""
    sector = compute_sector_expected_returns(tri)
    if sector.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No sector data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    heatmap_data = sector.set_index("industry")[
        [
            "mc_mean",
            "mc_median",
            "kalman_mean",
            "kalman_median",
            "pt_mean",
            "pt_median",
            "pct_bullish",
        ]
    ].rename(
        columns={
            "mc_mean": "MC Mean",
            "mc_median": "MC Median",
            "kalman_mean": "Kalman Mean",
            "kalman_median": "Kalman Median",
            "pt_mean": "Achiev. Mean",
            "pt_median": "Achiev. Median",
            "pct_bullish": "% All Bullish",
        }
    )

    fig = px.imshow(
        heatmap_data.round(1),
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto",
        title="Industry Expected Returns Heatmap (All Models)",
        labels={"color": "Value"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=max(600, len(sector) * 22))
    return fig


def create_var_analysis(mc: pd.DataFrame) -> go.Figure:
    """Two-panel VaR analysis: distribution + VaR vs upside scatter."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("VaR 5% Distribution", "VaR 5% vs Expected Upside"),
        vertical_spacing=0.12,
    )

    var_clipped = mc["var_5_pct"].clip(-150, 300)
    fig.add_trace(
        go.Histogram(
            x=var_clipped,
            nbinsx=80,
            marker_color=COLORS[3],
            opacity=0.75,
            name="VaR 5%",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="blue", row=1, col=1)

    sample = mc.sample(min(2000, len(mc)), random_state=42)
    fig.add_trace(
        go.Scatter(
            x=sample["var_5_pct"],
            y=sample["expected_upside_pct"],
            mode="markers",
            marker=dict(
                size=4,
                color=sample["prob_positive_upside"],
                colorscale="RdYlGn",
                colorbar=dict(title="P(+)"),
                opacity=0.5,
            ),
            text=sample.get("name"),
            hovertemplate="%{text}<br>VaR 5%: %{x:.1f}%<br>Upside: %{y:.1f}%<extra></extra>",
            name="Stocks",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Value-at-Risk (5%) Analysis",
        template=PLOTLY_TEMPLATE,
        height=800,
        showlegend=False,
    )
    fig.update_xaxes(title_text="VaR 5% (%)", row=1, col=1)
    fig.update_xaxes(title_text="VaR 5% (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Expected Upside (%)", row=2, col=1)
    return fig


def create_beat_vs_achievement_scatter(
    beat: pd.DataFrame, pt: pd.DataFrame
) -> go.Figure:
    """Scatter: P(Beat) vs P(Reach Price Target) coloured by return."""
    if beat.empty or pt.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    merged = beat[["ticker", "posterior_beat_prob", "confidence_score"]].merge(
        pt[["ticker", "achievement_probability", "expected_return_prob_weighted"]],
        on="ticker",
        how="inner",
    )
    if merged.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No overlapping tickers",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    fig = px.scatter(
        merged,
        x="posterior_beat_prob",
        y="achievement_probability",
        color="expected_return_prob_weighted",
        hover_name="ticker",
        title="Earnings Beat Probability vs Price Target Achievement",
        labels={
            "posterior_beat_prob": "P(Beat Next Quarter)",
            "achievement_probability": "P(Reach Price Target)",
        },
        color_continuous_scale="RdYlGn",
        template=PLOTLY_TEMPLATE,
        height=500,
    )
    return fig


def create_model_dispersion_dashboard(summary: pd.DataFrame) -> go.Figure:
    """Four-panel dashboard showing inter-model agreement/dispersion analytics."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Inter-Model Dispersion Distribution",
            "Discrete vs Weighted Agreement",
            "Sector Consensus Rate (% All Bullish)",
            "Highest Model Disagreement Stocks",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    return_cols = [
        "expected_upside_pct",
        "filtered_upside",
        "expected_return_prob_weighted",
    ]
    available = [c for c in return_cols if c in summary.columns]
    dispersion = pd.Series(dtype=float)

    if len(available) >= 2:
        returns_df = summary[available].dropna()
        row_mean = returns_df.mean(axis=1)
        dispersion = returns_df.sub(row_mean, axis=0).abs().mean(axis=1)

        fig.add_trace(
            go.Histogram(
                x=dispersion,
                nbinsx=60,
                marker_color=COLORS[0],
                opacity=0.75,
                name="Model Dispersion",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=dispersion.median(),
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {dispersion.median():.1f}",
            row=1,
            col=1,
        )

    if "agreement_score" in summary.columns and "weighted_agreement" in summary.columns:
        sample = summary.dropna(
            subset=["agreement_score", "weighted_agreement"]
        ).sample(
            min(2000, len(summary)),
            random_state=42,
        )
        fig.add_trace(
            go.Scatter(
                x=sample["agreement_score"],
                y=sample["weighted_agreement"],
                mode="markers",
                marker=dict(size=4, opacity=0.4, color=COLORS[1]),
                name="Stocks",
                hovertemplate="Score: %{x}<br>Weighted: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    group_col = "industry" if "industry" in summary.columns else "sector"
    if group_col in summary.columns and "agreement_score" in summary.columns:
        consensus = (
            summary.groupby(group_col)["agreement_score"]
            .apply(lambda x: (x == 4).mean() * 100)
            .sort_values(ascending=True)
            .tail(20)
        )
        fig.add_trace(
            go.Bar(
                x=consensus.values,
                y=consensus.index.astype(str),
                orientation="h",
                marker_color=COLORS[2],
                name="% Full Consensus",
            ),
            row=2,
            col=1,
        )

    if len(available) >= 2 and "ticker" in summary.columns and not dispersion.empty:
        summary_disp = summary.copy()
        summary_disp["_dispersion"] = dispersion
        top_disagree = summary_disp.nlargest(15, "_dispersion")
        fig.add_trace(
            go.Bar(
                x=top_disagree["_dispersion"],
                y=top_disagree["ticker"],
                orientation="h",
                marker_color=COLORS[3],
                name="Dispersion",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Cross-Model Dispersion & Agreement Dashboard",
        template=PLOTLY_TEMPLATE,
        height=900,
        showlegend=False,
    )
    return fig


def create_return_distribution_fit_chart(mc: pd.DataFrame) -> go.Figure:
    """Overlay histogram of MC returns with fitted parametric distribution."""
    upside = mc["expected_upside_pct"].dropna().clip(-100, 300)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=upside,
            nbinsx=100,
            histnorm="probability density",
            marker_color=COLORS[0],
            opacity=0.6,
            name="Observed",
        )
    )

    x_range = np.linspace(upside.min(), upside.max(), 300)
    for dist, color, label in [
        (sp_stats.norm, COLORS[1], "Normal"),
        (sp_stats.t, COLORS[2], "Student-t"),
        (sp_stats.skewnorm, COLORS[3], "Skew-Normal"),
    ]:
        try:
            params = dist.fit(upside)
            pdf = dist.pdf(x_range, *params)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=pdf,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,
                )
            )
        except Exception:
            continue

    var_5 = float(np.percentile(upside, 5))
    cvar_5 = float(upside[upside <= var_5].mean()) if (upside <= var_5).any() else var_5
    fig.add_vline(
        x=var_5,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR 5%: {var_5:.1f}%",
    )
    fig.add_vline(
        x=cvar_5,
        line_dash="dot",
        line_color="darkred",
        annotation_text=f"CVaR 5%: {cvar_5:.1f}%",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="MC Return Distribution — Parametric Fit Overlay",
        xaxis_title="Expected Upside (%)",
        yaxis_title="Density",
        template=PLOTLY_TEMPLATE,
        height=550,
    )
    return fig


def create_sector_return_analytics_heatmap(sector_analytics: pd.DataFrame) -> go.Figure:
    """Enhanced sector heatmap with confidence intervals, consensus rates, and beat probs."""
    if sector_analytics.empty or "sector" not in sector_analytics.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No sector analytics data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    display_cols = [
        c
        for c in [
            "mc_mean",
            "mc_median",
            "kalman_mean",
            "pt_mean",
            "pct_full_consensus",
            "mean_weighted_agreement",
            "risk_adjusted_return",
            "mean_beat_prob",
        ]
        if c in sector_analytics.columns
    ]
    if not display_cols:
        return go.Figure()

    heatmap_data = sector_analytics.set_index("sector")[display_cols]
    rename_map = {
        "mc_mean": "MC Mean %",
        "mc_median": "MC Median %",
        "kalman_mean": "Kalman Mean %",
        "pt_mean": "Achiev. Mean %",
        "pct_full_consensus": "Full Consensus %",
        "mean_weighted_agreement": "Wtd Agreement",
        "risk_adjusted_return": "Risk-Adj Return",
        "mean_beat_prob": "Mean P(Beat)",
    }
    heatmap_data = heatmap_data.rename(columns=rename_map)

    fig = px.imshow(
        heatmap_data.round(2),
        color_continuous_scale="RdYlGn",
        text_auto=True,
        aspect="auto",
        title="Sector Expected Returns — Enhanced Analytics Heatmap",
        labels={"color": "Value"},
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=max(600, len(sector_analytics) * 28)
    )
    return fig


def create_screening_summary_chart(screens: dict[str, pd.DataFrame]) -> go.Figure:
    """Bar chart summarizing stock counts from each screening strategy."""
    names = []
    counts = []
    for name, df in screens.items():
        if not df.empty:
            names.append(name.replace("_", " ").title())
            counts.append(len(df))

    if not names:
        fig = go.Figure()
        fig.add_annotation(
            text="No screening results",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        return fig

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=names,
            orientation="h",
            marker_color=COLORS[: len(names)],
            text=counts,
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Stock Screening Results Summary",
        xaxis_title="Number of Stocks Passing",
        template=PLOTLY_TEMPLATE,
        height=max(400, len(names) * 40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Export Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def export_expected_returns_results(
    mc: pd.DataFrame,
    pt: pd.DataFrame,
    kal: pd.DataFrame,
    tri: pd.DataFrame,
    strong: pd.DataFrame,
    beat: pd.DataFrame,
    summary: pd.DataFrame = None,
    credit: pd.DataFrame = None,
    div_safety: pd.DataFrame = None,
    screens: dict[str, pd.DataFrame] = None,
    output_dir: str = "outputs/analytics",
) -> dict[str, str]:
    """
    Export all expected returns analytics to the ``analytics`` schema.

    v3.0: Added dividend safety and screening results exports.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    _EXPORT_PAIRS = [
        (mc, "monte_carlo_simulation"),
        (pt, "price_target_achievement"),
        (kal, "kalman_filtered_price_targets"),
        (tri, "expected_returns_tri_model"),
        (strong, "strong_consensus_picks"),
        (beat, "earnings_probability_analysis"),
        (summary, "expected_returns_summary"),
        (credit, "credit_risk_analysis"),
        (div_safety, "dividend_safety_analysis"),
    ]

    for df, table in _EXPORT_PAIRS:
        if df is not None and not df.empty:
            try:
                reordered_df = reorder_with_identifiers(df)
                cfg = ExportConfig(table_name=table)
                export_to_db(reordered_df, cfg)
                export_to_csv(reordered_df, cfg)
                export_to_json(reordered_df, cfg)
                exports[table] = f"analytics.{table}"
                logger.info("Exported %d rows → analytics.%s", len(df), table)
            except Exception as e:
                logger.warning("Export failed for %s: %s", table, e)

    # Export screening results
    if screens:
        _SCREEN_TABLE_MAP = {
            "quality": "quality_stocks",
            "earnings_quality": "earnings_quality_stocks",
            "value": "value_stocks",
            "growth": "integrity_filtered_growth_stocks",
            "garp": "garp_stocks",
            "dividend": "dividend_safety_analysis",
            "healthy": "healthy_stocks",
            "valuation_reversion": "valuation_reversion_stocks",
            "integrity_growth": "integrity_filtered_growth_stocks",
            "high_yield_safe": "high_yield_safe_dividend_stocks",
            "sector_relative": "sector_relative_ranking",
        }
        for screen_name, df in screens.items():
            if df is not None and not df.empty:
                table = _SCREEN_TABLE_MAP.get(screen_name, f"screen_{screen_name}")
                try:
                    reordered_df = reorder_with_identifiers(df)
                    cfg = ExportConfig(table_name=table)
                    export_to_db(reordered_df, cfg)
                    exports[table] = f"analytics.{table}"
                    logger.info(
                        "Exported screen %s: %d rows → analytics.%s",
                        screen_name,
                        len(df),
                        table,
                    )
                except Exception as e:
                    logger.warning("Export failed for screen %s: %s", screen_name, e)

    return exports


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """
    Main expected returns analytics pipeline (v3.0).

    Steps:
        1.  Load feature data from materialized views (mv_expected_returns + mv_all_stock_features)
        2.  Run Monte Carlo simulation
        3.  Run Price Target Achievement model
        4.  Run Kalman filter
        5.  Run Earnings Beat analysis
        5b. Run Credit Risk & Dividend Safety analysis
        5c. Run Stock Screening (quality, value, growth, GARP, dividend, health)
        6.  Build tri-model & quad-model alignment
        7.  Build expected_returns_summary (4-model merge)
        7b. Run per-category Bayesian probability analytics
        8.  Generate visualizations
        9.  Build InferenceData (ArviZ)
        10. Export results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("Expected Returns Analytics Pipeline v3.0")
    print("=" * 80)
    print()

    output_dir = Path("outputs/analytics")
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = get_optimization_status()
    print(
        f"🔧 Numba: {opt.get('numba_available', False)}, "
        f"Joblib: {opt.get('joblib_available', False)}"
    )
    print()

    # ========================================================================
    # 1. DATA LOADING (v3.0: Materialized Views)
    # ========================================================================
    print("📦 Step 1: Loading feature data from materialized views...")
    print("-" * 80)

    df = load_expected_returns_data()
    if df.empty:
        print("✗ No data loaded from mv_expected_returns. Check DB_URL.")
        return

    print(
        f"✓ Loaded mv_expected_returns: {len(df):,} stocks × {len(df.columns)} features"
    )

    df_all = load_all_stock_features()
    if not df_all.empty:
        print(
            f"✓ Loaded mv_all_stock_features: {len(df_all):,} stocks × {len(df_all.columns)} features"
        )
    else:
        print(
            "⚠️ mv_all_stock_features not loaded — screening will use mv_expected_returns"
        )
        df_all = df  # Fallback: use expected returns MV for screening

    print()

    # ========================================================================
    # 2. MONTE CARLO SIMULATION
    # ========================================================================
    print("🎲 Step 2: Monte Carlo price target simulation (50K samples)...")
    print("-" * 80)

    hist_coverage = _resolve_available_historical_cols(df)
    hist_total = sum(len(v) for v in hist_coverage.values())
    print(
        f"  Historical price/target columns available: {hist_total} / {len(ALL_HISTORICAL_PRICE_TARGET_COLS)}"
    )
    for cat, cols in hist_coverage.items():
        if cols:
            print(f"    {cat}: {len(cols)} columns")

    mc = run_monte_carlo_analysis(df, n_simulations=50_000, use_historical_targets=True)
    if not mc.empty:
        print(f"✓ {len(mc):,} stocks simulated")
        print(f"  Mean upside:  {mc['expected_upside_pct'].mean():.1f}%")
        print(f"  Median upside: {mc['expected_upside_pct'].median():.1f}%")
        print(f"  Positive prob (mean): {mc['prob_positive_upside'].mean():.1f}%")

        mc = compute_price_target_mc(mc, df)
        if "price_target_mc" in mc.columns:
            valid_mc = mc.dropna(subset=["price_target_mc", "last_price"])
            if not valid_mc.empty:
                mean_price = valid_mc["last_price"].mean()
                mean_target = valid_mc["price_target_mc"].mean()
                implied_return = (
                    (mean_target / mean_price - 1) * 100 if mean_price > 0 else 0
                )
                print(
                    f"  Monte Carlo targets ({len(valid_mc):,} stocks): implied return={implied_return:.1f}%"
                )

        mc_stats = compute_model_detailed_statistics(
            mc,
            "Monte Carlo",
            [
                "expected_upside_pct",
                "prob_positive_upside",
                "var_5_pct",
                "risk_reward_ratio",
                "price_target_mc",
            ],
        )
        print_model_statistics(
            mc_stats, "Monte Carlo Simulation", show_sectors=True, top_n_sectors=10
        )

        dist_analytics = compute_return_distribution_analytics(mc)
        if dist_analytics.get("mc_distribution"):
            d = dist_analytics["mc_distribution"]
            print(
                f"\n  📐 Best-fit distribution: {d['name']} (AIC={d['aic']:.1f}, KS p={d['ks_pvalue']:.3f})"
            )
        if dist_analytics.get("risk_metrics"):
            rm = dist_analytics["risk_metrics"]
            print(
                f"\n  📉 VaR 1%: {rm['var_1_pct']:.1f}%  |  CVaR 5%: {rm['cvar_5_pct']:.1f}%"
            )
            print(f"     Downside deviation: {rm['downside_deviation']:.2f}")
            if rm.get("gain_loss_ratio"):
                print(f"     Gain/Loss ratio: {rm['gain_loss_ratio']:.2f}")
        if dist_analytics.get("opportunity_tiers"):
            t = dist_analytics["opportunity_tiers"]
            print(
                f"\n  🏷️  Tiers: High-conviction={t['high_conviction']}, "
                f"Moderate={t['moderate']}, Speculative={t['speculative']}, Avoid={t['avoid']}"
            )
    print()

    # ========================================================================
    # 3. PRICE TARGET ACHIEVEMENT
    # ========================================================================
    print("🎯 Step 3: Price target achievement model...")
    print("-" * 80)

    pt = run_price_target_achievement(df, use_historical_targets=True)
    if not pt.empty:
        print(f"✓ {len(pt):,} stocks analyzed")
        print(f"  Mean achievement prob: {pt['achievement_probability'].mean():.3f}")
        print(
            f"  Mean prob-weighted return: {pt['expected_return_prob_weighted'].mean():.1f}%"
        )

        pt = compute_price_target_prob_weighted(pt, df)
        if "price_target_prob_weighted" in pt.columns:
            valid_pt = pt.dropna(subset=["price_target_prob_weighted", "last_price"])
            if not valid_pt.empty:
                mean_price = valid_pt["last_price"].mean()
                mean_target = valid_pt["price_target_prob_weighted"].mean()
                implied_return = (
                    (mean_target / mean_price - 1) * 100 if mean_price > 0 else 0
                )
                print(
                    f"  Prob-weighted targets ({len(valid_pt):,} stocks): implied return={implied_return:.1f}%"
                )

        pt_stats = compute_model_detailed_statistics(
            pt,
            "Price Target Achievement",
            [
                "achievement_probability",
                "expected_return_prob_weighted",
                "analyst_conviction",
                "eps_revision_momentum",
                "price_target_prob_weighted",
            ],
        )
        print_model_statistics(pt_stats, "Price Target Achievement", show_sectors=True)
    print()

    # ========================================================================
    # 4. KALMAN FILTER
    # ========================================================================
    print("📐 Step 4: Kalman-filtered price targets...")
    print("-" * 80)

    kal = run_kalman_filter(df, use_historical_targets=True)
    if not kal.empty:
        print(f"✓ {len(kal):,} stocks filtered")
        print(f"  Mean filtered upside: {kal['filtered_upside'].mean():.1f}%")

        kal_stats = compute_model_detailed_statistics(
            kal,
            "Kalman Filter",
            ["filtered_upside", "kalman_variance", "signal_strength"],
        )
        print_model_statistics(kal_stats, "Kalman Filter", show_sectors=True)
    print()

    # ========================================================================
    # 5. EARNINGS BEAT ANALYSIS
    # ========================================================================
    print("📊 Step 5: Bayesian earnings beat analysis...")
    print("-" * 80)

    beat = run_earnings_beat_analysis(df)
    if not beat.empty:
        print(f"✓ {len(beat):,} stocks analyzed")
        print(f"  Mean P(beat): {beat['posterior_beat_prob'].mean():.3f}")
        if "beat_classification" in beat.columns:
            likely = (beat["beat_classification"] == "likely_beat").sum()
            print(f"  Classified as 'likely_beat': {likely}")
    print()

    # ========================================================================
    # 5b. CREDIT RISK & DIVIDEND SAFETY
    # ========================================================================
    print("🛡️ Step 5b: Credit risk & dividend safety analysis...")
    print("-" * 80)

    credit = run_credit_risk_analysis(df)
    if not credit.empty:
        high_risk = (
            credit["risk_level"].isin(["High", "Distressed"]).sum()
            if "risk_level" in credit.columns
            else 0
        )
        print(f"✓ Credit risk: {len(credit):,} stocks, {high_risk} high/distressed")
        if "ruin_probability" in credit.columns:
            print(f"  Mean ruin probability: {credit['ruin_probability'].mean():.3f}")

    div_safety = run_dividend_safety_analysis(df_all)
    if not div_safety.empty:
        at_risk = (
            (div_safety["risk_category"] == "At Risk").sum()
            if "risk_category" in div_safety.columns
            else 0
        )
        print(f"✓ Dividend safety: {len(div_safety):,} stocks, {at_risk} at risk")
    print()

    # ========================================================================
    # 5c. STOCK SCREENING (v3.0 — NEW)
    # ========================================================================
    print("🔍 Step 5c: Running stock screening strategies...")
    print("-" * 80)

    screens = run_stock_screening(df_all)
    for name, screen_df in screens.items():
        if not screen_df.empty:
            print(f"  ✓ {name}: {len(screen_df):,} stocks")
    print()

    # ========================================================================
    # 6. CROSS-MODEL ALIGNMENT
    # ========================================================================
    print("🔗 Step 6: Cross-model alignment...")
    print("-" * 80)

    tri = build_tri_model_alignment(mc, kal, pt)
    strong = extract_strong_consensus(tri)
    quad = build_quad_model_alignment(tri, beat)

    if not tri.empty:
        print(f"  Tri-model coverage: {len(tri):,} stocks")
        for label in _SIGNAL_LABELS.values():
            cnt = (tri["signal"] == label).sum()
            print(f"    {label}: {cnt}")
        print(f"  Strong consensus picks: {len(strong)}")

    if not quad.empty:
        full = (quad["quad_agreement"] == 4).sum()
        print(f"  Quad-model (4/4): {full} stocks")

    corr_info = compute_cross_model_correlation(mc, kal)
    if corr_info.get("correlation") is not None:
        print(f"  MC ↔ Kalman correlation: {corr_info['correlation']:.3f}")
    print()

    # ========================================================================
    # 7. EXPECTED RETURNS SUMMARY (4-MODEL MERGE)
    # ========================================================================
    print("📋 Step 7: Building expected_returns_summary (4-model merge)...")
    print("-" * 80)

    summary = build_expected_returns_summary(mc, kal, pt, beat, source_df=df_all)
    if not summary.empty:
        print(f"  ✓ {len(summary):,} stocks in expected_returns_summary")
        full_consensus = (summary["agreement_score"] == 4).sum()
        print(f"  Full consensus (4/4): {full_consensus} stocks")

        summary = filter_quality_stocks(summary, df_all)
        if "quality_tier" in summary.columns:
            high_quality_bullish = (
                (summary["agreement_score"] == 4)
                & (summary["quality_tier"].isin(["High", "Above Avg"]))
            ).sum()
            print(f"  High-quality full consensus: {high_quality_bullish} stocks")

        summary = compute_return_zscore_ranks(summary)

        print("  Agreement distribution:")
        for label in _SIGNAL_LABELS_4.values():
            cnt = (summary["signal"] == label).sum()
            if cnt > 0:
                print(f"    {label}: {cnt}")

        cross_diag = compute_cross_model_diagnostics(summary)
        if cross_diag:
            print("\n  🔬 Cross-Model Diagnostics:")
            print(f"     Direction agreement: {cross_diag['tail_agreement_pct']:.1f}%")
            print(
                f"     Mean inter-model dispersion: {cross_diag['mean_dispersion']:.2f}"
            )
            for pair, info in cross_diag.get("kendall_concordance", {}).items():
                print(
                    f"     Kendall τ ({pair}): {info['kendall_tau']:.3f} (p={info['p_value']:.4f})"
                )

        sector_analytics = compute_sector_return_analytics(summary)
        if not sector_analytics.empty:
            print(f"\n  🏢 Sector Analytics: {len(sector_analytics)} sectors")
            top_sectors = sector_analytics.head(5)
            for _, row in top_sectors.iterrows():
                consensus = row.get("pct_full_consensus", 0)
                ra = row.get("risk_adjusted_return", 0)
                print(
                    f"     {row['sector']}: MC mean={row.get('mc_mean', 0):.1f}%, "
                    f"consensus={consensus:.0f}%, risk-adj={ra:.2f}"
                )
    else:
        print("  ⚠️ Expected returns summary: no overlapping tickers across 4 models")
    print()

    # ========================================================================
    # 7b. PER-CATEGORY BAYESIAN PROBABILITY ANALYTICS (v3.0 — NEW)
    # ========================================================================
    print("🧮 Step 7b: Per-category Bayesian probability analytics...")
    print("-" * 80)

    category_analytics = run_category_probability_analysis(df)
    if category_analytics:
        print(f"  ✓ Analyzed {len(category_analytics)} categories")
        for cat_name, cat_result in category_analytics.items():
            n_feat = cat_result.get("features_analyzed", 0)
            bayesian_keys = list(cat_result.get("bayesian_results", {}).keys())
            print(
                f"    {cat_name}: {n_feat} features — {len(bayesian_keys)} posteriors"
            )
    print()

    # ========================================================================
    # 8. VISUALIZATIONS
    # ========================================================================
    print("📈 Step 8: Generating visualizations...")
    print("-" * 80)

    try:
        if not mc.empty:
            create_mc_return_distribution(mc).write_html(
                output_dir / "er_mc_distribution.html"
            )
            print("   ✓ er_mc_distribution.html")
            create_sector_risk_reward_scatter(mc).write_html(
                output_dir / "er_sector_risk_reward.html"
            )
            print("   ✓ er_sector_risk_reward.html")
            create_var_analysis(mc).write_html(output_dir / "er_var_analysis.html")
            print("   ✓ er_var_analysis.html")
            create_posterior_return_forest(mc, top_n=25).write_html(
                output_dir / "er_posterior_return_forest.html"
            )
            print("   ✓ er_posterior_return_forest.html")

        if not kal.empty:
            create_kalman_vs_raw_scatter(kal).write_html(
                output_dir / "er_kalman_vs_raw.html"
            )
            print("   ✓ er_kalman_vs_raw.html")

        if not tri.empty:
            create_tri_model_agreement_histogram(tri).write_html(
                output_dir / "er_tri_model_agreement.html"
            )
            print("   ✓ er_tri_model_agreement.html")
            create_sector_heatmap(tri).write_html(output_dir / "er_sector_heatmap.html")
            print("   ✓ er_sector_heatmap.html")

        if not strong.empty:
            create_strong_consensus_bar(strong).write_html(
                output_dir / "er_strong_consensus.html"
            )
            print("   ✓ er_strong_consensus.html")
            tri_cols = {
                "ticker",
                "expected_upside_pct",
                "filtered_upside",
                "expected_return_prob_weighted",
            }
            if tri_cols.issubset(strong.columns):
                create_tri_model_posterior_comparison(strong, top_n=12).write_html(
                    output_dir / "er_tri_model_posterior.html"
                )
                print("   ✓ er_tri_model_posterior.html")

        if not beat.empty and not pt.empty:
            create_beat_vs_achievement_scatter(beat, pt).write_html(
                output_dir / "er_beat_vs_achievement.html"
            )
            print("   ✓ er_beat_vs_achievement.html")

        if not beat.empty and "posterior_beat_prob" in beat.columns:
            create_beat_probability_posterior(beat, top_n=12).write_html(
                output_dir / "er_beat_probability_posterior.html"
            )
            print("   ✓ er_beat_probability_posterior.html")

        if not beat.empty:
            create_earnings_probability_dashboard(beat).write_html(
                output_dir / "er_earnings_probability_dashboard.html"
            )
            print("   ✓ er_earnings_probability_dashboard.html")

        if not df.empty:
            create_quality_risk_quadrant(df).write_html(
                output_dir / "er_quality_risk_quadrant.html"
            )
            print("   ✓ er_quality_risk_quadrant.html")
            create_distress_early_warning_dashboard(df).write_html(
                output_dir / "er_distress_early_warning.html"
            )
            print("   ✓ er_distress_early_warning.html")

        if not credit.empty and "ruin_probability" in credit.columns:
            create_ruin_probability_diagnostic(credit, top_n=20).write_html(
                output_dir / "er_ruin_probability_diagnostic.html"
            )
            print("   ✓ er_ruin_probability_diagnostic.html")

        if not summary.empty:
            tri_cols = {
                "ticker",
                "expected_upside_pct",
                "filtered_upside",
                "expected_return_prob_weighted",
            }
            if tri_cols.issubset(summary.columns):
                create_tri_model_posterior_comparison(summary, top_n=12).write_html(
                    output_dir / "er_expected_returns_summary_posterior.html"
                )
                print("   ✓ er_expected_returns_summary_posterior.html")

                create_model_dispersion_dashboard(summary).write_html(
                    output_dir / "er_model_dispersion_dashboard.html"
                )
                print("   ✓ er_model_dispersion_dashboard.html")

            if not mc.empty:
                create_return_distribution_fit_chart(mc).write_html(
                    output_dir / "er_return_distribution_fit.html"
                )
                print("   ✓ er_return_distribution_fit.html")

            sector_analytics = compute_sector_return_analytics(summary)
            if not sector_analytics.empty:
                create_sector_return_analytics_heatmap(sector_analytics).write_html(
                    output_dir / "er_sector_return_analytics.html"
                )
                print("   ✓ er_sector_return_analytics.html")

        # v3.0: Screening summary chart
        if screens:
            create_screening_summary_chart(screens).write_html(
                output_dir / "er_screening_summary.html"
            )
            print("   ✓ er_screening_summary.html")

        # Bayesian category ridge for analyst sentiment features
        sentiment_features = [
            f
            for f in [
                "analyst_bullish_pct",
                "upside_potential",
                "eps_revision_momentum",
                "analyst_conviction",
                "pt_consensus_convergence",
            ]
            if f in df.columns
        ]
        if sentiment_features:
            results = bayesian_category_analysis(
                df, "Analyst Sentiment", sentiment_features
            )
            create_bayesian_category_ridge(
                results, category_name="Analyst Sentiment"
            ).write_html(output_dir / "er_bayesian_sentiment_ridge.html")
            print("   ✓ er_bayesian_sentiment_ridge.html")

        # v3.0: Bayesian category ridge for profitability features
        profitability_features = [
            f
            for f in ["roe", "roa", "roic", "gross_margin_pct", "operating_margin_pct"]
            if f in df.columns
        ]
        if profitability_features:
            results = bayesian_category_analysis(
                df, "Profitability", profitability_features
            )
            create_bayesian_category_ridge(
                results, category_name="Profitability"
            ).write_html(output_dir / "er_bayesian_profitability_ridge.html")
            print("   ✓ er_bayesian_profitability_ridge.html")

    except Exception as e:
        print(f"   ⚠️ Visualization error: {e}")
        import traceback

        traceback.print_exc()

    print()

    # ========================================================================
    # 9. INFERENCE DATA (ArviZ)
    # ========================================================================
    if ARVIZ_AVAILABLE:
        print("🧪 Step 9: Building InferenceData (ArviZ)...")
        print("-" * 80)
        try:
            if not mc.empty:
                idata_mc = build_monte_carlo_inference_data(
                    mc, df_all, n_simulations=25_000
                )
                idata_summary = summarize_inference_data(idata_mc)
                print(
                    f"   ✓ MC InferenceData: {idata_summary.get('n_draws', 0)} draws, "
                    f"{idata_summary.get('n_equities', 0)} equities"
                )
                if idata_summary.get("r_hat"):
                    for var, rhat in idata_summary["r_hat"].items():
                        print(f"     R̂ ({var}): {rhat:.4f}")

            if not beat.empty and "posterior_alpha" in beat.columns:
                idata_beat = build_beat_probability_inference_data(
                    beat,
                    df_all,
                    n_posterior_samples=4000,
                    n_chains=4,
                )
                beat_summary = summarize_inference_data(idata_beat)
                print(
                    f"   ✓ Beat InferenceData: {beat_summary.get('n_chains', 0)} chains × "
                    f"{beat_summary.get('n_draws', 0)} draws"
                )
            if not credit.empty:
                idata_credit = build_credit_risk_inference_data(
                    credit, df_all,
                )
                credit_summary = summarize_inference_data(idata_credit)
                print(
                    f"   ✓ Credit Risk InferenceData: "
                    f"{credit_summary.get('n_equities', 0)} equities"
                )

            # Log EquityCoordinates for traceability
            if EquityCoordinates is not None and not df_all.empty:
                try:
                    coords = EquityCoordinates.from_dataframe(df_all)
                    print(
                        f"   ✓ EquityCoordinates: {len(coords.tickers)} tickers, "
                        f"{len(coords.sectors)} sectors"
                    )
                except Exception as e:
                    logger.debug("EquityCoordinates construction skipped: %s", e)

        except Exception as e:
            print(f"   ⚠️ InferenceData error: {e}")
        print()
    else:
        print("⏭️  Step 9: ArviZ not available — skipping InferenceData\n")

    # ========================================================================
    # 10. EXPORT RESULTS
    # ========================================================================
    print("💾 Step 10: Exporting results...")
    print("-" * 80)

    exports = export_expected_returns_results(
        mc=mc,
        pt=pt,
        kal=kal,
        tri=tri,
        strong=strong,
        beat=beat,
        summary=summary,
        credit=credit,
        div_safety=div_safety,
        screens=screens,
        output_dir=str(output_dir),
    )
    for name, dest in exports.items():
        print(f"   ✓ {name} → {dest}")

    # Export probability analytics results (beat + streak + credit + dividend)
    try:
        streak_analyzer = EPSStreakAnalyzer()
        streak_df = streak_analyzer.analyze_dataframe(df)
        prob_exports = export_probability_analytics_results(
            probability_df=beat,
            streak_df=streak_df,
            output_dir=output_dir,
            credit_risk_df=credit,
            dividend_safety_df=div_safety,
        )
        for pname, pdest in prob_exports.items():
            print(f"   ✓ {pname} → {pdest}")
    except Exception as e:
        logger.warning("Probability analytics export failed: %s", e)

    # Aggregate probability results for category analytics
    if category_analytics:
        try:
            for cat_name, cat_result in category_analytics.items():
                cat_prob = cat_result.get("conditional_probabilities")
                if isinstance(cat_prob, pd.DataFrame) and not cat_prob.empty:
                    aggregated = aggregate_probability_results(cat_prob)
                    if not aggregated.empty:
                        cfg = ExportConfig(table_name=f"prob_{cat_name.lower().replace(' ', '_')}")
                        export_to_db(aggregated, cfg)
                        logger.info("Aggregated probability export: %s (%d rows)", cat_name, len(aggregated))
        except Exception as e:
            logger.warning("Aggregated probability export failed: %s", e)

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✅ EXPECTED RETURNS ANALYTICS v3.0 COMPLETE")
    print("=" * 80)
    print()
    print("  Data sources:")
    print(
        f"    mv_expected_returns:       {len(df):,} stocks × {len(df.columns)} features"
    )
    print(
        f"    mv_all_stock_features:     {len(df_all):,} stocks × {len(df_all.columns)} features"
    )
    print()
    print("  Models:")
    print(f"    Monte Carlo simulations:   {len(mc):,}")
    print(f"    Price target achievements: {len(pt):,}")
    print(f"    Kalman-filtered targets:   {len(kal):,}")
    print(f"    Earnings beat analyses:    {len(beat):,}")
    print(f"    Credit risk analyses:      {len(credit):,}")
    print(f"    Dividend safety analyses:  {len(div_safety):,}")
    print()
    print("  Alignment:")
    print(f"    Tri-model aligned:         {len(tri):,}")
    print(f"    Strong consensus picks:    {len(strong):,}")
    if not quad.empty:
        print(f"    Quad-model full consensus: {(quad['quad_agreement'] == 4).sum()}")
    if not summary.empty:
        full_consensus = (summary["agreement_score"] == 4).sum()
        print(
            f"    Expected returns summary:  {len(summary):,} stocks, {full_consensus} full consensus"
        )
    if corr_info.get("correlation") is not None:
        print(f"    MC ↔ Kalman correlation:   {corr_info['correlation']:.3f}")
    print()
    print("  Screening:")
    for name, screen_df in screens.items():
        if not screen_df.empty:
            print(f"    {name}: {len(screen_df):,} stocks")
    print()
    print("  Probability Analytics:")
    print(f"    Categories analyzed:       {len(category_analytics)}")
    print()
    print(f"  Outputs: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
