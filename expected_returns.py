"""
Expected Returns Analytics Module (v2.5)

Automated pipeline for expected returns analysis using the v2.0+ analytics platform:
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions
- **Price Target Achievement** — Probability-weighted expected returns by sector
- **Kalman Filtered Targets** — Noise-reduced price target signals
- **Analyst Sentiment Features** — Feature-level probability analytics
- **Cross-Model Comparison** — MC vs Kalman vs Achievement model alignment
- **Quad-Model Agreement** — MC + Kalman + Achievement + Earnings Beat

Data sources:
    - public.vw_identifier_columns (identifier coordinates)
    - public.vw_features_analyst_sentiment, vw_features_profitability,
      vw_features_earnings, vw_features_temporal, vw_features_quality_risk,
      vw_features_growth, vw_features_momentum, vw_features_technical_analysis
    - analytics.monte_carlo_simulation
    - analytics.price_target_achievement
    - analytics.kalman_filtered_price_targets
    - analytics.expected_returns_tri_model
    - analytics.strong_consensus_picks
    - analytics.earnings_probability_analysis

Usage:
    python expected_returns.py
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
    load_feature_data_from_db,
    load_all_feature_views,
    load_identifier_columns,
    get_identifier_cols_set,
    backfill_feature_columns,
    export_to_analytics_db,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
    ProbExportPolicy,
    aggregate_probability_results,
)

# --- Statistical analysis ---
from finance_ml.analytics.statistical_analysis import (
    monte_carlo_price_target_simulation,
    kalman_filter_price_target,
    kalman_momentum_filter,
    fit_gaussian_copula,
    bayesian_category_analysis,
)

# --- Probability models ---
from finance_ml.analytics.probability_analytics import (
    EarningsBeatProbabilityModel,
    PriceTargetAchievementModel,
    CreditRiskProbabilityModel,
    EPSStreakAnalyzer,
    CategoryProbabilityAnalyzer,
    create_earnings_probability_dashboard,
    export_probability_analytics_results,
)

# --- Optimised operations ---
from finance_ml.analytics.optimized_ops import (
    fast_monte_carlo_simulation,
    get_optimization_status,
)

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from finance_ml.analytics.inference_schema import (
        ARVIZ_AVAILABLE,
        EquityCoordinates,
        build_monte_carlo_inference_data,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False

# --- Probabilistic visualizations (ArviZ-backed) ---
from finance_ml.analytics.visualizations.probability_viz import (
    create_posterior_return_forest,
    create_beat_probability_posterior,
    create_ruin_probability_diagnostic,
    create_bayesian_category_ridge,
    create_tri_model_posterior_comparison,
)

# --- Other visualizations ---
from finance_ml.analytics.visualizations._shared import PLOTLY_TEMPLATE, COLORS

px.defaults.template = PLOTLY_TEMPLATE

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

# Views to load and merge for expected returns analysis
_EXPECTED_RETURN_VIEWS = [
    "vw_features_analyst_sentiment",
    "vw_features_profitability",
    "vw_features_earnings",
    "vw_features_temporal",
    "vw_features_quality_risk",
    "vw_features_growth",
    "vw_features_momentum",
    "vw_features_technical_analysis",
]


def _discover_market_data_mapping(engine) -> dict[str, str]:
    """
    Query ``public.equities_schema_metadata`` to build an alias → column_name
    mapping for all columns with role = 'market_data'.

    Returns
    -------
    dict[str, str]
        Mapping from ``column_alias`` to ``column_name``.
        Empty dict if none found.
    """
    from sqlalchemy import text

    query = text(
        "SELECT column_alias " "FROM public.equities_schema_metadata " "WHERE role = 'market_data'"
    )
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)

    result = result.dropna(subset=["column_alias"])
    if result.empty:
        return {}
    return dict(zip(result["column_alias"], result["column_alias"]))


def _fetch_backfill_columns(
    engine,
    column_alias: list[str],
    join_col: str,
) -> pd.DataFrame:
    """
    Load the specified columns from ``public.mv_all_stock_features``
    along with *join_col* for merging.
    """
    select_cols = [f'"{join_col}"'] + [f'"{c}"' for c in column_alias]
    query = f"SELECT {', '.join(select_cols)} FROM public.mv_all_stock_features"
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _backfill_market_data_columns(
    df: pd.DataFrame,
    db_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Backfill missing market_data columns from ``public.mv_all_stock_features``
    using ``public.equities_schema_metadata`` to identify which columns have
    role = 'market_data'.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to backfill into.
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing market_data columns filled from equities.
    """
    from sqlalchemy import create_engine

    url = db_url or os.environ.get("DB_URL")
    if not url:
        logger.warning("DB_URL not configured — skipping market_data backfill")
        return df

    try:
        engine = create_engine(url)

        # 1. Discover alias → original column_name mapping from schema metadata
        alias_to_col = _discover_market_data_mapping(engine)
        if not alias_to_col:
            logger.info("No market_data columns found in equities_schema_metadata")
            return df

        # 2. Identify which market_data aliases are missing from df
        missing_aliases = [a for a in alias_to_col if a not in df.columns]
        if not missing_aliases:
            logger.info("All market_data columns already present — no backfill needed")
            return df

        logger.info(
            "Backfilling %d missing market_data columns from public.mv_all_stock_features",
            len(missing_aliases),
        )

        # 3. Map missing aliases back to original equities column names
        missing_original = [alias_to_col[a] for a in missing_aliases]

        # Determine join key (isin preferred, fallback to ticker)
        join_col = "isin" if "isin" in df.columns else "ticker"

        backfill_df = _fetch_backfill_columns(engine, missing_original, join_col)

        # 4. Merge into df
        df = df.merge(backfill_df, on=join_col, how="left", suffixes=("", "_eq"))

        # Drop any unexpected duplicate columns from merge
        duplicate_cols = [c for c in df.columns if c.endswith("_eq")]
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)

        logger.info(
            "Market data backfill complete: %d columns added",
            len(missing_aliases),
        )

    except Exception as e:
        logger.warning("Market data backfill failed: %s", e)

    return df


def load_expected_returns_data(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> pd.DataFrame:
    """
    Load and merge feature views required for expected returns analysis.

    Loads ``vw_identifier_columns`` as the base, then LEFT JOINs each
    feature view via ``isin``, dropping duplicate identifier columns.
    Missing market_data columns are backfilled directly from
    ``public.mv_all_stock_features`` based on ``public.equities_schema_metadata``.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. Falls back to DB_URL env var.
    schema : str, default "public"
        Schema containing the feature views.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with identifier + feature columns.
    """
    df = load_all_feature_views(
        db_url=db_url,
        schema=schema,
        views=_EXPECTED_RETURN_VIEWS,
        return_dict=False,
    )
    if df is not None and not df.empty:
        df = backfill_feature_columns(df)
        df = _backfill_market_data_columns(df, db_url=db_url)
        logger.info(
            "Loaded expected returns data: %d stocks × %d features",
            len(df),
            len(df.columns),
        )
    else:
        logger.warning("No data loaded from feature views")
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
        df = pd.read_sql(f"SELECT * FROM {schema}.{table_name}", engine)
        logger.info("Loaded %d rows from %s.%s", len(df), schema, table_name)
        return df
    except Exception as e:
        logger.warning("Failed to load %s.%s: %s", schema, table_name, e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Runners
# ═══════════════════════════════════════════════════════════════════════════════

_MC_REQUIRED_COLS = ["price_target", "price_target_high", "price_target_low", "last_price"]
_KALMAN_REQUIRED_COLS = ["last_price", "price_target"]


def run_monte_carlo_analysis(
    df: pd.DataFrame,
    n_simulations: int = 10_000,
    max_stocks: int = 10_000,
) -> pd.DataFrame:
    """
    Run Monte Carlo price target simulation on the feature DataFrame.

    Delegates to ``statistical_analysis.monte_carlo_price_target_simulation``
    with required columns: ``price_target``, ``price_target_high``,
    ``price_target_low``, ``last_price``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with price target columns.
    n_simulations : int
        Number of triangular distribution samples per stock.
    max_stocks : int
        Cap on number of stocks to simulate.

    Returns
    -------
    pd.DataFrame
        Monte Carlo results with ``expected_upside_pct``, ``var_5_pct``,
        ``prob_positive_upside``, ``risk_reward_ratio``, etc.
    """
    missing = [c for c in _MC_REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning("MC simulation skipped — missing columns: %s", missing)
        return pd.DataFrame()

    mc = monte_carlo_price_target_simulation(
        df,
        n_simulations=n_simulations,
        max_stocks=max_stocks,
    )
    logger.info("Monte Carlo simulation: %d stocks processed", len(mc))
    return mc


def run_price_target_achievement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate probability of reaching consensus price targets.

    Uses ``PriceTargetAchievementModel`` from probability_analytics.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with analyst sentiment columns.

    Returns
    -------
    pd.DataFrame
        Price target achievement results with ``achievement_probability``,
        ``expected_return_prob_weighted``, ``confidence_level``, etc.
    """
    model = PriceTargetAchievementModel()
    pt = model.analyze_dataframe(df)
    logger.info("Price target achievement: %d stocks processed", len(pt))
    return pt


def run_kalman_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth noisy analyst price targets.

    Delegates to ``statistical_analysis.kalman_filter_price_target``.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with ``last_price`` and ``price_target``.

    Returns
    -------
    pd.DataFrame
        Kalman-filtered results with ``kalman_estimate``, ``filtered_upside``,
        ``signal_strength``, etc.
    """
    missing = [c for c in _KALMAN_REQUIRED_COLS if c not in df.columns]
    if missing:
        logger.warning("Kalman filter skipped — missing columns: %s", missing)
        return pd.DataFrame()

    kal = kalman_filter_price_target(df)
    logger.info("Kalman filter: %d stocks processed", len(kal))
    return kal


def run_earnings_beat_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run enhanced three-layer Bayesian earnings beat probability model.

    Uses ``EarningsBeatProbabilityModel.analyze_dataframe_enhanced()``
    which fuses historical EPS, revision momentum, and GAAP quality layers.
    The model uses ``ReportedEPSHistory`` and ``ForwardEstimateSignals``
    dataclasses to extract quarterly/annual EPS series and forward estimates
    from the feature DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with earnings/EPS columns (from
        ``vw_features_earnings`` and ``vw_features_analyst_sentiment``).

    Returns
    -------
    pd.DataFrame
        Beat probability results with ``posterior_beat_prob``,
        ``posterior_alpha``, ``posterior_beta``, ``confidence_score``,
        ``beat_classification``, ``revision_momentum_score``, etc.
    """
    model = EarningsBeatProbabilityModel()
    sector_col = "sector" if "sector" in df.columns else "industry"
    beat = model.analyze_dataframe_enhanced(df, sector_col=sector_col)
    logger.info("Earnings beat analysis: %d stocks processed", len(beat))
    return beat


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tri-Model & Quad-Model Alignment
# ═══════════════════════════════════════════════════════════════════════════════

_SIGNAL_LABELS = {
    0: "Strong Bearish (0/3)",
    1: "Bearish (1/3)",
    2: "Bullish (2/3)",
    3: "Strong Bullish (3/3)",
}

# Quad-model signal labels (4 models)
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

    Columns retained from each model align with the ``analytics.expected_returns_tri_model``
    table schema exported to the database.

    Parameters
    ----------
    mc : pd.DataFrame
        Monte Carlo simulation results.
    kal : pd.DataFrame
        Kalman-filtered price target results.
    pt : pd.DataFrame
        Price target achievement results.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``agreement_score`` (0–3) and ``signal`` label.
    """
    if mc.empty or kal.empty or pt.empty:
        logger.warning("Tri-model alignment skipped — one or more inputs empty")
        return pd.DataFrame()

    # Select columns for merge — include identifier cols from MC (most complete)
    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]
    mc_select = list(set(mc_id_cols + ["ticker", "expected_upside_pct", "prob_positive_upside"]))

    tri = (
        mc[mc_select]
        .copy()
        .merge(kal[["ticker", "filtered_upside"]], on="ticker", how="inner")
        .merge(
            pt[
                [
                    "ticker",
                    "expected_return_prob_weighted",
                    "achievement_probability",
                    "confidence_level",
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
    """
    Extend tri-model alignment with earnings beat probability for 4-model scoring.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment DataFrame.
    beat : pd.DataFrame
        Earnings beat probability results (from ``analyze_dataframe_enhanced``).
    beat_threshold : float
        Posterior beat probability threshold for bullish classification.

    Returns
    -------
    pd.DataFrame
        Quad-model DataFrame with ``quad_agreement`` (0–4).
    """
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
) -> pd.DataFrame:
    """
    Merge four expected-return model results into a unified summary DataFrame.

    Combines Monte Carlo simulation, Kalman-filtered price targets,
    price target achievement, and earnings probability analysis into a
    single ``expected_returns_summary`` table with direction agreement
    scoring across all four models.

    Parameters
    ----------
    mc : pd.DataFrame
        Monte Carlo simulation results (``analytics.monte_carlo_simulation``).
    kal : pd.DataFrame
        Kalman-filtered price targets (``analytics.kalman_filtered_price_targets``).
    pt : pd.DataFrame
        Price target achievement results (``analytics.price_target_achievement``).
    earn : pd.DataFrame
        Earnings probability analysis (``analytics.earnings_probability_analysis``).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns from each model and:
        - ``mc_bullish``, ``kal_bullish``, ``pt_bullish``, ``earn_bullish``
        - ``agreement_score`` (0–4)
        - ``signal`` (human-readable label)
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

    # Select identifier + key metric columns from MC (most complete identifiers)
    id_cols_set = get_identifier_cols_set()
    mc_id_cols = [c for c in mc.columns if c in id_cols_set]

    # Additional columns to carry through from MC (market data & temporal metadata)
    extra_cols = [
        "trading_country",
        "size_class",
        "style_class",
        "unit",
        "fy_end",
        "next_earnings_report",
        "fy_end_date",
        "income_statement_report_date",
        "last_updated",
        "next_earnings",
        "next_fy_end_date",
        "next_income_statement_report_date",
        "next_earnings_status",
        "next_earnings_when",
        "next_fiscal_quarter",
        "reporting_interval",
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
    available_extra = [c for c in extra_cols if c in mc.columns]

    mc_select = list(
        set(
            mc_id_cols + ["ticker", "expected_upside_pct", "prob_positive_upside"] + available_extra
        )
    )

    summary = (
        mc[mc_select]
        .copy()
        .merge(
            kal[["ticker", "filtered_upside"]],
            on="ticker",
            how="inner",
        )
        .merge(
            pt[
                [
                    "ticker",
                    "expected_return_prob_weighted",
                    "achievement_probability",
                    "confidence_level",
                ]
            ],
            on="ticker",
            how="inner",
        )
        .merge(
            earn[["ticker", "posterior_beat_prob", "confidence_score", "beat_classification"]],
            on="ticker",
            how="inner",
        )
    )

    if summary.empty:
        logger.warning("Expected returns summary: no overlapping tickers across all 4 models")
        return summary

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
    """
    Filter strong consensus picks — all 3 models bullish with high confidence.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment DataFrame.
    min_prob_positive : float
        Minimum MC probability of positive upside (%).
    min_achievement : float
        Minimum price target achievement probability.
    top_n : int
        Number of top picks to return.

    Returns
    -------
    pd.DataFrame
        Strong consensus picks sorted by expected upside.
    """
    if tri.empty:
        return pd.DataFrame()

    strong = tri[
        (tri["agreement_score"] == 3)
        & (tri["prob_positive_upside"] >= min_prob_positive)
        & (tri["achievement_probability"] >= min_achievement)
    ].nlargest(top_n, "expected_upside_pct")

    logger.info("Strong consensus picks: %d stocks", len(strong))
    return strong


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analytical Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def compute_sector_expected_returns(tri: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate expected return metrics by industry sector across all models.

    Parameters
    ----------
    tri : pd.DataFrame
        Tri-model alignment DataFrame.

    Returns
    -------
    pd.DataFrame
        Sector-level aggregated return metrics.
    """
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


def compute_cross_model_correlation(
    mc: pd.DataFrame,
    kal: pd.DataFrame,
) -> dict:
    """
    Compute correlation and copula dependency between MC and Kalman returns.

    Returns
    -------
    dict
        ``correlation``, ``n_stocks``, and optional ``tail_dependence``.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization Functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_mc_return_distribution(mc: pd.DataFrame) -> go.Figure:
    """Two-panel figure: expected upside histogram + P(positive) bar chart."""
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Expected Upside Distribution", "Probability of Positive Return"),
        vertical_spacing=0.12,
    )

    upside = mc["expected_upside_pct"].clip(-100, 300)
    fig.add_trace(
        go.Histogram(
            x=upside, nbinsx=80, marker_color=COLORS[0], opacity=0.75, name="Expected Upside %"
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
        labels={"mean_var5": "Mean VaR 5% (%)", "mean_upside": "Mean Expected Upside (%)"},
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
    sample["raw_log"] = np.sign(sample["raw_upside"]) * np.log1p(np.abs(sample["raw_upside"]))

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
        sample["filtered_log"].abs().quantile(0.99), sample["raw_log"].abs().quantile(0.99)
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
        v: [COLORS[3], COLORS[1], COLORS[0], COLORS[2]][k] for k, v in _SIGNAL_LABELS.items()
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
            text="No sector data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
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
        go.Histogram(x=var_clipped, nbinsx=80, marker_color=COLORS[3], opacity=0.75, name="VaR 5%"),
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
        title="Value-at-Risk (5%) Analysis", template=PLOTLY_TEMPLATE, height=800, showlegend=False
    )
    fig.update_xaxes(title_text="VaR 5% (%)", row=1, col=1)
    fig.update_xaxes(title_text="VaR 5% (%)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Expected Upside (%)", row=2, col=1)
    return fig


def create_beat_vs_achievement_scatter(
    beat: pd.DataFrame,
    pt: pd.DataFrame,
) -> go.Figure:
    """Scatter: P(Beat) vs P(Reach Price Target) coloured by return."""
    if beat.empty or pt.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
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
            text="No overlapping tickers", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
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


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Export Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _reorder_with_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder DataFrame: identifier columns first (from vw_identifier_columns), then the rest."""
    id_cols_all = load_identifier_columns()
    id_present = [c for c in id_cols_all if c in df.columns]
    other = [c for c in df.columns if c not in id_present]
    return df[id_present + other]


def export_expected_returns_results(
    mc: pd.DataFrame,
    pt: pd.DataFrame,
    kal: pd.DataFrame,
    tri: pd.DataFrame,
    strong: pd.DataFrame,
    beat: pd.DataFrame,
    summary: pd.DataFrame = None,
    output_dir: str = "outputs/analytics",
) -> dict[str, str]:
    """
    Export all expected returns analytics to the ``analytics`` schema.

    Parameters
    ----------
    mc, pt, kal, tri, strong, beat, summary : pd.DataFrame
        Model result DataFrames.
    output_dir : str
        Directory for HTML visualization exports.

    Returns
    -------
    dict[str, str]
        Map of table name → export destination.
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
    ]
    for df, table in _EXPORT_PAIRS:
        if df is not None and not df.empty:
            try:
                reordered_df = _reorder_with_identifiers(df)
                cfg = ExportConfig(table_name=table)
                export_to_db(reordered_df, cfg)
                export_to_csv(reordered_df, cfg)
                export_to_json(reordered_df, cfg)
                exports[table] = f"analytics.{table}"
                logger.info("Exported %d rows → analytics.%s", len(df), table)
            except Exception as e:
                logger.warning("Export failed for %s: %s", table, e)

    return exports


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """
    Main expected returns analytics pipeline.

    Steps:
        1. Load feature data from merged views
        2. Run Monte Carlo simulation
        3. Run Price Target Achievement model
        4. Run Kalman filter
        5. Run Earnings Beat analysis
        6. Build tri-model & quad-model alignment
        7. Build expected_returns_summary (4-model merge)
        8. Generate visualizations
        9. Build InferenceData (ArviZ)
        10. Export results
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 80)
    print("Expected Returns Analytics Pipeline")
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
    # 1. DATA LOADING
    # ========================================================================
    print("📦 Step 1: Loading feature data...")
    print("-" * 80)

    df = load_expected_returns_data()
    if df.empty:
        print("✗ No data loaded. Check DB_URL and feature views.")
        return

    print(f"✓ Loaded {len(df):,} stocks × {len(df.columns)} features")
    print()

    # ========================================================================
    # 2. MONTE CARLO SIMULATION
    # ========================================================================
    print("🎲 Step 2: Monte Carlo price target simulation...")
    print("-" * 80)

    mc = run_monte_carlo_analysis(df, n_simulations=10_000)
    if not mc.empty:
        print(f"✓ {len(mc):,} stocks simulated")
        print(f"  Mean upside:  {mc['expected_upside_pct'].mean():.1f}%")
        print(f"  Median upside: {mc['expected_upside_pct'].median():.1f}%")
        print(f"  Positive prob (mean): {mc['prob_positive_upside'].mean():.1f}%")
    print()

    # ========================================================================
    # 3. PRICE TARGET ACHIEVEMENT
    # ========================================================================
    print("🎯 Step 3: Price target achievement model...")
    print("-" * 80)

    pt = run_price_target_achievement(df)
    if not pt.empty:
        print(f"✓ {len(pt):,} stocks analyzed")
        print(f"  Mean achievement prob: {pt['achievement_probability'].mean():.3f}")
        print(f"  Mean prob-weighted return: {pt['expected_return_prob_weighted'].mean():.1f}%")
    print()

    # ========================================================================
    # 4. KALMAN FILTER
    # ========================================================================
    print("📐 Step 4: Kalman-filtered price targets...")
    print("-" * 80)

    kal = run_kalman_filter(df)
    if not kal.empty:
        print(f"✓ {len(kal):,} stocks filtered")
        print(f"  Mean filtered upside: {kal['filtered_upside'].mean():.1f}%")
        if "signal_strength" in kal.columns:
            print(f"  Mean signal strength: {kal['signal_strength'].mean():.2f}")
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

    summary = build_expected_returns_summary(mc, kal, pt, beat)
    if not summary.empty:
        print(f"  ✓ {len(summary):,} stocks in expected_returns_summary")
        full_consensus = (summary["agreement_score"] == 4).sum()
        print(f"  Full consensus (4/4): {full_consensus} stocks")
        print("  Agreement distribution:")
        for label in _SIGNAL_LABELS_4.values():
            cnt = (summary["signal"] == label).sum()
            if cnt > 0:
                print(f"    {label}: {cnt}")
    else:
        print("  ⚠️ Expected returns summary: no overlapping tickers across 4 models")
    print()

    # ========================================================================
    # 8. VISUALIZATIONS
    # ========================================================================
    print("📈 Step 8: Generating visualizations...")
    print("-" * 80)

    try:
        if not mc.empty:
            create_mc_return_distribution(mc).write_html(output_dir / "er_mc_distribution.html")
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
            create_kalman_vs_raw_scatter(kal).write_html(output_dir / "er_kalman_vs_raw.html")
            print("   ✓ er_kalman_vs_raw.html")

        if not tri.empty:
            create_tri_model_agreement_histogram(tri).write_html(
                output_dir / "er_tri_model_agreement.html"
            )
            print("   ✓ er_tri_model_agreement.html")
            create_sector_heatmap(tri).write_html(output_dir / "er_sector_heatmap.html")
            print("   ✓ er_sector_heatmap.html")

        if not strong.empty:
            create_strong_consensus_bar(strong).write_html(output_dir / "er_strong_consensus.html")
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

        # Expected returns summary posterior comparison
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
            results = bayesian_category_analysis(df, "Analyst Sentiment", sentiment_features)
            create_bayesian_category_ridge(results, category_name="Analyst Sentiment").write_html(
                output_dir / "er_bayesian_sentiment_ridge.html"
            )
            print("   ✓ er_bayesian_sentiment_ridge.html")

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
                # Pass source df for EquityCoordinates (includes isin, sector,
                # industry, country, exchange from vw_identifier_columns)
                idata_mc = build_monte_carlo_inference_data(mc, df, n_simulations=10_000)
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
                    df,
                    n_posterior_samples=4000,
                    n_chains=4,
                )
                beat_summary = summarize_inference_data(idata_beat)
                print(
                    f"   ✓ Beat InferenceData: {beat_summary.get('n_chains', 0)} chains × "
                    f"{beat_summary.get('n_draws', 0)} draws"
                )
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
        output_dir=str(output_dir),
    )
    for name, dest in exports.items():
        print(f"   ✓ {name} → {dest}")

    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("✅ EXPECTED RETURNS ANALYTICS COMPLETE")
    print("=" * 80)
    print()
    print(f"  Total stocks loaded:          {len(df):,}")
    print(f"  Monte Carlo simulations:      {len(mc):,}")
    print(f"  Price target achievements:    {len(pt):,}")
    print(f"  Kalman-filtered targets:      {len(kal):,}")
    print(f"  Earnings beat analyses:       {len(beat):,}")
    print(f"  Tri-model aligned:            {len(tri):,}")
    print(f"  Strong consensus picks:       {len(strong):,}")
    if not quad.empty:
        print(f"  Quad-model full consensus:    {(quad['quad_agreement'] == 4).sum()}")
    if not summary.empty:
        full_consensus = (summary["agreement_score"] == 4).sum()
        print(
            f"  Expected returns summary:     {len(summary):,} stocks, {full_consensus} full consensus (4/4)"
        )
    if corr_info.get("correlation") is not None:
        print(f"  MC ↔ Kalman correlation:      {corr_info['correlation']:.3f}")
    print()
    print(f"  Outputs: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
