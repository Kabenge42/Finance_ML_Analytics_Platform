"""
Market Analytics - Refactored Version
This script demonstrates the refactored modular structure for feature analytics.
All functionality has been organized into logical modules:
- data_utils: Data loading, preprocessing, validation
- statistical_analysis: Bayesian, MCMC, Monte Carlo, distribution fitting
- screening: Multi-factor screening and quality scoring
- feature_analytics: Visualization dashboards (existing)
- visualizations: Additional visualization utilities
Usage:
    python market_analytics.py
"""
from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text

# --- Data utilities ---
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics,
    validate_feature_alignment,
    export_to_analytics_db,
    load_identifier_columns,
    get_identifier_cols_set,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
)

# --- Feature analytics dashboards ---
from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_summary_dashboard,
)

# --- Multi-factor screening ---
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum,
    screen_financial_health,
    screen_garp_opportunities,
    screen_high_yield_safe_dividends,
    screen_valuation_reversion_candidates,
    screen_integrity_filtered_growth,
    screen_earnings_quality,
    screen_dividend_quality,
    rank_stocks_by_composite_score,
    create_sector_relative_ranking,
)

# --- Statistical analysis ---
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    calculate_ruin_probability,
    analyze_employee_productivity_frontier,
    detect_accounting_anomalies,
    analyze_reporting_lag_sentiment,
    metropolis_hastings_sampler,
    mcmc_student_t,
    hierarchical_mcmc_by_sector,
    fit_distributions_by_category,
    calculate_conditional_probabilities,
    monte_carlo_price_target_simulation,
    kalman_filter_price_target,
    kalman_momentum_filter,
    fit_gaussian_copula,
    parallel_mcmc_chains,
    run_category_probability_analytics,
    export_probability_view_results,
)

# --- Performance-optimized operations ---
from finance_ml.analytics.optimized_ops import (
    load_feature_data_from_db_cached,
    fast_monte_carlo_simulation,
    fast_ruin_probability,
    vectorized_zscore,
    vectorized_percentile_rank,
    get_optimization_status,
    dataframe_hash,
)

# --- InferenceData schema (ArviZ / xarray bridge) ---
try:
    from finance_ml.analytics.inference_schema import (
        ARVIZ_AVAILABLE,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        build_monte_carlo_inference_data,
        build_category_analysis_inference_data,
        summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False

# --- Probability analytics (optional) ---
try:
    from finance_ml.analytics.probability_analytics import (
        EarningsBeatProbabilityModel,
        CreditRiskProbabilityModel,
        DividendCutProbabilityModel,
        PriceTargetAchievementModel,
        EPSStreakAnalyzer,
        ModelConfidenceEstimator,
        CategoryProbabilityAnalyzer,
        create_earnings_probability_dashboard,
        create_confidence_calibration_chart,
        create_eps_streak_analysis_chart,
        create_view_probability_dashboard,
        export_probability_analytics_results,
    )

    PROBABILITY_ANALYTICS_AVAILABLE = True
except ImportError:
    PROBABILITY_ANALYTICS_AVAILABLE = False
    logging.warning("Probability analytics module not available")

# --- Visualizations: category charts ---
from finance_ml.analytics.visualizations.category_charts import (
    create_analyst_sentiment_histogram,
    create_eps_surprise_histogram,
    create_fcf_margin_yield_scatter,
    create_rnd_intensity_boxplot,
    create_goodwill_concentration_boxplot,
    create_capex_growth_scatter,
    create_valuation_violin_plot,
    create_quality_risk_radar_chart,
    create_leverage_liquidity_bubble_chart,
    create_productivity_quadrant,
    create_accounting_quality_breakdown,
    create_valuation_range_visual,
)

# --- Visualizations: profitability ---
from finance_ml.analytics.visualizations.profitability import (
    create_margin_waterfall_chart,
    create_profitability_quadrant,
)

# --- Visualizations: technical analysis ---
from finance_ml.analytics.visualizations.technical import (
    create_momentum_ribbon_chart,
    create_52w_range_distribution,
)

# --- Visualizations: temporal analysis ---
from finance_ml.analytics.visualizations.temporal_analysis import (
    create_earnings_calendar_heatmap,
    create_dividend_streak_timeline,
)

# --- Visualizations: valuation ---
from finance_ml.analytics.visualizations.valuation import (
    create_valuation_multiples_comparison,
    create_valuation_distribution_dashboard,
    create_relative_valuation_matrix,
    create_valuation_vs_growth_quadrant,
    create_historical_valuation_percentile,
)

# --- Visualizations: earnings quality ---
from finance_ml.analytics.visualizations.earnings_quality import (
    create_earnings_surprise_dashboard,
    create_eps_trajectory_analysis,
    create_earnings_quality_decomposition,
    create_beat_rate_heatmap,
    create_earnings_consistency_matrix,
)

# --- Visualizations: quality & risk ---
from finance_ml.analytics.visualizations.quality_risk import (
    create_piotroski_fscore_breakdown,
    create_altman_zscore_distribution,
    create_quality_risk_quadrant,
    create_beneish_mscore_analysis,
    create_risk_tier_sunburst,
    create_distress_early_warning_dashboard,
)

# --- Visualizations: growth analysis ---
from finance_ml.analytics.visualizations.growth_analysis import (
    create_growth_waterfall_chart,
    create_growth_consistency_matrix,
    create_growth_vs_profitability_quadrant,
    create_growth_acceleration_chart,
    create_sustainable_growth_analysis,
)

# =============================================================================
# Configuration constants
# =============================================================================
_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DB_URL_ENV_VAR = "DB_URL"
_DEFAULT_DB_URL = "postgresql://user:password@localhost:5432/postgres"
_FEATURE_REGISTRY_QUERY = text("""
                               SELECT category, feature_alias
                               FROM public.calculated_features_registry
                               ORDER BY category, feature_alias
                               """)

# =============================================================================
# Global setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.float_format", "{:.2f}".format)
px.defaults.template = PLOTLY_TEMPLATE


# =============================================================================
# Database helpers
# =============================================================================


def _resolve_connection_string(connection_string: str | None) -> str:
    """Return the provided connection string or fall back to the environment / default."""
    if connection_string is not None:
        return connection_string
    return os.environ.get(_DB_URL_ENV_VAR, _DEFAULT_DB_URL)


def _fetch_feature_categories(connection_string: str) -> dict[str, list[str]]:
    """
    Query the feature registry and return a mapping of category → feature aliases.

    Raises on any database error so the caller can decide how to handle it.
    """
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        rows = conn.execute(_FEATURE_REGISTRY_QUERY).fetchall()

    categories: dict[str, list[str]] = defaultdict(list)
    for category, feature_alias in rows:
        categories[category].append(feature_alias)

    logging.info("Loaded %d feature categories from database", len(categories))
    return dict(categories)


def load_feature_categories(connection_string: str | None = None) -> dict[str, list[str]]:
    """
    Load feature categories from the calculated_features_registry table.

    Falls back to hardcoded defaults when the database is unreachable.

    Returns:
        Dictionary mapping category names to lists of feature aliases.
    """
    resolved_url = _resolve_connection_string(connection_string)
    try:
        return _fetch_feature_categories(resolved_url)
    except Exception as exc:
        logging.warning("Could not load categories from database: %s", exc)
        logging.warning("Falling back to hardcoded FEATURE_CATEGORIES")
        return _get_fallback_feature_categories()


def _get_fallback_feature_categories() -> dict[str, list[str]]:
    """Fallback hardcoded categories if database is unavailable."""
    return {
        "Valuation Ratios": [
            "p_e_ratio",
            "p_b_ratio",
            "ev_ebitda_ratio",
            "ev_sales_ratio",
            "dividend_yield",
            "peg_ratio",
            "price_to_tangible_book",
            "tangible_book_value_ltm",
        ],
        "Momentum & Technical": [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "price_momentum_1y",
            "price_momentum_3y",
            "price_momentum_5y",
            "range_52w_position",
            "long_term_trend_score",
            "secular_trend_flag",
        ],
        "Profitability": [
            "roe",
            "roa",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "ebitda_margin_pct",
            "roic",
            "net_margin_trend_yoy",
        ],
        "Quality & Risk": [
            "piotroski_f_score",
            "distress_risk_score",
            "altman_z_score",
            "accounting_quality_score",
            "earnings_quality_composite",
            "cash_flow_quality_score",
            "beta_stability_score",
        ],
        "Leverage & Liquidity": [
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "interest_coverage_ratio",
            "cash_ratio",
            "working_capital_ratio",
            "debt_deleveraging",
        ],
        "Analyst Sentiment": [
            "analyst_bullish_pct",
            "analyst_neutral_pct",
            "analyst_bearish_pct",
            "upside_potential",
            "analyst_rating_normalized",
            "eps_revision_momentum",
        ],
        "Earnings Quality": [
            "eps_surprise_pct",
            "eps_adjustment_ratio",
            "gaap_adj_eps_gap_pct",
            "eps_trajectory_score",
            "earnings_quality_score",
            "gaap_revision_momentum",
        ],
        "Growth Metrics": [
            "revenue_yoy_growth",
            "ebitda_growth_yoy",
            "eps_yoy_growth",
            "fcf_growth_yoy",
            "revenue_cagr_5y",
        ],
        "Cash Flow": [
            "fcf_positive_years",
            "fcf_margin",
            "fcf_yield",
            "cfo_to_net_income",
            "self_funding_ratio",
            "cash_flow_quality_score",
        ],
        "Dividend Features": [
            "dividend_streak",
            "dividend_yield_ltm",
            "dividend_payout_ratio",
            "fcf_dividend_coverage",
            "total_shareholder_yield",
        ],
        "R&D Investment": [
            "rnd_intensity_ltm",
            "rnd_yoy_growth",
            "rnd_per_employee",
            "high_rnd_intensity_flag",
        ],
        "Inventory Temporal": [
            "inventory_days",
            "inventory_turnover_itf",
            "inventory_yoy_change",
            "inventory_buildup_flag",
        ],
        "Goodwill & M&A": [
            "goodwill_concentration",
            "goodwill_3y_growth",
            "recent_acquisition_flag",
            "impairment_risk_score",
        ],
        "CapEx & Investment": [
            "capex_yoy_growth",
            "capex_vs_5y_avg",
            "acquisitions_ltm_total",
            "ma_intensity_score",
            "investment_efficiency",
        ],
    }


def compare_registry_with_local(
    db_categories: dict[str, list[str]], local_categories: dict[str, list[str]]
) -> dict:
    """
    Compare database registry with local/fallback categories.
    Useful for identifying missing features or new additions.
    """
    report = {
        "categories_only_in_db": [],
        "categories_only_in_local": [],
        "features_only_in_db": {},
        "features_only_in_local": {},
    }

    db_cats = set(db_categories.keys())
    local_cats = set(local_categories.keys())

    report["categories_only_in_db"] = list(db_cats - local_cats)
    report["categories_only_in_local"] = list(local_cats - db_cats)

    for cat in db_cats & local_cats:
        db_features = set(db_categories[cat])
        local_features = set(local_categories[cat])

        only_in_db = db_features - local_features
        only_in_local = local_features - db_features

        if only_in_db:
            report["features_only_in_db"][cat] = list(only_in_db)
        if only_in_local:
            report["features_only_in_local"][cat] = list(only_in_local)

    return report


# Load categories dynamically at module level (with fallback)
FEATURE_CATEGORIES = load_feature_categories()


# NOTE: load_feature_data_from_db, backfill_feature_columns, safe_get_column,
# load_feature_categories_from_db, _get_fallback_feature_categories, and
# compare_registry_with_local are imported from data_utils.py (canonical source).


def ensure_subplot_data(
    fig: go.Figure, row: int, col: int, has_data: bool, placeholder_text: str = "No data available"
) -> None:
    """
    Add placeholder annotation if subplot has no data.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to modify
    row : int
        Subplot row (1-indexed)
    col : int
        Subplot column (1-indexed)
    has_data : bool
        Whether the subplot has valid data
    placeholder_text : str
        Text to display if no data
    """
    if not has_data:
        # Calculate approximate position for annotation
        x_pos = (col - 0.5) / 2
        y_pos = 1 - (row - 0.5) / 2
        fig.add_annotation(
            text=placeholder_text,
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
        )


def create_interactive_momentum_dashboard(df: pd.DataFrame) -> Figure:
    """
    Create an interactive momentum analysis dashboard with hover details.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing momentum columns:
        - price_momentum_1m, price_momentum_3m, price_momentum_6m, price_momentum_1y
        - price_momentum_5d, price_momentum_3y, price_momentum_5y, pt_vs_price_momentum
        - range_52w_position
        - ticker, name, industry

    Returns
    -------
    Figure
        Plotly Figure with 4 subplot panels:
        1. Momentum distribution by period (all 8 momentum columns)
        2. 3-Month momentum by industry
        3. 52-Week range position
        4. Short vs medium-term momentum scatter
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Momentum Distribution by Period",
            "3-Month Momentum by Industry",
            "52-Week Range Position",
            "Short vs Medium-Term Momentum",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "box"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # All available momentum columns from mv_all_stock_features
    momentum_cols = [
        "price_momentum_5d",
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_momentum_1y",
        "price_momentum_3y",
        "price_momentum_5y",
        "pt_vs_price_momentum",
    ]
    colors = [
        "#1abc9c",  # 5-Day - teal
        "#3498db",  # 1-Month - blue
        "#e74c3c",  # 3-Month - red
        "#2ecc71",  # 6-Month - green
        "#9b59b6",  # 1-Year - purple
        "#f39c12",  # 3-Year - orange
        "#e91e63",  # 5-Year - pink
        "#00bcd4",  # PT vs Price - cyan
    ]
    labels = [
        "5-Day",
        "1-Month",
        "3-Month",
        "6-Month",
        "1-Year",
        "3-Year",
        "5-Year",
        "PT vs Price",
    ]

    # Panel 1: Overlaid momentum histograms
    panel1_has_data = False
    for col, color, label in zip(momentum_cols, colors, labels):
        if col in df.columns:
            data = df[col].dropna().clip(-50, 100)
            if len(data) > 0:
                panel1_has_data = True
                fig.add_trace(
                    go.Histogram(x=data, name=label, marker_color=color, opacity=0.6, nbinsx=50),
                    row=1,
                    col=1,
                )

    # Add placeholder if no data for panel 1
    if not panel1_has_data:
        fig.add_trace(
            go.Histogram(x=[0], name="No Data", marker_color="#adb5bd", opacity=0.3),
            row=1,
            col=1,
        )
        fig.add_annotation(
            text="No momentum data available",
            x=0.25,
            y=0.75,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )

    # Panel 2: Box plot by industry (interactive)
    panel2_has_data = False
    if "industry" in df.columns and "price_momentum_3m" in df.columns:
        valid_data = df.dropna(subset=["industry", "price_momentum_3m"])
        if len(valid_data) > 0:
            panel2_has_data = True
            fig.add_trace(
                go.Box(
                    x=valid_data["industry"],
                    y=valid_data["price_momentum_3m"].clip(-50, 100),
                    marker_color="#3498db",
                    name="3M Momentum",
                    boxpoints="outliers",
                ),
                row=1,
                col=2,
            )

    if not panel2_has_data:
        fig.add_trace(
            go.Box(x=["N/A"], y=[0], marker_color="#adb5bd", name="No Data"),
            row=1,
            col=2,
        )

    # Panel 3: 52-week range position
    panel3_has_data = False
    if "range_52w_position" in df.columns:
        range_data = df["range_52w_position"].dropna()
        if len(range_data) > 0:
            panel3_has_data = True
            fig.add_trace(
                go.Histogram(
                    x=range_data,
                    nbinsx=30,
                    marker_color="#9b59b6",
                    name="52W Position",
                    hovertemplate="Position: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_vline(
                x=range_data.median(),
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"Median: {range_data.median():.2f}",
                row=2,
                col=1,
            )

    if not panel3_has_data:
        fig.add_trace(
            go.Histogram(x=[0.5], name="No Data", marker_color="#adb5bd", opacity=0.3),
            row=2,
            col=1,
        )

    # Panel 4: Scatter with hover details
    panel4_has_data = False
    if "price_momentum_1m" in df.columns and "price_momentum_6m" in df.columns:
        valid_mask = df["price_momentum_1m"].notna() & df["price_momentum_6m"].notna()
        scatter_cols = ["price_momentum_1m", "price_momentum_6m"]
        optional_cols = ["ticker", "name", "industry"]
        available_cols = scatter_cols + [c for c in optional_cols if c in df.columns]

        scatter_df = df.loc[valid_mask, available_cols].copy()
        scatter_df["price_momentum_1m"] = scatter_df["price_momentum_1m"].clip(-50, 100)
        scatter_df["price_momentum_6m"] = scatter_df["price_momentum_6m"].clip(-50, 200)

        if len(scatter_df) > 0:
            panel4_has_data = True
            # Build hover template dynamically based on available columns
            hover_parts = []
            customdata_cols = []
            if "ticker" in scatter_df.columns:
                hover_parts.append("<b>%{text}</b>")
            if "name" in scatter_df.columns:
                hover_parts.append("%{customdata[0]}")
                customdata_cols.append("name")
            if "industry" in scatter_df.columns:
                idx = len(customdata_cols)
                hover_parts.append(f"Industry: %{{customdata[{idx}]}}")
                customdata_cols.append("industry")
            hover_parts.extend(["1M: %{x:.1f}%", "6M: %{y:.1f}%"])

            customdata = (
                np.stack([scatter_df[c] for c in customdata_cols], axis=-1)
                if customdata_cols
                else None
            )

            fig.add_trace(
                go.Scatter(
                    x=scatter_df["price_momentum_1m"],
                    y=scatter_df["price_momentum_6m"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.4, color="#3498db"),
                    text=scatter_df.get("ticker", None),
                    customdata=customdata,
                    hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
                    name="Stocks",
                ),
                row=2,
                col=2,
            )

            # Reference lines for scatter
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=2)

    if not panel4_has_data:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[0], mode="markers", marker=dict(size=10, color="#adb5bd"), name="No Data"
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        title_text="📈 Interactive Momentum Analysis Dashboard",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Momentum (%)", row=1, col=1)
    fig.update_xaxes(title_text="Industry", tickangle=-45, row=1, col=2)
    fig.update_xaxes(title_text="52W Range Position", row=2, col=1)
    fig.update_xaxes(title_text="1-Month Momentum (%)", row=2, col=2)
    fig.update_yaxes(title_text="6-Month Momentum (%)", row=2, col=2)

    return fig


def create_interactive_valuation_heatmap(df: pd.DataFrame) -> Figure:
    """
    Create an interactive valuation heatmap with click-to-filter capability.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing valuation columns:
        - p_e_ratio, p_b_ratio, ev_ebitda_ratio, ev_sales_ratio
        - industry

    Returns
    -------
    Figure
        Plotly Figure with heatmap showing median valuation metrics by industry
    """
    valuation_cols = ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "ev_sales_ratio"]
    val_labels = ["P/E", "P/B", "EV/EBITDA", "EV/Sales"]

    # Filter out null industries
    df_filtered = df[df["industry"].notna()] if "industry" in df.columns else df
    sectors = sorted(df_filtered["industry"].dropna().unique())

    # Build heatmap data
    heatmap_data = []
    hover_text = []

    for sector in sectors:
        sector_df = df_filtered[df_filtered["industry"] == sector]
        row_vals = []
        row_hover = []
        # Computes median, IQR, and count for each sector
        for col, label in zip(valuation_cols, val_labels):
            if col in sector_df.columns:
                median_val = sector_df[col].median()
                count = sector_df[col].notna().sum()
                q25 = sector_df[col].quantile(0.25)
                q75 = sector_df[col].quantile(0.75)
            else:
                median_val = 0
                count = 0
                q25 = 0
                q75 = 0
            row_vals.append(median_val if pd.notna(median_val) else 0)
            row_hover.append(
                f"<b>{sector}</b><br>{label}: {median_val:.1f}<br>"
                + f"IQR: [{q25:.1f}, {q75:.1f}]<br>N={count}"
            )
        heatmap_data.append(row_vals)
        hover_text.append(row_hover)

    heatmap_array = np.array(heatmap_data)

    # Normalize for color scale
    min_vals = heatmap_array.min(axis=0)
    max_vals = heatmap_array.max(axis=0)
    heatmap_norm = (heatmap_array - min_vals) / (max_vals - min_vals + 1e-10)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_norm,
            x=val_labels,
            y=sectors,
            text=np.round(heatmap_array, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            colorscale="RdYlGn_r",
            colorbar=dict(title="Relative<br>Valuation"),
        )
    )

    fig.update_layout(
        title="📊 Median Valuation Metrics by Industry<br><sup>Green=Cheaper, Red=Expensive (Normalized)</sup>",
        height=max(600, len(sectors) * 25),
        template=PLOTLY_TEMPLATE,
        xaxis_title="Valuation Metric",
        yaxis_title="Industry",
    )

    return fig


def create_leverage_liquidity_quadrant(df: pd.DataFrame) -> Figure:
    """
    Interactive quadrant analysis of leverage vs liquidity with distress coloring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - debt_to_equity, current_ratio, distress_risk_score
        - ticker, name, industry

    Returns
    -------
    Figure
        Plotly Figure with scatter plot showing leverage vs liquidity quadrants
    """
    required_cols = [
        "ticker",
        "name",
        "industry",
        "debt_to_equity",
        "current_ratio",
        "distress_risk_score",
    ]
    available_cols = [c for c in required_cols if c in df.columns]

    plot_df = df[available_cols].dropna().copy()

    if "debt_to_equity" in plot_df.columns:
        plot_df["debt_to_equity"] = plot_df["debt_to_equity"].clip(0, 3)
    if "current_ratio" in plot_df.columns:
        plot_df["current_ratio"] = plot_df["current_ratio"].clip(0, 5)

    fig = px.scatter(
        plot_df,
        x="debt_to_equity",
        y="current_ratio",
        color="distress_risk_score" if "distress_risk_score" in plot_df.columns else None,
        color_continuous_scale="RdYlGn",
        hover_data=(
            ["ticker", "name", "industry"]
            if all(c in plot_df.columns for c in ["ticker", "name", "industry"])
            else None
        ),
        title="📉 Leverage vs Liquidity Quadrant Analysis",
        labels={
            "debt_to_equity": "Debt-to-Equity Ratio",
            "current_ratio": "Current Ratio (Liquidity)",
            "distress_risk_score": "Distress Risk Score",
        },
        height=650,
    )

    # Add quadrant lines
    fig.add_hline(
        y=1.5, line_dash="dash", line_color="#2ecc71", annotation_text="Healthy Liquidity (CR=1.5)"
    )
    fig.add_vline(
        x=1.0, line_dash="dash", line_color="#e74c3c", annotation_text="High Leverage (D/E=1)"
    )

    # Add quadrant labels
    fig.add_annotation(
        x=0.3, y=4.5, text="✅ Low Risk", showarrow=False, font=dict(size=14, color="#2ecc71")
    )
    fig.add_annotation(
        x=2.5, y=0.5, text="⚠️ High Risk", showarrow=False, font=dict(size=14, color="#e74c3c")
    )
    fig.add_annotation(
        x=2.5, y=4.5, text="🔄 Mixed", showarrow=False, font=dict(size=12, color="#f39c12")
    )
    fig.add_annotation(
        x=0.3, y=0.5, text="💧 Illiquid", showarrow=False, font=dict(size=12, color="#3498db")
    )

    fig.update_traces(marker=dict(size=6, opacity=0.6))
    fig.update_layout(template=PLOTLY_TEMPLATE)

    return fig


def monte_carlo_price_target_simulation(
    df: pd.DataFrame,
    n_simulations: int = 10000,
    confidence_level: float = 0.99,
    max_stocks: int = 10000,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of price targets based on analyst spread.

    Uses the analyst price target range (high/low/median) to model
    uncertainty and generate probabilistic fair value estimates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - ticker, name, industry, last_price
        - price_target, price_target_high, price_target_low, price_target_median
    n_simulations : int, default 10000
        Number of Monte Carlo simulations per stock
    confidence_level : float, default 0.95
        Confidence level for VaR calculation
    max_stocks : int, default 2000
        Maximum number of stocks to simulate (for performance)
    confidence_level : float, default 0.95
        Confidence level for VaR calculation
    max_stocks : int, default 1000
        Maximum number of stocks to simulate (for performance)

    Returns
    -------
    pd.DataFrame
        DataFrame with simulation results including:
        - ticker, name, industry, last_price
        - expected_upside_pct, upside_std, var_5_pct
        - prob_positive_upside, risk_reward_ratio
    """
    np.random.seed(42)

    results = []

    required_cols = ["price_target", "price_target_high", "price_target_low", "last_price"]
    valid_df = df.dropna(subset=required_cols)

    for _, row in valid_df.head(max_stocks).iterrows():
        pt_low = row["price_target_low"]
        pt_high = row["price_target_high"]
        pt_median = row.get("price_target_median", row["price_target"])
        last_price = row["last_price"]

        if pt_high <= pt_low or last_price <= 0:
            continue

        # Model price target as triangular distribution (low, mode=median, high)
        simulated_pts = np.random.triangular(pt_low, pt_median, pt_high, n_simulations)

        # Calculate simulated upside
        simulated_upside = (simulated_pts - last_price) / last_price * 100

        # Statistics
        var_5 = np.percentile(simulated_upside, 5)
        expected_upside = np.mean(simulated_upside)
        upside_std = np.std(simulated_upside)
        prob_positive = (simulated_upside > 0).mean() * 100

        results.append(
            {
                "ticker": row.get("ticker", ""),
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "region": row.get("region", ""),
                "country": row.get("country", ""),
                "exchange": row.get("exchange", ""),
                "last_price": last_price,
                "pt_median": pt_median,
                "pt_spread": pt_high - pt_low,
                "expected_upside_pct": expected_upside,
                "upside_std": upside_std,
                "var_5_pct": var_5,  # 5% Value at Risk
                "prob_positive_upside": prob_positive,
                "risk_reward_ratio": expected_upside / upside_std if upside_std > 0 else 0,
            }
        )

    return pd.DataFrame(results)


def bayesian_earnings_beat_model(df: pd.DataFrame, n_total: int = 5) -> pd.DataFrame:
    """
    Bayesian model for earnings beat probability.

    Uses EPS positive streak as prior evidence and updates posterior
    based on recent performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - ticker, name, industry
        - eps_positive_streak (number of positive quarters in last n_total)
    n_total : int, default 5
        Total number of quarters in the observation window

    Returns
    -------
    pd.DataFrame
        DataFrame with Bayesian model results:
        - ticker, name, industry, eps_positive_streak
        - posterior_beat_prob, model_confidence, map_estimate
    """
    # Prior: Uniform belief across probability grid
    p_grid = np.arange(0.1, 1.0, 0.1)  # 9 parameter values
    uniform_prior = 1 / len(p_grid)

    results = []

    streak_col = "eps_positive_streak"
    if streak_col not in df.columns:
        return pd.DataFrame()

    for _, row in df.dropna(subset=[streak_col]).iterrows():
        n_beats = int(row[streak_col])
        n_beats = min(n_beats, n_total)  # Cap at n_total

        # Compute likelihood: P(data | p) = p^k * (1-p)^(n-k)
        likelihoods = p_grid**n_beats * (1 - p_grid) ** (n_total - n_beats)

        # Unnormalized posterior
        posterior_unnorm = uniform_prior * likelihoods

        # Normalize
        posterior = posterior_unnorm / posterior_unnorm.sum()

        # Posterior predictive: P(beat next quarter) = sum(p * posterior(p))
        prob_beat_next = np.sum(p_grid * posterior)

        # Confidence (inverse entropy proxy)
        entropy = -np.sum(posterior * np.log(posterior + 1e-10))
        confidence = 1 - entropy / np.log(len(p_grid))

        results.append(
            {
                "ticker": row.get("ticker", ""),
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "industry": row.get("industry", ""),
                "region": row.get("region", ""),
                "country": row.get("country", ""),
                "exchange": row.get("exchange", ""),
                "eps_positive_streak": n_beats,
                "posterior_beat_prob": prob_beat_next,
                "model_confidence": confidence,
                "map_estimate": p_grid[np.argmax(posterior)],  # Maximum a posteriori
            }
        )

    return pd.DataFrame(results)


def analyze_distress_distribution(df: pd.DataFrame) -> Figure:
    """
    Analyze distress risk score distribution with tail risk metrics.

    Uses concepts from MCMC sampling to understand distribution shape.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        - distress_risk_score
        - industry

    Returns
    -------
    Figure
        Plotly Figure with 4 panels:
        1. Distress risk score distribution with fitted normal
        2. Empirical CDF
        3. Q-Q plot vs normal
        4. Tail risk by industry
    """
    distress_data = df["distress_risk_score"].dropna()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Distress Risk Score Distribution",
            "Empirical CDF",
            "Q-Q Plot vs Normal",
            "Tail Risk by Industry",
        ],
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # Panel 1: Histogram with fitted distribution
    fig.add_trace(
        go.Histogram(
            x=distress_data,
            nbinsx=50,
            name="Observed",
            marker_color="#3498db",
            opacity=0.7,
            histnorm="probability density",
        ),
        row=1,
        col=1,
    )

    # Fit normal for comparison
    mu, std = distress_data.mean(), distress_data.std()
    x_range = np.linspace(0, 100, 100)
    normal_pdf = stats.norm.pdf(x_range, mu, std)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=normal_pdf,
            mode="lines",
            name="Normal Fit",
            line=dict(color="#e74c3c", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Panel 2: Empirical CDF
    sorted_data = np.sort(distress_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fig.add_trace(
        go.Scatter(x=sorted_data, y=ecdf, mode="lines", name="ECDF", line=dict(color="#00bc8c")),
        row=1,
        col=2,
    )
    # Add risk thresholds
    fig.add_vline(
        x=30, line_dash="dot", line_color="#e74c3c", row=1, col=2, annotation_text="High Risk (<30)"
    )
    fig.add_vline(
        x=70, line_dash="dot", line_color="#2ecc71", row=1, col=2, annotation_text="Low Risk (>70)"
    )

    # Panel 3: Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
    empirical_quantiles = np.percentile(distress_data, np.linspace(1, 99, 100))
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode="markers",
            marker=dict(size=4, color="#9b59b6"),
            name="Q-Q",
        ),
        row=2,
        col=1,
    )
    # Reference line
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[mu - 3 * std, mu + 3 * std],
            mode="lines",
            line=dict(dash="dash", color="white"),
            name="Normal Ref",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Tail risk by industry (% below 30)
    if "industry" in df.columns:
        tail_risk = (
            df.groupby("industry")
            .apply(lambda x: (x["distress_risk_score"] < 30).mean() * 100, include_groups=False)
            .sort_values(ascending=False)
        )

        fig.add_trace(
            go.Bar(
                x=tail_risk.values[:15],
                y=tail_risk.index[:15],
                orientation="h",
                marker_color="#e74c3c",
                name="High Risk %",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        title_text="📉 Financial Distress Risk Distribution Analysis",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
    )

    # Summary statistics annotation
    var_5 = distress_data.quantile(0.05)
    var_1 = distress_data.quantile(0.01)
    high_risk_pct = (distress_data < 30).mean() * 100

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"<b>Risk Metrics</b><br>"
        + f"VaR(5%): {var_5:.1f}<br>"
        + f"VaR(1%): {var_1:.1f}<br>"
        + f"High Risk (<30): {high_risk_pct:.1f}%",
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="white",
        borderwidth=1,
    )

    return fig


def create_composite_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite quality score combining multiple factors with probabilistic normalization.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing quality factor columns:
        - piotroski_f_score, earnings_quality_composite
        - cash_flow_quality_score, distress_risk_score
        - accounting_quality_score, dilution_score
        - beta_stability_score, long_term_trend_score, eps_trajectory_score

    Returns
    -------
    pd.DataFrame
        DataFrame with composite scores:
        - ticker, name, industry, market_cap
        - composite_quality_score (0-100)
        - quality_tier (Low, Below Avg, Above Avg, High)
        Sorted by composite_quality_score descending
    """

    # Define factor weights by category
    factor_weights = {
        "piotroski_f_score": 0.15,
        "earnings_quality_composite": 0.15,
        "cash_flow_quality_score": 0.12,
        "distress_risk_score": 0.12,
        "accounting_quality_score": 0.10,
        "dilution_score": 0.08,
        "beta_stability_score": 0.08,
        "long_term_trend_score": 0.10,
        "eps_trajectory_score": 0.10,
    }

    # Select base columns
    base_cols = [
        "ticker",
        "name",
        "sector",
        "industry",
        "region",
        "country",
        "exchange",
        "market_cap",
        "enterprise_value",
        "last_price",
        "price_target",
        "piotroski_f_score",
        "earnings_quality_composite",
        "cash_flow_quality_score",
        "distress_risk_score",
        "accounting_quality_score",
        "dilution_score",
        "beta_stability_score",
        "long_term_trend_score",
        "eps_trajectory_score",
    ]
    available_base = [c for c in base_cols if c in df.columns]
    result_df = df[available_base].copy()

    # Normalize each factor to 0-100 percentile rank
    for factor, weight in factor_weights.items():
        if factor in df.columns:
            # Percentile rank (0-100)
            result_df[f"{factor}_pctl"] = df[factor].rank(pct=True) * 100
        else:
            result_df[f"{factor}_pctl"] = 50  # Neutral if missing

    # Compute weighted composite score
    composite = np.zeros(len(result_df))
    total_weight = 0

    for factor, weight in factor_weights.items():
        pctl_col = f"{factor}_pctl"
        if pctl_col in result_df.columns:
            valid_mask = result_df[pctl_col].notna()
            composite[valid_mask] += result_df.loc[valid_mask, pctl_col] * weight
            total_weight += weight

    result_df["composite_quality_score"] = (
        composite / total_weight if total_weight > 0 else composite
    )

    # Add probability interpretation
    result_df["quality_tier"] = pd.cut(
        result_df["composite_quality_score"],
        bins=[0, 30, 50, 70, 100],
        labels=["Low", "Below Avg", "Above Avg", "High"],
    )

    # Drop percentile columns for cleaner output
    pctl_cols = [c for c in result_df.columns if c.endswith("_pctl")]
    result_df = result_df.drop(columns=pctl_cols)

    return result_df.sort_values("composite_quality_score", ascending=False).reset_index(drop=True)


def create_summary_dashboard(df: pd.DataFrame) -> Figure:
    """
    Create a KPI summary dashboard using Plotly indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing various financial metrics

    Returns
    -------
    Figure
        Plotly Figure with 8 KPI indicator panels
    """
    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[[{"type": "indicator"}] * 4, [{"type": "indicator"}] * 4],
        subplot_titles=[
            "Total Stocks",
            "Avg P/E",
            "Median Upside",
            "High Quality %",
            "Profitable %",
            "Strong F-Score",
            "Bullish Sentiment",
            "Low Distress %",
        ],
    )

    # Row 1 indicators
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=len(df),
            number={"suffix": "", "font": {"size": 40}},
            title={"text": "Stocks Analyzed"},
        ),
        row=1,
        col=1,
    )

    pe_median = df["p_e_ratio"].median() if "p_e_ratio" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=pe_median,
            number={"suffix": "x", "font": {"size": 40}},
            title={"text": "Median P/E"},
        ),
        row=1,
        col=2,
    )

    upside_median = df["upside_potential"].median() if "upside_potential" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=upside_median,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Median Upside"},
        ),
        row=1,
        col=3,
    )

    high_quality_pct = (
        (df["earnings_quality_composite"] > 70).mean() * 100
        if "earnings_quality_composite" in df.columns
        else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=high_quality_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "High Quality"},
        ),
        row=1,
        col=4,
    )

    # Row 2 indicators
    profitable_pct = (
        (df["net_margin_pct"] > 0).mean() * 100 if "net_margin_pct" in df.columns else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=profitable_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Profitable"},
        ),
        row=2,
        col=1,
    )

    strong_fscore_pct = (
        (df["piotroski_f_score"] >= 7).mean() * 100 if "piotroski_f_score" in df.columns else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=strong_fscore_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Strong F-Score"},
        ),
        row=2,
        col=2,
    )

    bullish_avg = df["analyst_bullish_pct"].mean() if "analyst_bullish_pct" in df.columns else 0
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=bullish_avg,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Avg Bullish %"},
        ),
        row=2,
        col=3,
    )

    low_distress_pct = (
        (df["distress_risk_score"] >= 70).mean() * 100 if "distress_risk_score" in df.columns else 0
    )
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=low_distress_pct,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Low Distress"},
        ),
        row=2,
        col=4,
    )

    fig.update_layout(
        height=400,
        title_text="📊 Feature Analytics Summary Dashboard",
        template=PLOTLY_TEMPLATE,
    )

    return fig


def main():
    """Main execution function demonstrating refactored modules."""

    print("=" * 80)
    print("MARKET ANALYTICS - REFACTORED VERSION")
    print("=" * 80)
    print()

    # ========================================================================
    # 0. OPTIMIZATION STATUS
    # ========================================================================
    opt_status = get_optimization_status()
    print("⚡ Optimization Status:")
    print(
        f"   Numba JIT:      {'✓ Available' if opt_status['numba_available'] else '✗ Fallback mode'}"
    )
    print(
        f"   Parallel:       {'✓ Available' if opt_status['parallel_available'] else '✗ Sequential'}"
    )
    print(
        f"   ArviZ:          {'✓ Available' if opt_status.get('arviz_available') else '✗ Not installed'}"
    )
    print(f"   DB Cache:       {opt_status['db_cache_size']} entries")
    print(f"   Stats Cache:    {opt_status['stats_cache_size']} entries")
    print()

    # ========================================================================
    # 1. DATA LOADING AND PREPROCESSING
    # ========================================================================
    print("📊 Step 1: Loading and preprocessing data...")
    print("-" * 80)

    try:
        # Use cached loader for performance (subsequent runs skip DB round-trip)
        df = load_feature_data_from_db_cached()
        print(f"✓ Loaded {len(df):,} stocks with {len(df.columns)} features")
        print(f"   DataFrame hash: {dataframe_hash(df)[:12]}...")

        # Backfill missing columns
        df = backfill_feature_columns(df)
        print(f"✓ Backfilled features, now have {len(df.columns)} columns")

        # Compare registry with fallback to identify drift
        fallback = _get_fallback_feature_categories()
        diff_report = compare_registry_with_local(FEATURE_CATEGORIES, fallback)
        if diff_report["features_only_in_db"]:
            print("   📌 New features in registry (not in fallback):")
            for cat, feats in diff_report["features_only_in_db"].items():
                print(f"      {cat}: {feats}")

    except Exception as e:
        print(f"⚠️  Could not load from database: {e}")
        print("   Using sample data for demonstration...")
        # Create sample data for demonstration
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "name": [
                    "Apple Inc.",
                    "Microsoft Corp.",
                    "Alphabet Inc.",
                    "Amazon.com Inc.",
                    "Tesla Inc.",
                ],
                "industry": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Consumer",
                    "Automotive",
                ],
                "market_cap": [3000000, 2800000, 1800000, 1600000, 800000],
                "p_e_ratio": [28.5, 32.1, 25.3, 45.2, 65.8],
                "piotroski_f_score": [8, 7, 8, 6, 5],
                "distress_risk_score": [85, 82, 88, 75, 60],
                "eps_trajectory_score": [75, 80, 78, 65, 55],
                "fcf_positive_years": [5, 5, 5, 4, 3],
            }
        )

    print()

    # ========================================================================
    # 2. FEATURE VALIDATION
    # ========================================================================
    print("📋 Step 2: Validating feature coverage...")
    print("-" * 80)

    validation = validate_feature_alignment(df, FEATURE_CATEGORIES)
    low_coverage = {k: v for k, v in validation.items() if v["coverage_pct"] < 80}

    if low_coverage:
        print("⚠️  Categories with <80% feature coverage:")
        for cat, info in low_coverage.items():
            print(f"   {cat}: {info['coverage_pct']:.1f}%")
    else:
        print("✓ All feature categories have ≥80% coverage")

    print()

    # ========================================================================
    # 3. FEATURE VIEW ANALYTICS
    # ========================================================================
    print("📊 Step 3: Loading all feature views...")
    print("-" * 80)

    from finance_ml.analytics.data_utils import (
        load_all_feature_views,
        get_view_category_labels,
        export_view_analytics_results,
    )
    from finance_ml.analytics.statistical_analysis import (
        run_all_views_probability_analytics,
        export_probability_view_results,
    )

    # Load all 17 vw_features views
    try:
        views_dict = load_all_feature_views(return_dict=True)
        print(f"✓ Loaded {len(views_dict)} feature views")

        for view_name, view_df in views_dict.items():
            if not view_df.empty:
                print(f"   {view_name}: {len(view_df):,} rows, {len(view_df.columns)} columns")

        # Run probability analytics on each view
        print("\n📈 Running view-based probability analytics...")
        view_mapping = get_view_category_labels()
        view_analytics = run_all_views_probability_analytics(views_dict, view_mapping)

        # Export results to database
        print("\n💾 Exporting analytics to database...")
        export_counts = export_view_analytics_results(view_analytics)
        for table, count in export_counts.items():
            print(f"   {table}: {count} rows")

        # --- NEW: Export per-feature probability metrics to prob_vw_features_* tables ---
        print("\n📊 Exporting per-feature probability metrics...")
        id_cols = load_identifier_columns()
        id_cols_set = set(id_cols)
        for view_name, view_df in views_dict.items():
            if view_df.empty:
                continue
            feature_cols = [c for c in view_df.columns if c not in id_cols_set]
            rows_exported = export_probability_view_results(
                view_df, view_name, feature_cols, id_cols
            )
            if rows_exported and rows_exported > 0:
                print(f"   ✓ prob_{view_name}: {rows_exported} rows")

        # Generate visualizations for each view
        print("\n📊 Generating view-specific visualizations...")
        output_dir_views = Path("outputs/analytics/views")
        output_dir_views.mkdir(parents=True, exist_ok=True)

        for view_name, view_df in views_dict.items():
            if view_df.empty:
                continue
            category = view_mapping.get(view_name, view_name)

            # Enhanced View Analytics using CategoryProbabilityAnalyzer
            if PROBABILITY_ANALYTICS_AVAILABLE:
                analyzer = CategoryProbabilityAnalyzer(category)
                id_cols_set_viz = get_identifier_cols_set()
                feature_cols = [c for c in view_df.columns if c not in id_cols_set_viz]
                view_prob_results = analyzer.analyze_view(view_df, feature_cols)
                if not view_prob_results.empty:
                    # Export view-specific probability results
                    export_to_analytics_db(view_prob_results, f"prob_{view_name}")

            fig = create_view_probability_dashboard(view_df, view_name, category)
            output_path = output_dir_views / f"{view_name}_probability.html"
            fig.write_html(str(output_path))
            print(f"   ✓ Saved {output_path.name}")
    except Exception as e:
        print(f"⚠️  Error in view-based analytics: {e}")

    print()

    # ========================================================================
    # 4. STATISTICAL ANALYSIS
    # ========================================================================
    print("📈 Step 4: Running statistical analysis...")
    print("-" * 80)

    # Bayesian analysis for profitability features
    if "roe" in df.columns and len(df) > 50:
        print("   Running Bayesian analysis on Profitability features...")
        profitability_features = [
            f for f in FEATURE_CATEGORIES.get("Profitability", []) if f in df.columns
        ][:3]
        bayesian_results = bayesian_category_analysis(df, "Profitability", profitability_features)
        if bayesian_results:
            print(f"   ✓ Analyzed {len(bayesian_results)} profitability metrics")

            # Build InferenceData for Bayesian category analysis
            if ARVIZ_AVAILABLE:
                try:
                    idata_cat = build_category_analysis_inference_data(
                        bayesian_results,
                        df,
                        "Profitability",
                        profitability_features,
                    )
                    cat_summary = summarize_inference_data(idata_cat)
                    print(
                        f"   ✓ InferenceData (category): {cat_summary.get('n_chains', 0)} chains × {cat_summary.get('n_draws', 0)} draws"
                    )
                except Exception as e:
                    logging.warning("InferenceData (category) build failed: %s", e)

    # --- NEW: Fast ruin probability (optimized_ops) replaces statistical_analysis ---
    if all(col in df.columns for col in ["market_cap", "distress_risk_score"]):
        print("   Calculating investor's ruin probabilities (Numba-accelerated)...")
        ruin_df = fast_ruin_probability(df, n_simulations=2000, n_days=252)
        high_risk_count = (ruin_df["ruin_probability"] > 0.6).sum()
        print(f"   ✓ Identified {high_risk_count} high-risk stocks ({len(ruin_df)} analyzed)")

        # Build InferenceData for credit risk / ruin probability
        if ARVIZ_AVAILABLE:
            try:
                idata_ruin = build_credit_risk_inference_data(
                    ruin_df,
                    df,
                    n_posterior_samples=2000,
                    n_chains=4,
                )
                ruin_summary = summarize_inference_data(idata_ruin)
                print(
                    f"   ✓ InferenceData (credit risk): {ruin_summary.get('n_chains', 0)} chains × {ruin_summary.get('n_draws', 0)} draws"
                )
            except Exception as e:
                logging.warning("InferenceData (credit risk) build failed: %s", e)

        # Also keep analytical ruin for comparison
        ruin_analytical = calculate_ruin_probability(df)
        print(
            f"   ✓ Analytical ruin model: {(ruin_analytical['ruin_probability'] > 0.6).sum()} high-risk"
        )

        # Monte Carlo simulation
        required_mc_cols = ["price_target", "price_target_high", "price_target_low", "last_price"]
        if all(col in df.columns for col in required_mc_cols):
            print("  - Running Monte Carlo price target simulation...")
            mc_results = monte_carlo_price_target_simulation(df, max_stocks=10000)
            if len(mc_results) > 0:
                export_to_analytics_db(mc_results, "monte_carlo_simulation")
                print(
                    f"    ✓ Exported {len(mc_results)} simulations to analytics.monte_carlo_simulation"
                )
                print(f"    Top 5 by risk-reward ratio:")
                top5 = mc_results.nlargest(5, "risk_reward_ratio")[
                    ["ticker", "name", "expected_upside_pct", "risk_reward_ratio"]
                ]
                print(top5.to_string(index=False))

                # Build InferenceData for Monte Carlo simulation
                if ARVIZ_AVAILABLE:
                    try:
                        idata_mc = build_monte_carlo_inference_data(
                            mc_results,
                            df,
                            n_simulations=10000,
                        )
                        mc_summary = summarize_inference_data(idata_mc)
                        print(
                            f"    ✓ InferenceData (MC): {mc_summary.get('n_draws', 0)} simulations × {mc_summary.get('n_equities', 0)} equities"
                        )
                    except Exception as e:
                        logging.warning("InferenceData (MC) build failed: %s", e)

    # --- NEW: Vectorized z-scores and percentile ranks (optimized_ops) ---
    print("   Computing vectorized z-scores and percentile ranks...")
    zscore_cols = [c for c in ["roe", "roa", "p_e_ratio", "debt_to_equity"] if c in df.columns]
    if zscore_cols:
        df = vectorized_zscore(df, zscore_cols, group_col="industry")
        df = vectorized_percentile_rank(df, zscore_cols, group_col="industry")
        print(f"   ✓ Z-scores and percentiles computed for {len(zscore_cols)} metrics by industry")

    # --- NEW: Kalman-filtered price targets (statistical_analysis) ---
    if "last_price" in df.columns and "price_target_median" in df.columns:
        print("   Applying Kalman filter to price targets...")
        kalman_pt = kalman_filter_price_target(
            df, observation_col="last_price", target_col="price_target_median"
        )
        if len(kalman_pt) > 0:
            print(f"   ✓ Kalman-filtered {len(kalman_pt)} price targets")
            high_signal = kalman_pt.nlargest(5, "signal_strength")
            for _, k in high_signal.iterrows():
                print(
                    f"      {k['ticker']}: Filtered upside={k['filtered_upside']:.1f}%, "
                    f"Signal={k['signal_strength']:.1f}"
                )

    # --- NEW: Kalman-filtered momentum (statistical_analysis) ---
    momentum_cols = [
        c
        for c in ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
        if c in df.columns
    ]
    if momentum_cols:
        print("   Applying Kalman filter to momentum indicators...")
        df = kalman_momentum_filter(df, momentum_cols=momentum_cols)
        print(f"   ✓ Smoothed {len(momentum_cols)} momentum columns")

    # --- NEW: Gaussian copula dependency analysis (statistical_analysis) ---
    copula_features = [
        c for c in ["roe", "debt_to_equity", "p_e_ratio", "current_ratio"] if c in df.columns
    ]
    if len(copula_features) >= 2:
        print("   Fitting Gaussian copula for dependency structure...")
        copula_result = fit_gaussian_copula(df, copula_features, n_simulations=5000)
        print(
            f"   ✓ Copula fitted on {len(copula_result['features'])} features "
            f"({copula_result['n_observations']} observations)"
        )

    # --- NEW: Parallel MCMC with convergence diagnostics (statistical_analysis) ---
    if "roe" in df.columns:
        roe_data = df["roe"].dropna().values
        if len(roe_data) > 100:
            print("   Running parallel MCMC chains with Gelman-Rubin diagnostic...")
            mcmc_result = parallel_mcmc_chains(roe_data, n_chains=4, n_samples=5000)
            print(
                f"   ✓ R-hat: {mcmc_result['r_hat']:.4f} "
                f"({'Converged ✓' if mcmc_result['converged'] else 'Not converged ✗'})"
            )
            print(
                f"     Posterior mean ROE: {mcmc_result['posterior_mean']:.2f} "
                f"[{mcmc_result['ci_95'][0]:.2f}, {mcmc_result['ci_95'][1]:.2f}]"
            )

    # --- NEW: Hierarchical MCMC by sector (statistical_analysis) ---
    if "roe" in df.columns and "industry" in df.columns:
        print("   Running hierarchical MCMC by sector...")
        hier_results = hierarchical_mcmc_by_sector(df, "roe", sector_col="industry")
        print(f"   ✓ Hierarchical analysis for {len(hier_results)} sectors")

    # --- NEW: Distribution fitting by category (statistical_analysis) ---
    for cat_name in ["Profitability", "Valuation Ratios"]:
        cat_features = [f for f in FEATURE_CATEGORIES.get(cat_name, []) if f in df.columns][:3]
        if cat_features:
            print(f"   Fitting distributions for {cat_name}...")
            dist_fits = fit_distributions_by_category(df, cat_name, cat_features)
            for feat, info in dist_fits.items():
                print(
                    f"      {feat}: best={info['best_distribution']}, "
                    f"VaR(5%)={info['var_5_pct']:.2f}"
                )

    # --- NEW: Conditional probabilities P(Distress|Feature) (statistical_analysis) ---
    if "distress_risk_score" in df.columns:
        print("   Calculating conditional distress probabilities...")
        cond_probs = calculate_conditional_probabilities(df, FEATURE_CATEGORIES)
        if len(cond_probs) > 0:
            top_predictors = cond_probs.nlargest(5, "separation")
            print(f"   ✓ Top distress predictors:")
            for _, p in top_predictors.iterrows():
                print(f"      {p['feature']} ({p['category']}): separation={p['separation']:.3f}")

    # --- NEW: Per-category probability analytics pipeline (statistical_analysis) ---
    print("   Running per-category probability analytics...")
    for cat_name, cat_feats in list(FEATURE_CATEGORIES.items())[:3]:
        available = [f for f in cat_feats if f in df.columns]
        if available:
            cat_results = run_category_probability_analytics(df, cat_name, available)
            print(f"   ✓ {cat_name}: {cat_results['features_analyzed']} features analyzed")

    # New Statistical Analyses (existing)
    print("   Running Productivity and Accounting analysis...")
    df = analyze_employee_productivity_frontier(df)
    df = detect_accounting_anomalies(df)
    lag_analysis = analyze_reporting_lag_sentiment(df)
    print(f"   ✓ Accounting anomaly detection complete")
    if lag_analysis["sample_size"] > 0:
        print(
            f"   ✓ Reporting lag correlation: {lag_analysis['correlation']:.2f} "
            f"(Hypothesis: {lag_analysis['hypothesis_confirmed']})"
        )

    # ========================================================================
    # 5. PROBABILITY & CONFIDENCE ANALYTICS (ENHANCED)
    # ========================================================================
    if PROBABILITY_ANALYTICS_AVAILABLE:
        print("📊 Step 5: Running probability & confidence analytics...")
        print("-" * 80)

        probability_results = None
        streak_results = None
        confidence_result = None
        credit_results = None
        dividend_results = None
        pt_results = None

        try:
            # Initialize models
            beat_model = EarningsBeatProbabilityModel()
            streak_analyzer = EPSStreakAnalyzer(mean_reversion_weight=0.3)
            confidence_estimator = ModelConfidenceEstimator(n_bins=10)

            # --- Enhanced: Use three-layer evidence fusion when forward data available ---
            print("   Computing Bayesian earnings beat probabilities (enhanced)...")

            # Try enhanced analysis first (uses forward estimates + GAAP quality)
            probability_results = beat_model.analyze_dataframe_enhanced(
                df,
                sector_col="sector" if "sector" in df.columns else "industry",
                ticker_col="ticker" if "ticker" in df.columns else "isin",
                name_col="name",
            )

            # Fallback to basic analysis if enhanced yields no results
            if probability_results is None or len(probability_results) == 0:
                print(
                    "   ⚠️  Enhanced analysis yielded no results, falling back to proxy method..."
                )
                if "eps_trajectory_score" in df.columns:
                    df_analysis = df.copy()
                    # Use eps_positive_years if available, else derive from trajectory score
                    beats_col_name = "eps_beat_count"
                    if "eps_positive_years" in df_analysis.columns:
                        beats_col_name = "eps_positive_years"
                    elif beats_col_name not in df_analysis.columns:
                        df_analysis[beats_col_name] = (
                            df_analysis["eps_trajectory_score"].fillna(50) / 100 * 5
                        ).astype(int)
                    total_col_name = "eps_total_reports"
                    if total_col_name not in df_analysis.columns:
                        df_analysis[total_col_name] = 5

                    probability_results = beat_model.analyze_dataframe(
                        df_analysis,
                        beats_col=beats_col_name,
                        total_col=total_col_name,
                        sector_col="sector" if "sector" in df.columns else "industry",
                        ticker_col="ticker" if "ticker" in df.columns else "isin",
                    )

            if probability_results is not None and len(probability_results) > 0:
                likely_beat_count = (
                    probability_results["beat_classification"] == "likely_beat"
                ).sum()
                print(f"   ✓ Analyzed {len(probability_results)} stocks")
                print(f"   ✓ {likely_beat_count} classified as 'likely beat'")
                print(
                    f"   ✓ Mean posterior beat probability: "
                    f"{probability_results['posterior_beat_prob'].mean():.1%}"
                )
                # Report data source breakdown if enhanced columns present
                if "data_source" in probability_results.columns:
                    source_counts = probability_results["data_source"].value_counts()
                    for src, cnt in source_counts.items():
                        print(f"     Data source '{src}': {cnt} stocks")

                # Build InferenceData for beat probability
                if ARVIZ_AVAILABLE:
                    try:
                        idata_beat = build_beat_probability_inference_data(
                            probability_results,
                            df,
                            n_posterior_samples=4000,
                            n_chains=4,
                        )
                        beat_summary = summarize_inference_data(idata_beat)
                        print(f"   ✓ InferenceData (beat prob): {beat_summary.get('groups', [])}")
                        if beat_summary.get("r_hat"):
                            for var, rhat_val in beat_summary["r_hat"].items():
                                print(f"     R-hat ({var}): {rhat_val:.4f}")
                    except Exception as e:
                        logging.warning("InferenceData (beat prob) build failed: %s", e)

            # EPS Streak Analysis
            print("   Analyzing EPS streaks and continuation probabilities...")

            if "eps_trajectory_score" in df.columns:
                streak_results = streak_analyzer.analyze_dataframe(
                    df,
                    trajectory_col="eps_trajectory_score",
                    streak_col=(
                        "eps_positive_streak" if "eps_positive_streak" in df.columns else None
                    ),
                    ticker_col="ticker" if "ticker" in df.columns else "isin",
                )

                if len(streak_results) > 0:
                    beat_streaks = (streak_results["streak_type"] == "beat").sum()
                    miss_streaks = (streak_results["streak_type"] == "miss").sum()
                    print(f"   ✓ Identified {beat_streaks} stocks on beat streaks")
                    print(f"   ✓ Identified {miss_streaks} stocks on miss streaks")
                    print(
                        f"   ✓ Mean continuation probability: "
                        f"{streak_results['continuation_probability'].mean():.1%}"
                    )
                    # Report dynamic total reports
                    if "dynamic_total_reports" in streak_results.columns:
                        avg_total = streak_results["dynamic_total_reports"].mean()
                        print(f"   ✓ Avg dynamic total reports per stock: {avg_total:.1f}")

            # Model Confidence Estimation (using simulated outcomes for demo)
            print("   Estimating model confidence metrics...")

            if probability_results is not None and len(probability_results) > 10:
                # Simulate actual outcomes based on posterior probability
                # In production, this would use actual historical outcomes
                np.random.seed(42)
                simulated_outcomes = (
                    np.random.random(len(probability_results))
                    < probability_results["posterior_beat_prob"].values
                ).astype(float)

                confidence_result = confidence_estimator.compute_confidence_metrics(
                    predicted_probs=probability_results["posterior_beat_prob"].values,
                    actual_outcomes=simulated_outcomes,
                    model_name="Bayesian Earnings Beat Model",
                )

                print(f"   ✓ Brier Score: {confidence_result.brier_score:.4f}")
                print(f"   ✓ Calibration Error: {confidence_result.calibration_error:.4f}")
                print(f"   ✓ AUC-ROC: {confidence_result.discrimination_auc:.3f}")
                print(f"   ✓ Overall Confidence: {confidence_result.overall_confidence:.1f}/100")

            # New Probability Models
            print("   Running Credit Risk, Dividend Cut, and Price Target models...")
            credit_model = CreditRiskProbabilityModel()
            dividend_model = DividendCutProbabilityModel()
            pt_model = PriceTargetAchievementModel()

            credit_results = credit_model.analyze_dataframe(df)
            dividend_results = dividend_model.analyze_dataframe(df)
            pt_results = pt_model.analyze_dataframe(df)

            print(f"   ✓ Credit Risk analysis complete ({len(credit_results)} stocks)")
            print(f"   ✓ Dividend Cut analysis complete ({len(dividend_results)} stocks)")
            print(f"   ✓ Price Target achievement analysis complete ({len(pt_results)} stocks)")

            # Build InferenceData for credit risk model results
            if (
                ARVIZ_AVAILABLE
                and len(credit_results) > 0
                and "ruin_probability" in credit_results.columns
            ):
                try:
                    idata_credit = build_credit_risk_inference_data(
                        credit_results,
                        df,
                        n_posterior_samples=4000,
                        n_chains=4,
                    )
                    credit_summary = summarize_inference_data(idata_credit)
                    print(
                        f"   ✓ InferenceData (credit model): {credit_summary.get('n_chains', 0)} chains × {credit_summary.get('n_draws', 0)} draws"
                    )
                except Exception as e:
                    logging.warning("InferenceData (credit model) build failed: %s", e)

            # --- FIX: Export ALL probability results including credit/dividend/PT ---
            if probability_results is not None and streak_results is not None:
                print("   Exporting probability analytics results...")
                output_dir_prob = Path("outputs/analytics")
                export_paths = export_probability_analytics_results(
                    probability_df=probability_results,
                    streak_df=streak_results,
                    output_dir=output_dir_prob,
                    confidence_result=confidence_result,
                    credit_risk_df=credit_results,
                    dividend_safety_df=dividend_results,
                    price_target_df=pt_results,
                )
                for name, path in export_paths.items():
                    print(f"   ✓ Saved: {path}")

        except Exception as e:
            print(f"   ⚠️  Probability analytics error: {e}")
            import traceback

            traceback.print_exc()

        print()
    else:
        # Ensure variables are defined even when probability analytics unavailable
        probability_results = None
        streak_results = None
        confidence_result = None
        credit_results = None
        dividend_results = None
        pt_results = None

    # ========================================================================
    # 6. STOCK SCREENING
    # ========================================================================
    print("🔍 Step 6: Running stock screens...")
    print("-" * 80)

    # Enhanced quality screener
    if all(col in df.columns for col in ["piotroski_f_score", "distress_risk_score"]):
        quality_stocks = create_enhanced_screener(df, min_fscore=7, min_fcf_positive_years=4)
        print(f"   ✓ Quality screen: {len(quality_stocks)} stocks")

    # Value opportunities
    if "p_e_ratio" in df.columns and "upside_potential" in df.columns:
        value_stocks = screen_value_opportunities(df, max_pe_ratio=100, min_upside_potential=20)
        print(f"   ✓ Value screen: {len(value_stocks)} stocks")

    # Growth momentum
    if "revenue_yoy_growth" in df.columns or "revenue_growth_yoy" in df.columns:
        growth_stocks = screen_growth_momentum(df, min_revenue_growth=5)
        print(f"   ✓ Growth screen: {len(growth_stocks)} stocks")

    # Financial health
    if "distress_risk_score" in df.columns:
        healthy_stocks = screen_financial_health(df, min_distress_score=80)
        print(f"   ✓ Financial health screen: {len(healthy_stocks)} stocks")

    # GARP opportunities
    if "peg_ratio" in df.columns:
        garp_stocks = screen_garp_opportunities(df)
        print(f"   ✓ GARP screen: {len(garp_stocks)} stocks")

    # High-yield safe dividends
    if "dividend_yield" in df.columns or "dividend_yield_ltm" in df.columns:
        safe_div_stocks = screen_high_yield_safe_dividends(df)
        print(f"   ✓ High-yield safe dividend screen: {len(safe_div_stocks)} stocks")

    # New Screens
    reversion_stocks = screen_valuation_reversion_candidates(df)
    growth_integrity_stocks = screen_integrity_filtered_growth(df)
    print(f"   ✓ Valuation reversion screen: {len(reversion_stocks)} stocks")
    print(f"   ✓ Integrity-filtered growth screen: {len(growth_integrity_stocks)} stocks")

    # --- NEW: Earnings quality screening (screening.py) ---
    if "earnings_quality_composite" in df.columns:
        eq_stocks = screen_earnings_quality(df, min_quality_score=65)
        print(f"   ✓ Earnings quality screen: {len(eq_stocks)} stocks")

    # --- NEW: Dividend quality screening (screening.py) ---
    div_yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in df.columns else "dividend_yield"
    if div_yield_col in df.columns:
        dq_stocks = screen_dividend_quality(df, min_dividend_yield=2.5, min_dividend_streak=3)
        print(f"   ✓ Dividend quality screen: {len(dq_stocks)} stocks")

    # --- NEW: Composite ranking (screening.py) ---
    ranked_df = rank_stocks_by_composite_score(df, export=True)
    print(f"   ✓ Composite ranking: top score = {ranked_df['composite_score'].max():.1f}")

    # --- NEW: Sector-relative rankings (screening.py) ---
    for metric in ["roe", "p_e_ratio"]:
        if metric in df.columns and "industry" in df.columns:
            df = create_sector_relative_ranking(df, metric, sector_col="industry")
            print(f"   ✓ Sector-relative ranking for {metric}")

    print()

    # ========================================================================
    # 7. VISUALIZATIONS
    # ========================================================================
    print("📊 Step 7: Generating visualizations...")
    print("-" * 80)

    output_dir = Path("outputs/analytics")
    output_dir.mkdir(exist_ok=True)

    try:
        # Interactive momentum dashboard
        if all(
            col in df.columns
            for col in [
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
            ]
        ):
            print("   Creating momentum dashboard...")
            fig = create_interactive_momentum_dashboard(df)
            fig.write_html(output_dir / "momentum_dashboard.html")
            print("   ✓ Saved: outputs/momentum_dashboard.html")

        # Valuation heatmap
        if "p_e_ratio" in df.columns and "industry" in df.columns:
            print("   Creating valuation heatmap...")
            fig = create_interactive_valuation_heatmap(df)
            fig.write_html(output_dir / "valuation_heatmap.html")
            print("   ✓ Saved: outputs/valuation_heatmap.html")

        # Summary dashboard
        print("   Creating summary dashboard...")
        fig = create_summary_dashboard(df)
        fig.write_html(output_dir / "summary_dashboard.html")
        print("   ✓ Saved: outputs/summary_dashboard.html")

        # --- NEW: Probability Analytics Visualizations ---
        if PROBABILITY_ANALYTICS_AVAILABLE:
            if probability_results is not None and len(probability_results) > 0:
                print("   Creating earnings probability dashboard...")
                fig = create_earnings_probability_dashboard(probability_results)
                fig.write_html(output_dir / "earnings_beat_probability_dashboard.html")
                print("   ✓ Saved: outputs/analytics/earnings_beat_probability_dashboard.html")

                if "confidence_result" in locals():
                    print("   Creating model confidence calibration chart...")
                    fig = create_confidence_calibration_chart(confidence_result)
                    fig.write_html(output_dir / "model_confidence_calibration.html")
                    print("   ✓ Saved: outputs/analytics/model_confidence_calibration.html")

            if streak_results is not None and len(streak_results) > 0:
                print("   Creating EPS streak analysis chart...")
                fig = create_eps_streak_analysis_chart(streak_results)
                fig.write_html(output_dir / "eps_streak_analysis.html")
                print("   ✓ Saved: outputs/analytics/eps_streak_analysis.html")

        # --- New Category-Specific Visualizations ---

        # Profitability visualizations
        if all(col in df.columns for col in ["roe", "roa", "industry"]):
            print("   Creating profitability quadrant...")
            fig = create_profitability_quadrant(df)
            fig.write_html(output_dir / "profitability_quadrant.html")
            print("   ✓ Saved: outputs/profitability_quadrant.html")

        if "gross_margin_pct" in df.columns:
            print("   Creating margin waterfall chart...")
            fig = create_margin_waterfall_chart(df)
            fig.write_html(output_dir / "margin_waterfall.html")
            print("   ✓ Saved: outputs/margin_waterfall.html")

        # Technical analysis visualizations
        if "range_52w_position" in df.columns:
            print("   Creating 52-week range distribution...")
            fig = create_52w_range_distribution(df)
            fig.write_html(output_dir / "52w_range_distribution.html")
            print("   ✓ Saved: outputs/52w_range_distribution.html")

        if all(
            col in df.columns
            for col in [
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
            ]
        ):
            print("   Creating momentum ribbon chart...")
            fig = create_momentum_ribbon_chart(df)
            fig.write_html(output_dir / "momentum_ribbon.html")
            print("   ✓ Saved: outputs/momentum_ribbon.html")

        # Temporal analysis visualizations
        if "next_earnings" in df.columns:
            print("   Creating earnings calendar heatmap...")
            fig = create_earnings_calendar_heatmap(df)
            fig.write_html(output_dir / "earnings_calendar.html")
            print("   ✓ Saved: outputs/earnings_calendar.html")

        if "dividend_streak" in df.columns:
            print("   Creating dividend streak timeline...")
            fig = create_dividend_streak_timeline(df)
            fig.write_html(output_dir / "dividend_streak_timeline.html")
            print("   ✓ Saved: outputs/dividend_streak_timeline.html")

        # New Visualizations
        print("   Creating new enhanced visualizations...")
        fig_prod = create_productivity_quadrant(df)
        fig_prod.write_html(output_dir / "productivity_quadrant.html")

        if len(df) > 0:
            sample_ticker = df["ticker"].iloc[0]
            fig_acc = create_accounting_quality_breakdown(df, sample_ticker)
            fig_acc.write_html(output_dir / f"accounting_quality_{sample_ticker}.html")

            fig_val = create_valuation_range_visual(df, sample_ticker)
            fig_val.write_html(output_dir / f"valuation_range_{sample_ticker}.html")

        print("   ✓ Saved enhanced visualizations to outputs/analytics/")

        # Category-specific charts
        if "analyst_bullish_pct" in df.columns:
            print("   Creating analyst sentiment histogram...")
            fig = create_analyst_sentiment_histogram(df)
            fig.write_html(output_dir / "analyst_sentiment.html")
            print("   ✓ Saved: outputs/analyst_sentiment.html")

        if "eps_surprise_pct" in df.columns:
            print("   Creating EPS surprise histogram...")
            fig = create_eps_surprise_histogram(df)
            fig.write_html(output_dir / "eps_surprise.html")
            print("   ✓ Saved: outputs/eps_surprise.html")

        if all(col in df.columns for col in ["fcf_margin", "fcf_yield"]):
            print("   Creating FCF margin vs yield scatter...")
            fig = create_fcf_margin_yield_scatter(df)
            fig.write_html(output_dir / "fcf_margin_yield.html")
            print("   ✓ Saved: outputs/fcf_margin_yield.html")

        if "rnd_intensity_ltm" in df.columns and "industry" in df.columns:
            print("   Creating R&D intensity boxplot...")
            fig = create_rnd_intensity_boxplot(df)
            fig.write_html(output_dir / "rnd_intensity.html")
            print("   ✓ Saved: outputs/rnd_intensity.html")

        if "goodwill_concentration" in df.columns and "industry" in df.columns:
            print("   Creating goodwill concentration boxplot...")
            fig = create_goodwill_concentration_boxplot(df)
            fig.write_html(output_dir / "goodwill_concentration.html")
            print("   ✓ Saved: outputs/goodwill_concentration.html")

        if all(col in df.columns for col in ["capex_yoy_growth", "capex_vs_5y_avg"]):
            print("   Creating CapEx growth scatter...")
            fig = create_capex_growth_scatter(df)
            fig.write_html(output_dir / "capex_growth.html")
            print("   ✓ Saved: outputs/capex_growth.html")

        # --- New Enhanced Visualizations ---
        if "p_e_ratio" in df.columns:
            print("   Creating valuation violin plot...")
            fig = create_valuation_violin_plot(df)
            fig.write_html(output_dir / "valuation_violin.html")
            print("   ✓ Saved: outputs/valuation_violin.html")

        if all(col in df.columns for col in ["current_ratio", "debt_to_equity"]):
            print("   Creating leverage vs liquidity bubble chart...")
            fig = create_leverage_liquidity_bubble_chart(df)
            fig.write_html(output_dir / "leverage_liquidity_bubble.html")
            print("   ✓ Saved: outputs/leverage_liquidity_bubble.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            print(f"   Creating quality radar chart for {ticker}...")
            fig = create_quality_risk_radar_chart(df, ticker)
            fig.write_html(output_dir / f"quality_radar_{ticker}.html")
            print(f"   ✓ Saved: outputs/quality_radar_{ticker}.html")

        # --- NEW: Valuation Analysis Visualizations ---
        print("   Creating valuation analysis visualizations...")

        fig = create_valuation_distribution_dashboard(df)
        fig.write_html(output_dir / "valuation_distribution_dashboard.html")
        print("   ✓ Saved: valuation_distribution_dashboard.html")

        fig = create_relative_valuation_matrix(df)
        fig.write_html(output_dir / "relative_valuation_matrix.html")
        print("   ✓ Saved: relative_valuation_matrix.html")

        fig = create_valuation_vs_growth_quadrant(df)
        fig.write_html(output_dir / "valuation_vs_growth_quadrant.html")
        print("   ✓ Saved: valuation_vs_growth_quadrant.html")

        fig = create_historical_valuation_percentile(df)
        fig.write_html(output_dir / "historical_valuation_percentile.html")
        print("   ✓ Saved: historical_valuation_percentile.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_valuation_multiples_comparison(df, ticker=ticker)
            fig.write_html(output_dir / f"valuation_multiples_{ticker}.html")
            print(f"   ✓ Saved: valuation_multiples_{ticker}.html")

        # --- NEW: Earnings Quality Visualizations ---
        print("   Creating earnings quality visualizations...")

        fig = create_earnings_surprise_dashboard(df)
        fig.write_html(output_dir / "earnings_surprise_dashboard.html")
        print("   ✓ Saved: earnings_surprise_dashboard.html")

        fig = create_eps_trajectory_analysis(df)
        fig.write_html(output_dir / "eps_trajectory_analysis.html")
        print("   ✓ Saved: eps_trajectory_analysis.html")

        fig = create_beat_rate_heatmap(df)
        fig.write_html(output_dir / "beat_rate_heatmap.html")
        print("   ✓ Saved: beat_rate_heatmap.html")

        fig = create_earnings_consistency_matrix(df)
        fig.write_html(output_dir / "earnings_consistency_matrix.html")
        print("   ✓ Saved: earnings_consistency_matrix.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_earnings_quality_decomposition(df, ticker=ticker)
            fig.write_html(output_dir / f"earnings_quality_decomposition_{ticker}.html")
            print(f"   ✓ Saved: earnings_quality_decomposition_{ticker}.html")

        # --- NEW: Quality & Risk Visualizations ---
        print("   Creating quality & risk visualizations...")

        fig = create_piotroski_fscore_breakdown(df)
        fig.write_html(output_dir / "piotroski_fscore_breakdown.html")
        print("   ✓ Saved: piotroski_fscore_breakdown.html")

        fig = create_altman_zscore_distribution(df)
        fig.write_html(output_dir / "altman_zscore_distribution.html")
        print("   ✓ Saved: altman_zscore_distribution.html")

        fig = create_quality_risk_quadrant(df)
        fig.write_html(output_dir / "quality_risk_quadrant.html")
        print("   ✓ Saved: quality_risk_quadrant.html")

        fig = create_beneish_mscore_analysis(df)
        fig.write_html(output_dir / "beneish_mscore_analysis.html")
        print("   ✓ Saved: beneish_mscore_analysis.html")

        fig = create_risk_tier_sunburst(df)
        fig.write_html(output_dir / "risk_tier_sunburst.html")
        print("   ✓ Saved: risk_tier_sunburst.html")

        fig = create_distress_early_warning_dashboard(df)
        fig.write_html(output_dir / "distress_early_warning_dashboard.html")
        print("   ✓ Saved: distress_early_warning_dashboard.html")

        # --- NEW: Growth Analysis Visualizations ---
        print("   Creating growth analysis visualizations...")

        fig = create_growth_consistency_matrix(df)
        fig.write_html(output_dir / "growth_consistency_matrix.html")
        print("   ✓ Saved: growth_consistency_matrix.html")

        fig = create_growth_vs_profitability_quadrant(df)
        fig.write_html(output_dir / "growth_vs_profitability_quadrant.html")
        print("   ✓ Saved: growth_vs_profitability_quadrant.html")

        fig = create_growth_acceleration_chart(df)
        fig.write_html(output_dir / "growth_acceleration_chart.html")
        print("   ✓ Saved: growth_acceleration_chart.html")

        fig = create_sustainable_growth_analysis(df)
        fig.write_html(output_dir / "sustainable_growth_analysis.html")
        print("   ✓ Saved: sustainable_growth_analysis.html")

        if len(df) > 0:
            ticker = df.iloc[0]["ticker"]
            fig = create_growth_waterfall_chart(df, ticker=ticker)
            fig.write_html(output_dir / f"growth_waterfall_{ticker}.html")
            print(f"   ✓ Saved: growth_waterfall_{ticker}.html")

    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")

    print()

    # ========================================================================
    # 8. EXPORT RESULTS
    # ========================================================================
    print("💾 Step 8: Exporting results...")
    print("-" * 80)

    # Export feature statistics
    stats_data = []
    for category, features in FEATURE_CATEGORIES.items():
        for feature in features:
            if feature in df.columns:
                stats = compute_metric_statistics(df[feature])
                if stats:
                    stats["category"] = category
                    stats["feature"] = feature
                    stats_data.append(stats)

    # Helper to reorder DataFrame columns: identifier cols first, then the rest
    identifier_cols = load_identifier_columns()

    def _reorder_with_identifiers(result_df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [c for c in identifier_cols if c in result_df.columns]
        other_cols = [c for c in result_df.columns if c not in id_cols]
        return result_df[id_cols + other_cols]

    if stats_data:
        stats_df = _reorder_with_identifiers(pd.DataFrame(stats_data))
        stats_cfg = ExportConfig(table_name="feature_statistics")
        export_to_db(stats_df, stats_cfg)
        export_to_csv(stats_df, stats_cfg)
        export_to_json(stats_df, stats_cfg)
        print(f"   ✓ Exported {len(stats_df)} features to analytics.feature_statistics")

    # Export screened stocks (all screening results from Step 6)
    screening_exports = {
        "quality_stocks": "quality_stocks",
        "value_stocks": "value_stocks",
        "growth_stocks": "growth_stocks",
        "healthy_stocks": "healthy_stocks",
        "garp_stocks": "garp_stocks",
        "safe_div_stocks": "safe_dividend_stocks",
        "reversion_stocks": "valuation_reversion_stocks",
        "growth_integrity_stocks": "integrity_filtered_growth_stocks",
        "eq_stocks": "earnings_quality_stocks",
        "dq_stocks": "dividend_quality_stocks",
    }

    for var_name, table in screening_exports.items():
        # Exports dataframes to database, CSV, and JSON formats
        if var_name in locals() and len(locals()[var_name]) > 0:
            screen_cfg = ExportConfig(table_name=table)
            screen_df = _reorder_with_identifiers(locals()[var_name])
            export_to_db(screen_df, screen_cfg)
            export_to_csv(screen_df, screen_cfg)
            export_to_json(screen_df, screen_cfg)
            print(f"   ✓ Exported {len(screen_df)} stocks to analytics.{table}")

    # --- NEW: Export composite scores ---
    if "ranked_df" in locals() and "composite_score" in ranked_df.columns:
        export_cols = ["composite_score"]
        id_cols_present = [c for c in identifier_cols if c in ranked_df.columns]
        available = id_cols_present + [c for c in export_cols if c in ranked_df.columns]
        composite_cfg = ExportConfig(table_name="composite_quality_scores")
        export_to_db(ranked_df[available].head(200), composite_cfg)
        export_to_csv(ranked_df[available].head(200), composite_cfg)
        export_to_json(ranked_df[available].head(200), composite_cfg)
        print(f"   ✓ Exported top 200 composite scores to analytics.composite_quality_scores")

    # --- NEW: Export Kalman-filtered price targets ---
    if "kalman_pt" in locals() and len(kalman_pt) > 0:
        kalman_cfg = ExportConfig(table_name="kalman_filtered_price_targets")
        kalman_df = _reorder_with_identifiers(kalman_pt)
        export_to_db(kalman_df, kalman_cfg)
        export_to_csv(kalman_df, kalman_cfg)
        export_to_json(kalman_df, kalman_cfg)
        print(f"   ✓ Exported {len(kalman_pt)} Kalman-filtered targets")

    print()

    # ========================================================================
    # SUMMARY (UPDATED)
    # ========================================================================
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Refactored modules used:")
    print("  • data_utils: Data loading and preprocessing")
    print("  • statistical_analysis: Bayesian, MCMC, Kalman, Copula, Monte Carlo")
    print("  • screening: Multi-factor stock screening (12 screeners)")
    print("  • feature_analytics: Interactive visualizations")
    print("  • optimized_ops: Numba MC, fast ruin, vectorized stats, caching")
    if ARVIZ_AVAILABLE:
        print("  • inference_schema: ArviZ InferenceData schema bridge (xarray/NetCDF)")
    print("  • visualizations.profitability: Margin and DuPont analysis")
    print("  • visualizations.technical: Momentum and range charts")
    print("  • visualizations.temporal_analysis: Earnings and dividend timelines")
    print("  • visualizations.category_charts: Category-specific charts")
    print("  • visualizations.valuation: Valuation ratio analysis")
    print("  • visualizations.earnings_quality: Earnings quality charts")
    print("  • visualizations.quality_risk: Quality & risk assessment")
    print("  • visualizations.growth_analysis: Growth metrics analysis")
    if PROBABILITY_ANALYTICS_AVAILABLE:
        print(
            "  • probability_analytics: Bayesian beat, credit risk, dividend safety, PT achievement"
        )
    print()
    print(f"Total stocks analyzed: {len(df):,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Feature categories: {len(FEATURE_CATEGORIES)}")
    if probability_results is not None:
        print(f"Probability analysis completed: {len(probability_results)} stocks")
    if streak_results is not None:
        print(f"Streak analysis completed: {len(streak_results)} stocks")
    if credit_results is not None:
        print(f"Credit risk analysis completed: {len(credit_results)} stocks")
    if dividend_results is not None:
        print(f"Dividend safety analysis completed: {len(dividend_results)} stocks")
    if pt_results is not None:
        print(f"Price target achievement analysis completed: {len(pt_results)} stocks")
    print()
    print("Check the 'outputs/analytics/' directory for generated files.")
    print()


if __name__ == "__main__":
    main()
