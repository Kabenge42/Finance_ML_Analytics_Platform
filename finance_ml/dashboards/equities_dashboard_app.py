"""Equities Dashboard (Plotly Dash)

Run:
    python finance_ml/dashboards/equities_dashboard_app.py

Design goals:
- No heavy work at import time (use create_app()).
- Safe fallbacks when data sources / artifacts are missing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from flask import send_from_directory

from finance_ml.dashboards.callbacks import register_all_callbacks
from finance_ml.dashboards.components import (
    _safe_options,
    _severity_style,
    compute_surprise,
    create_empty_state_figure,
    create_missing_columns_warning,
    validate_required_columns,
)
from finance_ml.dashboards.components.constants import (
    COLOR_PALETTE,
    FONT_FAMILY,
    FONT_SIZES,
    PLOTLY_TEMPLATE,
)
from finance_ml.dashboards.components.data_utils import (
    ARTIFACTS_DIR,
    ARTIFACTS_METADATA_PATH,
    DEFAULT_CSV_EXPORT_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_METADATA_PATH,
    PROJECT_ROOT,
    _validate_explorer_columns,
)
from finance_ml.dashboards.components.temporal_utils import (
    compute_days_to_earnings,
    get_reference_date,
)
from finance_ml.dashboards.earnings_widgets import (
    DATE_DISPLAY_FORMAT,
    _add_formatted_date_columns,
    _resolve_reference_date,
    create_analyst_recommendation_heatmap,
    create_category_comparison_chart,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
)
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_CATEGORIES
from finance_ml.ml_workflow.features.advanced import engineer_temporal_features
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_features,
    ETLConfig,
    DataExtractionConfig,
    SchemaValidationConfig,
    DtypeCastingConfig,
    SemanticClassificationConfig,
    ImputationConfig,
    SemanticTransformConfig,
    DataSanitizationConfig,
    ScalingConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    FinancialMetricsConfig,
)

# Type aliases
DataSource = Literal["auto", "csv", "db"]

# Logging setup
logger = logging.getLogger(__name__)


# Apply template globally to all Plotly figures
px.defaults.template = PLOTLY_TEMPLATE

# Standard Plotly layout configuration
PLOTLY_LAYOUT_DEFAULTS = {
    "font": {"family": FONT_FAMILY, "size": FONT_SIZES["caption"]},
    "title_font_size": FONT_SIZES["h3"],
    "showlegend": True,
    "legend": {
        "orientation": "v",
        "yanchor": "top",
        "xanchor": "right",
        "x": 1.02,
        "y": 1,
    },
    "hovermode": "closest",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
}

# Standard DataTable styles (aligned with code_guidelines.md Section 17.3)
TABLE_STYLE_CELL = {
    "backgroundColor": "#111",
    "color": "#ffffff",
    "border": f"1px solid {COLOR_PALETTE['secondary']}",
    "fontFamily": FONT_FAMILY,
    "fontSize": f"{FONT_SIZES['caption']}px",
    "padding": "8px",
    "whiteSpace": "normal",
    "height": "auto",
    "minWidth": "80px",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": COLOR_PALETTE["primary"],
    "fontWeight": "bold",
    "color": "#ffffff",
}

TABLE_STYLE_TABLE = {
    "overflowX": "auto",
    "maxHeight": "500px",
    "overflowY": "auto",
}

# Earnings calendar mode options
EARNINGS_MODE_OPTIONS = [
    {"label": "All Categories", "value": "all"},
    {"label": "Earnings Focus", "value": "earnings"},
    {"label": "Dividends Focus", "value": "dividends"},
    {"label": "Valuation", "value": "valuation"},
    {"label": "Profitability", "value": "profitability"},
    {"label": "Growth", "value": "growth"},
    {"label": "Momentum", "value": "momentum"},
    {"label": "Quality & Risk", "value": "quality_risk"},
]

# Default columns for the Data Explorer tab - always included in initial view
# (code_guidelines.md Section 8.1: Single Source of Truth for configuration constants)

# Re-export for backward compatibility
__all__ = [
    "get_reference_date",
    "compute_days_to_earnings",
    "parse_data_store",
    "load_data_csv_first",
    "validate_required_columns",
    "create_missing_columns_warning",
    "compute_surprise",
    "create_empty_state_figure",
    "export_equities_data",
    "generate_dashboard_artifacts",
    "load_data",
    "create_app",
    "main",
]


# =============================================================================
# Dashboard ETL Configuration (mirrors etl_data_explorer.ipynb)
# =============================================================================

# Winsorization bounds (code_guidelines.md Section 2.1)
WINSORIZE_LOWER = 0.10
WINSORIZE_UPPER = 0.90


def _get_dashboard_etl_config() -> ETLConfig:
    """
    Get ETL configuration for the dashboard, mirroring etl_data_explorer.ipynb.

    This configuration ensures proper data-loading and processing with correct
    data types and engineers the required features including earnings analytics.
    """
    return ETLConfig(
        extraction=DataExtractionConfig(
            normalize_column_names=True,
        ),
        validation=SchemaValidationConfig(
            validate_schema=True,
            require_target_column=False,  # Dashboard doesn't require target
            drop_rows_with_missing_critical_fields=True,
            validate_schema_alignment=True,
            schema_alignment_threshold=0.80,
        ),
        dtype_casting=DtypeCastingConfig(
            apply_dtype_casting=True,
            track_diagnostics=True,
        ),
        semantic_classification=SemanticClassificationConfig(
            enabled=True,
            preserve_price_columns=True,
        ),
        imputation=ImputationConfig(
            apply_imputation=False,
            strategy="6step",
            knn_neighbors=5,
            sector_column="sector",
            reference_price_column="last_price",
            impute_categorical_columns=True,
            impute_datetime_columns=True,  # Critical for earnings calendar
        ),
        semantic_transform=SemanticTransformConfig(
            apply_log_transforms=False,
            log_transform_method="log1p",
            log_transform_market_values=False,
            exclude_ratios_from_winsorization=True,
            exclude_percentages_from_winsorization=True,
            exclude_counts_from_scaling=True,
        ),
        sanitization=DataSanitizationConfig(
            sanitize_data=False,
            apply_winsorization=False,
            winsorize_lower_percentile=WINSORIZE_LOWER,
            winsorize_upper_percentile=WINSORIZE_UPPER,
        ),
        scaling=ScalingConfig(
            enabled=True,
            scaler_type="robust",
            scale_by_sector=True,
            exclude_price_columns=True,
        ),
        feature_engineering=FeatureEngineeringConfig(
            enabled=True,
            preset="comprehensive",
            engineer_earnings_analytics=True,  # Enable Earnings Quality features
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=False,
            method="both",
            min_importance_threshold=0.01,
            max_correlation_threshold=0.95,
        ),
        financial_metrics=FinancialMetricsConfig(
            compute_valuation_metrics=True,
            compute_profitability_metrics=True,
            compute_growth_metrics=True,
            compute_leverage_metrics=True,
            compute_target_vs_price_metrics=True,
            compute_sector_specific_metrics=True,
            generate_quality_alerts=True,
            generate_metrics_dashboard=True,
            output_directory="financial_metrics",
        ),
    )


# =============================================================================
# Helper Functions & Utilities (code_guidelines.md Section 8)
# =============================================================================


def _coerce_list(value: Any) -> List[str]:
    """Coerce value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context_name: str,
) -> Tuple[List[str], List[str]]:
    """Validate required columns exist in DataFrame.

    Args:
        df: DataFrame to check
        required_cols: List of required column names
        context_name: Name of the context (for logging)

    Returns:
        Tuple of (present_cols, missing_cols)
    """
    if df is None or df.empty:
        return [], required_cols

    present = [c for c in required_cols if c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]

    return present, missing


def create_missing_columns_warning(
    missing_cols: List[str],
    context_name: str,
) -> html.Div:
    """Create a warning panel for missing columns."""
    if not missing_cols:
        return html.Div()

    return html.Div(
        [
            html.H5(f"⚠️ Missing Columns for {context_name}", className="text-warning"),
            html.P(f"The following columns are unavailable: {', '.join(missing_cols[:10])}"),
            html.P("Some charts may be limited or unavailable.", className="text-muted"),
        ],
        style={
            "padding": "10px",
            "backgroundColor": "#2d2d2d",
            "borderRadius": "5px",
            "marginBottom": "10px",
        },
    )


def compute_surprise(
    actual: pd.Series,
    estimate: pd.Series,
    mode: Literal["pct", "abs"] = "pct",
    clip_bounds: Tuple[float, float] = (-100, 100),
) -> pd.Series:
    """Compute earnings surprise with safe handling.

    Args:
        actual: Actual values
        estimate: Estimated values
        mode: 'pct' for percentage, 'abs' for absolute
        clip_bounds: Bounds to clip extreme values

    Returns:
        Series of surprise values
    """
    actual_num = pd.to_numeric(actual, errors="coerce")
    estimate_num = pd.to_numeric(estimate, errors="coerce")

    if mode == "pct":
        # Use absolute estimate as denominator to avoid sign issues
        denom = estimate_num.abs().replace(0, np.nan)
        surprise = ((actual_num - estimate_num) / denom) * 100
    else:
        surprise = actual_num - estimate_num

    # Replace inf with NaN and clip
    surprise = surprise.replace([np.inf, -np.inf], np.nan)
    if clip_bounds:
        surprise = surprise.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    return surprise


def create_empty_state_figure(
    title: str,
    message: str = "No data available",
) -> go.Figure:
    """Create standardized empty state figure per code_guidelines.md Section 17.2.

    Args:
        title: Figure title
        message: Message to display in empty state

    Returns:
        Plotly Figure with empty state annotation
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(
            family=FONT_FAMILY,
            size=FONT_SIZES["body"],
            color=COLOR_PALETTE["neutral"],
        ),
    )
    fig.update_layout(**PLOTLY_LAYOUT_DEFAULTS, title=title)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return fig


def export_equities_data(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Export equities data to CSV with metadata.

    Args:
        df: DataFrame to export
        output_path: Path for CSV file (defaults to DEFAULT_CSV_EXPORT_PATH)
        metadata_path: Path for metadata JSON (defaults to DEFAULT_METADATA_PATH)

    Returns:
        Dict with export metadata
    """
    if output_path is None:
        output_path = DEFAULT_CSV_EXPORT_PATH
    if metadata_path is None:
        metadata_path = DEFAULT_METADATA_PATH

    # Ensure directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Export CSV
    df.to_csv(output_path, index=False)

    # Generate metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "file_path": str(output_path),
        "file_size_mb": output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0,
    }

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_data_store(store_data: Optional[str]) -> pd.DataFrame:
    """Safely parse data from dcc.Store component.

    Handles the case where store_data is None (when load_on_start=False
    or when initial data loading fails).

    Args:
        store_data: JSON string from dcc.Store or None

    Returns:
        DataFrame (empty if store_data is None or invalid)
    """
    if store_data is None:
        return pd.DataFrame()

    try:
        return pd.read_json(store_data, orient="split")
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to parse data store: {e}")
        return pd.DataFrame()


def _apply_temporal_enrichments(
    df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], List[str]]:
    """Apply temporal features and formatted date companions using unified reference date.

    Args:
        df: Input DataFrame to enrich.
        reference_date: Optional reference date override.

    Returns:
        Tuple of (enriched_df, resolved_reference_date, formatted_columns).
    """

    if df is None or df.empty:
        return df, reference_date, []

    resolved_reference_date = _resolve_reference_date(df, reference_date)

    temporal_date_col: Optional[str] = None
    if "reference_date" in df.columns:
        temporal_date_col = "reference_date"
    elif "next_earnings" in df.columns:
        temporal_date_col = "next_earnings"

    enriched_df = df
    if temporal_date_col:
        enriched_df = engineer_temporal_features(
            df, date_col=temporal_date_col, reference_date=resolved_reference_date
        )

    date_columns_for_format = [
        "_reference_date",
        "reference_date",
        "next_earnings",
        "income_statement_report_date",
        "dividend_record_ex_date",
        "fy_end",
    ]
    date_columns_for_format = [c for c in date_columns_for_format if c in enriched_df.columns]
    formatted_cols = _add_formatted_date_columns(enriched_df, date_columns_for_format)

    return enriched_df, resolved_reference_date, formatted_cols


# =============================================================================
# Dashboard Configuration Constants (code_guidelines.md Section 8.3)
# =============================================================================

# DataTable pagination defaults
DEFAULT_PAGE_SIZE_CALENDAR = 15
DEFAULT_PAGE_SIZE_ALERTS = 20
DEFAULT_PAGE_SIZE_EXPLORER = 20

# Earnings calendar defaults
DEFAULT_EARNINGS_DAYS_WINDOW = 10
MIN_EARNINGS_DAYS_WINDOW = 3
MAX_EARNINGS_DAYS_WINDOW = 30
EARNINGS_DAYS_MARKS = {3: "3", 7: "7", 10: "10", 14: "14", 21: "21", 30: "30"}

# Top N defaults
DEFAULT_TOP_N = 50
MIN_TOP_N = 10
MAX_TOP_N = 200
TOP_N_STEP = 10

# Alert thresholds (default values for UI inputs)
DEFAULT_EPS_MISS_THRESHOLD = 20.0
DEFAULT_DOWNGRADE_THRESHOLD = 5.0
DEFAULT_MIN_DOWNGRADE_PERIODS = 2
DEFAULT_TARGET_SPREAD_THRESHOLD = 30.0
DEFAULT_PRE_EARNINGS_WINDOW_DAYS = 7
DEFAULT_VOLATILITY_QUANTILE = 0.75
DEFAULT_MAX_TICKERS_PER_ALERT = 10

# Explorer defaults
DEFAULT_EXPLORER_ROW_LIMIT = 200
EXPLORER_ROW_LIMIT_STEP = 50
MIN_EXPLORER_ROW_LIMIT = 10


def load_data_csv_first(
    *,
    data_dir: Optional[Path] = None,
    db_url: Optional[str] = None,
    feature_preset: str = "comprehensive",
    force_etl: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load data preferring CSV export, falling back to ETL.

    Returns:
        Tuple of (DataFrame, source_label) where source_label is one of:
        'csv_export', 'etl_csv', 'etl_db'
    """
    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    resolved_db_url = db_url or os.getenv("DB_URL")

    # Fast path: load from exported CSV if it exists and is recent
    if not force_etl and DEFAULT_CSV_EXPORT_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_CSV_EXPORT_PATH)
            _validate_explorer_columns(df, "csv_export")
            return df, "csv_export"
        except Exception as e:
            logger.warning(f"Failed to load CSV export: {e}, falling back to ETL")

    # Slow path: run ETL pipeline with dashboard-specific config
    try:
        source: Literal["csv", "db"] = "db" if resolved_db_url else "csv"

        # Use dashboard ETL config (mirrors etl_data_explorer.ipynb)
        etl_config = _get_dashboard_etl_config()

        # Phase 9.1-9.3: Unified ETL Pipeline (STANDARD Pattern)
        df, metrics = etl_with_features(
            source=source,
            data_dir=resolved_data_dir,
            db_url=resolved_db_url,
            feature_preset=feature_preset,
            config=etl_config,
            return_metrics=True,
        )

        df, resolved_reference_date, formatted_cols = _apply_temporal_enrichments(df)
        if resolved_reference_date is not None:
            logger.info(
                "Reference date resolved to %s",
                resolved_reference_date.strftime(DATE_DISPLAY_FORMAT),
            )
        if formatted_cols:
            logger.debug("Formatted date columns added: %s", ", ".join(formatted_cols))

        # Export to CSV for next time
        if not df.empty:
            try:
                export_equities_data(df)
            except Exception as e:
                logger.debug(f"Non-critical: Failed to export equities data: {e}")

        # Return the metrics summary as the source label for the status bar
        source_label = metrics.summary()
        _validate_explorer_columns(df, f"etl_{source}")
        return df, source_label
    except Exception as e:
        logger.error(f"ETL Pipeline failed: {e}")
        return pd.DataFrame(), f"ETL failed: {str(e)[:50]}"


# Metric mappings for Est vs Actual tab
EST_ACTUAL_METRICS = {
    "EPS": {
        "actual": "eps_adj_ltm",
        "estimate": "eps_norm_est_avg_ntm",
        "adjusted": "eps_adj_ltm",
        "gaap": "net_eps_basic_ltm",
    },
    "Revenue": {
        "actual": "total_revenues_ltm",
        "estimate": "revenues_est_avg_ntm",
        "adjusted": None,
        "gaap": "total_revenues_ltm",
    },
    "EBITDA": {
        "actual": "ebitda_ltm",
        "estimate": "ebitda_est_avg_ntm",
        "adjusted": "ebitda_adj_ltm",
        "gaap": "ebitda_ltm",
    },
}


def generate_dashboard_artifacts(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate all dashboard artifacts using earnings_widgets.

    Args:
        df: Source DataFrame (equities_dash_df)
        output_dir: Directory for artifacts (defaults to ARTIFACTS_DIR)
        metadata_path: Path for artifacts metadata JSON

    Returns:
        Dict with artifact generation metadata
    """
    if output_dir is None:
        output_dir = ARTIFACTS_DIR
    if metadata_path is None:
        metadata_path = ARTIFACTS_METADATA_PATH

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    reference_date = _resolve_reference_date(df, None)
    timestamp = (reference_date or pd.Timestamp.now()).strftime(DATE_DISPLAY_FORMAT)
    reference_date_str = (
        reference_date.strftime(DATE_DISPLAY_FORMAT) if reference_date is not None else None
    )

    try:
        # Generate main dashboard widgets
        artifacts["earnings_surprise"] = {
            "file": "earnings_surprise_dashboard.html",
            "title": "Earnings Surprise Analysis",
            "section": "earnings",
        }
        create_earnings_surprise_dashboard(
            df,
            reference_date=reference_date,
            output_path=output_dir / artifacts["earnings_surprise"]["file"],
        )

        artifacts["analyst_heatmap"] = {
            "file": "analyst_recommendation_heatmap.html",
            "title": "Analyst Recommendations by Sector",
            "section": "earnings",
        }
        create_analyst_recommendation_heatmap(
            df, output_path=output_dir / artifacts["analyst_heatmap"]["file"]
        )

        artifacts["market_movers"] = {
            "file": "market_movers_dashboard.html",
            "title": "Market Movers Around Earnings",
            "section": "earnings",
        }
        create_market_movers_dashboard(
            df,
            reference_date=reference_date,
            output_path=output_dir / artifacts["market_movers"]["file"],
        )

        artifacts["price_target_analytics"] = {
            "file": "price_target_analytics.html",
            "title": "Price Target Analytics",
            "section": "analytics",
        }
        create_price_target_analytics(
            df, output_path=output_dir / artifacts["price_target_analytics"]["file"]
        )

        # Generate Phase 9.3 category charts
        phase93_categories = [
            "profitability",
            "valuation",
            "growth",
            "momentum",
            "quality_risk",
            "cash_flow",
            "dividends",
            "forecasts",
            "earnings_quality",
        ]

        for category in phase93_categories:
            key = f"earnings_metrics_{category}"
            artifacts[key] = {
                "file": f"earnings_metrics_{category}.html",
                "title": f"Earnings Metrics: {category.replace('_', ' ').title()}",
                "section": "phase93",
            }
            create_earnings_metrics_chart(
                df,
                metric_category=category,
                reference_date=reference_date,
                output_path=output_dir / artifacts[key]["file"],
            )

        # Generate category comparison chart
        artifacts["category_comparison"] = {
            "file": "phase93_category_comparison.html",
            "title": "Phase 9.3 Category Comparison",
            "section": "phase93",
        }
        create_category_comparison_chart(
            df,
            reference_date=reference_date,
            output_path=output_dir / artifacts["category_comparison"]["file"],
        )

        # Generate monitoring report (Task 8)
        report = {
            "timestamp": timestamp,
            "total_stocks": len(df),
            "kpis": {},
        }
        if "total_revenues_cagr_5y_fy" in df.columns:
            growth = pd.to_numeric(df["total_revenues_cagr_5y_fy"], errors="coerce")
            report["kpis"]["pct_positive_revenue_growth"] = float((growth > 0).mean() * 100)
            report["kpis"]["median_revenue_growth"] = float(growth.median())
        if "net_income_margin_pct_ltm" in df.columns:
            margin = pd.to_numeric(df["net_income_margin_pct_ltm"], errors="coerce")
            report["kpis"]["median_net_margin"] = float(margin.median())
        if "return_on_equity_pct_ltm" in df.columns:
            roe = pd.to_numeric(df["return_on_equity_pct_ltm"], errors="coerce")
            report["kpis"]["median_roe"] = float(roe.median())

        report_path = output_dir / "monitoring_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    except Exception as e:
        print(f"Warning: Error generating some artifacts: {e}")

    # Create metadata
    metadata = {
        "timestamp": timestamp,
        "total_stocks": len(df),
        "artifacts_dir": str(output_dir),
        "artifacts": artifacts,
        "generation_status": "completed",
        "reference_date": reference_date_str,
    }

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_data(
    *,
    data_source: DataSource = "auto",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    feature_preset: str = "comprehensive",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load equities data using the unified ETL + features pipeline.

    - auto: try DB if DB_URL is provided, otherwise fall back to CSV.
    - csv: load from CSV region files under data_dir.
    - db: load from database (requires db_url or DB_URL env var).

    Returns an empty DataFrame on failures.
    """

    resolved_data_dir: Path = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    resolved_db_url = db_url or os.getenv("DB_URL")

    # Get dashboard ETL config (mirrors etl_data_explorer.ipynb)
    etl_config = _get_dashboard_etl_config()

    def _etl(source: Literal["csv", "db"]) -> pd.DataFrame:
        result = etl_with_features(
            source=source,
            data_dir=resolved_data_dir,
            db_url=resolved_db_url,
            feature_preset=feature_preset,
            config=etl_config,
            return_metrics=True,
        )
        return result

    try:
        if data_source == "db":
            if not resolved_db_url:
                return pd.DataFrame()
            df = _etl("db")
        elif data_source == "csv":
            df = _etl("csv")
        else:
            if resolved_db_url:
                try:
                    df = _etl("db")
                except Exception as e:
                    logger.warning(f"Failed to load from DB, falling back to CSV: {e}")
                    df = _etl("csv")
            else:
                df = _etl("csv")

        if not df.empty:
            df, _, _ = _apply_temporal_enrichments(df)

        if limit is not None and limit > 0:
            return df.head(int(limit)).copy()
        return df
    except Exception as e:
        logger.error(f"Failed to load data in load_data: {e}")
        return pd.DataFrame()


def create_app(
    *,
    data_source: DataSource = "auto",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    load_on_start: bool = True,
) -> dash.Dash:
    """Create Dash app instance.

    Set load_on_start=True when running interactively.
    Keep it False in tests to avoid running ETL.
    """
    initial_df = pd.DataFrame()

    if load_on_start:
        try:
            initial_df = load_data(data_source=data_source, data_dir=data_dir, db_url=db_url)
            if initial_df is None:
                logger.warning("load_data returned None, using empty DataFrame")
                initial_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            initial_df = pd.DataFrame()

    app = dash.Dash(
        __name__,
        title="Equities Dashboard",
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    server = app.server

    @server.route("/app_assets/<path:filename>")
    def serve_outputs(filename: str):
        """Serve files from outputs directory with path traversal protection."""
        from flask import abort

        # Validate filename to prevent directory traversal attacks
        # Reject any path containing '..' or absolute paths
        if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
            abort(403)  # Forbidden

        # Ensure the resolved path is within the outputs directory
        outputs_dir = PROJECT_ROOT / "outputs"
        try:
            requested_path = (outputs_dir / filename).resolve()
            if not str(requested_path).startswith(str(outputs_dir.resolve())):
                abort(403)  # Forbidden
        except (ValueError, OSError):
            abort(400)  # Bad Request

        return send_from_directory(outputs_dir, filename)

    # Layout
    app.layout = html.Div(
        [
            html.H1("📈 Equities Analytics Dashboard", style={"textAlign": "center"}),
            dcc.Store(
                id="equities-data-store",
                # Note: For large datasets (>6000 rows, 300+ columns), consider:
                # 1. Using storage_type='session' or 'local' for persistence
                # 2. Implementing server-side caching with flask-caching
                # 3. Using pagination/lazy loading for initial data
                # Current implementation stores full DataFrame as JSON which may
                # cause performance issues with very large datasets.
                data=initial_df.to_json(orient="split") if not initial_df.empty else None,
                storage_type="memory",  # Explicit default for clarity
            ),
            html.Div(
                id="kpi-cards",
                style={
                    "display": "flex",
                    "justifyContent": "space-around",
                    "margin": "20px",
                },
            ),
            html.Div(
                [
                    html.H4("Filters", style={"marginBottom": "10px", "color": "white"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Sector", className="filter-label"),
                                    dcc.Dropdown(
                                        id="sector-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "sector"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Region", className="filter-label"),
                                    dcc.Dropdown(
                                        id="region-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "region"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Country", className="filter-label"),
                                    dcc.Dropdown(
                                        id="country-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "country"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Trading Country", className="filter-label"),
                                    dcc.Dropdown(
                                        id="trading-country-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "trading_country"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                        ],
                        className="filter-row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Industry", className="filter-label"),
                                    dcc.Dropdown(
                                        id="industry-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "industry"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Exchange", className="filter-label"),
                                    dcc.Dropdown(
                                        id="exchange-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "exchange"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Style Class", className="filter-label"),
                                    dcc.Dropdown(
                                        id="style-class-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "style_class"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Size Class", className="filter-label"),
                                    dcc.Dropdown(
                                        id="size-class-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "size_class"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                        ],
                        className="filter-row",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Load / Refresh Data",
                                id="refresh-data-btn",
                                color="primary",
                                style={"marginRight": "10px"},
                            ),
                            dbc.Button(
                                "Reset Filters",
                                id="reset-filters-btn",
                                color="secondary",
                                style={"marginRight": "10px"},
                            ),
                            dbc.Button(
                                "Generate Artifacts",
                                id="generate-artifacts-btn",
                                color="success",
                            ),
                            html.Span(
                                id="data-status",
                                style={
                                    "marginLeft": "10px",
                                    "color": COLOR_PALETTE["neutral"],
                                    "fontFamily": FONT_FAMILY,
                                },
                            ),
                        ],
                        style={"margin": "10px 0"},
                    ),
                ],
                style={"padding": "10px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="overview",
                children=[
                    dcc.Tab(
                        label="📋 Overview",
                        value="overview",
                        children=[
                            html.Div(
                                [
                                    dcc.Graph(id="target-vs-price-scatter"),
                                    dcc.Graph(id="market-cap-distribution"),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="📅 Earnings Analytics Dashboard",
                        value="earnings",
                        children=[
                            html.Div(
                                [
                                    # Alert summary panel (Task 3)
                                    html.Div(id="earnings-alert-summary"),
                                    # Alert filter dropdown (Task 4)
                                    html.Div(
                                        [
                                            html.Label(
                                                "Filter by alerts:",
                                                className="filter-label",
                                            ),
                                            dcc.Dropdown(
                                                id="earnings-alert-filter-dropdown",
                                                options=[
                                                    {
                                                        "label": "All tickers",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Only tickers with alerts",
                                                        "value": "alerts_only",
                                                    },
                                                ],
                                                value="all",
                                                clearable=False,
                                                style={"width": "250px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "gap": "10px",
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    # Generate artifacts button (Task 5)
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Generate Earnings Analytics Artifacts",
                                                id="generate-earnings-artifacts-btn",
                                                color="info",
                                            ),
                                            html.Span(
                                                id="earnings-artifacts-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "15px"},
                                    ),
                                    # Interactive Earnings Calendar Dashboard
                                    html.Div(
                                        [
                                            html.H5(
                                                "📅 Interactive Earnings Calendar",
                                                style={
                                                    "marginBottom": "10px",
                                                    "color": COLOR_PALETTE["info"],
                                                },
                                            ),
                                            # Calendar controls row
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Calendar Mode:",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Dropdown(
                                                                id="earnings-calendar-mode",
                                                                options=EARNINGS_MODE_OPTIONS,
                                                                value="all",
                                                                clearable=False,
                                                                style={"width": "180px"},
                                                            ),
                                                        ],
                                                        className="filter-item",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Days Window (±):",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Slider(
                                                                id="earnings-calendar-days",
                                                                min=MIN_EARNINGS_DAYS_WINDOW,
                                                                max=MAX_EARNINGS_DAYS_WINDOW,
                                                                step=1,
                                                                value=DEFAULT_EARNINGS_DAYS_WINDOW,
                                                                marks=EARNINGS_DAYS_MARKS,
                                                                tooltip={
                                                                    "placement": "bottom",
                                                                    "always_visible": False,
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "width": "200px",
                                                            "marginLeft": "20px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Label(
                                                                "Top N:",
                                                                className="filter-label",
                                                            ),
                                                            dcc.Input(
                                                                id="earnings-calendar-top-n",
                                                                type="number",
                                                                value=DEFAULT_TOP_N,
                                                                min=MIN_TOP_N,
                                                                max=MAX_TOP_N,
                                                                step=TOP_N_STEP,
                                                                style={"width": "80px"},
                                                            ),
                                                        ],
                                                        style={"marginLeft": "20px"},
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Checkbox(
                                                                id="earnings-calendar-apply-filters",
                                                                label="Apply Global Filters",
                                                                value=True,
                                                            ),
                                                        ],
                                                        style={
                                                            "marginLeft": "20px",
                                                            "display": "flex",
                                                            "alignItems": "center",
                                                        },
                                                    ),
                                                ],
                                                className="filter-row",
                                                style={"marginBottom": "10px"},
                                            ),
                                            # Calendar status
                                            html.Div(
                                                id="earnings-calendar-status",
                                                style={
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontSize": f"{FONT_SIZES['caption']}px",
                                                    "fontFamily": FONT_FAMILY,
                                                    "marginBottom": "5px",
                                                },
                                            ),
                                            # Earnings Calendar DataTable (code_guidelines.md Section 17.3)
                                            dash_table.DataTable(
                                                id="earnings-calendar-table",
                                                data=[],
                                                columns=[],
                                                style_table={
                                                    **TABLE_STYLE_TABLE,
                                                    "maxHeight": "400px",
                                                    "minWidth": "100%",
                                                },
                                                style_cell={
                                                    **TABLE_STYLE_CELL,
                                                    "width": "auto",
                                                    "maxWidth": "200px",
                                                    "textOverflow": "ellipsis",
                                                },
                                                style_header={
                                                    **TABLE_STYLE_HEADER,
                                                    "textTransform": "capitalize",
                                                    "fontWeight": "bold",
                                                },
                                                style_data_conditional=[
                                                    # Past earnings (red)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} < 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "color": COLOR_PALETTE["danger"],
                                                    },
                                                    # Today (warning background)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} = 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "backgroundColor": COLOR_PALETTE["warning"],
                                                        "color": "#000000",
                                                    },
                                                    # Future earnings (green)
                                                    {
                                                        "if": {
                                                            "filter_query": "{days_to_earnings} > 0",
                                                            "column_id": "days_to_earnings",
                                                        },
                                                        "color": COLOR_PALETTE["success"],
                                                    },
                                                ],
                                                sort_action="native",
                                                filter_action="native",
                                                page_action="native",
                                                page_size=DEFAULT_PAGE_SIZE_CALENDAR,
                                                row_selectable="multi",
                                                selected_rows=[],
                                            ),
                                        ],
                                        style={
                                            "padding": "15px",
                                            "backgroundColor": "#1a1a1a",
                                            "borderRadius": "8px",
                                            "marginBottom": "20px",
                                            "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                        },
                                    ),
                                    # Existing charts
                                    dcc.Graph(id="earnings-events-timeline"),
                                    dcc.Graph(id="earnings-surprise-fig"),
                                    dcc.Graph(id="analyst-heatmap-fig"),
                                    dcc.Graph(id="market-movers-fig"),
                                    dcc.Graph(id="price-target-analytics-fig"),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="🚨 Alerts",
                        value="alerts",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H4(
                                                "Earnings Quality Alerts",
                                                style={"marginTop": "10px"},
                                            ),
                                            html.Div(
                                                id="alerts-meta",
                                                style={
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ]
                                    ),
                                    html.H5("Regenerate"),
                                    html.Div(
                                        [
                                            html.Label("EPS miss threshold (%)"),
                                            dcc.Input(
                                                id="cfg-eps-miss",
                                                type="number",
                                                value=DEFAULT_EPS_MISS_THRESHOLD,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Downgrade threshold (%)",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-downgrade",
                                                type="number",
                                                value=DEFAULT_DOWNGRADE_THRESHOLD,
                                                step=0.5,
                                            ),
                                            html.Label(
                                                "Downgrade min periods",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-min-periods",
                                                type="number",
                                                value=DEFAULT_MIN_DOWNGRADE_PERIODS,
                                                step=1,
                                            ),
                                            html.Br(),
                                            html.Label("Target spread threshold (%)"),
                                            dcc.Input(
                                                id="cfg-target-spread",
                                                type="number",
                                                value=DEFAULT_TARGET_SPREAD_THRESHOLD,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Pre-earnings window (days)",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-window-days",
                                                type="number",
                                                value=DEFAULT_PRE_EARNINGS_WINDOW_DAYS,
                                                step=1,
                                            ),
                                            html.Label(
                                                "Volatility quantile",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-vol-quantile",
                                                type="number",
                                                value=DEFAULT_VOLATILITY_QUANTILE,
                                                step=0.05,
                                                min=0,
                                                max=1,
                                            ),
                                            html.Label(
                                                "Max tickers per alert",
                                                style={"marginLeft": "10px"},
                                            ),
                                            dcc.Input(
                                                id="cfg-max-tickers",
                                                type="number",
                                                value=DEFAULT_MAX_TICKERS_PER_ALERT,
                                                step=1,
                                            ),
                                            html.Br(),
                                            dbc.Button(
                                                "Generate Alerts",
                                                id="generate-alerts-btn",
                                                color="warning",
                                            ),
                                            html.Span(
                                                id="generate-alerts-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={
                                            "padding": "10px",
                                            "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                        },
                                    ),
                                    # Alerts DataTable (code_guidelines.md Section 17.3)
                                    dash_table.DataTable(
                                        id="alerts-table",
                                        columns=[
                                            {"name": "Severity", "id": "severity"},
                                            {"name": "Type", "id": "alert_type"},
                                            {"name": "Count", "id": "count"},
                                            {
                                                "name": "Description",
                                                "id": "description",
                                            },
                                            {"name": "Tickers", "id": "tickers"},
                                        ],
                                        data=[],
                                        style_table=TABLE_STYLE_TABLE,
                                        style_cell=TABLE_STYLE_CELL,
                                        style_header=TABLE_STYLE_HEADER,
                                        sort_action="native",
                                        filter_action="native",
                                        page_action="native",
                                        page_size=DEFAULT_PAGE_SIZE_ALERTS,
                                        style_data_conditional=[
                                            {
                                                "if": {"filter_query": '{severity} = "high"'},
                                                **_severity_style("high"),
                                            },
                                            {
                                                "if": {"filter_query": '{severity} = "medium"'},
                                                **_severity_style("medium"),
                                            },
                                            {
                                                "if": {"filter_query": '{severity} = "low"'},
                                                **_severity_style("low"),
                                            },
                                        ],
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="🔎 Data Explorer",
                        value="explorer",
                        children=[
                            html.Div(
                                [
                                    html.Label("Feature category"),
                                    dcc.Dropdown(
                                        id="feature-category-dropdown",
                                        options=[
                                            {"label": k, "value": k}
                                            for k in sorted(PHASE93_FEATURE_CATEGORIES.keys())
                                        ],
                                        multi=True,
                                        value=["profitability"],
                                    ),
                                    html.Label("Columns"),
                                    dcc.Dropdown(id="explorer-columns-dropdown", multi=True),
                                    html.Div(
                                        [
                                            html.Label("Row limit"),
                                            dcc.Input(
                                                id="explorer-row-limit",
                                                type="number",
                                                value=DEFAULT_EXPLORER_ROW_LIMIT,
                                                step=EXPLORER_ROW_LIMIT_STEP,
                                                min=MIN_EXPLORER_ROW_LIMIT,
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),
                                    dbc.Button(
                                        "Update Table",
                                        id="explorer-update-btn",
                                        color="secondary",
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style={
                                    "padding": "10px",
                                    "border": f"1px solid {COLOR_PALETTE['secondary']}",
                                },
                            ),
                            # Data Explorer DataTable (code_guidelines.md Section 17.3)
                            dash_table.DataTable(
                                id="explorer-table",
                                data=[],
                                columns=[],
                                style_table=TABLE_STYLE_TABLE,
                                style_cell=TABLE_STYLE_CELL,
                                style_header=TABLE_STYLE_HEADER,
                                sort_action="native",
                                filter_action="native",
                                page_action="native",
                                page_size=DEFAULT_PAGE_SIZE_EXPLORER,
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="🗂️ Artifacts",
                        value="artifacts",
                        children=[
                            html.Div(
                                [
                                    dcc.Dropdown(id="artifact-dropdown"),
                                    html.Div(
                                        id="artifact-viewer",
                                        style={"marginTop": "10px"},
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    # Task 6-7: Est. vs Actual vs Adjusted Tab
                    dcc.Tab(
                        label="📊 Est. vs Actual vs Adjusted",
                        value="est-actual",
                        children=[
                            html.Div(
                                [
                                    # Missing columns warning
                                    html.Div(id="est-actual-missing-cols-warning"),
                                    # Controls row
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Metric",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-metric-selector",
                                                        options=[
                                                            {"label": k, "value": k}
                                                            for k in EST_ACTUAL_METRICS.keys()
                                                        ],
                                                        value="EPS",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Surprise Calculation",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-surprise-method",
                                                        options=[
                                                            {
                                                                "label": "Percentage",
                                                                "value": "pct",
                                                            },
                                                            {
                                                                "label": "Absolute",
                                                                "value": "abs",
                                                            },
                                                        ],
                                                        value="pct",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Segment By",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="est-actual-segment-by",
                                                        options=[
                                                            {
                                                                "label": "Sector",
                                                                "value": "sector",
                                                            },
                                                            {
                                                                "label": "Region",
                                                                "value": "region",
                                                            },
                                                            {
                                                                "label": "Size Class",
                                                                "value": "size_class",
                                                            },
                                                            {
                                                                "label": "Style Class",
                                                                "value": "style_class",
                                                            },
                                                            {
                                                                "label": "Industry",
                                                                "value": "industry",
                                                            },
                                                            {
                                                                "label": "Trading Country",
                                                                "value": "trading_country",
                                                            },
                                                            {
                                                                "label": "Exchange",
                                                                "value": "exchange",
                                                            },
                                                        ],
                                                        value="sector",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="filter-item",
                                            ),
                                        ],
                                        className="filter-row",
                                    ),
                                    # Charts
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="est-actual-scatter-fig",
                                                style={"height": "400px"},
                                            ),
                                            dcc.Graph(
                                                id="est-actual-distribution-fig",
                                                style={"height": "400px"},
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="est-actual-adjusted-fig",
                                                style={"height": "400px"},
                                            ),
                                            dcc.Graph(
                                                id="est-actual-revision-fig",
                                                style={"height": "400px"},
                                            ),
                                        ],
                                        style={
                                            "display": "grid",
                                            "gridTemplateColumns": "1fr 1fr",
                                            "gap": "10px",
                                        },
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                    # Task 8: Earnings Monitoring Tab
                    dcc.Tab(
                        label="📈 Earnings Monitoring",
                        value="monitoring",
                        children=[
                            html.Div(
                                [
                                    # KPI cards row
                                    html.Div(id="monitoring-kpi-row", className="kpi-row"),
                                    # Controls
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Segment By",
                                                        className="filter-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="monitoring-segment-by",
                                                        options=[
                                                            {
                                                                "label": "Sector",
                                                                "value": "sector",
                                                            },
                                                            {
                                                                "label": "Region",
                                                                "value": "region",
                                                            },
                                                            {
                                                                "label": "Size Class",
                                                                "value": "size_class",
                                                            },
                                                        ],
                                                        value="sector",
                                                        clearable=False,
                                                        style={"width": "200px"},
                                                    ),
                                                ],
                                                style={"marginRight": "20px"},
                                            ),
                                            dbc.Button(
                                                "Generate Monitoring Report",
                                                id="generate-monitoring-report-btn",
                                                color="success",
                                            ),
                                            html.Span(
                                                id="monitoring-report-status",
                                                style={
                                                    "marginLeft": "10px",
                                                    "color": COLOR_PALETTE["neutral"],
                                                    "fontFamily": FONT_FAMILY,
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginBottom": "15px",
                                        },
                                    ),
                                    # Charts
                                    dcc.Graph(id="monitoring-growth-fig"),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                                id="monitoring-margin-fig",
                                                style={"flex": "1"},
                                            ),
                                            dcc.Graph(
                                                id="monitoring-quality-fig",
                                                style={"flex": "1"},
                                            ),
                                        ],
                                        style={"display": "flex", "gap": "10px"},
                                    ),
                                ],
                                style={"padding": "10px"},
                            )
                        ],
                    ),
                ],
            ),
        ]
    )

    # ---------------------- Callbacks ----------------------
    # Callbacks are now registered via the callbacks module (code_guidelines.md §6.2)
    try:
        register_all_callbacks(
            app=app,
            data_dir=data_dir,
            db_url=db_url,
            load_on_start=load_on_start,
            initial_df=initial_df,
            load_data_csv_first=load_data_csv_first,
            generate_dashboard_artifacts=generate_dashboard_artifacts,
        )
    except Exception as e:
        logger.error(f"Failed to register callbacks: {e}")
        # Add a fallback error display to the layout
        app.layout.children.append(
            html.Div(
                [
                    html.H4("⚠️ Application Error", style={"color": COLOR_PALETTE["danger"]}),
                    html.P(f"Some features may be unavailable: {str(e)[:200]}"),
                ],
                style={
                    "padding": "20px",
                    "backgroundColor": "#2d2d2d",
                    "margin": "20px",
                    "borderRadius": "5px",
                },
            )
        )

    return app


def main() -> None:
    """Run the Equities Dashboard application."""
    app = create_app(load_on_start=True)
    app.run(debug=False)


if __name__ == "__main__":
    main()
