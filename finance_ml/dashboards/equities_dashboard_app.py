"""Equities Dashboard (Plotly Dash)

Run:
    python finance_ml/dashboards/equities_dashboard_app.py

Design goals:
- No heavy work at import time (use create_app()).
- Safe fallbacks when data sources / artifacts are missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

# Load environment variables from environment_variables.txt at module import time
# This ensures DB_URL and other config are available before any functions run
_env_file = Path(__file__).parent.parent.parent / "environment_variables.txt"
if _env_file.exists():
    load_dotenv(_env_file, override=False)  # Don't override existing env vars

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash import dash_table, dcc, html
from flask import send_from_directory
from sqlalchemy import create_engine, text

from finance_ml.core.constants import (
    WINSORIZE_LOWER,
    WINSORIZE_UPPER,
)
from finance_ml.core.schema import normalize_column_name, PHASE93_FEATURE_CATEGORIES
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
    DEFAULT_DOWNGRADE_THRESHOLD,
    DEFAULT_EARNINGS_DAYS_WINDOW,
    DEFAULT_EPS_MISS_THRESHOLD,
    DEFAULT_EXPLORER_ROW_LIMIT,
    DEFAULT_MAX_TICKERS_PER_ALERT,
    DEFAULT_MIN_DOWNGRADE_PERIODS,
    DEFAULT_PAGE_SIZE_ALERTS,
    DEFAULT_PAGE_SIZE_CALENDAR,
    DEFAULT_PAGE_SIZE_EXPLORER,
    DEFAULT_PRE_EARNINGS_WINDOW_DAYS,
    DEFAULT_TARGET_SPREAD_THRESHOLD,
    DEFAULT_TOP_N,
    DEFAULT_VOLATILITY_QUANTILE,
    EARNINGS_DAYS_MARKS,
    EXPLORER_ROW_LIMIT_STEP,
    FONT_FAMILY,
    FONT_SIZES,
    MAX_EARNINGS_DAYS_WINDOW,
    MAX_TOP_N,
    MIN_EARNINGS_DAYS_WINDOW,
    MIN_EXPLORER_ROW_LIMIT,
    MIN_TOP_N,
    PLOTLY_TEMPLATE,
    TOP_N_STEP,
)
from finance_ml.dashboards.components.data_utils import (
    PROJECT_ROOT,
    export_equities_data,
    generate_dashboard_artifacts,
    load_data,
    load_data_csv_first,
    parse_data_store,
)
from finance_ml.dashboards.components.temporal_utils import (
    compute_days_to_earnings,
    get_reference_date,
)
from finance_ml.etl.config import (
    ETLConfig,
    DataExtractionConfig,
    SchemaValidationConfig,
    DtypeCastingConfig,
    SemanticClassificationConfig,
    ImputationConfig,
    CurrencyConversionConfig,
    SemanticTransformConfig,
    DataSanitizationConfig,
    ScalingConfig,
    FeatureEngineeringConfig,
    FeatureSelectionConfig,
    FinancialMetricsConfig,
)
from finance_ml.etl.pipeline import ETLPipeline
from finance_ml.etl.metrics import ETLMetrics

# Type aliases
DataSource = Literal["auto", "csv", "db"]

# Logging setup
logger = logging.getLogger(__name__)

# =============================================================================
# Dashboard Configuration & Constants
# =============================================================================


def _get_default_db_url() -> str:
    """
    Get database URL from environment variables.

    Returns:
        Database connection string from DB_URL or DATABASE_URL env var,
        or a placeholder if not configured.

    Raises:
        RuntimeError: If no database URL is configured and strict mode is enabled.
    """
    db_url = os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
    if db_url:
        return db_url

    # Fallback for local development only - logs warning
    logger.warning(
        "No DB_URL or DATABASE_URL environment variable set. "
        "Set DB_URL=postgresql+psycopg2://user:password@host:port/dbname"
    )
    return ""


DEFAULT_DB_URL = _get_default_db_url()

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
# Helper Functions & Utilities (code_guidelines.md Section 8)
# =============================================================================


def load_from_equities_table(db_url: str) -> pd.DataFrame:
    """
    Robustly load data directly from postgres.public.equities table.

    Args:
        db_url: PostgreSQL connection string

    Returns:
        DataFrame with all equities data

    Raises:
        RuntimeError: If table is empty or connection fails
        ValueError: If db_url is empty or invalid
    """
    if not db_url:
        raise ValueError("Database URL is required. Set DB_URL environment variable.")

    engine = create_engine(db_url)

    try:
        # Verify table exists and has data
        with engine.connect() as conn:
            # Check row count first
            count_result = conn.execute(text("SELECT COUNT(*) FROM public.equities"))
            row_count = count_result.scalar()

            if row_count == 0:
                raise RuntimeError("equities table is empty. Run import_equities_data.sql first.")

            logger.info(f"Found {row_count:,} rows in public.equities")

        # Load all data from equities table for current date
        query = text('SELECT * FROM public.equities WHERE "Reference Date" = current_date - 1')
        df = pd.read_sql(query, engine)

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns from equities table")

        # Quick validation
        if df.empty:
            raise RuntimeError("DataFrame is empty after loading from equities table")

        return df

    except Exception as e:
        logger.error(f"Database loading failed: {e}")
        raise

    finally:
        engine.dispose()
        logger.debug("Database engine disposed")


def get_dashboard_etl_config() -> ETLConfig:
    """
    Get ETL configuration matching etl_data_explorer.ipynb pipeline.

    Returns:
        ETLConfig with comprehensive feature engineering settings
    """
    return ETLConfig(
        extraction=DataExtractionConfig(
            normalize_column_names=True,
            source_type="database",
        ),
        validation=SchemaValidationConfig(
            validate_schema=True,
            require_target_column=True,
            drop_rows_with_missing_critical_fields=True,
            validate_schema_alignment=True,
            schema_alignment_threshold=0.95,
            validate_quality=True,
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
            knn_neighbors=25,
            sector_column="sector",
            reference_price_column="last_price",
            impute_categorical_columns=False,
            impute_datetime_columns=False,
        ),
        currency_conversion=CurrencyConversionConfig(
            enabled=False, target_currency="USD", suffix="_usd"
        ),
        semantic_transform=SemanticTransformConfig(
            apply_log_transforms=True,
            log_transform_method="log1p",
            log_transform_market_values=True,
            exclude_ratios_from_winsorization=True,
            exclude_percentages_from_winsorization=True,
        ),
        sanitization=DataSanitizationConfig(
            sanitize_data=False,
            apply_winsorization=False,
            winsorize_lower_percentile=WINSORIZE_LOWER,
            winsorize_upper_percentile=WINSORIZE_UPPER,
        ),
        scaling=ScalingConfig(
            enabled=False,
            scaler_type="robust",
            scale_by_sector=True,
            exclude_price_columns=True,
        ),
        feature_engineering=FeatureEngineeringConfig(
            enabled=True,
            preset="comprehensive",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=False,
            method="both",
            min_importance_threshold=0.05,
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


def run_etl_pipeline(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ETL transformation pipeline on raw equities data.

    Args:
        raw_data: Raw DataFrame from equities table

    Returns:
        Transformed DataFrame with engineered features
    """
    # Normalize column names (SQL has mixed case, Python needs lowercase)
    raw_data.columns = [normalize_column_name(col) for col in raw_data.columns]
    logger.info(f"Normalized {len(raw_data.columns)} column names")

    # Configure and run ETL pipeline
    etl_config = get_dashboard_etl_config()
    pipeline = ETLPipeline(config=etl_config)

    # Initialize metrics manually since we're not using run()
    pipeline.metrics = ETLMetrics(source_type="database")
    pipeline.metrics.rows_input = len(raw_data)
    pipeline.metrics.columns_input = len(raw_data.columns)

    # Transform and load
    df_transformed = pipeline.transform(raw_data)
    df_features = pipeline.load(df_transformed)

    # Log specific execution summary as requested (mirrors etl_data_explorer.ipynb)
    logger.info("=" * 60)
    logger.info("ETL Pipeline Summary:")
    logger.info("  Source: postgres.public.equities")
    logger.info(f"  Input rows: {pipeline.metrics.rows_input:,}")
    logger.info(f"  Output rows: {len(df_features):,}")
    logger.info(f"  Output columns: {len(df_features.columns)}")
    logger.info(f"  Missing values after imputation: {df_features.isna().sum().sum()}")
    logger.info(f"  Stages executed: {pipeline.metrics.stages_executed}")
    logger.info("=" * 60)

    logger.info(
        f"ETL Pipeline complete: {len(df_features):,} rows, " f"{len(df_features.columns)} columns"
    )

    return df_features


# =============================================================================
# Metric mappings for Est vs Actual tab
# =============================================================================
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


def create_app(
    *,
    data_source: DataSource = "db",
    data_dir: Optional[str | Path] = None,
    db_url: Optional[str] = None,
    load_on_start: bool = True,
) -> dash.Dash:
    """Create Dash app instance.

    Args:
        data_source: Where to load data from ('auto', 'csv', 'db')
        data_dir: Directory for CSV data files
        db_url: Database connection URL (falls back to DB_URL env var)
        load_on_start: Whether to load data immediately (False for tests)

    Returns:
        Configured Dash application instance

    Note:
        Set load_on_start=True when running interactively.
        Keep it False in tests to avoid running ETL.
    """
    initial_df = pd.DataFrame()

    # Resolve database URL with proper precedence
    effective_db_url = db_url or os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")

    if not effective_db_url and data_source == "db":
        logger.warning(
            "Database source requested but no DB_URL configured. " "Falling back to CSV source."
        )
        data_source = "csv"

    if load_on_start:
        try:
            # Try database-first loading with ETL pipeline (matching etl_data_explorer.ipynb)
            if data_source == "db" or (data_source == "auto" and effective_db_url):
                try:
                    logger.info("Loading data from postgres.public.equities...")
                    raw_data = load_from_equities_table(effective_db_url)
                    initial_df = run_etl_pipeline(raw_data)
                    logger.info(
                        f"ETL Pipeline complete: {len(initial_df):,} rows, "
                        f"{len(initial_df.columns)} features"
                    )
                except Exception as db_error:
                    logger.warning(f"Database loading failed: {db_error}, falling back to CSV")
                    initial_df = load_data(
                        data_source=data_source, data_dir=data_dir, db_url=effective_db_url
                    )
            else:
                initial_df = load_data(
                    data_source=data_source, data_dir=data_dir, db_url=effective_db_url
                )

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
                            html.Div(
                                [
                                    html.Label("Fiscal Quarter", className="filter-label"),
                                    dcc.Dropdown(
                                        id="fiscal-quarter-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "fiscal_quarter"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Fiscal Year", className="filter-label"),
                                    dcc.Dropdown(
                                        id="fiscal-year-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "fiscal_year"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Earnings Status", className="filter-label"),
                                    dcc.Dropdown(
                                        id="earnings-status-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "next_earnings_status"),
                                    ),
                                ],
                                className="filter-item",
                            ),
                            html.Div(
                                [
                                    html.Label("Earnings Report", className="filter-label"),
                                    dcc.Dropdown(
                                        id="earnings-report-dropdown",
                                        multi=True,
                                        options=_safe_options(initial_df, "next_earnings_report"),
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
                                    # Company Drilldown Section (Task 6)
                                    html.Div(
                                        id="ticker-drilldown-container",
                                        children=[
                                            html.H5(
                                                "🔍 Company Metric Detail",
                                                style={
                                                    "marginTop": "30px",
                                                    "marginBottom": "15px",
                                                    "color": COLOR_PALETTE["info"],
                                                },
                                            ),
                                            html.Div(id="ticker-drilldown-content"),
                                        ],
                                        style={"display": "none"},  # Hidden until ticker selected
                                    ),
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
            db_url=effective_db_url,
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
