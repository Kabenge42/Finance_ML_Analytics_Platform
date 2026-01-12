I'll provide a comprehensive fix for both files to ensure the refresh functionality and all components work correctly.
The main issues are:

1. **In `data_utils.py`**: The `load_data_csv_first` function doesn't properly resolve DB_URL at runtime
2. **In `equities_dashboard_app.py`**: The `run_etl_pipeline` function name conflicts with the imported one from
   `finance_ml.etl.pipeline`
3. **Missing proper error handling and fallback chains**

Here are the comprehensive fixes:

```python
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from finance_ml.core.constants import DATE_DISPLAY_FORMAT
from finance_ml.dashboards.widgets import (
    add_formatted_date_columns,
    resolve_reference_date,
    create_analyst_recommendation_heatmap,
    create_category_comparison_chart,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
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
from finance_ml.etl.pipeline import run_etl_pipeline as _run_etl_pipeline
from finance_ml.etl.metrics import ETLMetrics
from finance_ml.features.advanced import engineer_temporal_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DASHBOARD_ROOT = PROJECT_ROOT / "outputs" / "dashboards" / "equities_dashboard"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_EXPORT_PATH = DASHBOARD_ROOT / "equities_dash_df.csv"
DEFAULT_METADATA_PATH = DASHBOARD_ROOT / "metadata.json"
ARTIFACTS_DIR = DASHBOARD_ROOT / "artifacts"
ARTIFACTS_METADATA_PATH = DASHBOARD_ROOT / "artifacts_metadata.json"
DEFAULT_ALERTS_PATH = (
        PROJECT_ROOT
        / "outputs"
        / "eda"
        / "earnings_analytics"
        / "earnings_quality_alerts.json"
)

DEFAULT_EXPLORER_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "region",
    "last_price",
    "price_target",
    "market_cap",
]

# Winsorization bounds (code_guidelines.md Section 2.1)
WINSORIZE_LOWER = 0.10
WINSORIZE_UPPER = 0.90


def _resolve_db_url(db_url: Optional[str] = None) -> Optional[str]:
    """Resolve database URL from parameter or environment variables.
    
    Args:
        db_url: Optional explicit database URL
        
    Returns:
        Resolved database URL or None if not configured
        
    Note:
        Checks in order: explicit parameter, DB_URL env var, DATABASE_URL env var
    """
    resolved = db_url or os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")
    if not resolved:
        logger.debug("No database URL configured (DB_URL or DATABASE_URL)")
    return resolved


def _get_dashboard_etl_config() -> ETLConfig:
    """
    Get ETL configuration for the dashboard, mirroring etl_data_explorer.ipynb.

    This configuration ensures proper data-loading and processing with correct
    data types and engineers the required features including earnings analytics.
    """
    return ETLConfig(
        extraction=DataExtractionConfig(
            normalize_column_names=True,
            source_type="database",
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
        currency_conversion=CurrencyConversionConfig(
            enabled=False,
            target_currency="USD",
            suffix="_usd",
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
            enabled=False,
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
        "file_size_mb": (output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_data_store(store_data: Optional[str]) -> pd.DataFrame:
    """Parse Dash store JSON data back to DataFrame.

    Returns empty DataFrame on failure.
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

    resolved_reference_date = resolve_reference_date(df, reference_date)

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
    formatted_cols = add_formatted_date_columns(enriched_df, date_columns_for_format)

    return enriched_df, resolved_reference_date, formatted_cols


def load_data_csv_first(
        *,
        data_dir: Optional[Path] = None,
        db_url: Optional[str] = None,
        feature_preset: str = "comprehensive",
        force_etl: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Load data preferring CSV export, falling back to ETL.

    Args:
        data_dir: Directory containing CSV data files
        db_url: Database connection URL (resolves from env if not provided)
        feature_preset: Feature engineering preset ('comprehensive', 'basic', etc.)
        force_etl: If True, always run ETL; if False, try cached CSV first

    Returns:
        Tuple of (DataFrame, source_label) where source_label describes the data source
        and ETL metrics summary, or error message if failed.
        
    Note:
        DB_URL resolution happens at call time, not at module import time.
        This ensures environment variables set after import are respected.
    """
    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    # Resolve DB URL at runtime (code_guidelines.md Section 2.2)
    resolved_db_url = _resolve_db_url(db_url)

    # Fast path: load from exported CSV if it exists and is recent
    if not force_etl and DEFAULT_CSV_EXPORT_PATH.exists():
        try:
            df = pd.read_csv(DEFAULT_CSV_EXPORT_PATH)
            _validate_explorer_columns(df, "csv_export")
            logger.info(f"Loaded {len(df):,} rows from cached CSV: {DEFAULT_CSV_EXPORT_PATH}")
            return df, "csv_export"
        except Exception as e:
            logger.warning(f"Failed to load CSV export: {e}, falling back to ETL")

    # Slow path: run ETL pipeline with dashboard-specific config
    try:
        # Determine source based on available configuration
        source: Literal["csv", "db"] = "db" if resolved_db_url else "csv"

        if source == "db":
            logger.info(f"Loading data from database...")
        else:
            logger.info(f"Loading data from CSV directory: {resolved_data_dir}")

        # Use dashboard ETL config (mirrors etl_data_explorer.ipynb)
        etl_config = _get_dashboard_etl_config()
        etl_config.feature_engineering.preset = feature_preset

        # Run ETL pipeline using run_etl_pipeline (code_guidelines.md Section 7.5)
        if source == "db" and resolved_db_url:
            df, metrics = _run_etl_pipeline(
                source="db",
                db_url=resolved_db_url,
                config=etl_config,
                return_metrics=True,
            )
        else:
            df, metrics = _run_etl_pipeline(
                source="csv",
                data_dir=str(resolved_data_dir),
                config=etl_config,
                return_metrics=True,
            )

        # Validate ETL output
        if df is None or df.empty:
            error_msg = "ETL returned empty DataFrame"
            logger.error(error_msg)
            return pd.DataFrame(), error_msg

        # Apply temporal enrichments
        df, resolved_reference_date, formatted_cols = _apply_temporal_enrichments(df)
        if resolved_reference_date is not None:
            logger.info(
                "Reference date resolved to %s",
                resolved_reference_date.strftime(DATE_DISPLAY_FORMAT),
            )
        if formatted_cols:
            logger.debug("Formatted date columns added: %s", ", ".join(formatted_cols))

        # Export to CSV for next time (non-blocking)
        if not df.empty:
            try:
                export_equities_data(df)
                logger.debug(f"Exported data to {DEFAULT_CSV_EXPORT_PATH}")
            except Exception as e:
                logger.debug(f"Non-critical: Failed to export equities data: {e}")

        # Return the metrics summary as the source label for the status bar
        source_label = metrics.summary() if metrics else f"etl_{source}"
        _validate_explorer_columns(df, f"etl_{source}")

        logger.info(f"ETL complete: {len(df):,} rows, {len(df.columns)} columns from {source}")
        return df, source_label

    except Exception as e:
        error_msg = f"ETL failed: {str(e)[:100]}"
        logger.error(f"ETL pipeline failed during dashboard load: {e}", exc_info=True)
        return pd.DataFrame(), error_msg


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
    reference_date = resolve_reference_date(df, None)
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
        logger.error(f"Failed to generate dashboard artifacts: {e}")
        return {"error": str(e), "artifacts": {}}

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
        data_source: Literal["auto", "csv", "db"] = "auto",
        data_dir: Optional[str | Path] = None,
        db_url: Optional[str] = None,
        feature_preset: str = "comprehensive",
        limit: Optional[int] = None,
) -> pd.DataFrame:
    """Consolidated data loading entry point for dashboards.

    Handles CSV/DB selection, ETL triggering, and temporal enrichment.
    
    Args:
        data_source: Data source selection ('auto', 'csv', 'db')
        data_dir: Directory for CSV files
        db_url: Database URL (resolves from env if not provided)
        feature_preset: Feature engineering preset
        limit: Optional row limit for testing
        
    Returns:
        DataFrame with loaded and enriched data, or empty DataFrame on failure
    """
    # Resolve DB URL at runtime
    resolved_db_url = _resolve_db_url(db_url)

    try:
        if data_source == "auto":
            df, source = load_data_csv_first(
                data_dir=Path(data_dir) if data_dir else None,
                db_url=resolved_db_url,
                feature_preset=feature_preset,
                force_etl=False,
            )
        else:
            etl_config = _get_dashboard_etl_config()
            etl_config.feature_engineering.preset = feature_preset

            # Use run_etl_pipeline with correct source parameter
            if data_source == "db" and resolved_db_url:
                df, _ = _run_etl_pipeline(
                    source="db",
                    db_url=resolved_db_url,
                    config=etl_config,
                    return_metrics=True,
                )
            else:
                df, _ = _run_etl_pipeline(
                    source="csv",
                    data_dir=str(Path(data_dir) if data_dir else DEFAULT_DATA_DIR),
                    config=etl_config,
                    return_metrics=True,
                )

        if df is not None and not df.empty:
            df, _, _ = _apply_temporal_enrichments(df)

        if limit is not None and limit > 0 and df is not None:
            return df.head(int(limit)).copy()
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load data in load_data: {e}")
        return pd.DataFrame()


def load_alerts_payload(path: str | Path = DEFAULT_ALERTS_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_explorer_columns(df: pd.DataFrame, source_label: str) -> None:
    """Log warnings for missing DEFAULT_EXPLORER_COLUMNS in loaded data."""
    if df is None or df.empty:
        return

    missing_cols = [c for c in DEFAULT_EXPLORER_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"Data source '{source_label}' is missing explorer columns: {missing_cols}. "
            "Some Data Explorer features may be limited."
        )
    if "name" not in df.columns:
        logger.warning(
            f"Data source '{source_label}' is missing 'name' column. "
            "Stock names will not be displayed in the Data Explorer."
        )
```

```python
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
from sqlalchemy.exc import SQLAlchemyError

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
        or empty string if not configured.
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


def _resolve_db_url_runtime(db_url: Optional[str] = None) -> Optional[str]:
    """Resolve database URL at runtime.
    
    This function is called at runtime (not import time) to ensure
    environment variables set after module import are respected.
    
    Args:
        db_url: Optional explicit database URL
        
    Returns:
        Resolved database URL or None
    """
    return db_url or os.environ.get("DB_URL") or os.environ.get("DATABASE_URL")


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
            count_result = conn.execute(text("SELECT COUNT(*) as C FROM public.equities"))
            row_count = count_result.scalar()

            if row_count == 0:
                raise RuntimeError("equities table is empty. Run import_equities_data.sql first.")

            logger.info(f"Found {row_count:,} rows in public.equities")

        # Load all data from equities table for current date
        query = text('SELECT * FROM public.equities WHERE "Reference Date" = current_date - 1')
        df = pd.read_sql(query, engine)

        # If no data for yesterday, try loading all data
        if df.empty:
            logger.warning("No data for current_date - 1, loading most recent data")
            query = text('''
                SELECT * FROM public.equities 
                WHERE "Reference Date" = (SELECT MAX("Reference Date") FROM public.equities)
            ''')
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
            require_target_column=False,  # Dashboard doesn't require target
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


def run_dashboard_etl_pipeline(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ETL transformation pipeline on raw equities data.
    
    Note: This is a local ETL runner for the dashboard, distinct from
    the imported run_etl_pipeline from finance_ml.etl.pipeline.

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

# =============================================================================
# Constants for data size warnings
# =============================================================================
LARGE_DATASET_ROW_THRESHOLD = 6000
LARGE_DATASET_COL_THRESHOLD = 300


# =============================================================================
# Helper functions for create_app
# =============================================================================


def _load_initial_data(
        data_source: DataSource,
        effective_db_url: Optional[str],
        data_dir: Optional[str | Path],
) -> tuple[pd.DataFrame, DataSource]:
    """
    Load initial data with proper fallback chain.

    Args:
        data_source: Requested data source ('auto', 'csv', 'db')
        effective_db_url: Resolved database URL
        data_dir: Directory for CSV files

    Returns:
        Tuple of (DataFrame, actual_source_used)
    """
    # Handle missing DB URL upfront
    if not effective_db_url and data_source == "db":
        logger.warning(
            "Database source requested but no DB_URL configured. Falling back to CSV source."
        )
        data_source = "csv"

    # Attempt database loading
    if data_source == "db" or (data_source == "auto" and effective_db_url):
        try:
            logger.info("Loading data from postgres.public.equities...")
            raw_data = load_from_equities_table(effective_db_url)
            df = run_dashboard_etl_pipeline(raw_data)
            logger.info(
                f"ETL Pipeline complete: {len(df):,} rows, {len(df.columns)} features"
            )
            return df, "db"
        except SQLAlchemyError as db_error:
            logger.warning(f"Database connection failed: {db_error}")
            if data_source == "db":
                # Explicit db request failed, try csv fallback
                data_source = "csv"
            else:
                # Auto mode, continue to csv
                pass
        except (RuntimeError, ValueError) as data_error:
            logger.warning(f"Data loading/validation failed: {data_error}")
            data_source = "csv"
        except pd.errors.EmptyDataError:
            logger.warning("Database returned empty data, falling back to CSV")
            data_source = "csv"
        except Exception as e:
            logger.warning(f"Unexpected error loading from database: {e}")
            data_source = "csv"

    # CSV fallback or explicit CSV request
    try:
        df = load_data(data_source="csv", data_dir=data_dir, db_url=None)

        if df is None:
            logger.warning("load_data returned None, using empty DataFrame")
            return pd.DataFrame(), "none"

        if df.empty:
            logger.warning("load_data returned empty DataFrame")
            return pd.DataFrame(), "none"

        logger.info(f"Loaded {len(df):,} rows from CSV source")
        return df, "csv"
    except Exception as e:
        logger.error(f"CSV loading failed: {e}")
        return pd.DataFrame(), "none"


def _validate_safe_path(filename: str, base_dir: Path) -> Path:
    """
    Validate that a filename resolves to a path within the base directory.

    Args:
        filename: User-provided filename/path
        base_dir: Directory that must contain the resolved path

    Returns:
        Resolved safe path

    Raises:
        ValueError: If path is invalid or escapes base directory
    """
    from pathlib import PurePosixPath, PureWindowsPath

    # Reject obviously malicious patterns
    # Check both POSIX and Windows path interpretations
    for path_cls in (PurePosixPath, PureWindowsPath):
        parsed = path_cls(filename)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"Invalid path: {filename}")

    # Resolve and verify containment
    base_resolved = base_dir.resolve()
    requested_path = (base_dir / filename).resolve()

    # Use os.path.commonpath for reliable containment check
    try:
        common = Path(os.path.commonpath([base_resolved, requested_path]))
        if common != base_resolved:
            raise ValueError(f"Path escapes base directory: {filename}")
    except ValueError:
        # commonpath raises ValueError if paths are on different drives (Windows)
        raise ValueError(f"Invalid path: {filename}")

    return requested_path


def _prepare_data_store(df: pd.DataFrame) -> Optional[str]:
    """
    Prepare DataFrame for dcc.Store with size warnings.

    Args:
        df: DataFrame to serialize

    Returns:
        JSON string or None if empty
    """
    if df is None or df.empty:
        return None

    row_count, col_count = df.shape
    if row_count > LARGE_DATASET_ROW_THRESHOLD or col_count > LARGE_DATASET_COL_THRESHOLD:
        logger.warning(
            f"Large dataset detected ({row_count:,} rows, {col_count} columns). "
            "Consider implementing server-side caching or pagination for better performance."
        )

    return df.to_json(orient="split")


def create_app(
        *,
        data_source: DataSource = "db",
        data_dir: Optional[str | Path] = None,
        db_url: Optional[str] = None,
        load_on_start: bool = True,
) -> dash.Dash:
    """Create Dash app instance.

    Args:
        data_source: Where to load data from ('auto', 'csv', 'db'). Default is 'db'.
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
    # Resolve DB URL at runtime (code_guidelines.md Section 2.2)
    effective_db_url = _resolve_db_url_runtime(db_url)

    if load_on_start:
        try:
            initial_df, actual_source = _load_initial_data(data_source, effective_db_url, data_dir)
            logger.info(f"Initial data loaded from '{actual_source}': {len(initial_df):,} rows")
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

        outputs_dir = PROJECT_ROOT / "outputs"
        try:
            safe_path = _validate_safe_path(filename, outputs_dir)
            # Verify file exists before serving
            if not safe_path.is_file():
                abort(404)
        except ValueError:
            abort(403)
        except OSError:
            abort(400)

        return send_from_directory(outputs_dir, filename)

    # Determine initial status message
    if initial_df.empty:
        initial_status = "⚠ No data loaded. Click 'Load / Refresh Data' to load."
    else:
        initial_status = f"✓ Loaded {len(initial_df):,} rows"

    # Layout
    app.layout = html.Div(
        [
            html.H1("📈 Equities Analytics Dashboard", style={"textAlign": "center"}),
            dcc.Store(
                id="equities-data-store",
                data=_prepare_data_store(initial_df),
                storage_type="memory",
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
                                children=initial_status,
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
```

## Summary of Key Changes

### `data_utils.py`

1. **Added `_resolve_db_url()` helper** - Centralized DB URL resolution at runtime (not import time)
2. **Renamed import** - `from finance_ml.etl.pipeline import run_etl_pipeline as _run_etl_pipeline` to avoid naming
   conflicts
3. **Improved `load_data_csv_first()`**:
    - Resolves DB URL at call time using `_resolve_db_url()`
    - Added comprehensive logging
    - Better error messages with truncated details
    - Validates ETL output before returning
4. **Improved `load_data()`** - Uses the centralized `_resolve_db_url()` function

### `equities_dashboard_app.py`

1. **Renamed local ETL function** - `run_etl_pipeline()` → `run_dashboard_etl_pipeline()` to avoid conflict with the
   imported function
2. **Added `_resolve_db_url_runtime()`** - For runtime DB URL resolution
3. **Improved `load_from_equities_table()`** - Added fallback to load most recent data if no data for `current_date - 1`
4. **Improved `_load_initial_data()`** - Better exception handling with proper CSV fallback
5. **Added initial status message** - Shows helpful message when no data is loaded
6. **Fixed `_prepare_data_store()`** - Added None check for DataFrame

These changes ensure the refresh button properly resolves the database URL at runtime and provides clear error messages
when data loading fails.