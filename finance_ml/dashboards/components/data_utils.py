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
from finance_ml.features.advanced import engineer_temporal_features
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_features,
    ETLConfig,
)

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

    Returns:
        Tuple of (DataFrame, source_label) where source_label is one of:
        'csv_export', 'etl_csv', 'etl_db' or a metrics summary.
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
        logger.error(f"ETL pipeline failed during dashboard load: {e}")
        return pd.DataFrame(), f"ETL failed: {str(e)[:50]}"


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
    """
    try:
        if data_source == "auto":
            df, source = load_data_csv_first(
                data_dir=Path(data_dir) if data_dir else None,
                db_url=db_url,
                feature_preset=feature_preset,
                force_etl=False,
            )
        else:
            etl_config = _get_dashboard_etl_config()
            etl_config.feature_engineering.preset = feature_preset
            result = etl_with_features(
                source=data_source,
                data_dir=Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
                db_url=db_url,
                config=etl_config,
                return_metrics=False,  # Return DataFrame only, not tuple
            )
            df = result

        if not df.empty:
            df, _, _ = _apply_temporal_enrichments(df)

        if limit is not None and limit > 0:
            return df.head(int(limit)).copy()
        return df
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
