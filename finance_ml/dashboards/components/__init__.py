"""Dashboard component utilities extracted from equities_dashboard_app."""

from .data_utils import (
    ARTIFACTS_DIR,
    ARTIFACTS_METADATA_PATH,
    DEFAULT_CSV_EXPORT_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_METADATA_PATH,
    PROJECT_ROOT,
    _apply_temporal_enrichments,
    _get_dashboard_etl_config,
    _validate_explorer_columns,
    export_equities_data,
    generate_dashboard_artifacts,
    load_alerts_payload,
    load_data,
    load_data_csv_first,
    parse_data_store,
)
from .explorer import build_explorer_column_options
from .filters import _safe_options, apply_filters
from .kpi_cards import _kpi_cards, _monitoring_kpi_cards
from .charts import _target_vs_price_scatter, _market_cap_distribution
from .earnings import create_earnings_events_chart
from .artifacts import _list_artifacts, _render_artifact
from .utils import (
    compute_surprise,
    create_empty_state_figure,
    validate_required_columns,
    create_missing_columns_warning,
    _coerce_list,
    _severity_style,
    _alerts_to_rows,
)
from .temporal_utils import (
    get_reference_date,
    compute_days_to_earnings,
)

__all__ = [
    "ARTIFACTS_DIR",
    "ARTIFACTS_METADATA_PATH",
    "DEFAULT_CSV_EXPORT_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_METADATA_PATH",
    "PROJECT_ROOT",
    "_apply_temporal_enrichments",
    "_get_dashboard_etl_config",
    "load_alerts_payload",
    "_validate_explorer_columns",
    "export_equities_data",
    "generate_dashboard_artifacts",
    "load_data",
    "load_data_csv_first",
    "parse_data_store",
    "build_explorer_column_options",
    "_safe_options",
    "apply_filters",
    "_kpi_cards",
    "_monitoring_kpi_cards",
    "_target_vs_price_scatter",
    "_market_cap_distribution",
    "create_earnings_events_chart",
    "_list_artifacts",
    "_render_artifact",
    "compute_surprise",
    "create_empty_state_figure",
    "validate_required_columns",
    "create_missing_columns_warning",
    "_coerce_list",
    "_severity_style",
    "_alerts_to_rows",
    "get_reference_date",
    "compute_days_to_earnings",
]
