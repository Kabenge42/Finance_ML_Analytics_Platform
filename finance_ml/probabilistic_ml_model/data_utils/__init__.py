"""
Data utilities for the probabilistic_ml_model package.

Re-exports key functions from data_utils and inference_schema submodules
so callers can import from the subpackage directly:

    from finance_ml.probabilistic_ml_model.data_utils import (
        load_equities_data_from_db,
        load_feature_data_from_db,
        backfill_feature_columns,
        ARVIZ_AVAILABLE,
        build_monte_carlo_inference_data,
        summarize_inference_data,
    )
"""

from finance_ml.probabilistic_ml_model.data_utils.data_utils import (
    load_equities_data_from_db,
    load_feature_data_from_db,
    backfill_feature_columns,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
    reorder_with_identifiers,
    load_identifier_columns,
    get_identifier_cols_set,
    compute_metric_statistics,
    load_feature_categories_from_db,
)

# Inference schema re-exports (guarded — ArviZ is optional)
try:
    from finance_ml.probabilistic_ml_model.data_utils.inference_schema import (
        ARVIZ_AVAILABLE,
        EquityCoordinates,
        IdentifierCoordinates,
        FeatureCoordinates,
        build_monte_carlo_inference_data,
        build_beat_probability_inference_data,
        build_credit_risk_inference_data,
        build_accounting_anomaly_inference_data,
        build_category_analysis_inference_data,
        build_feature_view_inference_data,
        build_resampled_technical_inference_data,
        summarize_inference_data,
        load_equities_schema_metadata_from_db,
        load_feature_registry_metadata_from_db,
        load_mv_equities_spec_from_db,
        FEATURE_VIEW_REGISTRY,
    )
except ImportError:
    ARVIZ_AVAILABLE = False