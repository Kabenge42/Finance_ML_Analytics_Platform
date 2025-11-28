"""Clean public API for the Finance ML Analytics Platform.

This module provides a *stable*, higher-level facade over the internal
``finance_ml.ml_workflow`` subpackages. New code and notebooks should
prefer importing from :mod:`finance_ml.api` instead of reaching into
deep subpackage paths or relying on legacy re-exports from
``finance_ml.__init__``.

The facade is intentionally focused and opinionated: it surfaces the
most common end-to-end workflow steps (data loading, preprocessing,
feature engineering, models, analytics, reporting) while keeping the
underlying implementation details in the ml_workflow modules.

This file does **not** introduce new behavior; it simply re-exports
existing, well-tested functions under a clean namespace so that future
refactors can evolve internal structure without breaking user code.

Typical usage
-------------

Data loading and preprocessing::

    from finance_ml.api import load_from_csv, load_from_db, prepare_phase91_data

    df = load_from_csv("data/screening_us.csv")
    df_prepared = prepare_phase91_data(df)

Feature engineering::

    from finance_ml.api import build_features, PresetName

    features = build_features(df_prepared, preset="comprehensive")

Classification and regression::

    from finance_ml.api import (
        create_enhanced_event_labels,
        train_event_classifier,
        train_sector_specific_models,
        train_quantile_regressor,
    )

    labels = create_enhanced_event_labels(features)
    clf_result = train_event_classifier(features, labels)

    sector_models, sector_results = train_sector_specific_models(
        df=features,
        feature_cols=[c for c in features.columns if c not in {"sector", "target"}],
        target_col="target",
        sector_col="sector",
    )

Analytics and reporting::

    from finance_ml.api import (
        calculate_mispricing_score,
        rank_undervalued_stocks,
        calculate_financial_metrics_dashboard,
    )

    mispriced = calculate_mispricing_score(features)
    top_ideas = rank_undervalued_stocks(mispriced, top_n=20)

    dashboard = calculate_financial_metrics_dashboard(mispriced)

The exact argument signatures and behaviors of the underlying functions
are documented in their respective modules; this facade simply provides
convenient import paths.
"""

from __future__ import annotations

# NOTE: We keep imports focused and avoid pulling in heavy optional
# dependencies here. The symbols we re-export are already imported
# transitively when users do ``import finance_ml`` or when individual
# submodules are used directly, so this facade does not materially
# increase import cost.

# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------

from finance_ml.ml_workflow.preprocessing import (
    # Data loading and normalization
    load_from_csv,
    load_from_db,
    normalize_columns,
    # High-level preprocessing pipeline (Phase 9.1)
    prepare_phase91_data,
)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

from finance_ml.ml_workflow.features.api import (
    build_features,
    PresetName,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

from finance_ml.ml_workflow.classification import (
    create_enhanced_event_labels,
    prepare_classification_data,
    train_xgboost_classifier,
    train_lightgbm_classifier,
    train_catboost_classifier,
    train_neural_network_classifier,
    train_stacking_classifier,
    compare_classifiers,
)


# ---------------------------------------------------------------------------
# Regression (core + advanced sector models / quantiles)
# ---------------------------------------------------------------------------

# Legacy-named helpers mapped to structured classification modules to avoid
# importing deprecated ml_workflow.models (which pulls legacy internals).
from finance_ml.ml_workflow.classification.labels import (  # noqa: E402
    create_enhanced_event_labels as create_event_labels,
)
from finance_ml.ml_workflow.classification.models import (  # noqa: E402
    fit_classifier as train_event_classifier,
)

try:  # pragma: no cover - allow API import without full regression stack
    from finance_ml.ml_workflow.regression.dataset import (
        prepare_regression_data,
    )
    from finance_ml.ml_workflow.regression.models import (
        train_stacking_regressor,
        compare_regressors,
    )
    from finance_ml.ml_workflow.regression.quantile import (
        train_quantile_regressor,
    )
    from finance_ml.ml_workflow.regression.dataset import (
        train_sector_specific_models,
    )
except Exception:  # pragma: no cover - optional at import time
    # Defer errors until these symbols are actually accessed by callers that
    # need them. Report config tests and light imports shouldn't fail here.
    pass


# ---------------------------------------------------------------------------
# Analytics (mispricing, portfolio optimization)
# ---------------------------------------------------------------------------

from finance_ml.ml_workflow.analytics import (
    # Mispricing and ranking
    calculate_mispricing_score,
    calculate_risk_adjusted_mispricing,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
)
from finance_ml.ml_workflow.analytics.portfolio import (
    optimize_portfolio_max_sharpe as optimize_portfolio,
)


# ---------------------------------------------------------------------------
# Reporting / dashboards
# ---------------------------------------------------------------------------

from finance_ml.ml_workflow.reporting import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data,
)


__all__ = [
    # Data / preprocessing
    "load_from_csv",
    "load_from_db",
    "normalize_columns",
    "prepare_phase91_data",
    # Features
    "build_features",
    "PresetName",
    # Classification
    "create_enhanced_event_labels",
    "prepare_classification_data",
    "train_xgboost_classifier",
    "train_lightgbm_classifier",
    "train_catboost_classifier",
    "train_neural_network_classifier",
    "train_stacking_classifier",
    "compare_classifiers",
    # Regression
    "create_event_labels",
    "train_event_classifier",
    "prepare_regression_data",
    "train_stacking_regressor",
    "compare_regressors",
    "train_quantile_regressor",
    "train_sector_specific_models",
    # Analytics
    "calculate_mispricing_score",
    "calculate_risk_adjusted_mispricing",
    "rank_undervalued_stocks",
    "rank_overvalued_stocks",
    "rank_stocks_by_sector",
    "optimize_portfolio",
    # Reporting
    "calculate_financial_metrics_dashboard",
    "generate_data_quality_alerts",
    "prepare_plotly_dashboard_data",
]
