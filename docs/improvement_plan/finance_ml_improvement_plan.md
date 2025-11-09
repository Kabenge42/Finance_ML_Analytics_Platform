### Executive summary

The current `finance_ml/ml_workflow` package is powerful but fragmented. There is clear functional overlap and
split-brain APIs across pairs of modules:

- `classification.py` vs `classification_enhanced.py`
- `features.py` vs `advanced_features.py`
- `models.py` vs `advanced_models.py`
- `data.py` vs `advanced_preprocessing.py` (plus some preprocessing utilities in `data.py`)

Below I propose a consolidated, layered module structure aligned 1:1 with the business/phase objectives (9.1–9.8), plus
concrete refactor steps, public APIs, deprecation shims, and a migration checklist that will keep existing tests green
while reducing maintenance surface.

---

### Proposed target package layout (aligned to Phases 9.1–9.8)

- finance_ml/
    - ml_workflow/
        - core/
            - config.py (global constants, column naming, seeds)
            - types.py (TypedDicts/Protocols for dataset and model artifacts)
            - utils.py (small helpers, logging wrappers)
        - data/
            - loading.py (split from `data.load_from_csv`, `data.load_from_db`)
            - validation.py (split from `data.validate_schema`, range checks)
            - versioning.py (from `data_versioning.py`)
            - catalog.py (from `data_catalog.py`)
        - preprocessing/  [Phase 9.1]
            - imputation.py (from `advanced_preprocessing.apply_enhanced_imputation_strategy_4step`,
              KNN/median/zero/price)
            - outliers.py (from `advanced_preprocessing` and `data.py` iqr/zscore/isolation)
            - scaling.py (from `advanced_preprocessing.create_scaler_pipeline` + `scale_features`)
            - quality.py (from `advanced_preprocessing.DataQualityReport`, scoring)
            - pipeline.py (a high-level `prepare_phase91_data` wrapper; prev `prepare_phase95_data` remains for 9.5)
        - eda/  [Phase 9.2]
            - eda.py (quick summaries, distributions, corr, sector slices)
            - benchmarking.py (your existing `benchmarking` module; moved here)
            - reports.py (glue for static/HTML reports; optional)
        - features/  [Phase 9.3]
            - core.py (merge of `features.py` basic ratios, margins, vol, CAGR)
            - advanced.py (merge of `advanced_features.py`: sector-specific, microstructure, interactions, relative
              values)
            - selection.py (mutual info, RF, SHAP, RFE utilities)
            - api.py (`build_features(df, preset=...)` orchestrator)
        - classification/  [Phase 9.4]
            - models.py (unified from both `classification.py` and `classification_enhanced.py` – XGB/LGBM/CB/SVM/NN,
              voting, stacking)
            - tuning.py (Optuna hyperparam search + sector-stratified CV)
            - evaluation.py (metrics, calibration, confusion matrices, per-sector eval)
            - labels.py (from `classification.create_enhanced_event_labels`)
        - regression/  [Phase 9.5]
            - models.py (unified from `models.py` + `advanced_models.py`: linear, GBMs, histGB, RF/ET, NN,
              stacking/voting)
            - constraints.py (`NonNegativeRegressionWrapper` lives here)
            - quantile.py (quantile training and inference; current inner class extracted)
            - tuning.py (Optuna search for regressors)
            - dataset.py (prepare_regression_data, feature extraction helpers)
        - evaluation/  [Phase 9.6]
            - metrics.py (MAE/RMSE/MAPE/R2, sector-wise breakdowns)
            - analysis.py (residuals, error histograms, SHAP, ablation utilities)
        - analytics/  [Phase 9.7]
            - mispricing.py (mispricing score; rankings by sector/region)
            - analyst_comparison.py (existing module moved here)
            - visualizations.py (bar/scatter/heatmaps; reporting-friendly)
            - portfolio.py (existing `portfolio_optimization` moved here)
            - risk.py (moved `risk_metrics.py` now under `ml_workflow`)
        - reporting/  [Phase 9.8]
            - export.py (CSV/Excel with formatting)
            - dashboards.py (optional; stubs if not yet present)

Top-level convenience imports remain available via `finance_ml.__init__` with re-exports to avoid breaking
notebooks/CLI.

---

### Why this helps (and what we remove)

- Reduces duplication and drift:
    - Classification models + tuning live together with one API.
    - Features basic/advanced merge with a single place for sector-specific enrichments.
    - Regression models unified; quantile, stacking, and non-negative constraints sit side-by-side.
    - Outlier/winsorization/imputation logic consolidated under one preprocessing package.
- Aligns directly with the 8-phase roadmap: each subpackage maps to a phase, easing discoverability, onboarding, and
  test scoping.
- Allows clean test grouping by feature area (as your guidelines recommend).

---

### Module-by-module consolidation plan

#### 1) Preprocessing: `data.py` + `advanced_preprocessing.py`

- Move all advanced functions from `advanced_preprocessing.py` into `preprocessing/`:
    - Keep these public names (unchanged), but relocate:
        - `apply_enhanced_imputation_strategy_4step`
        - `apply_zero_imputation`, `apply_price_imputation`, `apply_knn_imputation_enhanced`, `apply_median_imputation`
        - `detect_outliers_iqr`, `detect_outliers_zscore`, `detect_outliers_isolation_forest`
        - `winsorize_by_sector`
        - `create_scaler_pipeline`, `scale_features`
    - Extract `DataQualityReport`, `calculate_data_quality_score` into `preprocessing/quality.py`.
- Remove duplicated helpers from `data.py` or re-export them from preprocessing:
    - `detect_outliers_*`, `winsorize_by_sector` currently appear in both `data.py` and `advanced_preprocessing.py`.
      Source-of-truth goes to `preprocessing/` and `data.py` re-exports with `DeprecationWarning`.
- Provide orchestration wrappers:
    - `preprocessing.pipeline.prepare_phase91_data(df, ...)` maps to Phase 9.1 baseline.
    - Preserve `prepare_phase95_data(df, ...)` for regression-specific prep; move it into `preprocessing.pipeline` or
      `regression.dataset` as appropriate.

Impact on Phases: firmly anchors Phase 9.1 and cleans up drift between `data.py` and advanced preprocessing.

#### 2) Features: `features.py` + `advanced_features.py`

- Merge into `features/` package:
    - `features/core.py` keeps: `_safe_div`, `engineer_basic_ratios`, `engineer_margin_features`,
      `engineer_volatility_features`, `engineer_revenue_cagr`, and `build_features_and_target`.
    - `features/advanced.py` keeps: valuation/profitability/leverage/liquidity/efficiency/growth, sector-specific,
      temporal, microstructure, nonlinear, interactions, relative values, and `build_comprehensive_features`.
    - `features/selection.py` keeps: mutual info, RF, SHAP, RFE importance.
- Add the single entry-point API:
    - `features/api.py` with:
        -
        `build_features(df, preset="basic|comprehensive|sector_optimized", include_interactions=True, include_relative=True)`
        - This composes `core` + `advanced` under consistent column naming conventions.
- Ensure sector column consistency (`sector` everywhere). If incoming data uses differently-cased names, normalize
  earlier (see `data.normalize_columns`).

Impact on Phases: Phase 9.3 gets a single cohesive API and feature-set presets.

#### 3) Classification: `classification.py` + `classification_enhanced.py`

- Unify into `classification/` with a crisp split:
    - `classification/models.py` contains trainers currently in `classification.py`: `train_xgboost_classifier`,
      `train_lightgbm_classifier`, `train_catboost_classifier`, `train_svm_classifier`,
      `train_neural_network_classifier`, `train_voting_classifier`, `train_stacking_classifier`, plus data prep helpers
      like `_prepare_categorical_features`, `prepare_classification_data`.
    - `classification/tuning.py` consolidates Optuna-based tuning (`optimize_classifier_hyperparameters`) and
      `cross_validate_with_sector_stratification`.
    - `classification/evaluation.py` consolidates `evaluate_classification`, `plot_confusion_matrices`,
      `evaluate_classification_by_sector`, `plot_learning_curves`, `analyze_calibration`, SHAP support.
    - `classification/labels.py` keeps `create_enhanced_event_labels` and any future label strategies.
- Provide top-level orchestrators:
    - `fit_classifier(X, y, model="xgboost", params=None, tuning=None, cv=None, class_weighting=None)`
    - `compare_classifiers(...)` remains but calls into the split modules.
- Harmonize return types: always return a dict with keys `{model, metrics, y_pred, y_proba, artifacts}` to ease
  downstream usage (Phase 9.5 meta-features).

Impact on Phases: Phase 9.4 gains one API with optional tuning and sector-aware CV from the enhanced module.

#### 4) Regression: `models.py` + `advanced_models.py`

- Merge into `regression/` with clear responsibilities:
    - `regression/models.py`: `train_ridge_regressor`, `train_lasso_regressor`, `train_elastic_net_regressor`,
      `train_bayesian_ridge_regressor`, `train_polynomial_regressor`, `train_xgboost_regressor`,
      `train_lightgbm_regressor`, `train_catboost_regressor`, `train_histgb_regressor`, `train_random_forest_regressor`,
      `train_extra_trees_regressor`, `train_neural_network_regressor`, `train_voting_regressor`,
      `train_stacking_regressor`, `compare_regressors`.
    - `regression/constraints.py`: `NonNegativeRegressionWrapper` (kept intact to satisfy tests like
      `test_phase95_nonnegative_predictions.py`).
    - `regression/quantile.py`: extract the inner `QuantileRegressionModel` class and the
      `train_quantile_regression(_by_sector)` functions.
    - `regression/tuning.py`: `optimize_hyperparameters_optuna` for grids/bayesian/Optuna.
    - `regression/dataset.py`: `prepare_regression_data`, `extract_numeric_feature_columns`,
      `create_classification_interactions`, `prepare_features_for_training`, `train_sector_specific_models`.
    - Model IO: `save_model`, `load_model` moved to `regression/io.py` or kept in `models.py` if preferred.
- Normalize outputs: return a standard object (dataclass)
  `RegressionResult(model, metrics, predictions, oof_predictions, feature_importance, artifacts)` to simplify Phase
  9.6/9.7 consumption.

Impact on Phases: Phase 9.5, 9.6, and 9.7 get a unified regression API with optional non-negative constraints and
quantile bands.

#### 5) EDA, Evaluation, Analytics, Reporting

- EDA (Phase 9.2): If `advanced_eda` already exists outside `ml_workflow`, move into `ml_workflow/eda/` and re-export.
- Evaluation (Phase 9.6): Extract generic metrics and diagnostics from scattered modules into `ml_workflow/evaluation/`.
- Analytics (Phase 9.7): Move `analyst_comparison`, `portfolio_optimization`, and `risk_metrics` (now under
  `ml_workflow`) into `analytics/`. Add `mispricing.py` with your standard
  `(Predicted_Target - Last_Price) / Last_Price` functions and ranking utilities.
- Reporting (Phase 9.8): centralize export utilities and plotting-to-file helpers.

---

### Public API proposals per Phase

#### Phase 9.1 – Preprocessing

```
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data
X_prep, stats = prepare_phase91_data(df, sector_column="sector", price_column="last_price", n_neighbors=5, return_stats=True)
```

- Guarantees: zero/price/KNN/median 4-step imputation applied; sector-wise robust scaling optional; returns quality
  stats.

#### Phase 9.2 – EDA/Benchmarking

```
from finance_ml.ml_workflow.eda.eda import eda_summary
from finance_ml.ml_workflow.eda.benchmarking import generate_benchmarking_report
summary = eda_summary(X_prep)
report_path = generate_benchmarking_report(X_prep, out_dir)
```

#### Phase 9.3 – Feature engineering

```
from finance_ml.ml_workflow.features.api import build_features
X_feat = build_features(X_prep, preset="sector_optimized", include_interactions=True, include_relative=True)
```

#### Phase 9.4 – Classification

```
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels
from finance_ml.ml_workflow.classification.models import fit_classifier
labels = create_enhanced_event_labels(X_feat, method="price_momentum", threshold_positive=10, threshold_negative=-10)
clf_res = fit_classifier(X_feat, labels, model="lightgbm", tuning={"n_trials": 50}, cv={"sector_stratified": True})
```

- `clf_res.artifacts["probabilities"]` can be fed into regression meta-features.

#### Phase 9.5 – Regression (with classification meta-features)

```
from finance_ml.ml_workflow.regression.dataset import integrate_classification_features
from finance_ml.ml_workflow.regression.models import fit_regressor

X_reg = integrate_classification_features(X_feat, clf_res.artifacts["probabilities"])  # wraps your existing helper
reg_res = fit_regressor(X_reg, y=target, model="xgboost", ensure_nonnegative=True)
```

- Quantile models via `regression.quantile.train_quantile_regression`.

#### Phase 9.6 – Evaluation

```
from finance_ml.ml_workflow.evaluation.metrics import regression_report
from finance_ml.ml_workflow.evaluation.analysis import residual_analysis
metrics = regression_report(y_true, reg_res.predictions)
residuals_fig = residual_analysis(reg_res)
```

#### Phase 9.7 – Mispricing and analytics

```
from finance_ml.ml_workflow.analytics.mispricing import mispricing_scores, rank_by_sector
scores = mispricing_scores(last_price, reg_res.predictions)
leaders = rank_by_sector(scores, sector_col="sector", top_n=10)
```

#### Phase 9.8 – Reporting

```
from finance_ml.ml_workflow.reporting.export import export_predictions
export_predictions(reg_res, out_path="outputs/regression_predictions.csv")
```

---

### Concrete refactor tasks (actionable checklist)

1) Create subpackages and move code:
    - preprocessing/{imputation,outliers,scaling,quality,pipeline}.py
    - features/{core,advanced,selection,api}.py
    - classification/{models,tuning,evaluation,labels}.py
    - regression/{models,constraints,quantile,tuning,dataset,io}.py
    - eda/{eda,benchmarking,reports}.py
    - evaluation/{metrics,analysis}.py
    - analytics/{mispricing,analyst_comparison,portfolio,risk}.py
    - reporting/{export,dashboards}.py
2) Extract duplicated helpers from `data.py` into preprocessing and re-export.
3) Standardize function signatures & return types:
    - All `train_*` return dicts with `{model, metrics, y_pred, y_proba?}`.
    - Dataset prep returns `(X_train, X_test, y_train, y_test, meta)` or a dataclass.
4) Column naming and schema:
    - Enforce `normalize_columns` early (keep `last_price`, `price_target`, `sector`, `region`), ensure all downstream
      modules assume normalized names.
5) Deprecation shims:
    - Keep old files `classification.py`, `classification_enhanced.py`, `features.py`, `advanced_features.py`,
      `models.py` with imports from the new locations and `warnings.warn(..., DeprecationWarning)` for 1–2 releases.
6) Update `finance_ml/__init__.py` to re-export new public APIs so notebooks and CLI continue to work.
7) Documentation: update README.md and the “Development Guidelines” with the new import paths and phase mappings.
8) Bump `MODEL_VERSION` (e.g., to `v9_8`) and note the structural changes in `IMPROVEMENT_PLAN.md`.
9) Refresh CLI mapping: ensure `finance-ml`, `finance-ml-analyze`, `finance-ml-validate` import new modules.

---

### Tests and backward compatibility

- Keep all existing test modules running by:
    - Providing deprecation shims with identical names and behavior.
    - Keeping `NonNegativeRegressionWrapper` semantics unchanged.
    - Ensuring `apply_enhanced_imputation_strategy_4step` path remains importable.
- As follow-ups, gradually migrate tests to the new imports:
    - Classification: use `ml_workflow.classification.*`
    - Regression: use `ml_workflow.regression.*`
    - Features: use `ml_workflow.features.api.build_features`
    - Preprocessing: `ml_workflow.preprocessing.pipeline.prepare_phase91_data`

Selective execution strategies (from your guidelines) map neatly onto the new structure, enabling per-package test
discovery.

---

### Mapping old → new (quick reference)

- classification.py
    - `create_enhanced_event_labels` → `classification.labels.create_enhanced_event_labels`
    - `train_xgboost_classifier` → `classification.models.train_xgboost_classifier`
    - `compare_classifiers` → `classification.models.compare_classifiers`
    - `evaluate_classification` → `classification.evaluation.evaluate_classification`
    - `compute_shap_values` → `classification.evaluation.compute_shap_values`
- classification_enhanced.py
    - `optimize_classifier_hyperparameters` → `classification.tuning.optimize_classifier_hyperparameters`
    - `cross_validate_with_sector_stratification` → `classification.tuning.cross_validate_with_sector_stratification`
    - `analyze_calibration` → `classification.evaluation.analyze_calibration`
- features.py / advanced_features.py
    - All basic ratios/margins/vol/CAGR → `features.core.*`
    - Advanced sector/microstructure/interactions/relative → `features.advanced.*`
    - Importance methods → `features.selection.*`
    - `build_comprehensive_features` → `features.advanced.build_comprehensive_features`
    - New orchestrator → `features.api.build_features`
- models.py / advanced_models.py
    - All regressors, stacking/voting → `regression.models.*`
    - `NonNegativeRegressionWrapper` → `regression.constraints.NonNegativeRegressionWrapper`
    - Quantile models → `regression.quantile.*`
    - Hyperparam tuning → `regression.tuning.optimize_hyperparameters_optuna`
    - Data prep/extraction → `regression.dataset.*`

---

### Notebook and CLI updates

- Notebook (`ml_finance_model_main.ipynb`):
    - Replace imports like `from finance_ml import advanced_features, advanced_models` with the new subpackages:
        - `from finance_ml.ml_workflow.features import api as feat_api`
        - `from finance_ml.ml_workflow.classification import models as cls_models, labels as cls_labels`
        - `from finance_ml.ml_workflow.regression import models as reg_models, quantile as reg_quant`
        - `from finance_ml.ml_workflow.preprocessing import pipeline as prep`
- CLI scripts (`finance-ml`, etc.) should import from the new modules; keep old imports working via re-exports for one
  release.

---

### Risks and mitigations

- Risk: circular imports after splitting modules.
    - Mitigate with lean utility modules, avoid cross-imports; pass artifacts explicitly (e.g., classification
      probabilities → regression dataset).
- Risk: test flakiness due to modified defaults.
    - Keep defaults identical; only relocate code.
- Risk: user code breakage.
    - Provide deprecation shims with warnings; document migration in README.

---

### Effort and sequencing

- Week 1: Create package skeleton, move features + preprocessing, add re-exports; run fast + medium tests.
- Week 2: Merge classification modules; add tuning/evaluation splits; update docs; run slow tests selectively.
- Week 3: Merge regression modules; extract quantile/constraints/dataset; update CLI + notebook; finalize migration
  guide.

---

### Final alignment to business objectives

- Phase 9.1: Dedicated `preprocessing/` with 4-step imputation and quality scoring.
- Phase 9.2: `eda/` and `benchmarking/` enable deeper diagnostics and stats.
- Phase 9.3: `features/` presets and selection utilities streamline sector-optimized feature construction.
- Phase 9.4: `classification/` centralizes models, tuning, evaluation, and labels.
- Phase 9.5: `regression/` centralizes all regressors, constraints, and quantile models.
- Phase 9.6: `evaluation/` provides consistent metrics and error analysis across tasks.
- Phase 9.7: `analytics/` consolidates mispricing, risk, portfolio, and analyst comparison.
- Phase 9.8: `reporting/` standardizes exports and artifacts for stakeholders.

This structure removes duplication, clarifies responsibility boundaries, and gives your team a clean, phase-aligned API
surface that supports both notebook-first exploration and CLI/automation workflows without breaking existing code.

---

### v9_8 Implementation Status (Completed 2025-01-08)

**All Phases Completed:**

- ✅ Phase 9.1: preprocessing/ subpackage (imputation, outliers, scaling, quality, pipeline)
- ✅ Phase 9.2: eda/ subpackage (eda, benchmarking, reports)
- ✅ Phase 9.3: features/ subpackage (core, advanced, selection, api)
- ✅ Phase 9.4: classification/ subpackage (labels, tuning, models, evaluation)
- ✅ Phase 9.5: regression/ subpackage (constraints, dataset, models, quantile, tuning, io)
- ✅ Phase 9.6: evaluation/ subpackage (metrics, analysis)
- ✅ Phase 9.7: analytics/ subpackage (mispricing, analyst_comparison, portfolio, risk)
- ✅ Phase 9.8: reporting/ subpackage (dashboard_data, export)

**Deprecation & Backward Compatibility:**

- ✅ Added deprecation warnings to data.py for duplicated preprocessing functions (detect_outliers_iqr,
  detect_outliers_zscore, detect_outliers_iqr_advanced, detect_outliers_by_sector, winsorize_by_sector)
- ✅ Added module-level deprecation to advanced_features.py → features.advanced + features.selection
- ✅ Added module-level deprecation to models.py → regression.models + classification.models
- ✅ classification.py already has Phase 9.4 refactor notices
- ✅ features.py already has Phase 9.3 deprecation wrappers
- ✅ All old imports continue to work with DeprecationWarning

**Configuration & Versioning:**

- ✅ Bumped MODEL_VERSION from v8_3 to v9_8 in finance_ml/config.py
- ✅ Updated both default and from_env() method

**Next Steps (Not in v9_8 scope):**

- Update ml_finance_model_main.ipynb with new import paths (deferred - backward compatible)
- Update ml_finance_model_main.py with new import paths (deferred - backward compatible)
- Update README.md with comprehensive phase mapping and import examples
- Update CLI entry points to use new modules (already using finance_ml top-level imports)
- Comprehensive integration testing with new import paths

**Status:** v9_8 structural refactoring complete. All subpackages created, deprecation notices in place, backward
compatibility maintained.