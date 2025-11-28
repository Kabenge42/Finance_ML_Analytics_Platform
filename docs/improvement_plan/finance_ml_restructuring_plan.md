# Finance_ML Package Restructuring Plan

**Date:** 2025-11-28  
**Version:** 1.0  
**Package Version:** 0.4.1 (actual) vs 0.9.1 (documented)  
**Audit Scope:** `finance_ml/` directory and `tests/` directory

---

## Executive Summary

The `finance_ml` package has significant architectural debt resulting from incremental Phase-based development (9.1-9.8)
without proper consolidation. This document provides a comprehensive restructuring plan addressing both package
architecture and test suite organization, with actionable TDD improvement tasks.

**Key Issues Identified:**

1. **Monster File:** `analytics/eval.py` at 8,428 lines requiring immediate decomposition
2. **Bloated API:** ~300+ exports in `__init__.py` with duplicate function names
3. **Code Duplication:** 17+ identical functions between legacy modules and new subpackages
4. **Test Fragmentation:** 136 test files with no subdirectory organization
5. **Version Inconsistency:** Package version mismatch (0.4.1 actual vs 0.9.1 documented)

---

## Part 1: Package Architecture Audit

### 1.1 Current Package Structure

```
finance_ml/
├── __init__.py           (35.6 KB, 1006 lines) - Bloated with 300+ exports
├── cli.py                (10.7 KB) - CLI entry points
├── config.py             (12.7 KB) - Configuration management
├── logging_config.py     (9.4 KB) - Logging setup
├── notebook_config.py    (5.4 KB) - Notebook configuration
├── notebook_utils.py     (6.1 KB) - Notebook helpers
├── dashboards/           - Interactive dashboards (Streamlit, Dash)
└── ml_workflow/          - Main ML workflow package
```

### 1.2 ml_workflow/ Subpackages Analysis

| Subpackage        | Files      | Purpose                           | Status                 |
|-------------------|------------|-----------------------------------|------------------------|
| `preprocessing/`  | 10 modules | Data loading, imputation, scaling | ✅ Well-structured      |
| `eda/`            | 4 modules  | Exploratory data analysis         | ⚠️ Partial duplication |
| `features/`       | 6 modules  | Feature engineering               | ✅ Well-structured      |
| `classification/` | 5 modules  | Event classification              | ⚠️ Heavy duplication   |
| `regression/`     | 14 modules | Regression models                 | ⚠️ Heavy duplication   |
| `evaluation/`     | 8 modules  | Model evaluation                  | ✅ Well-structured      |
| `analytics/`      | 12 modules | Stock analytics, portfolio        | ⚠️ `eval.py` bloated   |
| `reporting/`      | 3 modules  | Dashboard data, exports           | ✅ Well-structured      |
| `data/`           | 2 modules  | Schema, loaders                   | ✅ Well-structured      |
| `config/`         | 2 modules  | ML config                         | ✅ Minimal              |
| `core/`           | 1 module   | Utilities                         | ⚠️ Nearly empty        |
| `quality/`        | 2 modules  | Code review tools                 | ✅ Specialized          |
| `validation/`     | 2 modules  | Data validation                   | ⚠️ Minimal             |

### 1.3 Critical Issues

#### Issue 1: Massive Code Duplication (CRITICAL)

**Legacy modules duplicating subpackage functionality:**

| Legacy Module                | Size    | Duplicates                                                 | Duplication Level           |
|------------------------------|---------|------------------------------------------------------------|-----------------------------|
| `classification.py`          | 67.7 KB | `classification/models.py`, `classification/labels.py`     | **17+ identical functions** |
| `advanced_models.py`         | 76.8 KB | `regression/models.py`, `regression/quantile.py`           | High                        |
| `models.py`                  | 45.2 KB | `regression/`, `classification/`                           | High                        |
| `advanced_preprocessing.py`  | 40.7 KB | `preprocessing/imputation.py`, `preprocessing/outliers.py` | High                        |
| `advanced_features.py`       | 37.5 KB | `features/advanced.py`                                     | High                        |
| `advanced_eda.py`            | 20.4 KB | `eda/eda.py`, `analytics/eval.py`                          | Medium                      |
| `classification_enhanced.py` | 16 KB   | `classification/tuning.py`                                 | High                        |

**Confirmed Duplicate Functions (classification.py vs classification/models.py):**

- `_prepare_categorical_features`
- `prepare_classification_data`
- `train_xgboost_classifier`, `train_lightgbm_classifier`, `train_catboost_classifier`
- `train_svm_classifier`, `train_neural_network_classifier`
- `train_voting_classifier`, `train_stacking_classifier`
- `apply_smote`, `apply_adasyn`, `apply_undersampling`, `apply_combined_sampling`
- `export_classification_features`, `clean_extreme_values`, `validate_data_quality`
- `compare_classifiers`

#### Issue 2: Monster File - analytics/eval.py (CRITICAL)

**8,428 lines / 304 KB** containing ~100 functions spanning:

- Mispricing calculations (lines 40-296)
- EDA functions (lines 299-877)
- Correlation analysis (lines 1601-1735)
- SHAP/LIME explainability (lines 2934-3318)
- Learning/validation curves (lines 3452-3863)
- Hypothesis testing (lines 6661-7519)
- Report generation (lines 5063-5623, 7905-8287)

**This file violates single-responsibility principle and should be split into 10+ modules.**

#### Issue 3: Bloated Public API (~300+ exports)

The `__init__.py` exports **300+ symbols** with:

- **Duplicate exports with different names:** `train_ridge_regressor` AND `regression_train_ridge`
- **Duplicate exports from different sources:** `perform_pca` exported twice, `compare_sector_means` exported twice
- **Inconsistent naming:** Mix of `features_*`, `regression_*`, `analytics_*`, `evaluation_*` prefixes
- **try/except import guards** indicating fragile dependencies

#### Issue 4: Small Stub Modules (Unclear Purpose)

| Module                      | Size   | Duplicates                                  |
|-----------------------------|--------|---------------------------------------------|
| `analyst_comparison.py`     | 1.7 KB | `analytics/analyst_comparison.py` (17.5 KB) |
| `portfolio_optimization.py` | 1.6 KB | `analytics/portfolio.py` (34.2 KB)          |
| `risk_metrics.py`           | 1.3 KB | `analytics/risk.py` (15.6 KB)               |
| `benchmarking.py`           | 1.3 KB | `eda/benchmarking.py` (20.5 KB)             |

#### Issue 5: Version Inconsistency

- `__init__.py`: `__version__ = "0.4.1"`
- `code_guidelines.md`: Package Version 0.9.1
- Documented structure doesn't match actual implementation

#### Issue 6: Missing Documented Modules

Per `code_guidelines.md` Section 4.1, these modules should exist but don't:

- `eda/statistical_tests.py`
- `data/loaders.py`
- `config/settings.py`
- `core/utils.py`
- `validation/validators.py`

---

## Part 2: Test Suite Audit

### 2.1 Current Test Structure

**Total:** 136 test files in flat `tests/` directory (no subdirectory organization)

### 2.2 Test File Categories

#### By Test Focus Area

| Category                 | Files | Examples                                                                                           |
|--------------------------|-------|----------------------------------------------------------------------------------------------------|
| **Notebook Tests**       | 11    | `test_notebook_config.py`, `test_notebook_enhancements.py`, `test_ml_stock_prediction_notebook.py` |
| **Classification Tests** | 7     | `test_classification.py`, `test_classification_phase94.py`, `test_classification_models.py`        |
| **Phase Tests**          | 10    | `test_phase95_*.py`, `test_phase96_*.py`                                                           |
| **Portfolio Tests**      | 10    | `test_portfolio_*.py`                                                                              |
| **Fix Tests**            | 12    | `test_fix_*.py` (regression tests for bug fixes)                                                   |
| **Integration Tests**    | 4     | `test_integration_*.py`                                                                            |

#### By Feature Area (with fragmentation issues)

| Feature                  | Files | Issue                                                                                        |
|--------------------------|-------|----------------------------------------------------------------------------------------------|
| **Imputation**           | 5     | `test_enhanced_imputation.py`, `test_enhanced_imputation_phase93.py`, `test_imputation_*.py` |
| **Features**             | 14    | Scattered across momentum, cashflow, quality_risk, sentiment, sector_specific, etc.          |
| **Regression**           | 6     | Fragmented across phases and concerns                                                        |
| **EDA**                  | 7     | `test_advanced_eda.py`, `test_eda.py`, `test_eda_phase92.py`, `test_phase93_eda.py`, etc.    |
| **Preprocessing**        | 4     | Multiple overlapping files                                                                   |
| **Schema**               | 6     | `test_schema_completeness.py`, `test_schema_fix.py`, `test_schema_normalization.py`, etc.    |
| **Outlier**              | 4     | `test_outlier_*.py`, `test_preprocessing_outliers.py`, `test_robust_outlier_safety.py`       |
| **Quantile/Uncertainty** | 7     | Overlapping concerns across multiple files                                                   |

### 2.3 Test Suite Issues

1. **No subdirectory organization** - all 136 tests flat in `tests/`
2. **Phase-based naming** doesn't align with package structure
3. **Feature tests extremely fragmented** - 14 files for different feature types
4. **Multiple overlapping tests** for same functionality (e.g., 4 outlier tests, 6 schema tests)
5. **Fix tests not consolidated** into regression test suites
6. **Inconsistent naming conventions** - mix of `test_<feature>.py` and `test_<phase>_<feature>.py`

---

## Part 3: Proposed Restructured Architecture

### 3.1 Package Structure

```
finance_ml/
├── __init__.py                    # Minimal exports, version only
├── api.py                         # NEW: Clean public API facade
├── cli.py                         # Keep: CLI entry points
├── config.py                      # Keep: Configuration
├── logging_config.py              # Keep: Logging
├── notebook_config.py             # Keep: Notebook config
├── notebook_utils.py              # Keep: Notebook helpers
│
├── dashboards/                    # Keep as-is
│   ├── streamlit_app.py
│   ├── dash_app.py
│   └── portfolio_widgets.py
│
└── ml_workflow/
    ├── __init__.py                # Subpackage exports
    │
    ├── data/                      # Phase 9.1a: Data Loading
    │   ├── loaders.py             # CSV/DB loading (from preprocessing/data.py)
    │   ├── schema.py              # Column schema registry
    │   ├── catalog.py             # Data catalog (from data_catalog.py)
    │   └── versioning.py          # Data versioning (from data_versioning.py)
    │
    ├── preprocessing/             # Phase 9.1b: Preprocessing
    │   ├── imputation.py          # Keep
    │   ├── outliers.py            # Keep
    │   ├── scaling.py             # Keep
    │   ├── dtypes.py              # Keep
    │   ├── pipeline.py            # Keep
    │   ├── column_semantics.py    # Keep
    │   ├── transforms.py          # Keep
    │   └── quality.py             # Keep
    │
    ├── eda/                       # Phase 9.2: EDA
    │   ├── descriptive.py         # Basic statistics
    │   ├── correlations.py        # Correlation analysis
    │   ├── distributions.py       # Distribution analysis
    │   ├── benchmarking.py        # Keep
    │   └── reports.py             # EDA report generation
    │
    ├── features/                  # Phase 9.3: Feature Engineering
    │   ├── core.py                # Keep
    │   ├── advanced.py            # Keep
    │   ├── selection.py           # Keep
    │   ├── api.py                 # Keep
    │   ├── sector_specific.py     # Keep
    │   ├── validation.py          # Keep
    │   └── transformers.py        # Move from ml_workflow/transformers.py
    │
    ├── classification/            # Phase 9.4: Classification
    │   ├── labels.py              # Keep (13 methods)
    │   ├── models.py              # Keep (consolidated)
    │   ├── tuning.py              # Keep
    │   └── evaluation.py          # Keep
    │
    ├── regression/                # Phase 9.5: Regression
    │   ├── models.py              # Keep (consolidated)
    │   ├── quantile.py            # Keep
    │   ├── constraints.py         # Keep
    │   ├── stacking.py            # Keep (from evaluation/)
    │   ├── dataset.py             # Keep
    │   ├── tuning.py              # Keep
    │   ├── io.py                  # Keep
    │   └── sector_models.py       # Keep
    │
    ├── evaluation/                # Phase 9.6: Evaluation
    │   ├── metrics.py             # Regression/classification metrics
    │   ├── uncertainty.py         # Keep
    │   ├── calibration.py         # Keep
    │   ├── safety_rails.py        # Keep
    │   ├── splits.py              # Keep
    │   ├── explainability.py      # NEW: SHAP/LIME (from eval.py)
    │   ├── learning_curves.py     # NEW: (from eval.py)
    │   └── bias_variance.py       # NEW: (from eval.py)
    │
    ├── analytics/                 # Phase 9.7: Analytics
    │   ├── mispricing.py          # Keep
    │   ├── stock_selection.py     # Keep
    │   ├── portfolio.py           # Keep
    │   ├── risk.py                # Keep
    │   ├── ml_returns.py          # Keep
    │   ├── attribution.py         # Keep
    │   ├── analyst_comparison.py  # Keep (consolidated)
    │   ├── hypothesis_tests.py    # NEW: (from eval.py)
    │   └── peer_comparison.py     # NEW: (from eval.py)
    │
    ├── reporting/                 # Phase 9.8: Reporting
    │   ├── dashboard_data.py      # Keep
    │   ├── export.py              # Keep
    │   ├── excel_reports.py       # Keep
    │   ├── html_reports.py        # Keep
    │   ├── pdf_reports.py         # NEW: (from eval.py)
    │   └── quality_alerts.py      # NEW: (from eval.py)
    │
    └── archive/                   # NEW: Deprecated modules
        ├── README.md              # Deprecation notice
        ├── classification.py      # Archive with deprecation warning
        ├── models.py              # Archive with deprecation warning
        ├── advanced_models.py     # Archive with deprecation warning
        ├── advanced_preprocessing.py
        ├── advanced_features.py
        ├── advanced_eda.py
        └── classification_enhanced.py
```

### 3.2 Test Suite Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
│
├── unit/                          # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── test_imputation.py     # Consolidate 5 imputation test files
│   │   ├── test_outliers.py       # Consolidate 4 outlier test files
│   │   ├── test_scaling.py
│   │   ├── test_dtypes.py
│   │   └── test_column_semantics.py
│   ├── eda/
│   │   ├── test_descriptive.py
│   │   ├── test_correlations.py
│   │   └── test_benchmarking.py   # Consolidate EDA tests
│   ├── features/
│   │   ├── test_core.py           # Consolidate 14 feature test files
│   │   ├── test_advanced.py
│   │   ├── test_selection.py
│   │   └── test_sector_specific.py
│   ├── classification/
│   │   ├── test_labels.py
│   │   ├── test_models.py         # Consolidate classification tests
│   │   └── test_tuning.py
│   ├── regression/
│   │   ├── test_models.py         # Consolidate regression tests
│   │   ├── test_quantile.py
│   │   └── test_constraints.py
│   ├── evaluation/
│   │   ├── test_metrics.py
│   │   ├── test_uncertainty.py    # Consolidate 7 quantile/uncertainty tests
│   │   └── test_safety_rails.py
│   ├── analytics/
│   │   ├── test_mispricing.py
│   │   ├── test_portfolio.py      # Consolidate 10 portfolio tests
│   │   └── test_risk.py
│   └── reporting/
│       ├── test_dashboard_data.py
│       └── test_exports.py
│
├── integration/                   # Integration tests (slower, end-to-end)
│   ├── __init__.py
│   ├── test_cli_pipeline.py
│   ├── test_notebook_pipeline.py
│   └── test_production_scenarios.py
│
├── regression/                    # Regression tests (bug fix validation)
│   ├── __init__.py
│   └── test_bugfixes.py           # Consolidate 12 fix tests
│
└── fixtures/                      # Test data and fixtures
    ├── sample_data.csv
    └── expected_outputs/
```

---

## Part 4: Actionable TDD Improvement Tasks

### Phase 1: Critical - Monster File Decomposition (Priority: HIGH, Week 1-2)

#### Task 1.1: Split analytics/eval.py into focused modules

```
TDD Steps:
1. Create test_eval_mispricing.py - Test mispricing functions (lines 40-296)
2. Create test_eval_eda.py - Test EDA functions (lines 299-877)
3. Create test_eval_correlations.py - Test correlation functions (lines 1601-1735)
4. Create test_eval_explainability.py - Test SHAP/LIME functions (lines 2934-3318)
5. Create test_eval_learning_curves.py - Test learning curve functions (lines 3452-3863)
6. Create test_eval_hypothesis.py - Test hypothesis testing (lines 6661-7519)
7. Create test_eval_reports.py - Test report generation (lines 5063-5623, 7905-8287)

Implementation:
- Extract each function group to new module
- Update imports in __init__.py
- Verify all tests pass
- Target: 8 modules of ~500-1000 lines each
```

#### Task 1.2: Create deprecation shims for analytics/eval.py

```python
# analytics/eval.py (after refactor)
"""
DEPRECATION NOTICE: This module has been split into focused modules.
Import from specific modules instead:
- finance_ml.ml_workflow.analytics.mispricing
- finance_ml.ml_workflow.evaluation.explainability
- finance_ml.ml_workflow.reporting.pdf_reports
"""
from finance_ml.ml_workflow.analytics.mispricing import *  # noqa
# ... etc
```

### Phase 2: High - Consolidate Classification Duplicates (Week 3-4)

#### Task 2.1: Migrate classification.py to deprecation shim

```
TDD Steps:
1. Create test_classification_parity.py
   - Test that classification/models.py functions match classification.py signatures
   - Test output equivalence for all 17 duplicate functions
2. Replace classification.py with deprecation shim (like features.py)
3. Update __init__.py imports
4. Run full test suite
5. Archive original to ml_workflow/archive/
```

#### Task 2.2: Consolidate classification_enhanced.py

```
TDD Steps:
1. Verify classification/tuning.py has all classification_enhanced.py functions
2. Create deprecation shim
3. Update imports
```

### Phase 3: High - Consolidate Model Duplicates (Week 3-4)

#### Task 3.1: Migrate models.py to deprecation shim

```
TDD Steps:
1. Create test_models_parity.py
   - Verify regression/ and classification/ cover all models.py functions
2. Create deprecation shim
3. Archive original
```

#### Task 3.2: Migrate advanced_models.py to deprecation shim

```
TDD Steps:
1. Create test_advanced_models_parity.py
2. Verify regression/models.py has all training functions
3. Create deprecation shim
4. Archive original
```

### Phase 4: Medium - Consolidate Preprocessing Duplicates (Week 5-6)

#### Task 4.1: Migrate advanced_preprocessing.py to deprecation shim

```
TDD Steps:
1. Create test_preprocessing_parity.py
   - Compare preprocessing/imputation.py vs advanced_preprocessing.py
   - Compare preprocessing/outliers.py vs advanced_preprocessing.py
2. Create deprecation shim
3. Archive original
```

### Phase 5: Medium - Clean Up Small Stub Modules (Week 5-6)

#### Task 5.1: Remove or consolidate stub modules

```
Modules to evaluate:
- analyst_comparison.py (1.7 KB) → deprecation shim to analytics/analyst_comparison.py
- portfolio_optimization.py (1.6 KB) → deprecation shim to analytics/portfolio.py
- risk_metrics.py (1.3 KB) → deprecation shim to analytics/risk.py
- benchmarking.py (1.3 KB) → deprecation shim to eda/benchmarking.py

TDD Steps for each:
1. Verify subpackage module has all functions
2. Create deprecation shim
3. Update __init__.py
```

### Phase 6: Medium - Streamline Public API (Week 7-8)

#### Task 6.1: Create clean api.py facade

```python
# finance_ml/api.py
"""
Clean public API for Finance ML Analytics Platform.
Use this module for stable, documented imports.
"""
# Data Loading
from finance_ml.ml_workflow.data import load_from_csv, load_from_db, normalize_columns

# Preprocessing
from finance_ml.ml_workflow.preprocessing import (
    apply_enhanced_imputation_strategy,
    winsorize_by_sector,
    scale_features,
    )

# Features
from finance_ml.ml_workflow.features import build_features, PresetName

# Classification
from finance_ml.ml_workflow.classification import (
    create_enhanced_event_labels,
    train_event_classifier,
    )

# Regression
from finance_ml.ml_workflow.regression import (
    train_sector_specific_models,
    train_quantile_regressor,
    )

# Analytics
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_score,
    rank_undervalued_stocks,
    optimize_portfolio,
    )

# Reporting
from finance_ml.ml_workflow.reporting import generate_dashboard_data
```

#### Task 6.2: Reduce __init__.py exports

```
TDD Steps:
1. Create test_public_api.py - Test all documented public functions
2. Gradually move exports from __init__.py to api.py
3. Add deprecation warnings to __init__.py for removed exports
4. Target: <50 exports in __init__.py (version, config, key classes)
```

### Phase 7: Medium - Reorganize Test Suite (Week 7-8)

#### Task 7.1: Create test directory structure

```
TDD Steps:
1. Create tests/unit/, tests/integration/, tests/regression/, tests/fixtures/
2. Create conftest.py with shared fixtures
3. Move tests in batches by category
4. Update CI/CD configuration
```

#### Task 7.2: Consolidate fragmented tests

```
Consolidation Targets:
- 5 imputation tests → tests/unit/preprocessing/test_imputation.py
- 4 outlier tests → tests/unit/preprocessing/test_outliers.py
- 14 feature tests → tests/unit/features/test_*.py (4 files)
- 7 EDA tests → tests/unit/eda/test_*.py (3 files)
- 6 schema tests → tests/unit/data/test_schema.py
- 7 quantile tests → tests/unit/evaluation/test_uncertainty.py
- 10 portfolio tests → tests/unit/analytics/test_portfolio.py
- 12 fix tests → tests/regression/test_bugfixes.py
```

### Phase 8: Low - Create Missing Documented Modules (Week 9-10)

#### Task 8.1: Create placeholder modules

```
Create with minimal implementations:
- eda/statistical_tests.py (move from eval.py)
- data/loaders.py (extract from preprocessing/data.py)
- config/settings.py (extract from config.py)
- core/utils.py (collect scattered utilities)
- validation/validators.py (collect validation functions)
```

### Phase 9: Low - Version and Documentation Alignment (Week 9-10)

#### Task 9.1: Sync version numbers

```
1. Update __init__.py version to match code_guidelines.md (0.9.1)
2. Update pyproject.toml version
3. Create CHANGELOG entry
```

#### Task 9.2: Update code_guidelines.md Section 4

```
1. Document actual vs intended structure
2. Add migration guide for deprecated imports
3. Update import patterns with working examples
```

---

## Part 5: Test Coverage Requirements

| Task                    | New Tests Required            | Coverage Target            |
|-------------------------|-------------------------------|----------------------------|
| 1.1 Split eval.py       | 7 new test files              | 80% per new module         |
| 2.1 Classification shim | test_classification_parity.py | 100% function parity       |
| 3.1 Models shim         | test_models_parity.py         | 100% function parity       |
| 4.1 Preprocessing shim  | test_preprocessing_parity.py  | 100% function parity       |
| 6.1 Clean API           | test_public_api.py            | 100% API coverage          |
| 7.2 Test consolidation  | N/A (consolidation)           | Maintain existing coverage |

---

## Part 6: Estimated Impact

| Metric              | Current     | After Refactor     |
|---------------------|-------------|--------------------|
| Largest file        | 8,428 lines | <1,000 lines       |
| Public exports      | ~300        | ~50                |
| Duplicate functions | ~50+        | 0                  |
| Legacy modules      | 8           | 0 (archived)       |
| Test directories    | 1 (flat)    | 4 (organized)      |
| Test files          | 136         | ~60 (consolidated) |
| Code clarity        | Low         | High               |
| Import time         | Slow        | Fast               |

---

## Part 7: Implementation Timeline

| Phase | Tasks                        | Duration  | Priority |
|-------|------------------------------|-----------|----------|
| 1     | Split analytics/eval.py      | Week 1-2  | HIGH     |
| 2     | Classification consolidation | Week 3-4  | HIGH     |
| 3     | Models consolidation         | Week 3-4  | HIGH     |
| 4     | Preprocessing consolidation  | Week 5-6  | MEDIUM   |
| 5     | Stub module cleanup          | Week 5-6  | MEDIUM   |
| 6     | API streamlining             | Week 7-8  | MEDIUM   |
| 7     | Test suite reorganization    | Week 7-8  | MEDIUM   |
| 8     | Missing modules              | Week 9-10 | LOW      |
| 9     | Version alignment            | Week 9-10 | LOW      |

---

## Part 8: Risk Mitigation

1. **Breaking Changes:** Use deprecation warnings (not errors) for 2 release cycles
2. **Test Coverage:** Require parity tests before archiving any module
3. **Notebook Compatibility:** Test `ml_finance_model_main.ipynb` after each phase
4. **CI/CD:** Add import time benchmarks to prevent regression
5. **Documentation:** Update code_guidelines.md after each major phase

---

## Part 9: Success Criteria

- [ ] analytics/eval.py split into 8+ focused modules (<1000 lines each)
- [ ] All legacy modules archived with deprecation shims
- [ ] __init__.py exports reduced to <50 symbols
- [ ] Test suite organized into unit/integration/regression directories
- [ ] Test file count reduced from 136 to ~60 (consolidated)
- [ ] Version numbers aligned (0.9.1 or higher)
- [ ] All existing tests pass after restructuring
- [ ] Notebook executes successfully end-to-end
- [ ] Import time reduced by >50%

---

## Appendix A: Duplicate Function Mapping

### Classification Module Duplicates

| Function                          | classification.py | classification/models.py |
|-----------------------------------|-------------------|--------------------------|
| `_prepare_categorical_features`   | ✓                 | ✓                        |
| `prepare_classification_data`     | ✓                 | ✓                        |
| `train_xgboost_classifier`        | ✓                 | ✓                        |
| `train_lightgbm_classifier`       | ✓                 | ✓                        |
| `train_catboost_classifier`       | ✓                 | ✓                        |
| `train_svm_classifier`            | ✓                 | ✓                        |
| `train_neural_network_classifier` | ✓                 | ✓                        |
| `train_voting_classifier`         | ✓                 | ✓                        |
| `train_stacking_classifier`       | ✓                 | ✓                        |
| `apply_smote`                     | ✓                 | ✓                        |
| `apply_adasyn`                    | ✓                 | ✓                        |
| `apply_undersampling`             | ✓                 | ✓                        |
| `apply_combined_sampling`         | ✓                 | ✓                        |
| `export_classification_features`  | ✓                 | ✓                        |
| `clean_extreme_values`            | ✓                 | ✓                        |
| `validate_data_quality`           | ✓                 | ✓                        |
| `compare_classifiers`             | ✓                 | ✓                        |

### eval.py Function Distribution

| Target Module                   | Functions | Lines     |
|---------------------------------|-----------|-----------|
| `analytics/mispricing.py`       | 8         | 40-296    |
| `eda/descriptive.py`            | 15        | 299-877   |
| `eda/correlations.py`           | 6         | 1601-1735 |
| `evaluation/explainability.py`  | 12        | 2934-3318 |
| `evaluation/learning_curves.py` | 10        | 3452-3863 |
| `analytics/hypothesis_tests.py` | 18        | 6661-7519 |
| `reporting/pdf_reports.py`      | 8         | 5063-5623 |
| `reporting/quality_alerts.py`   | 6         | 7905-8287 |

---

## Appendix B: Test Consolidation Mapping

### Imputation Tests (5 → 1)

| Current File                          | Target                                        |
|---------------------------------------|-----------------------------------------------|
| `test_enhanced_imputation.py`         | `tests/unit/preprocessing/test_imputation.py` |
| `test_enhanced_imputation_phase93.py` | `tests/unit/preprocessing/test_imputation.py` |
| `test_imputation_6step.py`            | `tests/unit/preprocessing/test_imputation.py` |
| `test_imputation_report.py`           | `tests/unit/preprocessing/test_imputation.py` |
| `test_fix_imputation_*.py`            | `tests/regression/test_bugfixes.py`           |

### Feature Tests (14 → 4)

| Current Files                                                  | Target                                        |
|----------------------------------------------------------------|-----------------------------------------------|
| `test_features.py`, `test_build_features.py`                   | `tests/unit/features/test_core.py`            |
| `test_advanced_features.py`, `test_finance_ml_features.py`     | `tests/unit/features/test_advanced.py`        |
| `test_features_momentum.py`, `test_features_cashflow.py`, etc. | `tests/unit/features/test_sector_specific.py` |
| `test_feature_selection_*.py`                                  | `tests/unit/features/test_selection.py`       |

### Portfolio Tests (10 → 2)

| Current Files                                                               | Target                                         |
|-----------------------------------------------------------------------------|------------------------------------------------|
| `test_portfolio_optimization.py`, `test_portfolio_optimization_advanced.py` | `tests/unit/analytics/test_portfolio.py`       |
| `test_portfolio_ml_prediction.py`, `test_portfolio_risk_management.py`      | `tests/unit/analytics/test_portfolio.py`       |
| `test_portfolio_backtesting.py`, `test_portfolio_dashboards.py`             | `tests/integration/test_portfolio_pipeline.py` |

---

*Document generated: 2025-11-28*  
*Next review: After Phase 1 completion*
