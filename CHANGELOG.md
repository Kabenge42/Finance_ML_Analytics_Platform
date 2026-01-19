# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.22.1] - 2026-01-17

### Added

- SQL Feature Registry (`CalcFeatureRegistry.sql`) with SQL subquery functions that calculate features from the
  `postgres.public.equities` table, mirroring the Python feature registry in
  `finance_ml/features/advanced/__init__.py` ([efcb376](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/efcb376)).

### Changed

- Updated `create_equities_schema.sql`, `import_equities_data.sql`, and `schema.py` with schema
  enhancements ([e3ff0a4](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/e3ff0a4)).
- Enhanced `quality.py` with improved
  calculations ([e3ff0a4](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/e3ff0a4)).

## [1.22.0] - 2026-01-12

### Added

- Phase 9.3 v1.14 feature engineering enhancements aligned to the canonical schema (total 350
  features) ([437fcd3](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/437fcd3111302b0de9479f17eb9756c5619082ab)).
  - New generators: `engineer_price_target_dynamics`, `engineer_fiscal_calendar_features`,
    `engineer_dividend_timing_features`, `engineer_eps_trajectory_features`, and `engineer_cashflow_temporal_features`.
  - COLUMN_SCHEMA expanded with 54 temporal, sentiment, earnings, and cash flow features plus category updates.
  - FEATURE_REGISTRY and `__all__` exports updated for auto-discovery and tooling compatibility.
  - PHASE93 feature categories refreshed with revised counts (Analyst Sentiment 25, Cash Flow 17, Temporal 26, Earnings
    Quality 43, Dividend Reliability 20).
- Currency conversion stage added to ETL pipeline with support for
  `CurrencyConversionConfig` ([e25c5ff](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/e25c5ff8135c16f89db5ee3f72635b5e3facd271)).
- Missing feature generators implemented in
  `finance_ml/features/advanced/` ([f99ec99](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/f99ec992502fc28c92cbe0d1187d96add95e3433)).

### Changed

- Refactored `etl_data_explorer.ipynb` to align column references with the canonical schema via centralized helper
  module ([f99ec99](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/f99ec992502fc28c92cbe0d1187d96add95e3433)).
- Comprehensive schema alignment updates to `schema.py`, `import_equities_data.sql`, and
  `earnings.py` ([0503e02](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/0503e028e6b78f114261c03970665062e3b8c2f0), [54c56d0](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/54c56d084bcc4755cffeb4a5fd68d63fcccb6b26)).
- Revised `code_guidelines.md` and refactored
  `equities_dashboard_app.py` ([f99ec99](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/f99ec992502fc28c92cbe0d1187d96add95e3433), [0503e02](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/0503e028e6b78f114261c03970665062e3b8c2f0)).

### Testing

- Added TDD suite `tests/test_feature_enhancements_v114.py` covering schema alignment and registry integration for Phase
  9.3 ([437fcd3](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/437fcd3111302b0de9479f17eb9756c5619082ab)).

## [1.21.0] - 2025-12-30

### Added

- **Feature Engineering**: Implemented comprehensive enhancement plan for Phase 9.3 (Feature Engineering Registry).
  - **Earnings**: Added `gaap_revision_divergence` and `revenue_forecast_skew`. Updated `surprise_momentum_score` to
    include 1W and 1Y trends.
  - **Growth**: Added `forward_revenue_growth`, `revenue_cagr_5y`, and `growth_persistence_score`.
  - **Dividends**: Added `buyback_yield`, `total_shareholder_yield`, and `dividend_growth_expectation`.
  - **Profitability**: Added `rnd_intensity`, `marketing_efficiency`, `sga_ratio`, and explicit `equity_multiplier` for
    Dupont analysis.
  - **Quality**: Added `merger_impact_ratio`, `non_operating_income_share`, and `asset_sale_boost`.
  - **Revenue**: Implemented `revenue_estimate_momentum` and `revenue_surprise_volatility` placeholder.
  - **Momentum**: Added `beta_momentum` and `volatility_term_structure`.
  - **Sector**: Added `size_factor_percentile` and updated Tangible Book Value logic to use direct schema columns.

## [1.20.0] - 2025-12-27

### Added

- `preserve_columns` parameter to `FeatureSelectionConfig` in `finance_ml/etl/config.py` to support preserving
  identifier columns during feature selection (ml_workflow_guidelines.md Section 8.2).
- Default identifier columns constant `DEFAULT_IDENTIFIER_COLUMNS` in `finance_ml/etl/stages/feature_selection.py`
  containing `['ticker', 'isin', 'sector', 'region', 'country', 'industry']`.
- Validation checkpoint in notebook Phase 9.3 to verify critical identifiers exist after feature selection.

### Changed

- **Feature Selection Stage**: Updated `run_feature_selection_stage` in `finance_ml/etl/stages/feature_selection.py`
  to preserve identifier columns by default, preventing them from being stripped during automated feature selection.
- **ETL Pipeline**: Updated `ETLPipeline.transform()` in `finance_ml/etl/pipeline.py` to pass `preserve_columns`
  configuration to the feature selection stage.
- **Notebook Phase 9.3**: Comprehensive fix in `stock_price_target_prediction.ipynb` Section 2 to detect missing
  identifiers, reload them from source data if needed, and validate their presence before downstream phases.

### Fixed

- **Critical Bug**: Identifier columns (ticker, isin, sector, region, country, industry) were being stripped during
  ETL feature selection (Stage 10) because they are non-numeric. The code attempted to add them back after feature
  selection, but they no longer existed in `all_stocks_preprocessed`, causing `KeyError` exceptions.

## [1.19.0] - 2025-12-27

### Added

- Financial metrics dashboards with sector-wise aggregations and enhanced test coverage for temporal
  features ([7a04c15](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/7a04c1560add3e7b38042c26157383948dd147de)).
- Implementation of `reference_date` standardization across feature
  engineering ([7a04c15](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/7a04c1560add3e7b38042c26157383948dd147de)).

### Changed

- **Refactoring**: Created a single source of truth for all column definitions in
  `finance_ml/core/schema.py` ([dbf7f5a](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/dbf7f5a059b3e479b196af63f0bf4b9ccf38f484)).
- **Modularity**: Split `advanced.py` into domain-specific modules in
  `finance_ml/features/advanced/` ([dbf7f5a](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/dbf7f5a059b3e479b196af63f0bf4b9ccf38f484)).
- **ETL Config**: Extracted configuration dataclasses to
  `finance_ml/etl/config.py` ([dbf7f5a](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/dbf7f5a059b3e479b196af63f0bf4b9ccf38f484)).
- Updated `metrics_dashboard.json` to reflect latest financial data and
  timestamp ([fbc76af](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/fbc76af122897841646c5ef641933d3368d91826)).
- Optimized liquidity and valuation ratio
  calculations ([7a04c15](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/7a04c1560add3e7b38042c26157383948dd147de)).

### Removed

- Deprecated `ml_workflow.archive` package, including `advanced_eda.py` and
  `advanced_features.py` ([16630b0](https://github.com/Kabenge42/Finance_ML_Analytics_Platform/commit/16630b056b9ed31d7b674788724b8b6a9f7b4676)).
