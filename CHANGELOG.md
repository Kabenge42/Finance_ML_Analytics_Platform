# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
