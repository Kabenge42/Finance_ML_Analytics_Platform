# Notebook Refactoring Plan: ml_finance_model_main.ipynb

## Objective

Restructure the notebook to align with Phase 9.1-9.8 architecture specified in code_guidelines.md and
finance_ml_improvement_plan.md while maintaining all functionality and improving clarity.

---

## Refactoring Strategy

### Approach: Section Header Replacement + Content Reorganization

**Current Structure** (10 numbered sections):

```
1. Configuration and Setup
2. Loading and Preprocessing Financial Data
3. Exploratory Data Analysis of Financial Metrics
4. Advanced Feature Engineering with Sector-Specific Optimizations
5. Multi-Class Classification of Financial Events
6. Sector-Optimized Regression Models with Classification Features
7. Model Evaluation and Error Analysis
8. Identification of Under/Overvalued Stocks with Visualization
9. Comprehensive Analytics: Predicted vs. Analyst Price Target Comparison
10. Portfolio Optimization with Risk Metrics
```

**Target Structure** (8 phase-based sections + intro):

```
0. Configuration and Setup (unchanged)
Phase 9.1: Loading and Preprocessing with 6-Step Imputation Strategy
Phase 9.2: Enhanced Exploratory Data Analysis with Statistical Testing
Phase 9.3: Advanced Feature Engineering with Sector-Specific Optimizations
Phase 9.4: Multi-Class Event Classification
Phase 9.5: Sector-Optimized Regression Models with Quantile Predictions
Phase 9.6: Model Evaluation and Comprehensive Error Analysis
Phase 9.7: Stock Ranking, Analytics, and Analyst Comparison
Phase 9.8: Comprehensive Reporting and Portfolio Optimization
```

---

## Detailed Section Mapping

### Section 0: Configuration and Setup (Keep Current Section 1)

**Content**: Import, configuration, output directories
**Changes**:

- Keep current "## 1. Configuration and Setup" header
- Add note: "This notebook implements the 8-phase ML workflow (Phase 9.1-9.8)"
- Ensure all Phase 9.1-9.8 output directories are created

**No changes needed** - this section is already well-structured

---

### Phase 9.1: Loading and Preprocessing (Transform Current Section 2)

**New Header**:

```markdown
## Phase 9.1: Loading and Preprocessing with 6-Step Imputation Strategy

### Business Goal

Load multi-region equity data and apply comprehensive preprocessing to ensure high-quality inputs for downstream
modeling.

### Key Objectives

1. Load data from PostgreSQL/SQLite or CSV fallback
2. Apply 6-step imputation strategy (zero-fill, KNN, price-based, median)
3. Detect and handle outliers (IQR, z-score, isolation forest)
4. Apply sector-wise winsorization
5. Validate data quality and completeness

### Inputs

- Raw data: CSV files or database tables (US, EU, APAC, ROTW regions)

### Outputs

- `all_stocks_preprocessed`: Fully preprocessed DataFrame
- `outputs/preprocessing/`: Data quality reports, imputation stats
- `outputs/catalog/`: Data catalog metadata

### v1.2 Standards Applied

- ✅ 6-step imputation strategy
- ✅ Outlier safety rails (winsorization at [1st, 99th] percentiles)
- ✅ Data quality validation

### Validation Checkpoint

- Zero missing values after imputation
- Outliers capped within acceptable ranges
- All required columns present
```

**Content Structure**:

1. Data loading (CSV/DB)
2. Missing value analysis
3. Outlier detection
4. 6-step imputation
5. Winsorization
6. Validation checkpoint

**Code Verification Needed**:

- Ensure winsorization uses [1st, 99th] percentiles as per v1.2 standards
- Verify all imputation steps are documented

---

### Phase 9.2: Enhanced EDA (Transform Current Section 3)

**New Header**:

```markdown
## Phase 9.2: Enhanced Exploratory Data Analysis with Statistical Testing

### Business Goal

Understand data distributions, relationships, and sector-specific patterns to inform feature engineering and modeling
strategies.

### Key Objectives

1. Generate comprehensive statistical summaries
2. Analyze correlations and multicollinearity
3. Perform sector-wise comparisons with statistical tests
4. Create interactive visualizations
5. Generate benchmarking reports

### Inputs

- `all_stocks_preprocessed`: Preprocessed data from Phase 9.1

### Outputs

- `outputs/eda/`: EDA summary reports, correlation matrices, sector distributions
- Interactive visualizations (Plotly HTML files)

### Key Functions Used

- `generate_eda_report()` - Phase 9.2 EDA orchestrator
- `generate_benchmarking_report()` - Sector/region comparisons
- `compare_sector_distributions()` - Statistical testing

### Validation Checkpoint

- EDA report generated successfully
- Key metrics identified for feature engineering
- Sector patterns documented
```

**Content**: Keep current EDA content, add phase description

---

### Phase 9.3: Advanced Feature Engineering (Transform Current Section 4)

**New Header**:

```markdown
## Phase 9.3: Advanced Feature Engineering with Sector-Specific Optimizations

### Business Goal

Engineer comprehensive financial features including valuation ratios, profitability metrics, quality indicators, and
sector-specific features to maximize model predictive power.

### Key Objectives

1. Engineer valuation ratios (P/E, P/B, EV/EBITDA, PEG)
2. Engineer profitability features (margins, ROE, ROA, ROIC)
3. Create momentum and technical indicators
4. Engineer analyst quality features
5. Create accounting quality scores (Altman Z, Piotroski F)
6. Build sector-relative features
7. Create interaction features

### Inputs

- `all_stocks_preprocessed`: Preprocessed data from Phase 9.1

### Outputs

- `all_stocks_features`: Data with 400+ engineered features
- `outputs/features/`: Feature importance reports, correlation analysis

### Phase 9.3 API

```python
from finance_ml.ml_workflow.features.api import build_features
all_stocks_features = build_features(
    all_stocks_preprocessed, 
    preset="comprehensive",
    include_interactions=True,
    include_relative=True
)
```

### Validation Checkpoint

- 400+ features engineered
- No infinite values (replaced with NaN)
- Feature importance calculated
- Top features identified

```

**Content**: Keep current feature engineering, ensure API usage is highlighted

---

### Phase 9.4: Multi-Class Event Classification (Transform Current Section 5)

**New Header**:
```markdown
## Phase 9.4: Multi-Class Event Classification

### Business Goal
Classify stocks into financial event categories (Positive, Neutral, Negative) to provide sentiment signals that enhance regression model accuracy.

### Key Objectives
1. Create enhanced event labels using multiple methods
2. Prepare classification data with Phase 9.3 features
3. Train and optimize classifiers (XGBoost, LightGBM, CatBoost)
4. Evaluate classification performance
5. Extract classification probabilities as meta-features

### Inputs
- `all_stocks_features`: Feature-engineered data from Phase 9.3

### Outputs
- `clf_result`: Classification model result dict with probabilities
- `outputs/classification/`: Model artifacts, evaluation metrics, confusion matrices
- Event probability features for Phase 9.5

### Standardized Return Format (v1.2)
```python
clf_result = {
    'model': fitted_classifier,
    'metrics': {'accuracy': float, 'f1_macro': float, ...},
    'y_pred': np.ndarray,
    'y_proba': np.ndarray,  # 3-class probabilities
    'artifacts': {'feature_importance': pd.DataFrame, ...}
}
```

### Validation Checkpoint

- Classification accuracy > 60%
- All 3 classes represented in predictions
- Probabilities sum to 1.0
- Feature importance extracted

```

**Content**: Keep current classification, add standardized return format emphasis

---

### Phase 9.5: Sector-Optimized Regression (Transform Current Section 6)

**New Header**:
```markdown
## Phase 9.5: Sector-Optimized Regression Models with Quantile Predictions

### Business Goal
Predict stock price targets using regression models enhanced with classification meta-features, with uncertainty quantification via quantile regression.

### Key Objectives
1. Integrate classification probabilities as meta-features
2. Train multiple regression models (XGBoost, LightGBM, CatBoost, etc.)
3. Build stacking ensemble for robust predictions
4. Train quantile models for prediction intervals (p10, p50, p90)
5. Apply non-negative constraints (prices must be ≥ 0)
6. Perform time-series cross-validation

### Inputs
- `all_stocks_features`: From Phase 9.3
- `clf_result['y_proba']`: Classification probabilities from Phase 9.4

### Outputs
- `reg_result`: Regression result dict
- `outputs/regression/`: Model artifacts, quantile predictions
- `outputs/regression/regression_predictions_detailed.csv`: Standardized predictions
- `outputs/models/regression_metrics_by_sector.csv`: Per-sector metrics

### v1.2 Standards Applied
- ✅ Quantile regression (p10, p50, p90) with conformal calibration
- ✅ Monotonicity enforcement (p10 ≤ p50 ≤ p90)
- ✅ Non-negativity constraints
- ✅ Data split policy (TimeSeriesSplit → GroupKFold → Stratified)
- ✅ Standardized predictions schema
- ✅ Outlier safety rails (Huber loss, post-prediction clipping)

### Standardized Predictions Schema
Required columns:
- ticker, isin, sector, region, last_price, snapshot_date
- y_true, y_pred, y_pred_calibrated
- pred_p10, pred_p50, pred_p90, interval_width
- abs_error, pct_error
- model_version

### Validation Checkpoint
- MAE < 50% on validation set
- R² > 0.3
- Zero predictions < 1% (non-negativity enforced)
- Quantile monotonicity verified
- Prediction intervals coverage: 80% ± 5%
```

**Content**: Keep current regression, emphasize v1.2 standards

---

### Phase 9.6: Model Evaluation (Transform Current Section 7)

**New Header**:

```markdown
## Phase 9.6: Model Evaluation and Comprehensive Error Analysis

### Business Goal

Thoroughly evaluate regression model performance with comprehensive metrics, residual analysis, and segment-wise
breakdowns.

### Key Objectives

1. Calculate comprehensive regression metrics (MAE, RMSE, R², MAPE)
2. Perform segment analysis (by sector, region, market cap)
3. Generate residual plots and error distributions
4. Identify systematic biases
5. Analyze prediction errors by magnitude

### Inputs

- `reg_result`: Regression results from Phase 9.5
- `all_stocks_features`: Full dataset with predictions

### Outputs

- `outputs/evaluation/`: Comprehensive metrics, residual plots
- `outputs/evaluation/tscv_metrics.csv`: Time-series CV results
- Sector-wise performance analysis

### Key Metrics

- Overall: MAE, RMSE, R², MAPE, Median AE
- By sector: Per-sector performance comparison
- By region: Geographic performance patterns
- Error distribution: Histogram, percentiles

### Validation Checkpoint

- Comprehensive metrics calculated
- Residuals analyzed
- Sector biases identified
- Error patterns documented
```

**Content**: Keep current evaluation content

---

### Phase 9.7: Stock Ranking and Analytics (Transform Current Sections 8, 9, 10)

**New Header**:

```markdown
## Phase 9.7: Stock Ranking, Analytics, and Analyst Comparison

### Business Goal

Identify investment opportunities through mispricing scores, stock rankings, analyst comparison, and portfolio
optimization.

### Key Objectives

1. Calculate mispricing scores: (predicted_target - last_price) / last_price
2. Rank stocks by sector and region
3. Compare predictions vs analyst targets
4. Perform portfolio optimization (max Sharpe, min volatility)
5. Calculate risk metrics (VaR, CVaR, drawdown)
6. Generate investment recommendations

### Inputs

- `all_stocks_features`: Full dataset with predictions from Phase 9.5
- Analyst price targets

### Outputs

- `outputs/analytics/`: Mispricing rankings, analyst comparison reports
- `outputs/analytics/portfolio_optimization.csv`: Optimal portfolios
- `outputs/analytics/risk_metrics.csv`: Risk analysis
- Top undervalued/overvalued stocks by sector

### Key Functions

- `calculate_mispricing_score()` - Identify mispricing
- `rank_undervalued_stocks()` - Top opportunities
- `compare_prediction_vs_analyst_targets()` - Analyst agreement analysis
- `optimize_max_sharpe_portfolio()` - Portfolio optimization
- `calculate_portfolio_risk_metrics()` - Risk quantification

### Validation Checkpoint

- Mispricing scores calculated
- Top 20 undervalued stocks identified
- Analyst comparison complete
- Portfolio optimization converged
- Risk metrics within acceptable ranges
```

**Content**: Consolidate Sections 8, 9, 10 into this phase

**Subsections**:

1. Mispricing Score Calculation
2. Stock Ranking by Sector/Region
3. Analyst Comparison Analysis
4. Portfolio Optimization
5. Risk Metrics Calculation

---

### Phase 9.8: Comprehensive Reporting (Add to Current Section 9/10)

**New Header**:

```markdown
## Phase 9.8: Comprehensive Reporting and Dashboard Data

### Business Goal

Generate comprehensive reports, dashboards, and export final results for stakeholders and downstream applications.

### Key Objectives

1. Calculate financial metrics dashboard
2. Generate data quality alerts
3. Export predictions with standardized schema
4. Create interactive visualizations
5. Generate Excel/PDF reports

### Inputs

- All outputs from Phases 9.1-9.7

### Outputs

- `outputs/reporting/`: Final reports, dashboards
- `outputs/regression/regression_predictions_detailed.csv`: Final predictions export
- Excel reports with formatted tables and charts

### Key Functions

- `calculate_financial_metrics_dashboard()` - KPI reporting
- `generate_data_quality_alerts()` - Validation alerts
- `export_predictions()` - Standardized export
- `generate_prediction_analyst_excel_report()` - Excel report generation

### Validation Checkpoint

- All reports generated successfully
- Predictions exported with complete schema
- Dashboard data prepared
- Final artifacts persisted
```

**Content**: Consolidate reporting activities from Sections 9, 10

---

## Implementation Steps

### Step 1: Backup Current Notebook

- Create backup: `ml_finance_model_main.ipynb.backup`

### Step 2: Update Section Headers

For each section 2-10, replace headers with Phase 9.1-9.8 headers as specified above.

### Step 3: Add Phase Descriptions

Add business goal, objectives, inputs, outputs, and validation checkpoints to each phase.

### Step 4: Consolidate Sections 8, 9, 10

Reorganize content from these sections into Phases 9.7 and 9.8.

### Step 5: Add Validation Checkpoints

After each phase, add validation checkpoint cell with:

```python
print("=" * 80)
print(f"PHASE 9.X VALIDATION CHECKPOINT")
print("=" * 80)
# Validation logic
print(f"✅ Phase 9.X Complete")
```

### Step 6: Verify v1.2 Standards

Check each phase for v1.2 compliance:

- Phase 9.1: Outlier safety rails
- Phase 9.5: Uncertainty quantification, predictions schema, data split policy
- All phases: Standardized return types

### Step 7: Update Table of Contents

Update navigation links to reflect Phase 9.1-9.8 structure.

---

## Success Criteria

1. ✅ All sections use Phase 9.1-9.8 headers
2. ✅ Each phase has clear description, objectives, inputs, outputs
3. ✅ Validation checkpoints added for all phases
4. ✅ v1.2 standards verified and documented
5. ✅ Content logically organized within phases
6. ✅ All functionality preserved
7. ✅ Table of contents updated
8. ✅ Phase output directories properly utilized

---

## Post-Refactoring Validation

Run notebook end-to-end to verify:

1. All cells execute without errors
2. All phases produce expected outputs
3. Validation checkpoints pass
4. Final predictions file includes all required columns
5. Sector metrics persisted correctly
