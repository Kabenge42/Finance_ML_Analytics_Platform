### Comprehensive Notebook Restructuring Plan for `ml_finance_model_main.ipynb`

Based on the business objective and the reference notebook `ml_stock_prediction_model.ipynb`, here's a detailed
restructuring plan to streamline the current 5755-line notebook into a focused, production-ready workflow.

---

### Executive Summary

**Current State**: 5755 lines with phase-by-phase implementation (9.1-9.8+), extensive logging, and inline helper
functions  
**Target State**: ~800-1200 lines following clean 9-step workflow aligned with business objective  
**Key Strategy**: Leverage `finance_ml` package functions, remove redundant code, consolidate sections

---

### Restructured Notebook Outline (9 Sections)

#### **Section 0: Header and Business Objective** (Lines: ~30)

```markdown
# Stock Price Target Prediction — ML Analytics Platform

**Version 2.0.0** — Production-Ready Streamlined Workflow

## Business Objective
**Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support 
investment decisions and portfolio optimization.

**Target Variable**: "Predicted Price Target" for regression modeling

## Workflow Overview (9 Steps)
1. Loading and Preprocessing — Multi-region data with 4-step imputation
2. Exploratory Data Analysis — Financial metrics and benchmarking
3. Feature Engineering — Sector-specific optimizations
4. Multi-Class Classification — Financial event detection
5. Sector-Optimized Regression — Price target prediction with classification features
6. Model Evaluation — Comprehensive error analysis
7. Stock Valuation — Under/overvalued identification
8. Predicted vs. Analyst Analytics — Target comparison
9. Portfolio Optimization — Risk-adjusted portfolio construction

## Key Features
- 📊 **Data Management**: PostgreSQL/CSV with validation (data.py, data_catalog.py)
- 🔧 **Preprocessing**: 4-step imputation strategy (advanced_preprocessing.py)
- 📈 **EDA**: Statistical tests, benchmarking (advanced_eda.py, benchmarking.py, eval.py)
- 🔨 **Features**: Financial ratios, sector-specific (features.py, advanced_features.py, transformers.py)
- 🤖 **Models**: Classification + regression (classification.py, models.py, advanced_models.py)
- 📊 **Analytics**: Comprehensive evaluation (eval.py, analyst_comparison.py)
- 💼 **Portfolio**: Optimization with risk metrics (portfolio_optimization.py, risk_metrics.py)
```

**Modules**: `data.py`, `data_catalog.py`, `advanced_preprocessing.py`, `transformers.py`, `features.py`,
`advanced_features.py`, `advanced_eda.py`, `benchmarking.py`, `classification.py`, `models.py`, `advanced_models.py`,
`eval.py`, `analyst_comparison.py`, `portfolio_optimization.py`, `risk_metrics.py`

---

#### **Section 1: Configuration and Setup** (Lines: ~80)

```python
#%% md
## 1. Configuration and Setup

#%%
# Import configuration
from finance_ml import NotebookConfig

# Initialize with production settings
config = NotebookConfig(
    have_finance_prediction=True,
    have_database_connection=True,
    have_advanced_analytics=True,
    have_dim_reduction=True,
    debug_mode=False,
    enable_sector_analysis=True,
    enable_region_analysis=True,
    enable_interactive_plots=True,
    enable_excel_export=True,
)
config.display_summary()

#%%
# Core imports
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Finance ML package imports
from finance_ml import (
    data, features, advanced_features, classification, 
    advanced_models, eval as fm_eval, analyst_comparison,
    portfolio_optimization, risk_metrics
)

# Specific function imports
from finance_ml.data import load_from_csv, load_from_db, validate_schema
from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step
from finance_ml.advanced_eda import generate_eda_report
from finance_ml.benchmarking import generate_benchmarking_report

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
np.random.seed(RANDOM_SEED)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "regression").mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)

print("✓ Configuration and imports complete")
```

**Action**: Remove extensive utility function definitions (print_section_header, checkpoint system) — keep only
essential setup

---

#### **Section 2: Loading and Preprocessing** (Lines: ~120)

**Key Focus**: Use `data.py`, `data_catalog.py`, `advanced_preprocessing.py` (4-Step Imputation only), `transformers.py`

```python
#%% md
## 2. Loading and Preprocessing Financial Data

Multi-region data loading with 4-step imputation strategy:
1. Zero imputation for metrics that can be zero
2. Price-based imputation for price-derived metrics
3. KNN imputation (sector-aware) for complex relationships
4. Median imputation (sector-aware) as final fallback

#%%
# Load data (auto-detect from DB or CSV)
from finance_ml.data import load_from_db, load_from_csv
from finance_ml.data_catalog import DataCatalog

DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:@localhost:5432/postgres')

try:
    all_stocks = load_from_db(DB_URL, limit=None)
    print(f"✓ Loaded {len(all_stocks)} stocks from database")
except Exception as e:
    print(f"⚠ Database load failed: {e}. Falling back to CSV.")
    all_stocks = load_from_csv(Path("data"), limit=None)
    print(f"✓ Loaded {len(all_stocks)} stocks from CSV")

# Validate schema
validate_schema(all_stocks, require_target=True)

#%%
# Apply 4-step imputation strategy
from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step

print("\n📊 Applying Enhanced 4-Step Imputation Strategy...")
all_stocks = apply_enhanced_imputation_strategy_4step(
    all_stocks, 
    sector_column='sector',
    n_neighbors=5,
    price_column='last_price'
)

print(f"✓ Preprocessing complete: {all_stocks.shape}")
print(f"  Missing values remaining: {all_stocks.isnull().sum().sum()}")
```

**Remove**: Extensive Phase 9.1 sections (9.1.1-9.1.6) with detailed logging, helper functions, and redundant validation
code (~600 lines → ~120 lines)

---

#### **Section 3: Exploratory Data Analysis** (Lines: ~150)

**Key Focus**: Use `advanced_eda.py`, `benchmarking.py`, `eval.py`

```python
#%% md
## 3. Exploratory Data Analysis of Financial Metrics

Comprehensive statistical analysis including:
- Distribution analysis, outlier detection, normality tests
- Correlation matrices (Pearson, Spearman, Kendall)
- Sector and region comparisons with hypothesis tests
- Benchmarking and peer analysis

#%%
# Generate comprehensive EDA report
from finance_ml.advanced_eda import generate_eda_report

eda_output_dir = OUTPUT_DIR / "eda"
eda_output_dir.mkdir(exist_ok=True)

eda_report = generate_eda_report(
    all_stocks,
    target_col='price_target',
    sector_col='sector',
    output_dir=eda_output_dir
)

print(f"✓ EDA Report Generated")
print(f"  Correlations: {len(eda_report.correlations) if eda_report.correlations else 0} features")
print(f"  Statistical tests: {len(eda_report.statistical_tests)} performed")

#%%
# Benchmarking analysis
from finance_ml.benchmarking import generate_benchmarking_report

metrics_to_benchmark = ['p_e', 'p_b', 'ev_ebitda', 'operating_margin', 'roe']
available_metrics = [m for m in metrics_to_benchmark if m in all_stocks.columns]

benchmark_report = generate_benchmarking_report(
    all_stocks,
    metrics=available_metrics,
    sector_column='sector',
    region_column='region'
)

print(f"✓ Benchmarking Report Generated")
print(f"  Sectors analyzed: {benchmark_report['summary']['n_sectors']}")
print(f"  Regions analyzed: {benchmark_report['summary']['n_regions']}")

#%%
# Key visualizations
from finance_ml.eval import simple_eda

simple_eda(
    all_stocks, 
    out_dir=eda_output_dir, 
    save_plots=True,
    target_column='price_target',
    include_multivariate=True
)
```

**Remove**: Phase 9.2 and 9.2 Enhanced sections with redundant helper functions and extensive inline display code (~1000
lines → ~150 lines)

---

#### **Section 4: Advanced Feature Engineering** (Lines: ~100)

**Key Focus**: Use `features.py`, `advanced_features.py`, `eval.py`

```python
#%% md
## 4. Advanced Feature Engineering with Sector-Specific Optimizations

Features include:
- Financial ratios (valuation, profitability, leverage, liquidity, efficiency)
- Sector-specific features (Financials, Energy, Tech, Healthcare, etc.)
- Growth metrics and temporal features
- Relative value features (sector-normalized)
- Feature importance analysis

#%%
# Build comprehensive features
from finance_ml.advanced_features import build_comprehensive_features

all_stocks_features = build_comprehensive_features(
    all_stocks,
    include_interactions=True,
    include_relative_values=True,
    sector_col='sector'
)

print(f"✓ Feature Engineering Complete")
print(f"  Original features: {all_stocks.shape[1]}")
print(f"  Engineered features: {all_stocks_features.shape[1]}")
print(f"  New features added: {all_stocks_features.shape[1] - all_stocks.shape[1]}")

#%%
# Feature importance analysis
from finance_ml.advanced_features import calculate_feature_importance_rf

exclude_cols = ['ticker', 'sector', 'region', 'price_target', 'last_price']
feature_cols = [c for c in all_stocks_features.columns if c not in exclude_cols]

if 'price_target' in all_stocks_features.columns:
    X = all_stocks_features[feature_cols].select_dtypes(include=[np.number])
    y = all_stocks_features['price_target']
    
    importance_df = calculate_feature_importance_rf(X, y, top_k=20)
    print("\n🎯 Top 20 Most Important Features:")
    print(importance_df)
```

**Remove**: Phase 9.3 verbose feature engineering sections with inline helper classes and extensive logging (~800
lines → ~100 lines)

---

#### **Section 5: Multi-Class Classification of Financial Events** (Lines: ~150)

**Key Focus**: Use `classification.py`, `eval.py`

```python
#%% md
## 5. Multi-Class Classification of Financial Events

Train sophisticated classification models to predict financial events:
- Event labeling: Neutral, Positive, Negative (price momentum method)
- Multiple classifiers: XGBoost, LightGBM, CatBoost, Neural Networks, Ensembles
- Export classification probabilities as meta-features for regression

#%%
# Create event labels
from finance_ml.classification import create_enhanced_event_labels, prepare_classification_data

labels = create_enhanced_event_labels(
    all_stocks_features,
    method='price_momentum',
    threshold_positive=10.0,
    threshold_negative=-10.0,
    use_sector_adjustment=True
)

print(f"✓ Event Labels Created")
print(f"  Class distribution:")
print(f"    Neutral (0): {(labels == 0).sum()} ({(labels == 0).sum() / len(labels) * 100:.1f}%)")
print(f"    Positive (1): {(labels == 1).sum()} ({(labels == 1).sum() / len(labels) * 100:.1f}%)")
print(f"    Negative (2): {(labels == 2).sum()} ({(labels == 2).sum() / len(labels) * 100:.1f}%)")

#%%
# Compare multiple classifiers
from finance_ml.classification import compare_classifiers

numeric_cols = all_stocks_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = ['sector', 'region'] if 'sector' in all_stocks_features.columns else []
numeric_cols = [c for c in numeric_cols if c not in ['price_target', 'last_price'] + categorical_cols]

X_train, X_test, y_train, y_test = prepare_classification_data(
    all_stocks_features, labels, test_size=0.2, random_state=RANDOM_SEED
)

comparison_results = compare_classifiers(
    X_train, y_train, X_test, y_test,
    numeric_cols, categorical_cols
)

print("\n📊 Classifier Comparison Results:")
print(comparison_results.sort_values('F1-Score', ascending=False))

#%%
# Export classification features
from finance_ml.classification import train_stacking_classifier, export_classification_features

best_model = train_stacking_classifier(
    X_train, y_train, X_test, y_test,
    numeric_cols, categorical_cols
)

# Get predictions for all data
X_all = pd.concat([X_train, X_test], axis=0)
y_proba_all = best_model['model'].predict_proba(X_all)

all_stocks_with_classification = export_classification_features(
    all_stocks_features,
    y_proba_all,
    class_names=["Neutral", "Positive", "Negative"]
)

print(f"✓ Classification meta-features added: {all_stocks_with_classification.shape}")
```

**Remove**: Phase 9.4 redundant preprocessing, validation, and helper code (~900 lines → ~150 lines)

---

#### **Section 6: Sector-Optimized Regression Models** (Lines: ~120)

**Key Focus**: Use `models.py`, `advanced_models.py`

```python
#%% md
## 6. Training Sector-Optimized Regression Models Enhanced with Classification Features

Train sector-specific regression models with:
- Multiple algorithms: Ridge, Lasso, Elastic Net, XGBoost, LightGBM, CatBoost
- Classification features integrated
- Sector-specific optimization
- Ensemble methods (Stacking, Voting)

#%%
# Prepare regression data
from finance_ml.advanced_models import prepare_regression_data

exclude_cols = ['ticker', 'sector', 'region', 'last_price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_regression_data(
    all_stocks_with_classification,
    target_col='price_target',
    exclude_cols=exclude_cols,
    test_size=0.2,
    random_state=RANDOM_SEED
)

print(f"✓ Regression data prepared: {X_train_reg.shape}")

#%%
# Train sector-specific regression
from finance_ml.advanced_models import train_sector_specific_models

sector_models = train_sector_specific_models(
    all_stocks_with_classification,
    feature_cols=X_train_reg.columns.tolist(),
    target_col='price_target',
    sector_col='sector',
    model_type='xgboost',
    random_state=RANDOM_SEED,
    ensure_nonnegative=True
)

print(f"✓ Sector-Specific Models Trained")
print(f"  Sectors: {list(sector_models.keys())}")

#%%
# Train ensemble regression
from finance_ml.advanced_models import train_stacking_regressor

stacking_model = train_stacking_regressor(
    X_train_reg, y_train_reg,
    cv=5,
    random_state=RANDOM_SEED,
    ensure_nonnegative=True
)

y_pred_stacking = stacking_model.predict(X_test_reg)
print(f"✓ Stacking Ensemble Trained")
```

**Remove**: Phase 9.5 verbose model training with excessive logging (~800 lines → ~120 lines)

---

#### **Section 7: Model Evaluation and Error Analysis** (Lines: ~100)

**Key Focus**: Use `eval.py`

```python
#%% md
## 7. Model Evaluation and Error Analysis

Comprehensive evaluation including:
- Regression metrics (MAE, RMSE, MAPE, R²)
- Residual analysis
- Sector and region performance breakdown
- SHAP analysis for explainability
- Learning curves and bias-variance diagnosis

#%%
# Comprehensive regression metrics
from finance_ml.eval import comprehensive_regression_metrics, compute_metrics_by_segment

metrics = comprehensive_regression_metrics(y_test_reg, y_pred_stacking)
print("📊 Overall Model Performance:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

#%%
# Segment analysis (by sector and region)
from finance_ml.eval import compute_metrics_by_segment

# Prepare test data with predictions
test_data = all_stocks_with_classification.loc[X_test_reg.index].copy()
test_data['predicted_price_target'] = y_pred_stacking

sector_metrics = compute_metrics_by_segment(
    test_data, 'price_target', 'predicted_price_target', 'sector'
)
print("\n📊 Performance by Sector:")
print(sector_metrics)

#%%
# SHAP analysis
from finance_ml.eval import compute_shap_values, create_shap_summary_plot

shap_output_dir = OUTPUT_DIR / "shap"
shap_output_dir.mkdir(exist_ok=True)

create_shap_summary_plot(
    stacking_model,
    X_test_reg,
    output_path=shap_output_dir / "shap_summary.png",
    model_type="tree",
    n_samples=100
)
print("✓ SHAP analysis complete")
```

**Remove**: Phase 9.6 and 9.8 with redundant evaluation code (~700 lines → ~100 lines)

---

#### **Section 8: Identification of Under/Overvalued Stocks** (Lines: ~100)

**Key Focus**: Use `eval.py` visualization functions

```python
#%% md
## 8. Identification of Under/Overvalued Stocks with Visualization

Calculate mispricing scores and identify investment opportunities:
- Mispricing score: (Predicted - Current) / Current
- Valuation categories: Severely Undervalued, Undervalued, Fair, Overvalued, Severely Overvalued
- Sector-relative rankings
- Multi-factor scoring (valuation + quality + growth)

#%%
# Calculate mispricing scores
from finance_ml.eval import calculate_mispricing_score, assign_valuation_category

all_stocks_valued = all_stocks_with_classification.copy()
all_stocks_valued['predicted_price_target'] = stacking_model.predict(
    all_stocks_valued[X_train_reg.columns]
)

all_stocks_valued['mispricing_score'] = calculate_mispricing_score(
    all_stocks_valued,
    predicted_col='predicted_price_target',
    current_col='last_price'
)

all_stocks_valued['valuation_category'] = assign_valuation_category(
    all_stocks_valued['mispricing_score']
)

print(f"✓ Valuation Analysis Complete")

#%%
# Rank stocks
from finance_ml.eval import rank_undervalued_stocks, rank_overvalued_stocks

top_undervalued = rank_undervalued_stocks(all_stocks_valued, top_n=20)
top_overvalued = rank_overvalued_stocks(all_stocks_valued, top_n=20)

print("\n🏆 Top 20 Undervalued Stocks (Buy Opportunities):")
print(top_undervalued[['ticker', 'sector', 'mispricing_score', 'valuation_category']].head(20))

print("\n⚠️  Top 20 Overvalued Stocks (Sell Opportunities):")
print(top_overvalued[['ticker', 'sector', 'mispricing_score', 'valuation_category']].head(20))

#%%
# Visualizations
from finance_ml.eval import create_valuation_scatter_plot, create_sector_heatmap

plots_dir = OUTPUT_DIR / "plots"

create_valuation_scatter_plot(
    all_stocks_valued,
    out_path=plots_dir / "valuation_scatter.png",
    color_by='sector'
)

create_sector_heatmap(
    all_stocks_valued,
    out_path=plots_dir / "sector_heatmap.png",
    metric='mispricing_score'
)

print("✓ Visualizations created")
```

**Remove**: Phase 9.7 verbose helper functions and redundant display code (~600 lines → ~100 lines)

---

#### **Section 9: Comprehensive Analytics - Predicted vs. Analyst Price Target Comparison** (Lines: ~120)

**Key Focus**: Use `eval.py`, `analyst_comparison.py`

```python
#%% md
## 9. Comprehensive Analytics: Predicted vs. Analyst Price Target Comparison

Compare ML predictions with analyst consensus targets:
- Agreement rate and directional accuracy
- Systematic bias analysis
- Disagreement opportunities (contrarian plays)
- Segment analysis by sector/region
- Calibration and confidence metrics

#%%
# Prediction vs Analyst comparison
from finance_ml.analyst_comparison import PredictionAnalystAnalytics

analytics = PredictionAnalystAnalytics(all_stocks_valued)
analytics.run_full_analysis(
    disagreement_threshold=10.0,
    top_n=50
)

#%%
# Generate comprehensive Excel report
from finance_ml.eval import generate_prediction_analyst_excel_report

reports_dir = OUTPUT_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

generate_prediction_analyst_excel_report(
    all_stocks_valued,
    excel_path=reports_dir / "prediction_analyst_comparison.xlsx",
    top_n_opportunities=50
)

print("✓ Excel report generated")

#%%
# Generate PDF report
from finance_ml.eval import generate_enhanced_pdf_report

generate_enhanced_pdf_report(
    all_stocks_valued,
    pdf_path=reports_dir / "stock_valuation_report.pdf",
    title="Stock Price Target Analysis - Comprehensive Report",
    include_financial_dashboard=True,
    include_quality_alerts=True,
    include_hypothesis_tests=True,
    include_charts=False
)

print("✓ PDF report generated")
```

**Remove**: Scattered analytics sections and redundant reporting code (~400 lines → ~120 lines)

---

#### **Section 10: Portfolio Optimization** (Lines: ~100)

**Key Focus**: Use `portfolio_optimization.py`, `risk_metrics.py`

```python
#%% md
## 10. Portfolio Optimization with Risk Metrics

Construct optimized portfolios based on predictions:
- Maximum Sharpe ratio optimization
- Minimum volatility optimization
- Target return optimization
- Risk metrics (VaR, CVaR, Sharpe, Sortino, Max Drawdown)

#%%
# Prepare portfolio data (top undervalued stocks)
from finance_ml.portfolio_optimization import (
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
    generate_efficient_frontier
)

top_candidates = rank_undervalued_stocks(all_stocks_valued, top_n=50)

# Calculate expected returns (mispricing as proxy)
expected_returns = top_candidates['mispricing_score'].values / 100

# Estimate covariance (simplified - use historical returns in production)
n_stocks = len(top_candidates)
cov_matrix = np.eye(n_stocks) * 0.04  # Simplified example

#%%
# Optimize for maximum Sharpe ratio
optimal_portfolio = optimize_portfolio_max_sharpe(
    expected_returns,
    cov_matrix,
    risk_free_rate=0.02,
    allow_short=False,
    max_weight=0.15
)

print("✓ Portfolio Optimization Complete")
print(f"  Expected Return: {optimal_portfolio['portfolio_return']:.2%}")
print(f"  Portfolio Volatility: {optimal_portfolio['portfolio_volatility']:.2%}")
print(f"  Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}")

#%%
# Calculate risk metrics
from finance_ml.risk_metrics import calculate_portfolio_risk_metrics

# Simulated portfolio returns (use actual historical data in production)
portfolio_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 252)

risk_metrics = calculate_portfolio_risk_metrics(
    pd.Series(portfolio_returns),
    risk_free_rate=0.02,
    confidence_levels=[0.95, 0.99]
)

print("\n📊 Portfolio Risk Metrics:")
for metric, value in risk_metrics.items():
    print(f"  {metric}: {value}")

print("\n✅ Portfolio Optimization Complete")
```

**Action**: Add this new section (was missing or scattered in current notebook)

---

### Summary of Changes

| Current Section              | Lines    | Streamlined Section | Lines     | Reduction |
|------------------------------|----------|---------------------|-----------|-----------|
| Header + Config              | ~200     | Header + Setup      | ~110      | -45%      |
| Phase 9.1 (Preprocessing)    | ~600     | Section 2           | ~120      | -80%      |
| Phase 9.2 (EDA)              | ~1000    | Section 3           | ~150      | -85%      |
| Phase 9.3 (Features)         | ~800     | Section 4           | ~100      | -87%      |
| Phase 9.4 (Classification)   | ~900     | Section 5           | ~150      | -83%      |
| Phase 9.5 (Regression)       | ~800     | Section 6           | ~120      | -85%      |
| Phase 9.6 + 9.8 (Evaluation) | ~700     | Section 7           | ~100      | -86%      |
| Phase 9.7 (Valuation)        | ~600     | Section 8           | ~100      | -83%      |
| Analytics (scattered)        | ~400     | Section 9           | ~120      | -70%      |
| Portfolio (missing)          | ~0       | Section 10          | ~100      | New       |
| Helper code                  | ~755     | Removed             | ~0        | -100%     |
| **Total**                    | **5755** | **Total**           | **~1170** | **-80%**  |

---

### Key Restructuring Principles

#### 1. **Leverage Package Functions**

- Remove all inline helper functions — use `finance_ml` package
- Import functions directly rather than reimplementing
- Trust tested package implementations

#### 2. **Remove Redundancy**

- Eliminate phase-by-phase summaries
- Consolidate validation and logging
- Remove duplicate display code
- Simplify checkpoint systems

#### 3. **Focus on Business Objective**

- Each section maps to one business requirement
- Clear linear flow: Data → EDA → Features → Models → Analytics → Portfolio
- Target variable always: "Predicted Price Target"

#### 4. **Meaningful Analytics Integration**

- Use `eval.py` functions throughout for visualization
- Leverage `analyst_comparison.py` for comprehensive analytics
- Add `portfolio_optimization.py` and `risk_metrics.py` for investment decisions

#### 5. **Production-Ready Code**

- Configuration via `NotebookConfig`
- Error handling without verbose logging
- Clean outputs with professional formatting
- Modular cells for easy execution

---

### Implementation Steps

1. **Create backup**: `cp ml_finance_model_main.ipynb ml_finance_model_main_backup.ipynb`

2. **Start with reference template**: Use `ml_stock_prediction_model.ipynb` structure as foundation

3. **Migrate sections sequentially**:
    - Copy header and configuration (Section 0-1)
    - Implement Section 2 (Data Loading) with 4-step imputation
    - Implement Section 3 (EDA) with benchmarking
    - Implement Section 4 (Features) with importance analysis
    - Implement Section 5 (Classification) with ensemble models
    - Implement Section 6 (Regression) with sector optimization
    - Implement Section 7 (Evaluation) with comprehensive metrics
    - Implement Section 8 (Valuation) with visualizations
    - Implement Section 9 (Analytics) with analyst comparison
    - Implement Section 10 (Portfolio) with optimization

4. **Test each section**: Run cells sequentially to validate

5. **Remove legacy code**: Delete all Phase 9.x sections and helper functions

6. **Document**: Add clear markdown cells explaining each step

---

### Module Mapping to Sections

| Section                    | Primary Modules                                                              | Key Functions                                                                                  |
|----------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| 2. Loading & Preprocessing | `data.py`, `data_catalog.py`, `advanced_preprocessing.py`, `transformers.py` | `load_from_db()`, `apply_enhanced_imputation_strategy_4step()`                                 |
| 3. EDA                     | `advanced_eda.py`, `benchmarking.py`, `eval.py`                              | `generate_eda_report()`, `generate_benchmarking_report()`, `simple_eda()`                      |
| 4. Features                | `features.py`, `advanced_features.py`, `eval.py`                             | `build_comprehensive_features()`, `calculate_feature_importance_rf()`                          |
| 5. Classification          | `classification.py`, `eval.py`                                               | `create_enhanced_event_labels()`, `compare_classifiers()`, `train_stacking_classifier()`       |
| 6. Regression              | `models.py`, `advanced_models.py`                                            | `train_sector_specific_models()`, `train_stacking_regressor()`                                 |
| 7. Evaluation              | `eval.py`                                                                    | `comprehensive_regression_metrics()`, `compute_shap_values()`, `compute_metrics_by_segment()`  |
| 8. Valuation               | `eval.py`                                                                    | `calculate_mispricing_score()`, `rank_undervalued_stocks()`, `create_valuation_scatter_plot()` |
| 9. Analytics               | `eval.py`, `analyst_comparison.py`                                           | `PredictionAnalystAnalytics`, `generate_prediction_analyst_excel_report()`                     |
| 10. Portfolio              | `portfolio_optimization.py`, `risk_metrics.py`                               | `optimize_portfolio_max_sharpe()`, `calculate_portfolio_risk_metrics()`                        |

---

### Expected Benefits

1. **80% code reduction**: 5755 → ~1170 lines
2. **Improved maintainability**: All logic in tested package modules
3. **Clearer business alignment**: 9 sections match 9 business requirements
4. **Better readability**: Clean flow without excessive logging
5. **Professional presentation**: Production-ready notebook
6. **Complete workflow**: Adds missing portfolio optimization
7. **Easier execution**: Linear dependency chain, clear checkpoints
8. **Better analytics**: Leverages full `eval.py` and `analyst_comparison.py` capabilities

This restructuring transforms the notebook from a verbose development artifact into a streamlined, production-ready
analytical workflow that directly supports the business objective of predicting stock price targets for portfolio
optimization.