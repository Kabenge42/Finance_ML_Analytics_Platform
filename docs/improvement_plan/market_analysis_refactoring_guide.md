# Market Analytics Refactoring Guide

## Overview

The original `market_analytics.py` notebook code (5208 lines) has been refactored into a modular, maintainable
structure. This guide explains the new architecture, how to use the refactored modules, and how to migrate from the
original code.

## Refactored Structure

### Module Organization

```
finance_ml/analytics/
├── __init__.py                 # Package exports (128 lines)
├── data_utils.py               # Data loading and preprocessing (300 lines)
├── statistical_analysis.py     # Advanced statistical methods (1282 lines)
├── screening.py                # Stock screening functions (691 lines)
├── feature_analytics.py        # Visualization dashboards (1644 lines)
├── probability_analytics.py    # Probability models (1827 lines)
├── optimized_ops.py            # Performance optimizations (622 lines)
└── visualizations/
    ├── __init__.py             # Visualization package exports (257 lines)
    ├── profitability.py        # Margin and profitability charts (647 lines)
    ├── technical.py            # Technical analysis charts (565 lines)
    ├── temporal_analysis.py    # Time series analysis (769 lines)
    ├── category_charts.py      # Category-specific charts
    ├── valuation.py            # Valuation analysis charts (669 lines) [NEW]
    ├── earnings_quality.py     # Earnings quality charts (658 lines) [NEW]
    ├── quality_risk.py         # Quality & risk charts (852 lines) [NEW]
    ├── growth_analysis.py      # Growth metrics charts (631 lines) [NEW]
    └── probability_viz.py      # Probabilistic ArviZ-backed charts (857 lines) [NEW]

market_analytics.py             # Main demonstration script (935 lines)
```

### Total Lines of Code

- **Original**: 5208 lines (monolithic)
- **Refactored**: ~10,500+ lines (modular, reusable, with enhanced features)
- **Enhancement**: +100% additional functionality through new visualization, statistical, and optimization modules
- **New Visualization Modules**: 5 modules with 27 new visualization functions covering valuation, earnings quality,
  quality/risk, growth analysis, and probabilistic Bayesian diagnostics

---

## Module Descriptions

### 1. `data_utils.py`

**Purpose**: Data loading, preprocessing, and validation

**Key Functions**:

- `load_feature_data_from_db()` - Load data from PostgreSQL materialized view
- `backfill_feature_columns()` - Fill missing columns with calculated values
- `compute_metric_statistics()` - Calculate comprehensive statistics for features
- `validate_feature_alignment()` - Check feature coverage by category
- `safe_get_column()` - Safely retrieve columns with fallback options

**Example Usage**:

```python
from finance_ml.analytics.data_utils import (
    load_feature_data_from_db,
    backfill_feature_columns,
    compute_metric_statistics
)

# Load data
df = load_feature_data_from_db(earnings_date_filter="2026-01-01")

# Backfill missing columns
df = backfill_feature_columns(df)

# Get statistics
stats = compute_metric_statistics(df['p_e_ratio'])
print(f"Mean P/E: {stats['mean']:.2f}")
```

---

### 2. `statistical_analysis.py`

**Purpose**: Advanced statistical analysis including Bayesian methods, MCMC, and Monte Carlo simulations

**Key Functions**:

- `bayesian_category_analysis()` - Bayesian parameter estimation with conjugate priors
- `metropolis_hastings_sampler()` - MCMC sampling for posterior distributions
- `mcmc_student_t()` - MCMC with Student's t distribution (for heavy tails)
- `hierarchical_mcmc_by_sector()` - Hierarchical Bayesian modeling by sector
- `fit_distributions_by_category()` - Fit and select best distribution using AIC
- `calculate_ruin_probability()` - Investor's ruin probability (Gambler's Ruin)
- `calculate_conditional_probabilities()` - P(Distress | Feature) analysis

**Example Usage**:

```python
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis,
    calculate_ruin_probability,
    calculate_conditional_probabilities
)

# Bayesian analysis
results = bayesian_category_analysis(
    df,
    'Profitability',
    ['roe', 'roa', 'roic']
)
print(f"ROE posterior mean: {results['roe']['posterior_mean']:.2f}")

# Ruin probability
ruin_df = calculate_ruin_probability(df)
high_risk = ruin_df[ruin_df['ruin_probability'] > 0.6]
print(f"High risk stocks: {len(high_risk)}")

# Conditional probabilities
cond_probs = calculate_conditional_probabilities(df, FEATURE_CATEGORIES)
top_predictors = cond_probs.nlargest(10, 'separation')
```

---

### 3. `screening.py`

**Purpose**: Multi-factor stock screening and quality scoring

**Key Functions**:

- `create_enhanced_screener()` - Multi-factor quality and momentum screening
- `screen_earnings_quality()` - Filter by earnings quality metrics
- `screen_value_opportunities()` - Find undervalued stocks
- `screen_growth_momentum()` - Identify growth stocks
- `screen_dividend_quality()` - Quality dividend stock screening
- `screen_financial_health()` - Filter financially healthy companies
- `rank_stocks_by_composite_score()` - Composite quality ranking
- `create_sector_relative_ranking()` - Sector-relative performance ranking

**Example Usage**:

```python
from finance_ml.analytics.screening import (
    create_enhanced_screener,
    screen_value_opportunities,
    screen_growth_momentum
)

# Quality screening
quality_stocks = create_enhanced_screener(
    df,
    min_fscore=7,
    min_fcf_positive_years=4,
    require_deleveraging=True
)

# Value screening
value_stocks = screen_value_opportunities(
    df,
    max_pe_ratio=20,
    min_upside_potential=25
)

# Growth screening
growth_stocks = screen_growth_momentum(
    df,
    min_revenue_growth=15,
    min_eps_growth=10
)
```

---

### 4. `feature_analytics.py` (Existing)

**Purpose**: Interactive visualization dashboards

**Key Functions**:

- `create_interactive_momentum_dashboard()` - Momentum analysis dashboard
- `create_interactive_valuation_heatmap()` - Valuation by industry heatmap
- `create_leverage_liquidity_quadrant()` - Leverage vs liquidity analysis
- `monte_carlo_price_target_simulation()` - Monte Carlo price target simulation
- `bayesian_earnings_beat_model()` - Bayesian earnings beat probability
- `analyze_distress_distribution()` - Financial distress distribution
- `create_composite_quality_score()` - Composite quality scoring
- `create_summary_dashboard()` - KPI summary dashboard

**Example Usage**:

```python
from finance_ml.analytics.feature_analytics import (
    create_interactive_momentum_dashboard,
    monte_carlo_price_target_simulation,
    create_summary_dashboard
)

# Create dashboards
momentum_fig = create_interactive_momentum_dashboard(df)
momentum_fig.write_html("outputs/momentum.html")

# Monte Carlo simulation
mc_results = monte_carlo_price_target_simulation(df, n_simulations=10000)
top_opportunities = mc_results.nlargest(20, 'risk_reward_ratio')

# Summary dashboard
summary_fig = create_summary_dashboard(df)
summary_fig.show()
```

---

### 5. `visualizations/profitability.py` (New)

**Purpose**: Margin and profitability analysis visualizations

**Key Functions**:

- `create_margin_waterfall_chart()` - Revenue to net income margin breakdown
- `create_dupont_decomposition_dashboard()` - ROE = Net Margin × Asset Turnover × Leverage
- `create_profitability_quadrant()` - ROE vs ROIC with margin bubble size
- `create_margin_trend_heatmap()` - Margin trends by industry

**Example Usage**:

```python
from finance_ml.analytics.visualizations.profitability import (
    create_margin_waterfall_chart,
    create_dupont_decomposition_dashboard,
    create_profitability_quadrant
)

# DuPont analysis dashboard
dupont_fig = create_dupont_decomposition_dashboard(df)
dupont_fig.write_html("outputs/dupont_analysis.html")

# Profitability quadrant (ROE vs ROIC)
quadrant_fig = create_profitability_quadrant(df)
quadrant_fig.show()

# Margin waterfall for a specific stock
waterfall_fig = create_margin_waterfall_chart(df[df['ticker'] == 'AAPL'])
```

---

### 6. `visualizations/technical.py` (New)

**Purpose**: Technical analysis and momentum visualizations

**Key Functions**:

- `create_momentum_ribbon_chart()` - Multi-period momentum overlay (1m-5y)
- `create_52w_range_distribution()` - Overbought/oversold analysis by sector
- `create_trend_strength_matrix()` - Trend score heatmap by industry
- `create_momentum_divergence_scatter()` - Short vs long-term momentum divergence

**Example Usage**:

```python
from finance_ml.analytics.visualizations.technical import (
    create_momentum_ribbon_chart,
    create_52w_range_distribution,
    create_momentum_divergence_scatter
)

# Momentum ribbon chart
ribbon_fig = create_momentum_ribbon_chart(df)
ribbon_fig.write_html("outputs/momentum_ribbon.html")

# 52-week range distribution by sector
range_fig = create_52w_range_distribution(df)
range_fig.show()

# Identify momentum divergences
divergence_fig = create_momentum_divergence_scatter(df)
```

---

### 7. `visualizations/temporal_analysis.py` (New)

**Purpose**: Time series and temporal pattern visualizations

**Key Functions**:

- `create_earnings_calendar_heatmap()` - Earnings dates with quality overlay
- `create_inventory_cycle_analysis()` - Inventory days and turnover trends
- `create_fcf_trajectory_chart()` - FCF positive years visualization
- `create_dividend_streak_timeline()` - Dividend sustainability analysis

**Example Usage**:

```python
from finance_ml.analytics.visualizations.temporal_analysis import (
    create_earnings_calendar_heatmap,
    create_fcf_trajectory_chart,
    create_dividend_streak_timeline
)

# Earnings calendar with quality scores
calendar_fig = create_earnings_calendar_heatmap(df)
calendar_fig.write_html("outputs/earnings_calendar.html")

# FCF trajectory analysis
fcf_fig = create_fcf_trajectory_chart(df)
fcf_fig.show()

# Dividend streak timeline
dividend_fig = create_dividend_streak_timeline(df)
```

---

### 8. `visualizations/valuation.py` (New)

**Purpose**: Comprehensive valuation ratio analysis and visualization

**Key Functions**:

- `create_valuation_multiples_comparison()` - Spider/radar chart comparing P/E, P/B, EV/EBITDA vs sector median
- `create_valuation_distribution_dashboard()` - Multi-panel violin plots for valuation metrics by sector
- `create_relative_valuation_matrix()` - Heatmap of Z-scores identifying cheap/expensive sectors
- `create_valuation_vs_growth_quadrant()` - PEG-style scatter with quadrants (cheap+growing, expensive+slow)
- `create_historical_valuation_percentile()` - Distribution showing current valuations vs historical ranges

**Example Usage**:

```python
from finance_ml.analytics.visualizations.valuation import (
   create_valuation_multiples_comparison,
   create_valuation_vs_growth_quadrant,
   create_relative_valuation_matrix
)

# Radar chart for specific stock vs sector
radar_fig = create_valuation_multiples_comparison(df, ticker='AAPL')
radar_fig.write_html("outputs/valuation_radar.html")

# PEG-style quadrant analysis
quadrant_fig = create_valuation_vs_growth_quadrant(df)
quadrant_fig.show()

# Sector valuation heatmap
matrix_fig = create_relative_valuation_matrix(df, group_col='industry')
```

---

### 9. `visualizations/earnings_quality.py` (New)

**Purpose**: Deep-dive earnings quality and predictability analysis

**Key Functions**:

- `create_earnings_surprise_dashboard()` - Multi-panel: surprise distribution, beat rate by sector
- `create_eps_trajectory_analysis()` - Trajectory score with improvement counts and streak analysis
- `create_earnings_quality_decomposition()` - Waterfall: accruals ratio, cash conversion, persistence
- `create_beat_rate_heatmap()` - Historical beat rates by sector
- `create_earnings_consistency_matrix()` - eps_positive_streak vs eps_improvement_count by sector

**Example Usage**:

```python
from finance_ml.analytics.visualizations.earnings_quality import (
   create_earnings_surprise_dashboard,
   create_eps_trajectory_analysis,
   create_earnings_quality_decomposition
)

# Earnings surprise analysis
surprise_fig = create_earnings_surprise_dashboard(df)
surprise_fig.write_html("outputs/earnings_surprise.html")

# EPS trajectory for top performers
trajectory_fig = create_eps_trajectory_analysis(df, top_n=30)
trajectory_fig.show()

# Quality decomposition for specific stock
quality_fig = create_earnings_quality_decomposition(df, ticker='MSFT')
```

---

### 10. `visualizations/quality_risk.py` (New)

**Purpose**: Comprehensive quality scoring and risk assessment visualization

**Key Functions**:

- `create_piotroski_fscore_breakdown()` - F-Score distribution with pass/fail indicators
- `create_altman_zscore_distribution()` - Distribution with distress zones (safe/gray/distress)
- `create_quality_risk_quadrant()` - Piotroski F-Score vs Altman Z-Score scatter
- `create_beneish_mscore_analysis()` - M-Score with manipulation probability zones
- `create_risk_tier_sunburst()` - Sector → Industry → Risk Tier hierarchy
- `create_distress_early_warning_dashboard()` - Companies approaching distress thresholds

**Example Usage**:

```python
from finance_ml.analytics.visualizations.quality_risk import (
   create_piotroski_fscore_breakdown,
   create_altman_zscore_distribution,
   create_quality_risk_quadrant
)

# F-Score analysis
fscore_fig = create_piotroski_fscore_breakdown(df, ticker='AAPL')
fscore_fig.write_html("outputs/fscore_analysis.html")

# Z-Score distribution with risk zones
zscore_fig = create_altman_zscore_distribution(df, group_col='industry')
zscore_fig.show()

# Quality vs Risk quadrant
quadrant_fig = create_quality_risk_quadrant(df)
```

---

### 11. `visualizations/growth_analysis.py` (New)

**Purpose**: Comprehensive growth metrics analysis similar to profitability.py structure

**Key Functions**:

- `create_growth_waterfall_chart()` - Revenue → EBITDA → EPS growth decomposition
- `create_growth_consistency_matrix()` - Growth metrics consistency (YoY, 3Y CAGR, 5Y CAGR) by sector
- `create_growth_vs_profitability_quadrant()` - BCG-style: Revenue growth vs ROE with margin bubble
- `create_growth_acceleration_chart()` - Growth acceleration (current vs historical) ranked
- `create_sustainable_growth_analysis()` - SGR = ROE × Retention Rate analysis by sector

**Example Usage**:

```python
from finance_ml.analytics.visualizations.growth_analysis import (
   create_growth_waterfall_chart,
   create_growth_vs_profitability_quadrant,
   create_growth_acceleration_chart
)

# Growth decomposition waterfall
waterfall_fig = create_growth_waterfall_chart(df, ticker='GOOGL')
waterfall_fig.write_html("outputs/growth_waterfall.html")

# BCG-style growth vs profitability
bcg_fig = create_growth_vs_profitability_quadrant(df)
bcg_fig.show()

# Growth acceleration analysis
accel_fig = create_growth_acceleration_chart(df, top_n=25)
```

---

### 12. `visualizations/probability_viz.py` (New)

**Purpose**: Probabilistic financial analysis visualizations with ArviZ-enhanced Bayesian diagnostics

**Key Functions**:

- `create_posterior_return_forest()` - Forest plot of posterior expected returns (HDI + R-hat from InferenceData or CI
  from DataFrame)
- `create_beat_probability_posterior()` - Posterior density for earnings beat probability (Beta PDF / KDE / bar
  fallback)
- `create_ruin_probability_diagnostic()` - Four-panel diagnostic dashboard: top ruin probabilities, risk tier pie,
  scatter vs distress score, sector medians
- `create_mcse_convergence_panel()` - Monte Carlo Standard Error convergence per chain with ESS/R-hat annotations (ArviZ
  required)
- `create_bayesian_category_ridge()` - Ridge plot of posterior feature distributions from `bayesian_category_analysis()`
  output
- `create_tri_model_posterior_comparison()` - Overlaid Normal posteriors from Monte Carlo / Kalman / Achievement models

**Design**: All functions accept either `arviz.InferenceData` (full ArviZ path with HDI, R-hat, ESS diagnostics) or
`pd.DataFrame` (graceful fallback using `scipy.stats`). Uses `PLOTLY_TEMPLATE` and `COLORS` from `_shared.py`.

**Example Usage**:

```python
from finance_ml.analytics.visualizations.probability_viz import (
    create_posterior_return_forest,
    create_beat_probability_posterior,
    create_ruin_probability_diagnostic,
    create_mcse_convergence_panel,
    create_bayesian_category_ridge,
    create_tri_model_posterior_comparison,
)
from finance_ml.analytics.statistical_analysis import bayesian_category_analysis

# Forest plot from Monte Carlo DataFrame
fig = create_posterior_return_forest(mc_results, top_n=25)
fig.write_html("outputs/analytics/posterior_return_forest.html")

# Beat probability posterior from Bayesian earnings model
fig = create_beat_probability_posterior(bayesian_results, top_n=12)
fig.show()

# Ruin probability diagnostic from credit risk DataFrame
fig = create_ruin_probability_diagnostic(ruin_df, top_n=20)
fig.write_html("outputs/analytics/ruin_probability_diagnostic.html")

# Category ridge plot from Bayesian analysis output
prof_results = bayesian_category_analysis(df, 'Profitability', ['roe', 'roa', 'roic'])
fig = create_bayesian_category_ridge(prof_results, category_name='Profitability')
fig.show()

# Tri-model comparison from expected_returns_tri_model table
fig = create_tri_model_posterior_comparison(tri_df, top_n=8)
fig.write_html("outputs/analytics/tri_model_posterior_comparison.html")
```

---

### 13. Enhanced Statistical Methods (New in `statistical_analysis.py`)

**Purpose**: Advanced time series filtering, dependency modeling, and parallel MCMC

**New Functions**:

- `kalman_filter_price_target()` - Kalman filter for smoothing price targets
- `kalman_momentum_filter()` - Smooth noisy momentum indicators
- `fit_gaussian_copula()` - Dependency structure modeling with tail dependence
- `parallel_mcmc_chains()` - Multi-chain MCMC with Gelman-Rubin diagnostic

**Example Usage**:

```python
from finance_ml.analytics.statistical_analysis import (
    kalman_filter_price_target,
    fit_gaussian_copula,
    parallel_mcmc_chains
)

# Kalman filter for price targets
kalman_results = kalman_filter_price_target(df)
smoothed_targets = kalman_results['kalman_estimate']

# Copula dependency modeling
copula_result = fit_gaussian_copula(
    df,
    features=['roe', 'roa', 'debt_to_equity', 'current_ratio']
)
print(f"Tail dependence: {copula_result['tail_dependence']}")

# Parallel MCMC with convergence diagnostics
mcmc_result = parallel_mcmc_chains(
    data=df['roe'].dropna().values,
    n_chains=4,
    n_samples=10000
)
print(f"R-hat convergence: {mcmc_result['r_hat']:.3f}")
print(f"Converged: {mcmc_result['converged']}")
```

---

### 9. `optimized_ops.py` (New)

**Purpose**: Performance-optimized operations with caching and vectorization

**Key Functions**:

- `dataframe_hash()` - Generate hash for DataFrame caching
- `load_feature_data_from_db_cached()` - Cached database queries
- `fast_monte_carlo_simulation()` - Numba-accelerated Monte Carlo
- `fast_ruin_probability()` - Vectorized ruin probability calculation
- `vectorized_zscore()` - Efficient z-score computation
- `vectorized_percentile_rank()` - Fast percentile ranking
- `get_optimization_status()` - Check optimization feature availability

**Example Usage**:

```python
from finance_ml.analytics.optimized_ops import (
    load_feature_data_from_db_cached,
    fast_monte_carlo_simulation,
    vectorized_zscore,
    get_optimization_status
)

# Check available optimizations
status = get_optimization_status()
print(f"Numba available: {status['numba_available']}")
print(f"Joblib available: {status['joblib_available']}")

# Cached data loading (subsequent calls use cache)
df = load_feature_data_from_db_cached(earnings_date_filter="2026-01-01")

# Fast Monte Carlo simulation
expected_upside, upside_std, var_5, prob_positive = fast_monte_carlo_simulation(
    pt_low=df['price_target_low'].values,
    pt_median=df['price_target_median'].values,
    pt_high=df['price_target_high'].values,
    last_price=df['last_price'].values,
    n_simulations=10000
)

# Vectorized z-score calculation
z_scores = vectorized_zscore(df['p_e_ratio'].values)
```

---

## Migration Guide

### From Original Notebook to Refactored Code

#### Before (Original Notebook):

```python
# All code in one file, ~5200 lines
# Repeated data loading logic
# Inline statistical calculations
# Mixed visualization and analysis code
```

#### After (Refactored):

```python
# Import only what you need
from finance_ml.analytics.data_utils import load_feature_data_from_db
from finance_ml.analytics.screening import create_enhanced_screener
from finance_ml.analytics.feature_analytics import create_summary_dashboard

# Clean, modular workflow
df = load_feature_data_from_db()
quality_stocks = create_enhanced_screener(df, min_fscore=7)
fig = create_summary_dashboard(quality_stocks)
fig.show()
```

### Key Changes

1. **Data Loading**:
    - Old: Inline SQL queries and backfill logic scattered throughout
    - New: Centralized in `data_utils.py`

2. **Statistical Analysis**:
    - Old: MCMC and Bayesian code embedded in notebook cells
    - New: Reusable functions in `statistical_analysis.py`

3. **Screening**:
    - Old: Inline filtering with repeated logic
    - New: Parameterized screening functions in `screening.py`

4. **Visualizations**:
    - Old: Mixed with analysis code
    - New: Separated in `feature_analytics.py` and `visualizations/`

---

## Feature Categories

The refactored code uses standardized feature categories:

```python
FEATURE_CATEGORIES = {
    'Valuation Ratios': ['p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', ...],
    'Momentum & Technical': ['price_momentum_1m', 'price_momentum_3m', ...],
    'Profitability': ['roe', 'roa', 'gross_margin_pct', ...],
    'Quality & Risk': ['piotroski_f_score', 'distress_risk_score', ...],
    'Leverage & Liquidity': ['debt_to_equity', 'current_ratio', ...],
    'Analyst Sentiment': ['analyst_bullish_pct', 'upside_potential', ...],
    'Earnings Quality': ['eps_surprise_pct', 'eps_adjustment_ratio', ...],
   'Growth Metrics': ['revenue_yoy_growth', 'ebitda_growth_yoy', ...],
    'Cash Flow': ['fcf_positive_years', 'fcf_margin', 'fcf_yield', ...],
    'Dividend Features': ['dividend_streak', 'dividend_yield_ltm', ...],
    'R&D Investment': ['rnd_intensity_ltm', 'rnd_yoy_growth', ...],
   'Inventory Temporal': ['inventory_days', 'inventory_turnover_itf', ...],
    'Goodwill & M&A': ['goodwill_concentration', 'goodwill_3y_growth', ...],
    'CapEx & Investment': ['capex_yoy_growth', 'capex_vs_5y_avg', ...],
}
```

---

## Running the Refactored Code

### Quick Start

```bash
# Run the main refactored script
python market_analytics.py
```

### Custom Analysis

```python
from finance_ml.analytics.data_utils import load_feature_data_from_db, backfill_feature_columns
from finance_ml.analytics.screening import create_enhanced_screener
from finance_ml.analytics.statistical_analysis import calculate_ruin_probability

# Load and prepare data
df = load_feature_data_from_db(earnings_date_filter="2026-01-01")
df = backfill_feature_columns(df)

# Screen for quality stocks
quality_stocks = create_enhanced_screener(
    df,
    min_fscore=7,
    min_fcf_positive_years=4,
    require_deleveraging=True
)

# Analyze risk
ruin_analysis = calculate_ruin_probability(quality_stocks)
low_risk = ruin_analysis[ruin_analysis['risk_tier'] == 'Low Risk']

print(f"Found {len(low_risk)} low-risk, high-quality stocks")
```

---

## Benefits of Refactoring

### 1. **Modularity**

- Each module has a single, clear responsibility
- Easy to test individual components
- Reusable across different projects

### 2. **Maintainability**

- Easier to locate and fix bugs
- Clear separation of concerns
- Better code organization

### 3. **Extensibility**

- Add new screening criteria without touching existing code
- Extend statistical methods independently
- Create new visualizations without affecting analysis

### 4. **Performance**

- Import only what you need
- Reduced memory footprint
- Faster development iteration

### 5. **Testability**

- Each function can be unit tested
- Mock dependencies easily
- Better code coverage

---

## Testing

### Unit Tests (Recommended)

```python
# tests/test_screening.py
import pandas as pd
from finance_ml.analytics.screening import create_enhanced_screener


def test_enhanced_screener():
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT'],
        'piotroski_f_score': [8, 6],
        'distress_risk_score': [85, 70],
        'eps_trajectory_score': [75, 60],
        'fcf_positive_years': [5, 3],
    })

    result = create_enhanced_screener(df, min_fscore=7)
    assert len(result) == 1
    assert result.iloc[0]['ticker'] == 'AAPL'
```

### Integration Tests

```python
# tests/test_integration.py
from finance_ml.analytics.data_utils import load_feature_data_from_db
from finance_ml.analytics.screening import create_enhanced_screener


def test_full_workflow():
    df = load_feature_data_from_db(limit=100)
    quality_stocks = create_enhanced_screener(df, min_fscore=5)
    assert len(quality_stocks) > 0
```

### Test Coverage Summary

The refactored modules have comprehensive test coverage:

| Test File                              | Tests         | Coverage                       |
|----------------------------------------|---------------|--------------------------------|
| `test_screening.py`                    | 42 tests      | Screening functions            |
| `test_data_utils.py`                   | 38 tests      | Data loading and preprocessing |
| `test_statistical_analysis.py`         | 39 tests      | Bayesian, MCMC, distributions  |
| `test_market_analytics_integration.py` | 18 tests      | Cross-module workflows         |
| `test_visualizations.py`               | 35 tests      | All 12 visualization functions |
| `test_enhanced_statistics.py`          | 40 tests      | Kalman, Copula, parallel MCMC  |
| **Total**                              | **212 tests** | All modules covered            |

Run all tests:

```bash
pytest tests/test_screening.py tests/test_data_utils.py tests/test_statistical_analysis.py tests/test_market_analytics_integration.py tests/test_visualizations.py tests/test_enhanced_statistics.py -v
```

---

## Implemented Enhancements

The following enhancements have been implemented as of version 2.0.0:

### ✅ 1. Additional Visualization Modules (Completed)

- `visualizations/profitability.py` - Margin and profitability charts (561 lines)
- `visualizations/technical.py` - Technical analysis charts (561 lines)
- `visualizations/temporal_analysis.py` - Time series analysis (769 lines)

### ✅ 2. Enhanced Statistical Methods (Completed)

- Kalman filtering for time series (`kalman_filter_price_target`, `kalman_momentum_filter`)
- Copula-based dependency modeling (`fit_gaussian_copula`)
- Parallel MCMC with convergence diagnostics (`parallel_mcmc_chains`)

### ✅ 3. Performance Optimization (Completed)

- Caching for expensive calculations (`load_feature_data_from_db_cached`, `dataframe_hash`)
- Parallel processing for MCMC (`parallel_mcmc_chains` with joblib)
- Vectorized operations (`vectorized_zscore`, `vectorized_percentile_rank`)
- Numba-accelerated Monte Carlo (`fast_monte_carlo_simulation`, `fast_ruin_probability`)

---

## Future Enhancements

### Planned Improvements

1. **API Development**
    - REST API for screening functions
    - WebSocket for real-time updates
    - GraphQL interface

2. **Machine Learning Integration**
    - ML-based quality classification
    - Price target regression models
    - Feature importance analysis

3. **Real-time Data Integration**
    - Live market data feeds
    - Streaming analytics
    - Alert system for screening triggers

---

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'finance_ml.analytics'`
**Solution**: Ensure you're running from the project root and the package is installed:

```bash
pip install -e .
```

**Issue**: Database connection errors
**Solution**: Set environment variables:

```bash
export DB_URL="postgresql+psycopg2://user:pass@host:5432/db"
export DB_EQUITIES_SCHEMA="public"
```

**Issue**: Missing features in DataFrame
**Solution**: Use `backfill_feature_columns()` to create derived features:

```python
from finance_ml.analytics.data_utils import backfill_feature_columns

df = backfill_feature_columns(df)
```

---

## Contributing

### Adding New Screening Functions

1. Add function to `screening.py`
2. Follow existing naming conventions
3. Include comprehensive docstring
4. Add example usage
5. Write unit tests

### Adding New Statistical Methods

1. Add function to `statistical_analysis.py`
2. Include mathematical documentation
3. Provide references to papers/methods
4. Add validation tests

---

## References

### Original Code

- `market_analytics.py` - Original notebook (5208 lines)
- `feature_analytics.ipynb` - Jupyter notebook version

### Core Refactored Modules

- `finance_ml/analytics/data_utils.py` - Data loading and preprocessing
- `finance_ml/analytics/statistical_analysis.py` - Bayesian, MCMC, Kalman, Copula
- `finance_ml/analytics/screening.py` - Stock screening functions
- `finance_ml/analytics/feature_analytics.py` - Interactive visualizations
- `finance_ml/analytics/optimized_ops.py` - Performance optimizations

### Visualization Modules (New in v2.0)

- `finance_ml/analytics/visualizations/profitability.py` - Margin analysis
- `finance_ml/analytics/visualizations/technical.py` - Technical analysis
- `finance_ml/analytics/visualizations/temporal_analysis.py` - Time series

### Visualization Modules (New in v2.1)

- `finance_ml/analytics/visualizations/valuation.py` - Valuation ratio analysis (669 lines)
- `finance_ml/analytics/visualizations/earnings_quality.py` - Earnings quality charts (658 lines)
- `finance_ml/analytics/visualizations/quality_risk.py` - Quality & risk assessment (852 lines)
- `finance_ml/analytics/visualizations/growth_analysis.py` - Growth metrics analysis (631 lines)

### Test Files

- `tests/test_screening.py` - Screening function tests
- `tests/test_data_utils.py` - Data utility tests
- `tests/test_statistical_analysis.py` - Statistical method tests
- `tests/test_market_analytics_integration.py` - Integration tests
- `tests/test_visualizations.py` - Visualization tests (31 tests)
- `tests/test_visualizations_valuation.py` - Valuation visualization tests (19 tests)
- `tests/test_visualizations_earnings_quality.py` - Earnings quality tests (17 tests)
- `tests/test_visualizations_quality_risk.py` - Quality & risk tests (17 tests)
- `tests/test_visualizations_growth_analysis.py` - Growth analysis tests (15 tests)
- `tests/test_enhanced_statistics.py` - Enhanced statistics tests

### Main Script

- `market_analytics.py` - Main demonstration script (1074 lines, updated with new visualizations)

### Jupyter Notebooks (Updated in v2.1)

- `feature_analytics.ipynb` - Feature analytics notebook (209 cells, updated with new visualizations)
- `financial_market_statistical_analysis.ipynb` - Statistical analysis notebook (900 lines, updated with new
  visualizations)

### Documentation

- `README.md` - Project overview
- `docs/code_guidelines.md` - Coding standards
- `docs/improvement_plan/market_analysis_refactoring_guide.md` - This document

---

## Integration Summary (v2.1)

The following scripts and notebooks have been updated to integrate the new visualization modules:

### Scripts Updated

| Script                                      | Changes                                                                                 |
|---------------------------------------------|-----------------------------------------------------------------------------------------|
| `finance_ml/analytics/feature_analytics.py` | Added imports for 21 new visualization functions; main() generates 18 additional charts |
| `market_analytics.py`                       | Added imports for 21 new visualization functions; generates 21 additional charts        |

### Notebooks Updated

| Notebook                                      | Changes                                                            |
|-----------------------------------------------|--------------------------------------------------------------------|
| `feature_analytics.ipynb`                     | Added 37 new import lines; 22 new visualization cells              |
| `financial_market_statistical_analysis.ipynb` | Added 37 new import lines; 22 new visualization cells in Section 6 |

### New Visualizations Available

#### Valuation Analysis (5 functions)

- `create_valuation_multiples_comparison()` - Spider/radar chart vs sector median
- `create_valuation_distribution_dashboard()` - Multi-panel violin plots
- `create_relative_valuation_matrix()` - Z-score heatmap by industry
- `create_valuation_vs_growth_quadrant()` - PEG-style scatter analysis
- `create_historical_valuation_percentile()` - Distribution with percentile markers

#### Earnings Quality (5 functions)

- `create_earnings_surprise_dashboard()` - Multi-panel surprise analysis
- `create_eps_trajectory_analysis()` - Trajectory score visualization
- `create_earnings_quality_decomposition()` - Waterfall decomposition
- `create_beat_rate_heatmap()` - Beat rates by sector
- `create_earnings_consistency_matrix()` - Streak vs improvement matrix

#### Quality & Risk (6 functions)

- `create_piotroski_fscore_breakdown()` - F-Score distribution
- `create_altman_zscore_distribution()` - Z-Score with distress zones
- `create_quality_risk_quadrant()` - F-Score vs Z-Score scatter
- `create_beneish_mscore_analysis()` - M-Score manipulation analysis
- `create_risk_tier_sunburst()` - Hierarchical risk visualization
- `create_distress_early_warning_dashboard()` - Early warning system

#### Growth Analysis (5 functions)

- `create_growth_waterfall_chart()` - Growth decomposition
- `create_growth_consistency_matrix()` - Consistency by sector
- `create_growth_vs_profitability_quadrant()` - BCG-style analysis
- `create_growth_acceleration_chart()` - Acceleration ranking
- `create_sustainable_growth_analysis()` - SGR analysis

---

## Contact & Support

For questions or issues with the refactored code:

1. Check this guide first
2. Review module docstrings
3. Examine example usage in `market_analytics.py`
4. Create an issue in the project repository

---

**Last Updated**: 2026-02-12
**Version**: 2.4.0
**Status**: Production Ready (Probabilistic Visualization Integration)

---

## Changelog (v2.4.0)

### Probabilistic Visualization Module (`probability_viz.py`)

New `finance_ml/analytics/visualizations/probability_viz.py` module providing 6 ArviZ-backed
probabilistic visualization functions with graceful DataFrame fallback:

| Function                                | ArviZ Path                                 | DataFrame Fallback            | Database Source                           |
|-----------------------------------------|--------------------------------------------|-------------------------------|-------------------------------------------|
| `create_posterior_return_forest`        | HDI + R-hat from `InferenceData.posterior` | CI from `upside_std`          | `analytics.monte_carlo_simulation`        |
| `create_beat_probability_posterior`     | Beta KDE from posterior samples            | `Beta(α, β)` PDF from columns | `analytics.earnings_probability_analysis` |
| `create_ruin_probability_diagnostic`    | 4-panel with ESS/R-hat                     | Risk tier pie + sector bars   | `analytics.credit_risk_analysis`          |
| `create_mcse_convergence_panel`         | Running MCSE per chain                     | *ArviZ required*              | Any `InferenceData`                       |
| `create_bayesian_category_ridge`        | — (works from `dict`)                      | `Normal(μ, σ)` KDE            | `bayesian_category_analysis()` output     |
| `create_tri_model_posterior_comparison` | —                                          | Normal approximation          | `analytics.expected_returns_tri_model`    |

### Integration into Main Scripts and Notebooks

| File                                                         | Changes                                                                                              |
|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `market_analytics.py`                                        | Added `probability_viz` imports; generates 4 probabilistic visualizations in Step 7; updated summary |
| `feature_analytics.py`                                       | Added `probability_viz` imports; generates probabilistic visualizations after growth analysis        |
| `ExpectedReturnsAnalytics.ipynb`                             | Added probability_viz import cell and posterior forest + tri-model visualization cells               |
| `financial_market_statistical_analysis.ipynb`                | Added probability_viz section (5.5) with category ridge and beat probability cells                   |
| `feature_analytics.ipynb`                                    | Added probability_viz section with posterior forest, beat probability, and ruin diagnostic cells     |
| `docs/improvement_plan/market_analysis_refactoring_guide.md` | Added section 12 for `probability_viz.py`; updated version to 2.4.0                                  |

---

## Changelog (v2.3.0)

### DRY Identifier Columns Refactoring (`feature_registry.sql`)

All 17 `vw_features_*` views and the `mv_all_stock_features` materialized view have been refactored
to inherit identifier columns from `vw_identifier_columns` via `id.*` instead of hardcoding 9 columns
(`isin`, `ticker`, `name`, `industry`, `sector`, `trading_country`, `region`, `country`, `exchange`).

| What changed                                   | Before                                                                        | After                                                                                      |
|------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Identifier columns in 17 `vw_features_*` views | Hardcoded 9 columns (`id.isin, id.ticker, ...`)                               | `id.*` — inherits all 31 columns from `vw_identifier_columns`                              |
| Identifier columns in `mv_all_stock_features`  | Hardcoded 9 identifier columns + separate `e.` selects for dates/categoricals | `id.*` — all identifier, categorical, and date columns from single source                  |
| Single source of truth                         | `vw_identifier_columns` defined but not fully utilized                        | `vw_identifier_columns` is the **sole** source for all identifier/categorical/date columns |
| Adding a new identifier column                 | Required editing 17 views + 1 MV                                              | Edit only `vw_identifier_columns` — all views inherit automatically                        |

### Materialized View Duplicate Column Removal

Duplicate date columns that overlap with `vw_identifier_columns` have been removed from the
`mv_all_stock_features` materialized view:

- `e."FY End Date"` → already provided as `fy_end_date` via `id.*`
- `e."Next FY End Date"` → already provided as `next_fy_end_date` via `id.*`
- `e."Next Earnings"` → already provided as `next_earnings` via `id.*`
- `e."Income Statement Report Date"` → already provided as `income_statement_report_date` via `id.*`
- `e."Next Income Statement Report Date"` → already provided as `next_income_statement_report_date` via `id.*`

Non-overlapping equities columns (`market_cap`, `enterprise_value`, `last_price`, price targets,
`volume_shrs`, `shares_outstanding`) are retained as explicit selects from `e.`.

### Python Analytics Alignment

- `probability_analytics.py`: Replaced hardcoded 5-column identifier list in
  `ViewProbabilityAnalyzer.analyze_view()` with `load_identifier_columns()` from `data_utils`.
- All other analytics modules (`data_utils.py`, `statistical_analysis.py`, `screening.py`,
  `feature_analytics.py`, `optimized_ops.py`, `market_analytics.py`) already use the dynamic
  `load_identifier_columns()` / `get_identifier_cols_set()` utilities and required no changes.

### Files Updated

| File                                                         | Changes                                                             |
|--------------------------------------------------------------|---------------------------------------------------------------------|
| `feature_registry.sql`                                       | 17 views + 1 MV refactored to use `id.*`; section header updated    |
| `finance_ml/analytics/probability_analytics.py`              | Replaced hardcoded identifier list with `load_identifier_columns()` |
| `docs/improvement_plan/market_analysis_refactoring_guide.md` | Updated version to 2.3.0; added this changelog                      |

---

## Changelog (v2.2.0)

### Column Name Alignment (MV Schema Sync)

All visualization modules and dependent scripts/notebooks have been updated to use the correct
materialized view (`mv_all_stock_features`) column names:

| Old Name                                                        | New Name                                                                        | Affected Modules                                        |
|-----------------------------------------------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------|
| `revenue_growth_yoy`                                            | `revenue_yoy_growth`                                                            | `growth_analysis.py`, `market_analytics.py`, notebooks  |
| `revenue_growth_3y_cagr`                                        | `revenue_cagr_3y`                                                               | `growth_analysis.py`                                    |
| `revenue_growth_5y_cagr`                                        | `revenue_cagr_5y`                                                               | `growth_analysis.py`                                    |
| `eps_growth_3y_cagr`                                            | `eps_cagr_3y`                                                                   | `growth_analysis.py`                                    |
| `net_income_growth`                                             | `net_income_growth_yoy`                                                         | `growth_analysis.py`                                    |
| `inventory_turnover_mv`                                         | `inventory_turnover_itf`                                                        | `temporal_analysis.py`, `category_charts.py`, notebooks |
| `beneish_m_score`                                               | `accounting_quality_score` (fallback)                                           | `quality_risk.py`                                       |
| `eps_beat_count` / `eps_total_reports`                          | `eps_positive_years` / `eps_positive_streak`                                    | `earnings_quality.py`                                   |
| `accruals_ratio`, `cash_earnings_ratio`, `earnings_persistence` | `earnings_quality_composite`, `ni_adjustment_ratio`, `accounting_quality_score` | `earnings_quality.py`                                   |

### Earnings Quality Decomposition Remap

`create_earnings_quality_decomposition()` now uses columns actually present in the MV:

- `earnings_quality_composite`, `ni_adjustment_ratio`, `eps_adjustment_ratio`,
  `accounting_quality_score`, `earnings_quality_impact`

### Beat Rate & Consistency Remap

`create_beat_rate_heatmap()` and `create_earnings_consistency_matrix()` now use:

- `eps_positive_years`, `eps_positive_streak`, `eps_improvement_count`, `eps_trajectory_score`

### Beneish M-Score Fallback

`create_beneish_mscore_analysis()` now falls back through:
`beneish_m_score` → `accounting_quality_score` → `accruals_quality`
with adaptive thresholds and labels per resolved column.

### Shared Utilities (`_shared.py`)

New shared module `finance_ml/analytics/visualizations/_shared.py` provides:

- `PLOTLY_TEMPLATE`, `COLORS` — centralized constants
- `MV_COLUMN_ALIASES` — canonical alias map for MV column resolution
- `resolve_column(df, logical_name)` — resolve logical column names to actual DataFrame columns
- `create_no_data_figure(title)` — DRY replacement for per-module `_create_no_data_figure()`

### Data Guard Clauses (`category_charts.py`)

All 23+ functions in `category_charts.py` now include column-existence checks
and return graceful "No Data" placeholder figures instead of raising `ValueError`.

### Files Updated

| File                                                       | Changes                                                      |
|------------------------------------------------------------|--------------------------------------------------------------|
| `finance_ml/analytics/visualizations/earnings_quality.py`  | Remapped decomposition, beat rate, consistency to MV columns |
| `finance_ml/analytics/visualizations/growth_analysis.py`   | Fixed all growth metric column names                         |
| `finance_ml/analytics/visualizations/quality_risk.py`      | Added M-Score fallback chain                                 |
| `finance_ml/analytics/visualizations/temporal_analysis.py` | Fixed inventory turnover column                              |
| `finance_ml/analytics/visualizations/category_charts.py`   | Added data guard clauses to all functions                    |
| `finance_ml/analytics/visualizations/profitability.py`     | Added `total_asset_turnover` fallback for DuPont             |
| `finance_ml/analytics/visualizations/_shared.py`           | New shared utilities module                                  |
| `finance_ml/analytics/visualizations/__init__.py`          | Exports shared utilities                                     |
| `market_analytics.py`                                      | Updated metric references                                    |
| `feature_analytics.ipynb`                                  | Updated metric references                                    |
| `financial_market_statistical_analysis.ipynb`              | Updated metric references                                    |
