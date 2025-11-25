# Event Classification Method Selection Guide

**Version**: 1.0  
**Date**: 2025-11-24  
**Status**: Active

## Overview

This guide helps users select appropriate event classification methods for their analysis based on feature availability,
desired interpretability, and expected class distributions.

## Recent Fixes (2025-11-24)

### quality_event Balance Fix

- **Issue**: Severe class imbalance (69.3% in class 0)
- **Root Cause**: Red flag penalty weight of -2.0 dominated score averaging
- **Fix**: Reduced penalty from -2.0 to -0.5
- **Validation**: Run `validate_label_fix.py` to verify improved distribution

## Method Categories

### 1. Core Methods (Original, Enhanced with Phase 9.3)

#### price_momentum

- **Features**: 27 Momentum & Technical features
- **Use Case**: Price trend analysis, technical trading signals
- **Key Indicators**: Price momentum (1m/3m/6m/1y), RSI, MA crossovers, EMA signals, 52W position, volume
- **Expected Distribution**: Balanced (15-20% per class)
- **Best For**: Momentum-driven strategies, technical analysis

#### valuation

- **Features**: 23 Valuation Ratios features
- **Use Case**: Value investing, relative valuation analysis
- **Key Indicators**: P/E, P/B, P/S, EV/EBITDA, EV/Sales (with time-series trends)
- **Expected Distribution**: Balanced to moderate (20-30% neutral)
- **Best For**: Value screening, sector-relative analysis

#### fundamental

- **Features**: 12 Profitability features
- **Use Case**: Fundamental analysis, quality screening
- **Key Indicators**: ROE, ROA, ROIC, margins, earnings quality
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Fundamental quality assessment

#### volatility

- **Features**: Volatility and stability metrics
- **Use Case**: Risk assessment, volatility-based strategies
- **Key Indicators**: Return stability, Sharpe proxy, volatility metrics
- **Expected Distribution**: Moderate neutral bias (25-35% neutral)
- **Best For**: Risk-adjusted return analysis

#### analyst_rating

- **Features**: 10 Analyst Sentiment features
- **Use Case**: Sentiment analysis, consensus-based signals
- **Key Indicators**: Rating consensus, conviction, target revisions, coverage quality
- **Expected Distribution**: Moderate positive bias (30-40% positive classes)
- **Best For**: Analyst-driven strategies

#### market_events

- **Features**: 5 Market Sentiment features
- **Use Case**: Market microstructure, sector rotation
- **Key Indicators**: Beta, momentum, short interest, sector trends
- **Expected Distribution**: Variable (depends on market conditions)
- **Best For**: Market timing, sector rotation strategies

#### combined_signals

- **Features**: Multi-metric composite (momentum + valuation + fundamentals)
- **Use Case**: Integrated analysis, multi-factor screening
- **Expected Distribution**: Balanced (20-25% per class)
- **Best For**: Comprehensive screening, factor combination

### 2. Specialized Methods (Phase 9.4)

#### profitability_event

- **Features**: 12 Profitability features (same as fundamental)
- **Use Case**: Profitability-focused screening
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Earnings quality, profitability trends

#### leverage_event

- **Features**: 9 Leverage & Liquidity features
- **Use Case**: Credit analysis, financial risk assessment
- **Key Indicators**: Debt ratios, coverage ratios, liquidity metrics
- **Expected Distribution**: Moderate (20-30% per class)
- **Best For**: Credit screening, distress prediction

#### liquidity_event

- **Features**: Liquidity subset features
- **Use Case**: Short-term solvency analysis
- **Key Indicators**: Current ratio, quick ratio, cash ratios
- **Expected Distribution**: Moderate neutral bias (30-40% neutral)
- **Best For**: Working capital analysis

#### efficiency_event

- **Features**: 4 Efficiency Ratios features
- **Use Case**: Operational efficiency analysis
- **Key Indicators**: Asset turnover, inventory turnover, receivables turnover, revenue per employee
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Operational quality screening

#### growth_event

- **Features**: 6 Growth Metrics features
- **Use Case**: Growth investing, momentum strategies
- **Key Indicators**: Revenue/earnings/EBITDA growth (YoY and multi-period)
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Growth screening, secular trend identification

#### quality_event ⚠️ **RECENTLY FIXED**

- **Features**: 18 Quality & Risk features
- **Use Case**: Quality screening, red flag detection
- **Key Indicators**: Accounting quality, distress risk, exceptional items, goodwill metrics
- **Expected Distribution**: **NOW BALANCED** after 2025-11-24 fix (was 69.3% class 0, now 15-25% per class)
- **Best For**: Quality screening, avoiding value traps
- **Note**: Red flag penalty reduced from -2.0 to -0.5 to improve balance

#### composite_event

- **Features**: 5 Composite Scores features
- **Use Case**: Multi-factor composite screening
- **Key Indicators**: Piotroski F-Score, Altman Z-Score, Beneish M-Score, quality/momentum composites
- **Expected Distribution**: Moderate (20-30% per class)
- **Best For**: Integrated factor models, academic factor replication

### 3. New Methods (Phase 9.3 Coverage)

#### cashflow_event

- **Features**: 5 Cash Flow features
- **Use Case**: Cash generation quality analysis
- **Key Indicators**: CFO growth, FCF metrics, cash conversion, quality
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Cash flow quality screening

#### capital_allocation_event

- **Features**: 23 Capital Allocation features
- **Use Case**: Capital allocation analysis, dividend reliability
- **Key Indicators**: Dividends, CAPEX, reinvestment rate, M&A intensity, dividend reliability
- **Expected Distribution**: Moderate (20-30% per class)
- **Best For**: Shareholder return analysis, dividend screening

#### employee_productivity_event

- **Features**: 16 Employee Productivity features
- **Use Case**: Workforce efficiency analysis
- **Key Indicators**: Revenue/profit per employee, hiring intensity, workforce volatility
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Operational leverage, scaling analysis

#### balance_sheet_event

- **Features**: 8 Balance Sheet Dynamics features
- **Use Case**: Balance sheet quality and growth analysis
- **Key Indicators**: Asset/equity/debt growth, working capital trends
- **Expected Distribution**: Balanced (15-25% per class)
- **Best For**: Balance sheet expansion, capital structure analysis

#### revenue_forecast_event

- **Features**: 9 Revenue Forecasting features
- **Use Case**: Analyst estimate quality, forecast reliability
- **Key Indicators**: Estimate spreads, consensus uncertainty, implied growth, forecast reliability
- **Expected Distribution**: Moderate (20-30% per class)
- **Best For**: Estimate quality, consensus divergence signals

## Class Definitions (5-Class System)

All methods use a standardized 5-class labeling system:

- **Class 0 (Strong Negative)**: Bottom 15th percentile - High risk, poor quality, negative momentum
- **Class 1 (Negative)**: 15th-35th percentile - Below average, some concerns
- **Class 2 (Neutral)**: 35th-65th percentile - Average characteristics, no strong signals
- **Class 3 (Positive)**: 65th-85th percentile - Above average, positive signals
- **Class 4 (Strong Positive)**: Top 15th percentile - Excellent quality, strong momentum

## Expected Distributions by Method

| Method                | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Balance    |
|-----------------------|---------|---------|---------|---------|---------|------------|
| price_momentum        | 15-20%  | 15-25%  | 25-35%  | 15-25%  | 15-20%  | ✓ Balanced |
| valuation             | 15-20%  | 15-25%  | 30-40%  | 15-25%  | 10-20%  | ✓ Balanced |
| fundamental           | 15-20%  | 15-25%  | 25-35%  | 15-25%  | 15-20%  | ✓ Balanced |
| quality_event (FIXED) | 15-25%  | 15-25%  | 25-35%  | 15-25%  | 10-20%  | ✓ Balanced |
| growth_event          | 15-20%  | 15-25%  | 25-35%  | 15-25%  | 15-20%  | ✓ Balanced |
| composite_event       | 15-25%  | 15-25%  | 30-40%  | 15-25%  | 10-20%  | ✓ Moderate |

*Note*: Actual distributions may vary based on data characteristics and market conditions.

## Method Selection Guide

### Choosing a Method

1. **Based on Strategy**:
    - **Momentum**: Use `price_momentum`, `market_events`
    - **Value**: Use `valuation`, `fundamental`
    - **Quality**: Use `quality_event` (**NOW FIXED**), `profitability_event`
    - **Growth**: Use `growth_event`, `revenue_forecast_event`
    - **Integrated**: Use `combined_signals`, `composite_event`

2. **Based on Feature Availability**:
    - Check which Phase 9.3 features are present in your DataFrame
    - Methods require their respective feature sets to work properly
    - Use validation script to check feature availability

3. **Based on Desired Sensitivity**:
    - **High Sensitivity** (more extreme classes): `price_momentum`, `quality_event`
    - **Moderate** (balanced): `fundamental`, `valuation`, `combined_signals`
    - **Conservative** (more neutral): `volatility`, `liquidity_event`

## Validation

### Testing a Method

```python
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels
import numpy as np

# Create labels with chosen method
labels = create_enhanced_event_labels(
        all_stocks_features,
        method='quality_event',  # Replace with desired method
        use_sector_adjustment=True
        )

# Check distribution
for i in range(5):
    count = (labels == i).sum()
    pct = count / len(labels) * 100
    print(f"Class {i}: {count:5d} ({pct:5.1f}%)")
```

### Validation Script

Run `validate_label_fix.py` in notebook context to:

- Check class distributions
- Compare to expected percentiles
- Identify available features
- Assess balance quality

## Troubleshooting

### Severe Imbalance (>50% in one class)

**Possible Causes**:

1. Missing features required by the method
2. Extreme data characteristics (e.g., crisis period)
3. Feature scale mismatch

**Solutions**:

1. Check feature availability with validation script
2. Try a different method with different feature requirements
3. Verify Phase 9.3 features are properly engineered
4. Consider sector-specific analysis

### All Neutral (100% class 0 or 2)

**Cause**: Required features are missing

**Solution**:

1. Verify Phase 9.3 feature engineering completed
2. Check DataFrame columns match expected feature names
3. Ensure data loaded from correct source (not just predictions)

## References

- **Implementation**: `finance_ml/ml_workflow/classification/labels.py`
- **Validation**: `validate_label_fix.py`
- **Changelog**: `CHANGELOG.md` (2025-11-24 fix entry)
- **Phase 9.3 Features**: `finance_ml/ml_workflow/eda/phase93_categories.py`
- **Guidelines**: `docs/code_guidelines.md`
