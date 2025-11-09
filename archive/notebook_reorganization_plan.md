# Notebook Reorganization Plan

## Current Issues

- Phase 9.2 appears before Phase 9.1 (wrong order)
- Phase 9.3, 9.6, 9.7 are missing
- Workflow doesn't follow IMPROVEMENT_PLAN.md sequence

## Target Structure

### Section 1: Setup (Lines 1-325)

- Imports with all Phase 9 modules ✓
- Configuration
- Data Loading
- Data Validation

### Section 2: Phase 9.1 - Preprocessing (Lines 1210-1462)

- Data quality assessment
- Outlier detection (IQR, Z-score, Isolation Forest)
- Winsorization
- Missing value imputation
- ✓ Already implemented

### Section 3: Phase 9.2 - EDA (Currently at 326-556, move after 9.1)

- Enhanced simple_eda with statistical analysis
- Correlation matrices
- Normality tests
- Sector comparisons
- ✓ Already implemented, needs repositioning

### Section 4: Phase 9.3 - Feature Engineering (NEW - ADD)

- Valuation ratios (P/E, P/B, EV/EBITDA)
- Profitability ratios (ROE, ROA, margins)
- Leverage ratios
- Efficiency ratios
- Growth metrics
- Sector-specific features
- Feature interactions
- Relative value features

### Section 5: Phase 9.4 - Classification (Lines 1473+)

- Event label creation
- Multi-classifier comparison
- Export classification probabilities
- ✓ Already implemented

### Section 6: Phase 9.5 - Regression (Lines ~1800+)

- Multiple regression models
- Sector-specific models
- Quantile regression
- Stacking ensemble
- ✓ Already implemented

### Section 7: Phase 9.6 - Evaluation (NEW - ADD)

- Comprehensive regression metrics
- Metrics by segment (sector, region, market cap)
- Residual analysis
- Error bucketing
- Cross-validation strategies

### Section 8: Phase 9.7 - Valuation (NEW - ADD)

- Valuation categories (Strong Buy/Buy/Hold/Sell/Strong Sell)
- Sector z-scores
- Multi-factor scoring
- Stock filtering
- Interactive visualizations
- Rankings (undervalued/overvalued)
- Excel export

## Action Steps

1. Delete misplaced Phase 9.2 (lines 326-556)
2. Add Phase 9.3 after Phase 9.1
3. Re-insert Phase 9.2 after Phase 9.3
4. Add Phase 9.6 after Phase 9.5
5. Add Phase 9.7 at end
