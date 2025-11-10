# Phase 9.3 Feature Enhancement Plan

**Date:** 2025-11-10  
**Status:** DRAFT  
**Version:** 1.0  
**Model Version Target:** v9_9

---

## Executive Summary

This document proposes enhancements to the feature engineering pipeline by identifying underutilized columns in the
`equities` table and suggesting new calculated features. The plan follows Test-Driven Development (TDD) principles and
integrates with the existing `finance_ml.ml_workflow.features` module structure (core.py and advanced.py).

**Current State:**

- `core.py`: Basic ratios (P/E, P/B, debt/equity, ROE, ROA), margins, volatility, revenue CAGR
- `advanced.py`: 14 feature engineering functions covering valuation, profitability, leverage, liquidity, efficiency,
  growth, sector-specific, temporal, microstructure, analyst quality, accounting quality, and employee productivity

**Gap Analysis:**
The equities table contains 246 columns with rich temporal variations (FQ, FY, LTM, -1FY, 5YAVG) and quality indicators
that are currently underutilized for ML features.

---

## Feature Categories & Proposed Enhancements

### 1. Momentum & Technical Features

**Database columns to leverage:**

- `Price Chg. % (1M)`, `Price Chg. % (3M)`, `1-Day %`
- `Price (5D Ago)`, `Price (1W Ago)`, `Price (1M Ago)`, `Price (3M Ago)`, `Price (6M Ago)`, `Price (1Y Ago)`,
  `Price (3Y Ago)`, `Price (5Y Ago)`, `Price (QTD Ago)`
- `Total Return (YTD)`, `Total Return (5Y)`, `Total Return (10Y)`
- `Tot. Return %/CAGR (3Y)`, `Tot. Return %/CAGR (10Y)`

**Proposed features:**

1. **Price Momentum Indicators**
    - `price_momentum_1m`: 1-month price change %
    - `price_momentum_3m`: 3-month price change %
    - `price_momentum_6m`: 6-month price change (calculated)
    - `price_momentum_1y`: 1-year price change (calculated)
    - `price_acceleration_3m`: Rate of change in momentum (3M vs 1M)

2. **Relative Strength Index (RSI)**
    - `rsi_14d`: 14-day RSI using available price history
    - `rsi_30d`: 30-day RSI

3. **Moving Average Features**
    - `ma_crossover_signal`: Last_Price vs calculated moving averages
    - `price_distance_from_ma`: % distance from moving average

4. **Return Consistency**
    - `return_stability_score`: Ratio of total return to volatility
    - `sharpe_proxy`: (Total Return - Risk Free Rate) / Volatility

**Implementation approach:**

- Add `engineer_momentum_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_momentum_features.py`

---

### 2. Quality & Risk Signals

**Database columns to leverage:**

- `Altman Z-Score (FY)`, `Altman Z-Score (FQ)`, `Altman Z-Score (LTM)`
- `Impairment of Goodwill (FQ)`, `Impairment of Goodwill (LTM)`, `Impairment of Goodwill (-1FY)`,
  `Impairment of Goodwill (FY)`, `Impairment of Goodwill (5YAVGFQ)`
- `Asset Writedown (LTM)`, `Asset Writedown (FY)`, `Asset Writedown (-1FY)`, `Asset Writedown (FQ)`,
  `Asset Writedown (5YAVGFQ)`
- `Restructuring Charges (LTM)`, `Restructuring Charges (FQ)`, `Restructuring Charges (-1FY)`,
  `Restructuring Charges (FY)`, `Restructuring Charges (5YAVGFQ)`
- `Merger & Restructuring Charges (LTM)`, `Merger & Restructuring Charges (FQ)`, `Merger & Restructuring Charges (FY)`,
  `Merger & Restructuring Charges (5YAVGFQ)`
- `Other Unusual Items/Total (LTM)`

**Proposed features:**

1. **Financial Distress Indicators**
    - `altman_z_trend`: Change in Altman Z-Score (FY vs -1FY or FQ trend)
    - `distress_risk_score`: Composite score from Z-scores across periods
    - `z_score_volatility`: Variation across FQ, FY, LTM

2. **Accounting Quality Composite**
    - `total_exceptional_items_ltm`: Sum of impairments + writedowns + restructuring
    - `exceptional_items_to_ebitda`: Exceptional items / EBITDA (persistence indicator)
    - `exceptional_items_trend`: YoY change in exceptional items
    - `goodwill_impairment_flag`: Binary indicator of recent impairment
    - `restructuring_intensity`: Restructuring charges / Total Assets

3. **Asset Quality Metrics**
    - `goodwill_to_assets`: Goodwill / Total Assets (intangible concentration)
    - `goodwill_change_rate`: (Goodwill_LTM - Goodwill_-1FY) / Goodwill_-1FY
    - `intangible_intensity`: Gross Intangible Assets / Total Assets

**Implementation approach:**

- Extend `engineer_accounting_quality_features()` in advanced.py
- Add `engineer_financial_distress_features()` to advanced.py
- Test module: `tests/test_quality_risk_features.py`

---

### 3. Cash Flow & Capital Allocation Features

**Database columns to leverage:**

- `CFO (LTM)`, `CFO (FY)`, `CFO (-1FY)`, `CFO (FQ)`, `CFO (5YAVGFQ)`
- `CFI (LTM)`, `CFI (FY)`, `CFI (-1FY)`, `CFI (FQ)`, `CFI (5YAVGFQ)`
- `CFF (LTM)`, `CFF (FY)`, `CFF (-1FY)`, `CFF (FQ)`, `CFF (5YAVGFQ)`
- `FCF (LTM)`, `FCF (FY)`, `FCF (FQ)`, `FCF (5YAVGFQ)`, `FCF (5YAVGFQ)`
- `Capital Expenditure (LTM)`, `Capital Expenditure (-1FY)`, `Capital Expenditure (FY)`, `Capital Expenditure (FQ)`,
  `Capital Expenditure (5YAVGFQ)`
- `Cash Acquisitions (LTM)`, `Cash Acquisitions (FY)`, `Cash Acquisitions (-1FY)`, `Cash Acquisitions (FQ)`,
  `Cash Acquisitions (5YAVGFQ)`
- `Buyback Yield (LTM)`
- `Dividend Per Share (LTM)`, `Div Yield (Ind)`, `Div Yield (LTM)`, `Div Yield (TTM)`, `Div Yield (NTM)`,
  `Div Yield (5YAVGLTM)`, `Div Yield (-1FYInd)`

**Proposed features:**

1. **Cash Flow Quality**
    - `cfo_to_net_income`: CFO / Net Income (accruals quality)
    - `fcf_to_net_income`: FCF / Net Income
    - `fcf_margin`: FCF / Revenue
    - `cfo_growth_yoy`: (CFO_LTM - CFO_-1FY) / CFO_-1FY
    - `fcf_stability`: Std deviation of FCF across available periods

2. **Capital Intensity & Efficiency**
    - `capex_intensity`: CapEx / Revenue
    - `capex_to_depreciation`: CapEx / Depreciation (growth indicator)
    - `capex_growth_rate`: (CapEx_LTM - CapEx_-1FY) / CapEx_-1FY
    - `capex_volatility`: Coefficient of variation across 5Y

3. **Capital Allocation Score**
    - `total_shareholder_return_yield`: Dividend Yield + Buyback Yield
    - `payout_ratio`: (Dividend + Buyback) / Net Income
    - `reinvestment_rate`: (CapEx + Acquisitions) / CFO
    - `acquisition_intensity`: Cash Acquisitions / Total Assets

4. **Cash Conversion Metrics**
    - `cash_conversion_cycle`: Days (if inventory turnover available)
    - `working_capital_efficiency`: Revenue / Working Capital
    - `working_capital_trend`: (WC_LTM - WC_-1FY) / Revenue

**Implementation approach:**

- Add `engineer_cash_flow_quality_features()` to advanced.py
- Add `engineer_capital_allocation_features()` to advanced.py
- Test module: `tests/test_cashflow_capital_features.py`

---

### 4. Market Sentiment & Analyst Features

**Database columns to leverage:**

- `Analyst Rating`
- `# Strong Sell Ratings`, `# Strong Buys Ratings`, `# Hold Ratings`, `# Buys Ratings`, `# Sell Ratings`
- `Price Target`, `Price Target - Low`, `Price Target - Median`, `Price Target - High`, `Price Target - #`
- `Price Target (YTD Ago)`
- `Short Int. (%)`
- `Beta (1Y)`, `Beta (2Y)`, `Beta (5Y)`

**Proposed features:**

1. **Analyst Consensus Metrics**
    - `analyst_bullish_pct`: (Strong Buy + Buy) / Total Ratings
    - `analyst_bearish_pct`: (Strong Sell + Sell) / Total Ratings
    - `analyst_dispersion`: Std dev of ratings (buy=2, hold=1, sell=0)
    - `analyst_conviction`: Absolute difference between bulls and bears

2. **Price Target Features**
    - `upside_potential`: (Price Target Median - Last Price) / Last Price
    - `price_target_range`: (High - Low) / Median (uncertainty)
    - `price_target_revision`: (Current Target - YTD Ago Target) / YTD Ago
    - `analyst_coverage_quality`: # Analysts / Market Cap (log scaled)

3. **Market Sentiment**
    - `short_interest_ratio`: Short Interest % as signal
    - `beta_stability`: Variance across 1Y, 2Y, 5Y betas
    - `systematic_risk_trend`: Beta_1Y - Beta_5Y (changing risk profile)

**Implementation approach:**

- Extend `engineer_analyst_quality_features()` in advanced.py
- Add `engineer_market_sentiment_features()` to advanced.py
- Test module: `tests/test_sentiment_analyst_features.py`

---

### 5. Profitability Trends & Margins

**Database columns to leverage:**

- `EBITDA (FQ)`, `EBITDA (LTM)`, `EBITDA (FY)`, `EBITDA (-1FY)`, `EBITDA (5YAVGFQ)`, `EBITDA (5YAVGLTM)`
- `EBITDA/Adj. (LTM)`, `EBITDA/Adj. (FY)`, `EBITDA/Adj. (-1FY)`
- `EBIT (FQ)`, `EBIT (LTM)`, `EBIT (FY)`, `EBIT (-1FY)`, `EBIT (5YAVGFQ)`, `EBIT (5YAVGLTM)`
- `EBIT/Adj. (-1FY)`, `EBIT/Adj. (FY)`, `EBIT/Adj. (LTM)`
- `EBIT - Est Med (FY1E)`, `EBIT - Est Med (NTM)`
- `Net Income Margin % (FY)`, `Net Income Margin % (LTM)`
- `Gross Profit Margin % (FY)`, `Gross Profit Margin % (LTM)`
- `Operating Income (LTM)`, `Operating Income (FY)`, `Operating Income (FQ)`, `Operating Income (5YAVGFQ)`

**Proposed features:**

1. **Margin Evolution**
    - `ebitda_margin_trend`: (EBITDA_Margin_LTM - EBITDA_Margin_-1FY)
    - `gross_margin_trend`: (Gross_Margin_LTM - Gross_Margin_FY)
    - `operating_leverage`: % change in EBIT / % change in Revenue
    - `margin_stability_5y`: Std dev of margins across 5Y averages

2. **Profitability Quality**
    - `ebitda_adjustment_ratio`: EBITDA_Adj / EBITDA (adjustment intensity)
    - `ebit_adjustment_ratio`: EBIT_Adj / EBIT
    - `earnings_quality_score`: Composite of adjustments and cash conversion

3. **Forward-Looking Profitability**
    - `ebit_estimate_vs_ltm`: (EBIT_Est_NTM - EBIT_LTM) / EBIT_LTM
    - `ebit_estimate_surprise_potential`: (EBIT_Est - EBIT_LTM) / EBIT_volatility

**Implementation approach:**

- Extend `engineer_profitability_ratios()` in advanced.py
- Add `engineer_margin_trends()` to advanced.py
- Test module: `tests/test_profitability_trends.py`

---

### 6. Balance Sheet Strength & Temporal Patterns

**Database columns to leverage:**

- `Total Debt (FY)`, `Total Debt (LTM)`
- `Total Equity (FY)`, `Total Equity (LTM)`
- `Total Assets (LTM)`, `Total Assets (FY)`
- `Current Ratio (FY)`, `Current Ratio (LTM)`
- `Working Capital (LTM)`, `Working Capital (FQ)`, `Working Capital (FY)`, `Working Capital (5YAVGFY)`
- `Cash And Equivalents (LTM)`, `Cash And Equivalents (FQ)`, `Cash And Equivalents (FY)`,
  `Cash And Equivalents (5YAVGFQ)`
- `Inventory (LTM)`, `Inventory (FQ)`, `Inventory (FY)`, `Inventory (5YAVGFQ)`
- `Retained Earnings (LTM)`, `Retained Earnings (FQ)`, `Retained Earnings (FY)`, `Retained Earnings (5YAVGFQ)`

**Proposed features:**

1. **Balance Sheet Trends**
    - `debt_growth_rate`: (Total_Debt_LTM - Total_Debt_FY) / Total_Debt_FY
    - `equity_growth_rate`: (Total_Equity_LTM - Total_Equity_FY) / Total_Equity_FY
    - `asset_growth_rate`: (Total_Assets_LTM - Total_Assets_FY) / Total_Assets_FY
    - `balance_sheet_expansion`: Composite of debt/equity/asset growth

2. **Liquidity Trends**
    - `current_ratio_trend`: Current_Ratio_LTM - Current_Ratio_FY
    - `cash_ratio`: Cash / Current Liabilities
    - `cash_burn_rate`: (Cash_FY - Cash_LTM) / Quarters (if declining)
    - `working_capital_ratio`: Working Capital / Total Assets

3. **Retained Earnings Patterns**
    - `retained_earnings_growth`: (RE_LTM - RE_FY) / Total Equity
    - `earnings_retention_rate`: (RE_change) / Net_Income

**Implementation approach:**

- Extend `engineer_leverage_ratios()` and `engineer_liquidity_ratios()` in advanced.py
- Add `engineer_balance_sheet_trends()` to advanced.py
- Test module: `tests/test_balance_sheet_trends.py`

---

### 7. Time-Series & Seasonality Features

**Database columns to leverage:**

- All 5YAVG columns for baseline comparison
- FQ vs FY vs LTM variations
- `Next Earnings`, `Income Statement Report Date`, `Last Updated`

**Proposed features:**

1. **Quarterly Patterns**
    - `ltm_vs_5yavg_revenue`: (Revenue_LTM - Revenue_5YAVG) / Revenue_5YAVG
    - `fq_vs_5yavg_ebitda`: (EBITDA_FQ - EBITDA_5YAVGFQ) / EBITDA_5YAVGFQ
    - `quarterly_volatility_score`: Coefficient of variation across quarters

2. **Earnings Date Features**
    - `days_to_earnings`: Days until Next Earnings
    - `earnings_report_recency`: Days since Last Updated
    - `reporting_lag`: Days between Income Statement Date and Last Updated

**Implementation approach:**

- Extend `engineer_temporal_features()` in advanced.py
- Test module: `tests/test_temporal_seasonality.py`

---

### 8. Composite & Interaction Features

**Proposed features:**

1. **Quality Score Composite**
    - Combine: Altman Z, Cash Flow Quality, Margin Stability, Low Exceptional Items
    - Weighted score: 0-100 scale

2. **Growth-Quality Score**
    - Combine: Revenue Growth + FCF Growth + Margin Expansion - Exceptional Items

3. **Value Score**
    - Combine: P/E, P/B, EV/EBITDA percentiles (sector-relative)

4. **Momentum Score**
    - Combine: Price momentum + Earnings momentum + Analyst revisions

5. **Sector-Relative Interactions**
    - For each key metric: `metric_vs_sector_median`, `metric_vs_sector_top_quartile`

**Implementation approach:**

- Add `engineer_composite_scores()` to advanced.py
- Add `engineer_sector_relative_interactions()` to advanced.py
- Test module: `tests/test_composite_interactions.py`

---

## Test-Driven Development (TDD) Implementation Plan

### Phase 1: Test Infrastructure (Week 1)

**1.1 Create Test Fixtures**

- File: `tests/fixtures/feature_engineering_samples.py`
- Comprehensive sample DataFrames with all relevant columns
- Edge cases: missing values, zeros, negative values, outliers

**1.2 Create Test Utilities**

- File: `tests/utils/feature_test_helpers.py`
- Validation functions: check feature ranges, NaN handling, inf handling
- Assertion helpers for feature engineering tests

### Phase 2: Momentum & Technical Features (Week 2)

**2.1 Write Tests First**

- `tests/test_momentum_features.py`
- Test cases:
    - Basic momentum calculations
    - RSI edge cases (all up, all down, flat)
    - Moving average crossover signals
    - Return stability with varying volatility

**2.2 Implement Features**

- `finance_ml/ml_workflow/features/advanced.py`
- Add `engineer_momentum_features()`
- Ensure all tests pass

**2.3 Integration Test**

- Update `test_advanced_features.py` to include momentum features
- Validate with real data samples

### Phase 3: Quality & Risk Signals (Week 3)

**3.1 Write Tests First**

- `tests/test_quality_risk_features.py`
- Test Altman Z-Score trends
- Test exceptional items aggregation
- Test accounting quality composite scoring

**3.2 Implement Features**

- Extend `engineer_accounting_quality_features()`
- Add `engineer_financial_distress_features()`
- All tests pass

**3.3 Integration Test**

- Cross-sector validation
- Edge case validation (financial sector exceptions)

### Phase 4: Cash Flow Features (Week 4)

**4.1 Write Tests First**

- `tests/test_cashflow_capital_features.py`
- Test cash flow quality ratios
- Test capital allocation metrics
- Test working capital efficiency

**4.2 Implement Features**

- Add `engineer_cash_flow_quality_features()`
- Add `engineer_capital_allocation_features()`
- All tests pass

### Phase 5: Market Sentiment (Week 5)

**5.1 Write Tests First**

- `tests/test_sentiment_analyst_features.py`
- Test analyst consensus calculations
- Test price target features
- Test short interest integration

**5.2 Implement Features**

- Extend `engineer_analyst_quality_features()`
- Add `engineer_market_sentiment_features()`

### Phase 6: Profitability Trends (Week 6)

**6.1 Write Tests First**

- `tests/test_profitability_trends.py`
- Test margin evolution
- Test operating leverage calculation
- Test earnings quality scoring

**6.2 Implement Features**

- Extend `engineer_profitability_ratios()`
- Add `engineer_margin_trends()`

### Phase 7: Balance Sheet Features (Week 7)

**7.1 Write Tests First**

- `tests/test_balance_sheet_trends.py`
- Test debt/equity/asset growth rates
- Test liquidity trends
- Test retained earnings patterns

**7.2 Implement Features**

- Add `engineer_balance_sheet_trends()`

### Phase 8: Temporal & Composite Features (Week 8)

**8.1 Write Tests First**

- `tests/test_temporal_seasonality.py`
- `tests/test_composite_interactions.py`

**8.2 Implement Features**

- Extend `engineer_temporal_features()`
- Add `engineer_composite_scores()`
- Add `engineer_sector_relative_interactions()`

### Phase 9: Integration & Documentation (Week 9)

**9.1 Update `build_comprehensive_features()`**

- Integrate all new feature functions
- Add preset options: "momentum", "quality", "full_enhanced"

**9.2 Update API**

- `features/api.py`: Add new presets
- Ensure backward compatibility

**9.3 Documentation**

- Update docstrings with new features
- Update README.md with feature catalog
- Update code_guidelines.md

**9.4 Performance Testing**

- Benchmark feature engineering pipeline
- Optimize slow operations

### Phase 10: Validation (Week 10)

**10.1 Cross-validation**

- Run full test suite: `python -m unittest discover -s tests -p "test_*.py" -v`
- Ensure 100% pass rate

**10.2 Integration with Notebook**

- Update `ml_finance_model_main.ipynb`
- Add feature engineering demonstration cells

**10.3 Production Readiness**

- Update `MODEL_VERSION` to v9_9
- Tag release in version control

---

## Testing Standards & Requirements

### Test Coverage Requirements

- Minimum 85% code coverage for new functions
- 100% coverage for core feature calculation logic
- Edge case coverage: NaN, inf, zero, negative values

### Test Categories

**Unit Tests:**

- Individual feature calculation functions
- Input validation
- Output type checking
- NaN/inf handling

**Integration Tests:**

- `build_comprehensive_features()` with all presets
- Cross-function dependencies
- Large DataFrame performance

**Validation Tests:**

- Real data samples (1000+ rows)
- Sector-specific edge cases
- Column name variations (normalized vs raw)

### Test Naming Convention

```python
def test_<feature_group>_<specific_feature>_<scenario>():
    """Test <specific_feature> under <scenario> conditions."""
    pass
```

Example:

```python
def test_momentum_rsi_14d_bullish_trend():
    """Test RSI calculation with bullish trend data."""
    pass
```

---

## Column Naming Standards

### Normalized Column Names (Expected in DataFrames)

Follow existing conventions from `docs/code_guidelines.md`:

- Lowercase with underscores: `last_price`, `price_target`, `sector`, `region`
- Temporal suffixes: `_ltm`, `_fy`, `_fq`, `_1fy` (for -1FY)
- Calculated features: descriptive names like `ebitda_margin_trend`, `upside_potential`

### Database Column Mapping

- Use `normalize_columns()` from `data.py` early in pipeline
- Map: `"Last Price"` → `last_price`, `"EBITDA (LTM)"` → `ebitda_ltm`

---

## Risk Mitigation

### Technical Risks

**Risk 1: Performance Degradation**

- Mitigation: Vectorized operations only, avoid loops
- Benchmark: Feature engineering should complete in <30s for 10K rows

**Risk 2: NaN Propagation**

- Mitigation: Use `_safe_div()` helper, explicit NaN handling in all functions
- Validation: Check NaN counts before/after each feature group

**Risk 3: Memory Issues with Large Datasets**

- Mitigation: Process features in groups, clear intermediate results
- Validation: Test with 100K+ rows

### Business Risks

**Risk 4: Overfitting from Too Many Features**

- Mitigation: Implement feature selection in Phase 9.3
- Validation: Track feature importance, remove low-value features

**Risk 5: Sector-Specific Issues**

- Mitigation: Extensive sector-based testing (Financials, Energy, Tech, Healthcare)
- Validation: Per-sector feature distributions

---

## Success Criteria

### Quantitative Metrics

1. **Feature Count:** Add 50-75 new features across 8 categories
2. **Test Coverage:** ≥85% for new code, 100% for core calculations
3. **Performance:** Feature engineering <30s for 10K rows, <5min for 100K rows
4. **Test Pass Rate:** 100% of unit and integration tests
5. **Model Performance:** Improve R² by 3-5% in regression tasks

### Qualitative Metrics

1. Code adheres to existing style guidelines
2. Comprehensive docstrings for all new functions
3. Clear error messages and logging
4. Backward compatibility maintained
5. Documentation updated (README, code_guidelines, this plan)

---

## Integration with Existing Modules

### core.py (No Changes Required)

- Existing functions remain stable
- New features go into advanced.py

### advanced.py (Extensions)

- Add 8 new feature engineering functions
- Extend 3 existing functions
- Update `build_comprehensive_features()` orchestrator

### api.py (Minor Updates)

- Add new presets: "momentum", "quality", "cashflow", "comprehensive_v2"
- Update `build_features()` signature (optional parameters)

### Backward Compatibility

- All existing function signatures unchanged
- New parameters are optional with sensible defaults
- Deprecation warnings for any breaking changes (none expected)

---

## Appendix A: Feature Function Signatures

```python
def engineer_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price momentum and technical indicators."""
    pass

def engineer_financial_distress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Altman Z-Score trends and distress indicators."""
    pass

def engineer_cash_flow_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cash flow quality and conversion metrics."""
    pass

def engineer_capital_allocation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate capital allocation efficiency metrics."""
    pass

def engineer_market_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate analyst consensus and market sentiment features."""
    pass

def engineer_margin_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate profitability margin trends and quality."""
    pass

def engineer_balance_sheet_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate balance sheet growth and liquidity trends."""
    pass

def engineer_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite quality, growth, value, and momentum scores."""
    pass

def engineer_sector_relative_interactions(
    df: pd.DataFrame, 
    sector_col: str = "sector"
) -> pd.DataFrame:
    """Create sector-relative features for key metrics."""
    pass
```

---

## Appendix B: Database Column Utilization Matrix

| Category          | Current Utilization   | Target Utilization  | Priority |
|-------------------|-----------------------|---------------------|----------|
| Price History     | 20% (volatility only) | 80% (momentum, RSI) | High     |
| Altman Z-Score    | 0%                    | 100%                | High     |
| Beta Variations   | 0%                    | 100%                | Medium   |
| 5Y Averages       | 10%                   | 90%                 | High     |
| Exceptional Items | 40%                   | 100%                | High     |
| Cash Flows        | 30%                   | 90%                 | High     |
| Analyst Ratings   | 50%                   | 100%                | Medium   |
| Short Interest    | 0%                    | 100%                | Medium   |
| Buyback Yield     | 0%                    | 100%                | Medium   |

---

## Appendix C: Phased Rollout Schedule

**Week 1-2:** Test infrastructure + Momentum features  
**Week 3-4:** Quality & Risk + Cash Flow features  
**Week 5-6:** Sentiment + Profitability Trends  
**Week 7-8:** Balance Sheet + Temporal/Composite  
**Week 9:** Integration, optimization, documentation  
**Week 10:** Validation and production release

**Total Duration:** 10 weeks  
**Effort Estimate:** 1-2 full-time developers

---

## Version History

| Version | Date       | Author | Changes                                                |
|---------|------------|--------|--------------------------------------------------------|
| 1.0     | 2025-11-10 | System | Initial draft - comprehensive feature enhancement plan |

---

## Approval & Sign-off

- [ ] Technical Lead Review
- [ ] Data Science Team Review
- [ ] Code Quality Review
- [ ] Documentation Review
- [ ] Final Approval for Implementation

---

**END OF DOCUMENT**
