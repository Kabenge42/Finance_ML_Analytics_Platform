# Phase 9.3 Feature Enhancement Plan

**Date:** 2025-11-18  
**Status:** ACTIVE  
**Version:** 1.1  
**Model Version Target:** v9_9  
**Schema Version:** 1.3 (310 columns)

**Implementation status (Phase 9.3, current repo snapshot):**

- Phase 1 infrastructure (fixtures and helpers) implemented and covered by tests.
- Phase 2–5 feature groups (momentum, technical analysis, quality & distress, cash flow quality, capital allocation,
  sentiment/analyst) implemented with dedicated unit and integration tests.
- Schema 1.3 feature functions for valuation time-series, revenue forecasts, dividend reliability, and employment
  dynamics implemented with strict TDD coverage.
- Phase 6–8 trend, balance-sheet, temporal, and composite features wired into the advanced pipeline and validated via
  targeted tests.
- Phase 9 integration of valuation/estimate/quality feature families into
  Phase 9.4 classification labels and Phase 9.5 regression interactions
  is COMPLETE:
   - Event labels: `finance_ml.ml_workflow.classification.labels.create_enhanced_event_labels`
     now consume Schema 1.3 P/E, P/B, EV/EBITDA timelines and quality
     scores (e.g., `p_e_ntm`, `p_b_ltm`, `ev_ebitda_ltm`,
     `accounting_quality_score`, `distress_risk_score`).
   - Notebook: `ml_finance_model_main.ipynb` Phase 9.3/9.4 sections use an
     enhanced `valuation_candidates_by_method` mapping that aligns with
     the preprocessed metadata catalog
     (`outputs/catalog/preprocessed_stocks_metadata.json`).
   - Regression: `finance_ml.ml_workflow.regression.dataset.create_classification_interactions`
     creates interaction features between classification probabilities
     and the canonical Phase 9.3 valuation groups, safely ignoring
     columns that are absent in a given dataset.

---

## Version History

### Version 1.1 (2025-11-18)

- **Major Update:** Schema expansion from 262 to 310 columns (+48 new columns)
- Added 7 new feature categories based on revised CSV data structure
- Enhanced TDD specifications with comprehensive test requirements
- Updated all column references to reflect Schema Version 1.3
- Added technical indicators, valuation multiples time-series, and dividend analytics

### Version 1.0 (2025-11-10)

- Initial draft with baseline feature categories
- Original schema with 262 columns

---

## Executive Summary

This document proposes enhancements to the feature engineering pipeline by identifying underutilized columns in the
`equities` table and suggesting new calculated features. The plan follows Test-Driven Development (TDD) principles and
integrates with the existing `finance_ml.ml_workflow.features` module structure (core.py and advanced.py).

**Current State:**

- `core.py`: Basic ratios (P/E, P/B, debt/equity, ROE, ROA), margins, volatility, revenue CAGR
- `advanced.py`: 14 feature engineering functions covering valuation, profitability, leverage, liquidity, efficiency,
  growth, sector-specific, temporal, microstructure, analyst quality, accounting quality, and employee productivity

**Schema Version 1.3 Update:**

The equities table now contains **310 columns** (expanded from 262) with 48 new columns added in Schema Version 1.3:

- Revenue forecasting estimates (4 columns)
- EV/Sales valuation multiples time-series (11 columns)
- Employment metrics (2 columns)
- Technical indicators: 52W high/low, EMAs (6 columns)
- EV/EBITDA valuation multiples time-series (6 columns)
- Extended P/E ratio history (11 columns)
- Dividend record information (8 columns)

**Gap Analysis:**
The equities table contains rich temporal variations (FQ, FY, LTM, -1FY, 5YAVG, NTM, FY1E) and quality indicators
that present significant opportunities for advanced ML feature engineering. The new Schema 1.3 columns enable:

- Technical analysis integration with fundamental data
- Forward-looking valuation metrics
- Dividend reliability scoring
- Employee dynamics tracking
- Multi-period valuation trend analysis

---

## Feature Categories & Proposed Enhancements

## NEW in Version 1.1: Schema 1.3 Feature Categories

The following feature categories leverage the 48 new columns added in Schema Version 1.3.

### NEW 1. Technical Analysis Integration Features

**Database columns to leverage (Schema 1.3 additions):**

- `EMA (20D)`, `EMA (50D)`, `EMA (100D)`, `EMA (250D)` — Exponential moving averages
- `52W High/Adj`, `52W Low/Adj` — 52-week price extremes
- `Rel. Volume` — Relative volume indicator

**Proposed features:**

1. **EMA-Based Signals**
   - `ema_crossover_20_50`: Binary signal when 20D crosses above/below 50D
   - `ema_crossover_50_250`: Binary signal for longer-term trend changes
   - `price_vs_ema_20d`: (Last Price - EMA 20D) / EMA 20D
   - `price_vs_ema_250d`: Long-term price deviation from trend
   - `ema_slope_20d`: Rate of change in EMA 20D
   - `ema_trend_consistency`: Alignment of all EMAs (bullish/bearish/mixed)

2. **52-Week Position Features**
   - `pct_off_52w_high`: (52W High - Last Price) / 52W High
   - `pct_above_52w_low`: (Last Price - 52W Low) / 52W Low
   - `52w_range_position`: Position within 52W range (0-1 normalized)
   - `near_52w_high_flag`: Binary indicator if within 5% of 52W high
   - `near_52w_low_flag`: Binary indicator if within 5% of 52W low

3. **Volume & Momentum Composite**
   - `volume_momentum_score`: Rel. Volume * Price Momentum
   - `breakout_signal`: Combined EMA + 52W high proximity indicator

**Implementation approach:**

- Add `engineer_technical_analysis_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_technical_analysis_features.py`
- TDD requirements: Test EMA crossovers, 52W position calculations, edge cases (missing EMAs)

---

### NEW 2. Valuation Multiples Time-Series Features

**Database columns to leverage (Schema 1.3 additions):**

- **EV/Sales:** `ev_sales_est_fy1`, `ev_sales_ltm`, `ev_sales_ntm`, `ev_sales_1fyltm`, `ev_sales_2fyltm`,
  `ev_sales_3fyltm`, `ev_sales_3yavgltm`, `ev_sales_1fqltm` through `ev_sales_4fqltm`
- **EV/EBITDA:** `ev_ebitda_ltm`, `ev_ebitda_ntm`, `ev_ebitda_1fyltm`, `ev_ebitda_1fqltm`, `ev_ebitda_3yavgltm`,
  `ev_ebitda_est_fy1`
- **P/E Extended:** `p_e_est_fy1`, `p_e_2fyltm`, `p_e_3fyltm`, `p_e_3yavgltm`, `p_e_1fqltm`, `p_e_2fqltm`, `p_e_3fqltm`,
  `p_e_0fqqoqltm`, `p_e_0fyyoyltm`, `p_e_1fyyoyltm`, `p_e_0fqyoyltm`

**Proposed features:**

1. **Valuation Momentum Indicators**
   - `ev_sales_trend_1y`: (EV/Sales LTM - EV/Sales 1FYLTM) / EV/Sales 1FYLTM
   - `ev_sales_trend_3y`: 3-year trend slope using multiple points
   - `ev_ebitda_momentum`: Rate of change in EV/EBITDA
   - `p_e_momentum_yoy`: Year-over-year P/E change
   - `p_e_momentum_qoq`: Quarter-over-quarter P/E change

2. **Valuation Mean Reversion Features**
   - `ev_sales_vs_3y_avg`: Current vs 3-year average (z-score)
   - `ev_ebitda_vs_3y_avg`: Deviation from historical average
   - `p_e_vs_3y_avg`: P/E mean reversion indicator
   - `valuation_extreme_flag`: Binary flag for extreme deviations (>2 std)

3. **Forward vs Trailing Valuation**
   - `ev_sales_forward_discount`: (EV/Sales NTM - EV/Sales LTM) / EV/Sales LTM
   - `ev_ebitda_forward_discount`: Forward vs trailing comparison
   - `p_e_forward_discount`: (P/E EST FY1 - P/E LTM) / P/E LTM
   - `growth_implied_by_valuation`: Implied growth from forward multiples

4. **Quarterly Valuation Stability**
   - `ev_sales_quarterly_volatility`: Std dev across 4 quarters
   - `valuation_stability_score`: Inverse of quarterly volatility
   - `valuation_trend_consistency`: Monotonicity measure across quarters

**Implementation approach:**

- Add `engineer_valuation_timeseries_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_valuation_timeseries_features.py`
- TDD requirements: Test trend calculations, mean reversion z-scores, forward/trailing spreads, edge cases (negative
  multiples)

---

### NEW 3. Revenue Forecasting & Analyst Consensus Features

**Database columns to leverage (Schema 1.3 additions):**

- `Revenues - Est Avg (NTM)`, `Revenues - Est Avg (FY1E)`
- `Revenues - Est Med (NTM)`, `Revenues - Est Med (FY1E)`
- Existing: `Revenues - Est YoY % (FY1E)`

**Proposed features:**

1. **Analyst Consensus Metrics**
   - `revenue_estimate_spread_ntm`: (Est Avg - Est Med) / Est Med (disagreement indicator)
   - `revenue_estimate_spread_fy1e`: Analyst disagreement for FY1E
   - `revenue_consensus_uncertainty_score`: Composite spread metric

2. **Forward Revenue Expectations**
   - `revenue_growth_implied_ntm`: (Est Avg NTM - Total Revenues LTM) / Total Revenues LTM
   - `revenue_growth_implied_fy1e`: Expected growth rate from estimates
   - `revenue_growth_acceleration`: FY1E growth vs historical CAGR

3. **Estimate Quality Indicators**
   - `avg_vs_median_bias`: Systematic difference between avg and median
   - `estimate_confidence_flag`: Low spread = high confidence
   - `growth_surprise_potential`: Gap between estimate and trend

**Implementation approach:**

- Add `engineer_revenue_forecast_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_revenue_forecast_features.py`
- TDD requirements: Test spread calculations, growth rates, missing estimate handling

---

### NEW 4. Dividend Reliability & Income Stock Features

**Database columns to leverage (Schema 1.3 additions):**

- `Dividend Record (Announce Date)`, `Dividend Record (Ex Date)`, `Dividend Record (Payable Date)`,
  `Dividend Record (Record Date)`
- `Dividend Record (Frequency)` — TEXT: quarterly, annual, etc.
- `Dividend Record (Currency)` — TEXT
- `Dividend Record (Amount)` — NUMERIC
- `Dividend Streak` — Number of consecutive payments

**Proposed features:**

1. **Dividend Consistency Metrics**
   - `dividend_streak_years`: Dividend Streak (already numeric)
   - `dividend_consistency_score`: Weighted score based on streak + frequency
   - `income_stock_flag`: Binary flag for reliable dividend payers (streak > threshold)

2. **Dividend Coverage & Safety**
   - `dividend_payout_ratio`: Dividend Amount / EPS (sustainability)
   - `fcf_dividend_coverage`: FCF LTM / Total Dividends Paid
   - `dividend_safety_score`: Composite coverage metric

3. **Dividend Growth Features**
   - `dividend_growth_trend`: Change in Dividend Amount over time
   - `dividend_yield_vs_sector`: Sector-relative yield ranking
   - `dividend_aristocrat_flag`: Long streak + growth indicator

4. **Dividend Event Features**
   - `days_since_ex_date`: Recency of last dividend
   - `dividend_frequency_encoded`: Numerical encoding of frequency (quarterly=4, annual=1)
   - `currency_risk_flag`: Non-USD dividend currency indicator

**Implementation approach:**

- Add `engineer_dividend_reliability_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_dividend_reliability_features.py`
- TDD requirements: Test streak calculations, coverage ratios, date parsing, frequency encoding

---

### NEW 5. Employment Dynamics & Growth Signals

**Database columns to leverage (Schema 1.3 additions):**

- `Total Employees (FY)`, `Total Employees (FQ)`
- Existing: `Avg Employees (LTM)`, `Avg Employees (FY)`, `Avg Employees (5YAVGFY)`

**Proposed features:**

1. **Employee Growth Metrics**
   - `employee_growth_yoy`: (Total Employees FY - prior year) / prior year
   - `employee_growth_qoq`: Quarter-over-quarter employee change
   - `employee_growth_cagr_5y`: Long-term employee growth rate
   - `employee_growth_acceleration`: Change in growth rate

2. **Productivity & Efficiency**
   - `revenue_per_employee_fy`: Total Revenues FY / Total Employees FY
   - `revenue_per_employee_ltm`: Total Revenues LTM / Avg Employees LTM
   - `revenue_per_employee_trend`: YoY change in productivity
   - `profit_per_employee`: Net Income / Total Employees

3. **Scale & Workforce Indicators**
   - `employee_base_scale_flag`: Large employer indicator (>10k employees)
   - `workforce_volatility`: Std dev of employee counts across quarters
   - `hiring_intensity_score`: Employee growth relative to sector

**Implementation approach:**

- Add `engineer_employment_dynamics_features(df: pd.DataFrame) -> pd.DataFrame` to advanced.py
- Test module: `tests/test_employment_dynamics_features.py`
- TDD requirements: Test growth calculations, productivity ratios, missing employee data handling

---

## Original Feature Categories (Version 1.0)

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

- Run full test suite: `python -m unittest discover -s tests -p "test_*.py" -v`.
- Ensure 100% pass rate.

**10.2 Integration with Notebook**

- Update `ml_finance_model_main.ipynb`.
- Add feature engineering demonstration cells.

**10.3 Production Readiness**

- Confirm `MODEL_VERSION` is set to v9_9 in config and notebooks.
- Tag release in version control (out of scope for automated changes).

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

Use descriptive test names of the form::

    test_<feature_group>_<specific_feature>_<scenario>

Example::

    def test_momentum_rsi_14d_bullish_trend():
        """Test RSI calculation with bullish trend data."""
        ...

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

## Appendix A: Feature Function Signatures (Implemented)

The following feature functions are implemented in
``finance_ml/ml_workflow/features/advanced.py``. Signatures are shown here
for quick reference; see the module docstrings for detailed behavior.

```python
def engineer_momentum_features(df):
    """Calculate price momentum and technical indicators."""


def engineer_financial_distress_features(df):
    """Calculate Altman Z-Score trends and distress indicators."""


def engineer_cash_flow_quality_features(df):
    """Calculate cash flow quality and conversion metrics."""


def engineer_capital_allocation_features(df):
    """Calculate capital allocation efficiency metrics."""


def engineer_market_sentiment_features(df):
    """Calculate analyst consensus and market sentiment features."""


def engineer_margin_trends(df):
    """Calculate profitability margin trends and quality."""


def engineer_balance_sheet_trends(df):
    """Calculate balance sheet growth and liquidity trends."""


def engineer_composite_scores(df):
    """Calculate composite quality, growth, value, and momentum scores."""


def engineer_sector_relative_interactions(df, sector_col: str = "sector"):
    """Create sector-relative features for key metrics."""
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
