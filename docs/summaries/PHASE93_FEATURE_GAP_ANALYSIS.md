# Phase 9.3 Feature Engineering Gap Analysis

**Date:** 2025-11-22  
**Current Coverage:** 29/129 features (22.5%)  
**Target Coverage:** ≥60% (77+ features)  
**Status:** ANALYSIS COMPLETE

---

## Executive Summary

The low Phase 9.3 feature coverage (22.5%) is caused by:

1. **Feature naming mismatches** between generator functions and registry (60% of gaps)
2. **Missing orchestrator calls** for existing functions (20% of gaps)
3. **Conditional execution** of temporal features (10% of gaps)
4. **Truly missing feature implementations** (10% of gaps)

**High-Impact Quick Wins:**

- Update registry feature names → +35-40 features detected
- Add microstructure function call → +5-6 features
- Fix temporal execution → +4-5 features
- **Total estimated improvement:** 29 → 74+ features (57% coverage)

---

## Detailed Findings by Category

### 1. Analyst Sentiment (0/7 coverage, 0%)

**Function:** `engineer_analyst_quality_features()` at line 976  
**Status:** Implemented and called ✓  
**Issue:** Feature naming mismatches

| Registry Expectation          | Actual Feature Generated | Status        |
|-------------------------------|--------------------------|---------------|
| `analyst_consensus_strength`  | `consensus_strength`     | Rename needed |
| `analyst_dispersion`          | ❌ Missing                | Add feature   |
| `price_target_implied_return` | `upside_potential`       | Rename needed |
| `analyst_revision_momentum`   | `price_target_revision`  | Rename needed |
| `analyst_coverage_breadth`    | ❌ Missing                | Add feature   |
| `analyst_quality_score`       | ❌ Missing                | Add feature   |
| `target_revision_trend`       | ❌ Missing                | Add feature   |

**Additional features generated (not in registry):**

- `price_target_spread_pct`, `price_target_range`, `analyst_bullish_pct`, `analyst_bearish_pct`, `analyst_conviction`,
  `target_price_upside_pct`, `analyst_coverage_quality`

**Fix Strategy:**

1. Update registry to use actual names: `consensus_strength`, `upside_potential`, `price_target_revision`,
   `analyst_coverage_quality`
2. Add missing features: `analyst_dispersion`, `analyst_quality_score` (composite)
3. Add `analyst_bullish_pct`, `analyst_bearish_pct`, `analyst_conviction` to registry

**Estimated Post-Fix Coverage:** 7/10 features (70%)

---

### 2. Market Sentiment (0/10 coverage, 0%)

**Functions:**

- `engineer_market_sentiment_features()` at line 1666 (called ✓)
- `engineer_market_microstructure_features()` at line 764 (NOT called ❌)

**Issue:** Registry incorrectly categorizes analyst features + missing orchestrator call

| Registry Expectation       | Source Function                                  | Status         |
|----------------------------|--------------------------------------------------|----------------|
| `analyst_rating_score`     | Should be in Analyst                             | Wrong category |
| `strong_buy_ratio`         | Should be in Analyst                             | Wrong category |
| `sell_ratio`               | Should be in Analyst                             | Wrong category |
| `analyst_optimism`         | Should be in Analyst                             | Wrong category |
| `price_target_spread`      | Should be in Analyst (`price_target_spread_pct`) | Wrong category |
| `consensus_momentum`       | ❌ Missing                                        | Add feature    |
| `short_interest_ratio`     | Generated ✓                                      | Match!         |
| `volume_price_correlation` | Microstructure function not called               | Add call       |
| `liquidity_score`          | Microstructure function not called               | Add call       |
| `market_impact_estimate`   | Microstructure function not called               | Add call       |

**Market Sentiment Actually Generates:**

- `short_interest_ratio` ✓
- `beta_stability` ✓
- `systematic_risk_trend` ✓

**Microstructure Function Generates (NOT CALLED):**

- `price_range_pct`, `volatility_30d`, `volatility_60d`, `volatility_90d`, `momentum_20d`, `ma_20d`, `ma_50d`

**Fix Strategy:**

1. Move analyst-related features from Market Sentiment to Analyst Sentiment category
2. Update Market Sentiment registry with actual features: `short_interest_ratio`, `beta_stability`,
   `systematic_risk_trend`
3. Add microstructure call to orchestrator after line 1614
4. Update registry with microstructure features OR rename registry expectations

**Estimated Post-Fix Coverage:** 3/6 features (50%)

---

### 3. Temporal Patterns (0/8 coverage, 0%)

**Function:** `engineer_temporal_features()` at line 530  
**Status:** Implemented but conditionally called ❌  
**Issue:** Only called if "next_earnings" column exists (line 1634)

| Registry Expectation      | Actual Feature Generated     | Status        |
|---------------------------|------------------------------|---------------|
| `revenue_seasonality`     | `ltm_vs_5yavg_revenue`       | Rename needed |
| `earnings_seasonality`    | `fq_vs_5yavg_ebitda`         | Rename needed |
| `quarterly_consistency`   | `quarterly_volatility_score` | Rename needed |
| `yoy_growth_stability`    | ❌ Missing                    | Add feature   |
| `revenue_trend_strength`  | ❌ Missing                    | Add feature   |
| `earnings_trend_strength` | ❌ Missing                    | Add feature   |
| `cyclicality_indicator`   | ❌ Missing                    | Add feature   |
| `temporal_quality_score`  | ❌ Missing                    | Add feature   |

**Additional features generated:**

- `fiscal_quarter`, `month`, `year`, `days_to_earnings`, `reporting_lag`, `earnings_report_recency`

**Fix Strategy:**

1. Make temporal function call unconditional (check for alternative date columns)
2. Update registry with actual names
3. Add `fiscal_quarter`, `reporting_lag` to registry
4. Implement missing growth stability features

**Estimated Post-Fix Coverage:** 6/11 features (55%)

---

### 4. Profitability (1/15 coverage, 6.7%)

**Functions:** Multiple (`engineer_profitability_ratios()`, `engineer_margin_trends()`)  
**Status:** Called ✓  
**Issue:** Registry expects more features than generated

**Currently Detected:**

- `gross_margin_trend` ✓

**Registry Expects Additional:**

- `roe_ltm`, `roa_ltm`, `roic_ltm`, `net_margin_ltm`, `operating_margin_ltm`, `gross_margin_ltm`, `ebitda_margin_ltm`,
  `fcf_margin_ltm`, `margin_stability_score`, `margin_trend_consistency`, `profitability_composite`,
  `operating_margin_trend`, `net_margin_trend`, `margin_quality_score`

**Fix Strategy:**

1. Audit `engineer_profitability_ratios()` to see what it actually generates
2. Update registry or add missing features
3. Likely many features exist but with different names (e.g., `return_on_equity_pct_ltm` vs `roe_ltm`)

**Estimated Post-Fix Coverage:** 8/15 features (53%)

---

### 5. Cash Flow (1/10 coverage, 10%)

**Function:** `engineer_cash_flow_quality_features()` at line (need to locate)  
**Status:** Called ✓  
**Issue:** Feature naming mismatches

**Currently Detected:**

- `capex_intensity` ✓

**Registry Expects:**

- `cfo_to_ni_ratio`, `fcf_to_ni_ratio`, `cash_conversion_cycle`, `operating_cash_flow_ratio`, `fcf_yield`,
  `cash_flow_stability`, `fcf_growth_rate`, `cash_reinvestment_rate`, `cash_generation_quality`

**Fix Strategy:**

1. Audit cash flow function implementation
2. Update registry with actual names
3. Implement missing features

**Estimated Post-Fix Coverage:** 6/10 features (60%)

---

### 6. Capital Allocation (4/13 coverage, 30.8%)

**Function:** `engineer_capital_allocation_features()` + `engineer_dividend_reliability_features()`  
**Status:** Both called ✓  
**Currently Detected:** 4 features

**Registry Expects:**

- `capital_allocation_efficiency`, `reinvestment_rate` ✓, `payout_ratio_total`, `dividend_sustainability`,
  `buyback_intensity`, `dividend_growth_rate`, `total_shareholder_return`, `capital_deployment_score`,
  `acquisition_intensity` ✓, `capex_to_sales_ratio`, `dividend_consistency_score`, `dividend_coverage_ratio`,
  `dividend_streak_years` ✓

**Fix Strategy:**

1. Audit both functions to see actual feature names
2. Update registry
3. Best coverage already, focus on other categories first

**Estimated Post-Fix Coverage:** 10/13 features (77%)

---

### 7. Composite Scores (1/8 coverage, 12.5%)

**Function:** `engineer_composite_scores()` (need to locate and verify)  
**Currently Detected:** `composite_quality_score` ✓

**Registry Expects:**

- `composite_quality_score` ✓, `composite_growth_score`, `composite_value_score`, `composite_momentum_score`,
  `overall_fundamental_score`, `risk_adjusted_quality_score`, `growth_quality_score`, `value_quality_score`

**Fix Strategy:**

1. Verify composite function exists and is called
2. Implement missing composite scores
3. These are high-value features for model performance

**Estimated Post-Fix Coverage:** 5/8 features (62%)

---

### 8. Categories with Good Coverage (No immediate action needed)

**Leverage & Liquidity:** 7/12 (58.3%) ✓  
**Valuation Ratios:** 7/16 (43.8%) ✓  
**Momentum & Technical:** 6/17 (35.3%) ✓  
**Quality & Risk:** 2/13 (15.4%) - moderate priority

---

## Implementation Plan

### Phase 1: Registry Updates (High Impact, Low Effort)

**Estimated Time:** 1-2 hours  
**Expected Coverage Gain:** +35-40 features

1. Update `phase93_categories.py` PHASE93_FEATURE_CATEGORIES dict:
    - Analyst Sentiment: Add actual feature names, remove duplicates
    - Market Sentiment: Remove analyst features, add actual market features
    - Temporal Patterns: Update with actual feature names
    - All categories: Align with actual generator outputs

2. Move misplaced features:
    - `analyst_rating_score`, `strong_buy_ratio`, `sell_ratio`, `analyst_optimism`, `price_target_spread` → Analyst
      Sentiment

### Phase 2: Orchestrator Fixes (Medium Impact, Low Effort)

**Estimated Time:** 30 minutes  
**Expected Coverage Gain:** +5-6 features

1. Add missing function call in `build_comprehensive_features()`:
   ```python
   # After line 1614 (engineer_market_sentiment_features)
   result = engineer_market_microstructure_features(result)
   ```

2. Fix temporal features conditional:
   ```python
   # Replace lines 1634-1635
   # Try multiple date columns, not just next_earnings
   date_col = next((c for c in ["next_earnings", "last_updated", "income_statement_report_date"] if c in result.columns), None)
   if date_col:
       result = engineer_temporal_features(result, date_col=date_col)
   ```

### Phase 3: Add Missing Features (Lower Priority)

**Estimated Time:** 3-4 hours  
**Expected Coverage Gain:** +5-8 features

1. Analyst Sentiment: Add `analyst_dispersion`, `analyst_quality_score` (composite)
2. Temporal Patterns: Add growth stability features
3. Composite Scores: Implement missing composite scores

### Phase 4: Testing & Validation

**Estimated Time:** 1 hour

1. Run Phase 9.3 feature engineering in notebook
2. Execute benchmarking analysis
3. Verify coverage ≥60%
4. Run test suites: `python -m unittest tests.test_phase93_eda_integration -v`

---

## Success Criteria

- ✅ Feature coverage increases from 22.5% to ≥60%
- ✅ All tests pass with ≥80% coverage
- ✅ No regressions in existing functionality
- ✅ Documentation updated (CHANGELOG.md, Phase_9.3_feature_enhancement_plan.md)

---

**Next Steps:** Proceed with Phase 1 (Registry Updates) immediately.
