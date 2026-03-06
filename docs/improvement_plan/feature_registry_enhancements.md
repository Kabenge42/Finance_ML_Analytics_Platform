After comparing the `mv_equities` schema (700 lines, ~675 source columns) against the current `mv_all_stock_features` materialized view (54 `calc_*` functions, 17 categories), the following gaps and enhancement opportunities were identified. These are grouped by theme.

---

#### Enhancement 1: Add Missing Direct Reference Columns from `equities`

The current direct-select block (lines 5684–5693) only pulls 9 columns from `e`. Several high-value columns from `mv_equities` should be added:

```sql
-- Add to the direct reference columns block (after line 5693):
e."Current Fiscal Quarter"           AS current_fiscal_quarter,
e."Dividend Record (Currency)"       AS dividend_record_currency,
e."Dividend Record (Amount)"         AS dividend_record_amount,
e."Dividend Per Share (LTM)"         AS dividend_per_share_ltm,
e."Description"                      AS description,
e."Market Cap (Country R)"           AS market_cap_country_r,
e."Rel. Volume"                      AS rel_volume,
e."1-Day %"                          AS one_day_pct,
e."Total Return (YTD)"               AS total_return_ytd,
e."Total Return (5Y)"                AS total_return_5y,
e."Total Return (10Y)"               AS total_return_10y,
e."Tot. Return %/CAGR (3Y)"          AS tot_return_pct_cagr_3y,
e."Tot. Return %/CAGR (10Y)"         AS tot_return_pct_cagr_10y,
e."Total Revenues/CAGR (5Y FY)"      AS total_revenues_cagr_5y_fy,
e."Shrs Out (-1FY)"                  AS shrs_out_1fy,
e."Analyst Rating"                   AS analyst_rating,
e."Price Target - #"                 AS price_target_count,
```

**Rationale**: Total return metrics, relative volume, 1-day change, and analyst rating are fundamental screening/analytics columns not derivable from existing `calc_*` functions.

---

#### Enhancement 2: Expose Volatility Columns (New `calc_volatility_surface_features` Function)

`mv_equities` has `volatility_1m`, `volatility_3m`, `volatility_6m`, `volatility_1y` — none are directly surfaced. The existing `calc_technical_analysis_features` computes `volatility_compression` and `volatility_term_structure` but doesn't expose the raw values.

```sql
-- New function suggestion:
CREATE OR REPLACE FUNCTION calc_volatility_surface_features(p_isin TEXT DEFAULT NULL)
RETURNS TABLE (
    isin                    TEXT,
    volatility_1m           NUMERIC,
    volatility_3m           NUMERIC,
    volatility_6m           NUMERIC,
    volatility_1y           NUMERIC,
    volatility_trend_short  NUMERIC,  -- 1m/3m ratio
    volatility_trend_long   NUMERIC,  -- 3m/1y ratio
    low_vol_regime_flag     INTEGER
) ...

-- Add to Section 3 JOINs:
LEFT JOIN calc_volatility_surface_features() vsf ON id.isin = vsf.isin
```

---

#### Enhancement 3: Add Beta 2Y and Total Return Features to Quality & Risk

`mv_equities` has `beta_2y` which is not used anywhere. The existing `calc_beta_risk_features` only uses `beta_1y` and `beta_5y`.

```sql
-- Enhance calc_beta_risk_features to include:
e."Beta (2Y)"::NUMERIC AS beta_2y,
-- And derive:
(e."Beta (1Y)"::NUMERIC - e."Beta (2Y)"::NUMERIC) AS beta_short_term_shift,
```

---

#### Enhancement 4: Expose Effective Tax Rate Features (New Function)

`mv_equities` has 11 `effective_tax_rate_*` columns (LTM, FQ, FY, -1FY through -4FY, -1FQFQ through -4FQFQ) — **none are used in any `calc_*` function**.

```sql
-- New function suggestion:
CREATE OR REPLACE FUNCTION calc_tax_rate_features(p_isin TEXT DEFAULT NULL)
RETURNS TABLE (
    isin                    TEXT,
    effective_tax_rate_ltm  NUMERIC,
    effective_tax_rate_fy   NUMERIC,
    tax_rate_yoy_change     NUMERIC,  -- FY vs -1FY
    tax_rate_qoq_change     NUMERIC,  -- FQ vs -1FQFQ
    tax_rate_stability      NUMERIC,  -- stddev across periods
    low_tax_flag            INTEGER,  -- < 10%
    tax_rate_trend_4q       NUMERIC   -- trend across 4 quarters
) ...

-- Add as new Section 18 or within Section 7 (Quality & Risk):
LEFT JOIN calc_tax_rate_features() txf ON id.isin = txf.isin
```

**Rationale**: Tax rate dynamics are critical for earnings quality assessment and forward earnings modeling.

---

#### Enhancement 5: Expose Operating Expenses & SGA Temporal Data

`mv_equities` has `total_operating_expenses_*` (LTM, FQ, FY, -1FQFQ through -4FQFQ, -1FY through -4FY) — 11 columns. The current `calc_cost_structure_features` computes ratios like `opex_to_revenue` and `sga_to_revenue` but doesn't expose the raw temporal series or operating leverage trends.

```sql
-- Enhance calc_cost_structure_features or create calc_opex_temporal_features:
CREATE OR REPLACE FUNCTION calc_opex_temporal_features(p_isin TEXT DEFAULT NULL)
RETURNS TABLE (
    isin                     TEXT,
    opex_fq                  NUMERIC,
    opex_ltm                 NUMERIC,
    opex_fy                  NUMERIC,
    opex_qoq_growth          NUMERIC,
    opex_yoy_growth          NUMERIC,
    opex_vs_revenue_trend    NUMERIC,  -- change in opex/revenue ratio
    sga_qoq_growth           NUMERIC,
    sga_yoy_growth           NUMERIC,
    operating_leverage_score  NUMERIC   -- revenue growth - opex growth
) ...
```

---

#### Enhancement 6: Expand Analyst Coverage with Rating Breakdown

`mv_equities` has `num_strong_sell_ratings`, `num_strong_buys_ratings`, `num_hold_ratings`, `num_buys_ratings`, `num_sell_ratings`, `num_no_opinion_ratings` — these are the raw inputs for `calc_sentiment_features` but the breakdown itself isn't exposed. Also, `eps_norm_est_num_fy1e` (analyst count) and historical `price_target_num_*` columns aren't surfaced.

```sql
-- Add to direct reference columns or enhance calc_sentiment_features:
e."# Strong Buys Ratings"   AS num_strong_buys_ratings,
e."# Buys Ratings"          AS num_buys_ratings,
e."# Hold Ratings"          AS num_hold_ratings,
e."# Sell Ratings"          AS num_sell_ratings,
e."# Strong Sell Ratings"   AS num_strong_sell_ratings,
e."EPS Norm - Est # (FY1E)" AS eps_norm_est_num_fy1e,
```

---

#### Enhancement 7: Add GAAP EPS Estimate Revision Columns

`mv_equities` has `eps_gaap_est_avg_rev_pct_fy1e_1m/3m/6m/1y` — these are partially used in `calc_gaap_revision_features` but the raw GAAP estimate values (`eps_gaap_est_avg_ntm`, `eps_gaap_est_avg_fy1e`) aren't exposed. Similarly, `eps_norm_est_avg_ntm` and `eps_norm_est_avg_fy1e` should be surfaced.

```sql
-- Add to Section 5 (Earnings) or direct reference:
e."EPS GAAP - Est Avg (NTM)"    AS eps_gaap_est_avg_ntm,
e."EPS GAAP - Est Avg (FY1E)"   AS eps_gaap_est_avg_fy1e,
e."EPS Norm - Est Avg (NTM)"    AS eps_norm_est_avg_ntm,
e."EPS Norm - Est Avg (FY1E)"   AS eps_norm_est_avg_fy1e,
```

---

#### Enhancement 8: Add Gain/Loss on Asset Sales Features

`mv_equities` has `gain_loss_on_sale_of_assets_*` across 11 time periods — currently only `gain_loss_on_sale_of_assets_ltm` is indirectly referenced in unusual items. A dedicated function would improve coverage:

```sql
-- Enhance calc_unusual_items_features or create new:
gain_loss_on_sale_of_assets_ltm  NUMERIC,
asset_sale_frequency             INTEGER,  -- count of non-zero periods
asset_sale_trend                 NUMERIC,  -- trend across quarters
```

---

#### Enhancement 9: Expose FCF Estimate Forward Curve

`mv_equities` has `fcf_est_avg_fy1e` through `fcf_est_avg_fy5e` — a 5-year FCF estimate curve. None are currently surfaced.

```sql
-- Add to Section 12 (Cash Flow) or create calc_fcf_estimate_features:
e."FCF - Est Avg (FY1E)" AS fcf_est_avg_fy1e,
e."FCF - Est Avg (FY2E)" AS fcf_est_avg_fy2e,
e."FCF - Est Avg (FY3E)" AS fcf_est_avg_fy3e,
e."FCF - Est Avg (FY4E)" AS fcf_est_avg_fy4e,
e."FCF - Est Avg (FY5E)" AS fcf_est_avg_fy5e,
-- Derived:
fcf_est_cagr_5y          NUMERIC,  -- implied FCF growth from estimates
fcf_est_trend            NUMERIC,  -- linear trend across 5 estimates
```

**Rationale**: Forward FCF estimates are essential for DCF-based valuation and intrinsic value screening.

---

#### Enhancement 10: Add Dividend Historical Yield Breakdown

`mv_equities` has `div_yield_2fyind` through `div_yield_5fyind` — historical indicated dividend yields. Only `div_yield_ind` and `div_yield_1fyind` are currently used in `calc_dividend_yield_comprehensive`.

```sql
-- Enhance calc_dividend_yield_comprehensive to include:
e."Div Yield (-2FYInd)"::NUMERIC AS div_yield_2fyind,
e."Div Yield (-3FYInd)"::NUMERIC AS div_yield_3fyind,
e."Div Yield (-4FYInd)"::NUMERIC AS div_yield_4fyind,
e."Div Yield (-5FYInd)"::NUMERIC AS div_yield_5fyind,
-- Derived:
div_yield_5y_trend       NUMERIC,  -- trend across 5 years
div_yield_stability      NUMERIC,  -- coefficient of variation
```

---

#### Enhancement 11: Surface `Interest And Investment Income` Temporal Series

`mv_equities` has 11 `interest_and_investment_income_*` columns. The current `calc_interest_income_features` only uses the LTM value. Adding temporal coverage:

```sql
-- Enhance calc_interest_income_features:
interest_income_fq              NUMERIC,
interest_income_fy              NUMERIC,
interest_income_yoy_growth      NUMERIC,
interest_income_qoq_growth      NUMERIC,
interest_income_to_revenue_trend NUMERIC,
```

---

#### Enhancement 12: Add Share Dilution Tracking

`mv_equities` has `shrs_out` (current) and `shrs_out_1fy` (prior year). Currently `calc_composite_scores` computes `dilution_score` but the raw data isn't exposed.

```sql
-- Add to direct reference or enhance calc_composite_scores:
e."Shrs Out (-1FY)"::NUMERIC AS shrs_out_1fy,
-- Derived:
shares_yoy_change_pct    NUMERIC,  -- (current - 1fy) / 1fy
net_buyback_flag         INTEGER,  -- shares decreased YoY
```

---

### Summary of Coverage Impact

| Enhancement | New Columns | mv_equities Columns Consumed | Category |
|:--|:--|:--|:--|
| Direct reference columns | ~17 | 17 | Multiple |
| Volatility surface | ~7 | 4 | Technical |
| Beta 2Y | ~2 | 1 | Quality & Risk |
| Tax rate features | ~7 | 11 | New (Quality) |
| OpEx temporal | ~9 | 11 | Cost Structure |
| Analyst breakdown | ~6 | 6 | Analyst Sentiment |
| GAAP EPS estimates | ~4 | 4 | Earnings |
| Asset sale features | ~3 | 11 | Unusual Items |
| FCF estimate curve | ~7 | 5 | Cash Flow |
| Dividend yield history | ~6 | 4 | Dividends |
| Interest income temporal | ~5 | 10 | Cost Structure |
| Share dilution | ~3 | 1 | Composite |

**Estimated total**: ~76 new output columns, consuming ~85 previously unused `mv_equities` source columns. This would significantly improve column coverage and alignment between the raw equities data and the feature materialized view.