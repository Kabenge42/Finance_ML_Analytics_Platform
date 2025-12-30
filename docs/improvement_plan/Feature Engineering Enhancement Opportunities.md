### Feature Engineering Enhancement Opportunities

Based on the analysis of the `postgres.public.equities` schema and the existing feature engineering modules in
`finance_ml/features/advanced/`, the following improvements and new features are recommended.

#### 1. Earnings Analytics (`earnings.py`)

**Current State:** Focuses on EPS/Revenue surprise and 1M/3M/6M revision momentum.
**Enhancement Opportunities:**

* **Short & Long-Term Revision Momentum:** The schema includes `EPS Est Avg Rev % (FY1E - 1W)` and
  `EPS Est Avg Rev % (FY1E - 1Y)`.
    * *Update:* Include the **1-week** revision trend to capture immediate analyst reactions and the **1-year** trend
      for long-term sentiment shifts in `surprise_momentum_score`.
* **GAAP vs. Non-GAAP Revisions:** The schema provides `EPS GAAP Est Avg Rev %`.
    * *New Feature:* `gaap_revision_divergence` - Compare the revision magnitude of GAAP estimates vs. Normalized
      estimates to identify if analysts are adjusting "quality" expectations differently from "headline" numbers.
* **Revenue Forecast Skew:** Using `Revenues - Est Med (NTM)` vs `Revenues - Est Avg (NTM)`.
    * *New Feature:* `revenue_forecast_skew` - Difference between Mean and Median estimates normalized by the mean.
      Large differences indicate outlier analyst expectations.

#### 2. Growth Metrics (`growth.py`)

**Current State:** Calculates historical YoY growth for Revenue, EPS, EBITDA, TBV.
**Enhancement Opportunities:**

* **Forward Growth Indicators:** The schema includes `Revenues - Est YoY % (FY1E)` and `Revenues - Est Avg (NTM)`.
    * *New Feature:* `forward_revenue_growth` - Use the explicit estimate field rather than calculating it manually from
      raw estimates.
* **Long-Term Consistency:** The schema has `Total Revenues/CAGR (5Y FY)`.
    * *New Feature:* `revenue_cagr_5y` - Directly use the pre-calculated 5-year CAGR.
    * *New Feature:* `growth_persistence_score` - Compare `revenue_growth_yoy` (current) vs `revenue_cagr_5y` to check
      if growth is accelerating or decelerating relative to the long-term trend.

#### 3. Dividends & Capital Allocation (`dividends.py`)

**Current State:** Focuses on dividend streak, payout ratio, and coverage.
**Enhancement Opportunities:**

* **Buyback Yield (Major Addition):** The schema explicitly provides `Buyback Yield (LTM)`.
    * *New Feature:* `buyback_yield` - Direct mapping.
    * *New Feature:* `total_shareholder_yield` = `dividend_yield` + `buyback_yield`. This is a critical total return
      metric missing from the current implementation.
* **Dividend Yield Term Structure:** Schema has `Div Yield (NTM)` vs `Div Yield (LTM)`.
    * *New Feature:* `dividend_growth_expectation` - Derived from the spread between NTM and LTM yields.

#### 4. Profitability & Efficiency (`profitability.py`)

**Current State:** Standard margins (Gross, Operating, Net) and Returns (ROE, ROA, ROIC).
**Enhancement Opportunities:**

* **Operational Efficiency Ratios:** The schema includes detailed expense lines like `R&D Expenses (LTM)` and
  `Marketing Expenses (FY)`.
    * *New Feature:* `rnd_intensity` = `R&D Expenses (LTM)` / `Total Revenues (LTM)`. Crucial for Tech/Healthcare
      sectors.
    * *New Feature:* `marketing_efficiency` = `Total Revenues` / `Marketing Expenses`.
    * *New Feature:* `sga_ratio` = `Selling General & Admin Expenses` / `Total Revenues`.
* **Dupont Analysis Components:**
    * *Update:* Refine `roe` decomposition into `net_profit_margin` * `asset_turnover` * `equity_multiplier` explicitly
      to allow for factor-based screening.

#### 5. Quality & Risk (`quality.py`)

**Current State:** Accounting quality (goodwill, writedowns), Altman Z-Score, composite scores.
**Enhancement Opportunities:**

* **Merger & Restructuring Impact:** Schema includes `Merger & Restructuring Charges`.
    * *New Feature:* `merger_impact_ratio` - Charges relative to Market Cap or EBITDA to assess the scale of corporate
      reorganization.
* **Investment Income Dependence:** Schema has `Interest Income On Investments`.
    * *New Feature:* `non_operating_income_share` - `Interest Income` / `Net Income`. High values indicate earnings are
      driven by financial engineering rather than core operations.
* **Asset Sale Reliance:** Schema has `Gain (Loss) On Sale Of Assets`.
    * *New Feature:* `asset_sale_boost` - Flag periods where earnings are artificially inflated by one-off asset sales.

#### 6. Revenue Forecast (`revenue.py`)

**Current State:** Placeholder module.
**Enhancement Opportunities:**

* **Implementation:** Populate this module using the rich estimate data.
* *New Feature:* `revenue_estimate_momentum` - ( `Revenues - Est Avg (NTM)` - `Total Revenues (LTM)` ) /
  `Total Revenues (LTM)`.
* *New Feature:* `revenue_surprise_volatility` - If historical surprise data is accessible, calculate the standard
  deviation of revenue surprises.

#### 7. Momentum & Technicals (`momentum.py`)

**Current State:** RSI, Momentum (1m-1y), MA crossovers.
**Enhancement Opportunities:**

* **Beta Dynamics:** Schema has `Beta (1Y)`, `Beta (2Y)`, `Beta (5Y)`.
    * *New Feature:* `beta_momentum` = `Beta (1Y)` - `Beta (5Y)`. Positive values indicate the stock is becoming
      riskier/more volatile relative to the market.
* **Volatility Term Structure:** Schema has `Volatility (1M)` ... `Volatility (1Y)`.
    * *New Feature:* `volatility_term_structure` = `Volatility (1M)` / `Volatility (1Y)`. Ratio > 1 indicates short-term
      fear/uncertainty significantly higher than long-term baseline.

#### 8. Sector Specific (`sector.py`)

**Current State:** Basic sector masks and simple TBV logic.
**Enhancement Opportunities:**

* **Direct TBV Usage:** The schema has `TBV (LTM)` and `TBV (FY)` directly.
    * *Update:* Replace calculated tangible book value logic with the direct database column for higher accuracy,
      falling back to calculation only if missing.
* **Relative Valuation:** Schema contains `Market Cap (Country R)` (Rank).
    * *New Feature:* `size_factor_percentile` - Utilize the country rank to normalize size factor across different
      regions.

### Summary of New Data Points to Utilize

* **Estimates:** `Revenues - Est YoY %`, `Revenues - Est Med`, `EPS Est Avg Rev % (1W/1Y)`, `EPS GAAP Est Avg Rev %`.
* **Cash Flow/Yield:** `Buyback Yield (LTM)`.
* **Expenses:** `R&D Expenses`, `Marketing Expenses`, `Merger & Restructuring Charges`.
* **Risk:** `Beta (1Y/2Y/5Y)`, `Volatility (1M/3M/6M/1Y)`.