# Phase 9.3 → v1.14 Feature Engineering Enhancement Plan

## Refined Feature Assignment to Existing Categories

Based on your guidance, here's the refined mapping of new temporal features to existing categories and submodules:

| New Features          | Count | Target Category                  | Target Module                     |
|-----------------------|-------|----------------------------------|-----------------------------------|
| Price Target Dynamics | 15    | **Analyst Sentiment** (10→25)    | `sentiment.py`                    |
| Cash Flow Temporal    | 12    | **Cash Flow** (5→17)             | `leverage.py` (cash flow section) |
| EPS Trajectory        | 10    | **Earnings Quality** (33→43)     | `earnings.py`                     |
| Fiscal Calendar       | 8     | **Temporal Patterns** (17→25)    | `temporal.py`                     |
| Dividend Timing       | 8     | **Dividend Reliability** (12→20) | `dividends.py`                    |

**Updated Total: 296 + 53 = 349 features across 21 categories**

---

## Part 1: Schema.py Refactoring

```python
# ... existing code ...

# =============================================================================
# NEW FEATURES: Phase 9.3 v1.14 Temporal Enhancements
# =============================================================================

# --- Price Target Dynamics (Analyst Sentiment) ---
"pt_momentum_1w": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum (1-week change %)",
},
"pt_momentum_1m": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum (1-month change %)",
},
"pt_momentum_3m": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum (3-month change %)",
},
"pt_momentum_6m": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum (6-month change %)",
},
"pt_momentum_1y": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum (1-year change %)",
},
"pt_acceleration_short": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum acceleration (1M vs 3M)",
},
"pt_acceleration_long": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target momentum acceleration (3M vs 1Y)",
},
"pt_consensus_convergence": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Analyst consensus convergence (spread narrowing)",
},
"analyst_coverage_change_1m": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Change in analyst coverage count (1-month)",
},
"analyst_coverage_change_3m": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Change in analyst coverage count (3-month)",
},
"pt_vs_price_momentum": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target vs price momentum divergence",
},
"pt_qtd_momentum": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target quarter-to-date momentum",
},
"pt_ytd_momentum": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Price target year-to-date momentum",
},
"pt_skew_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Analyst estimate skewness trend (mean vs median)",
},
"pt_high_low_spread_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "High-low price target spread evolution",
},

# --- Cash Flow Temporal (Cash Flow category) ---
"fcf_quarterly_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "FCF trend slope across last 5 quarters (normalized)",
},
"fcf_quarterly_volatility": {
    "dtype": "Float64",
    "role": "feature",
    "description": "FCF coefficient of variation across quarters",
},
"fcf_positive_ratio": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Ratio of positive FCF quarters (0-1)",
},
"cfo_quarterly_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "CFO trend slope across last 5 quarters",
},
"cfo_yoy_quarterly": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "CFO year-over-year quarterly growth",
},
"investment_intensity_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "CFI investment intensity trend (normalized)",
},
"cfo_5y_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "CFO 5-year trend slope",
},
"cfo_5y_stability": {
    "dtype": "Float64",
    "role": "feature",
    "description": "CFO 5-year stability score (1 - CV)",
},
"cfo_margin_current": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "Current CFO margin (CFO/Revenue)",
},
"cfo_margin_trend": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "CFO margin trend (current vs prior year)",
},
"acquisition_activity_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Cash acquisition activity trend",
},
"acquisition_quarters_active": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Number of quarters with acquisition activity",
},

# --- EPS Trajectory (Earnings Quality category) ---
"eps_quarterly_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "EPS quarterly trend slope (normalized)",
},
"eps_quarterly_volatility": {
    "dtype": "Float64",
    "role": "feature",
    "description": "EPS quarterly coefficient of variation",
},
"eps_yoy_quarterly_growth": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "EPS year-over-year quarterly growth",
},
"eps_qoq_growth": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "EPS quarter-over-quarter growth",
},
"eps_positive_streak": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Count of positive EPS quarters (last 5)",
},
"eps_cagr_5y": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "EPS 5-year compound annual growth rate",
},
"eps_cagr_3y": {
    "dtype": "Float64",
    "role": "percentage",
    "description": "EPS 3-year compound annual growth rate",
},
"eps_annual_trend": {
    "dtype": "Float64",
    "role": "feature",
    "description": "EPS annual trend slope (normalized)",
},
"eps_vs_5y_avg": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Current EPS vs 5-year average ratio",
},
"eps_growth_acceleration": {
    "dtype": "Float64",
    "role": "feature",
    "description": "EPS growth acceleration (3Y CAGR - 5Y CAGR)",
},

# --- Fiscal Calendar (Temporal Patterns category) ---
"fiscal_year_progress": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Progress through fiscal year (0-1 scale)",
},
"days_to_quarter_end": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Days until fiscal quarter end",
},
"fiscal_half": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Fiscal half indicator (1 or 2)",
},
"reporting_lag_zscore": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Reporting lag z-score vs median",
},
"late_reporter_flag": {
    "dtype": "boolean",
    "role": "feature",
    "description": "Flag for late reporting (>60 days)",
},
"days_since_fy_end": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Days since fiscal year end",
},
"days_to_next_fy_end": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Days until next fiscal year end",
},
"earnings_imminent": {
    "dtype": "boolean",
    "role": "feature",
    "description": "Earnings within 14 days flag",
},
"pre_earnings_window": {
    "dtype": "boolean",
    "role": "feature",
    "description": "Within 30-day pre-earnings window",
},

# --- Dividend Timing (Dividend Reliability category) ---
"days_to_dividend_ex_date": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Days until dividend ex-date",
},
"days_to_dividend_record_date": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Days until dividend record date",
},
"days_to_dividend_payable_date": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Days until dividend payable date",
},
"approaching_ex_date": {
    "dtype": "boolean",
    "role": "feature",
    "description": "Within 7 days of ex-dividend date",
},
"recently_ex_dividend": {
    "dtype": "boolean",
    "role": "feature",
    "description": "Went ex-dividend within last 7 days",
},
"dividend_cycle_days": {
    "dtype": "Int64",
    "role": "feature",
    "description": "Days in dividend payment cycle",
},
"dividend_cycle_position": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Position within dividend cycle (0-1)",
},
"dividend_announcement_recency": {
    "dtype": "Float64",
    "role": "feature",
    "description": "Days since last dividend announcement",
},

# ... existing code ...
```

---

## Part 2: PHASE93_FEATURE_CATEGORIES Updates

```python
# ... existing code ...

# Phase 9.3 Feature Input Categorization (v1.14)
# Total: 349 features across 21 categories
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
    # ... existing categories ...
    
    # =========================================================================
    # ANALYST SENTIMENT (25 features) - Updated from 10
    # =========================================================================
    "Analyst Sentiment": [
        # Existing (10)
        "analyst_bullish_pct",
        "analyst_bearish_pct",
        "analyst_conviction",
        "analyst_coverage_quality",
        "consensus_strength",
        "price_target_range",
        "price_target_revision",
        "price_target_spread_pct",
        "target_price_upside_pct",
        "upside_potential",
        # NEW: Price Target Dynamics (15)
        "pt_momentum_1w",
        "pt_momentum_1m",
        "pt_momentum_3m",
        "pt_momentum_6m",
        "pt_momentum_1y",
        "pt_acceleration_short",
        "pt_acceleration_long",
        "pt_consensus_convergence",
        "analyst_coverage_change_1m",
        "analyst_coverage_change_3m",
        "pt_vs_price_momentum",
        "pt_qtd_momentum",
        "pt_ytd_momentum",
        "pt_skew_trend",
        "pt_high_low_spread_trend",
    ],
    
    # =========================================================================
    # CASH FLOW (17 features) - Updated from 5
    # =========================================================================
    "Cash Flow": [
        # Existing (5)
        "cfo_growth_yoy",
        "cfo_to_net_income",
        "fcf_margin",
        "fcf_stability",
        "fcf_to_net_income",
        # NEW: Cash Flow Temporal (12)
        "fcf_quarterly_trend",
        "fcf_quarterly_volatility",
        "fcf_positive_ratio",
        "cfo_quarterly_trend",
        "cfo_yoy_quarterly",
        "investment_intensity_trend",
        "cfo_5y_trend",
        "cfo_5y_stability",
        "cfo_margin_current",
        "cfo_margin_trend",
        "acquisition_activity_trend",
        "acquisition_quarters_active",
    ],
    
    # =========================================================================
    # TEMPORAL PATTERNS (25 features) - Updated from 17
    # =========================================================================
    "Temporal Patterns": [
        # Existing (17)
        "reference_date",
        "days_since_reference",
        "days_to_dividend",
        "days_to_earnings",
        "earnings_report_recency",
        "fiscal_quarter",
        "fiscal_year",
        "month",
        "reporting_lag",
        "year",
        "quarter_end_flag",
        "month_end_flag",
        "week_of_year",
        "day_of_week",
        "ltm_vs_5yavg_revenue",
        "fq_vs_5yavg_ebitda",
        "quarterly_volatility_score",
        # NEW: Fiscal Calendar (8)
        "fiscal_year_progress",
        "days_to_quarter_end",
        "fiscal_half",
        "reporting_lag_zscore",
        "late_reporter_flag",
        "days_since_fy_end",
        "days_to_next_fy_end",
        "earnings_imminent",
        "pre_earnings_window",
    ],
    
    # =========================================================================
    # EARNINGS QUALITY (43 features) - Updated from 33
    # =========================================================================
    "Earnings Quality": [
        # Existing (33)
        "accelerating_upgrades_flag",
        "consensus_uncertainty_score",
        "earnings_beat_indicator",
        "eps_surprise_magnitude",
        "eps_surprise_pct",
        "estimate_revision_acceleration",
        "ebitda_surprise_pct",
        "positive_revision_momentum",
        "revenue_beat_indicator",
        "revenue_surprise_pct",
        "surprise_momentum_score",
        "adjustment_consistency_score",
        "earnings_quality_score_composite",
        "earnings_quality_warning_flag",
        "ebit_adjustment_pct_ltm",
        "ebit_adjustment_ratio_fy",
        "ebit_adjustment_ratio_ltm",
        "ebit_adjustment_spread_fy",
        "ebit_adjustment_spread_ltm",
        "ebitda_adjustment_pct_ltm",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_adjustment_spread_fy",
        "ebitda_adjustment_spread_ltm",
        "eps_adjustment_pct_fy",
        "eps_adjustment_pct_ltm",
        "eps_adjustment_ratio_fy",
        "eps_adjustment_ratio_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_spread_ltm",
        "eps_quality_flag_ltm",
        "exceptional_items_impact_ratio",
        "net_income_adjustment_pct_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_spread_fy",
        "net_income_adjustment_spread_ltm",
        # NEW: EPS Trajectory (10)
        "eps_quarterly_trend",
        "eps_quarterly_volatility",
        "eps_yoy_quarterly_growth",
        "eps_qoq_growth",
        "eps_positive_streak",
        "eps_cagr_5y",
        "eps_cagr_3y",
        "eps_annual_trend",
        "eps_vs_5y_avg",
        "eps_growth_acceleration",
    ],
    
    # =========================================================================
    # DIVIDEND RELIABILITY (20 features) - Updated from 12
    # =========================================================================
    "Dividend Reliability": [
        # Existing (12)
        "days_to_dividend",
        "dividend_coverage_ratio",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_reliability_score",
        "dividend_streak",
        "dividend_yield_stability",
        "div_yield_5yavgltm",
        "fcf_dividend_coverage",
        "payout_consistency_score",
        "sustainable_dividend_flag",
        # NEW: Dividend Timing (8)
        "days_to_dividend_ex_date",
        "days_to_dividend_record_date",
        "days_to_dividend_payable_date",
        "approaching_ex_date",
        "recently_ex_dividend",
        "dividend_cycle_days",
        "dividend_cycle_position",
        "dividend_announcement_recency",
    ],
    
    # ... remaining categories unchanged ...
}

# ... existing code ...
```

---

## Part 3: Implementation Tasks

### Task 1: Update `sentiment.py` - Add Price Target Dynamics

**File:** `finance_ml/features/advanced/sentiment.py`

```python
# ... existing code ...

def engineer_price_target_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price target temporal dynamics features.
    
    Leverages historical price target data (1W, 1M, 3M, 6M, MTD, QTD, YTD, 1Y ago)
    to derive momentum, acceleration, and consensus evolution features.
    
    Features computed (15):
    - pt_momentum_* (1w, 1m, 3m, 6m, 1y): Price target percentage changes
    - pt_acceleration_short/long: Momentum acceleration
    - pt_consensus_convergence: Analyst spread narrowing
    - analyst_coverage_change_*: Coverage count changes
    - pt_vs_price_momentum: Target vs price divergence
    - pt_qtd/ytd_momentum: Period-specific momentum
    - pt_skew_trend: Mean vs median evolution
    - pt_high_low_spread_trend: Range evolution
    
    Args:
        df: Input DataFrame with price target historical columns
        
    Returns:
        DataFrame with price target dynamics features added
    """
    result = df.copy()
    
    # === Price Target Momentum ===
    momentum_pairs = [
        ("pt_momentum_1w", "price_target", "price_target_1w_ago"),
        ("pt_momentum_1m", "price_target", "price_target_1m_ago"),
        ("pt_momentum_3m", "price_target", "price_target_3m_ago"),
        ("pt_momentum_6m", "price_target", "price_target_6m_ago"),
        ("pt_momentum_1y", "price_target", "price_target_1y_ago"),
        ("pt_qtd_momentum", "price_target", "price_target_qtd_ago"),
        ("pt_ytd_momentum", "price_target", "price_target_ytd_ago"),
    ]
    
    for feature_name, current_col, prior_col in momentum_pairs:
        if current_col in df.columns and prior_col in df.columns:
            result[feature_name] = _safe_pct_change(
                df[current_col].astype(float),
                df[prior_col].astype(float)
            )
    
    # === Momentum Acceleration ===
    if "pt_momentum_1m" in result.columns and "pt_momentum_3m" in result.columns:
        result["pt_acceleration_short"] = (
            result["pt_momentum_1m"] - result["pt_momentum_3m"]
        )
    
    if "pt_momentum_3m" in result.columns and "pt_momentum_1y" in result.columns:
        result["pt_acceleration_long"] = (
            result["pt_momentum_3m"] - result["pt_momentum_1y"]
        )
    
    # === Consensus Range Evolution ===
    spread_cols_current = ["price_target_high", "price_target_low", "price_target_median"]
    spread_cols_3m = ["price_target_high_3m_ago", "price_target_low_3m_ago", "price_target_median_3m_ago"]
    
    if all(c in df.columns for c in spread_cols_current + spread_cols_3m):
        current_spread = _safe_div(
            df["price_target_high"].astype(float) - df["price_target_low"].astype(float),
            df["price_target_median"].astype(float)
        )
        spread_3m = _safe_div(
            df["price_target_high_3m_ago"].astype(float) - df["price_target_low_3m_ago"].astype(float),
            df["price_target_median_3m_ago"].astype(float)
        )
        # Positive = spread narrowing (converging)
        result["pt_consensus_convergence"] = spread_3m - current_spread
        result["pt_high_low_spread_trend"] = current_spread - spread_3m
    
    # === Analyst Coverage Trajectory ===
    count_col = "price_target_count" if "price_target_count" in df.columns else "price_target_num"
    if count_col in df.columns:
        if "price_target_count_1m_ago" in df.columns:
            result["analyst_coverage_change_1m"] = (
                df[count_col].astype(float) - df["price_target_count_1m_ago"].astype(float)
            )
        if "price_target_count_3m_ago" in df.columns:
            result["analyst_coverage_change_3m"] = (
                df[count_col].astype(float) - df["price_target_count_3m_ago"].astype(float)
            )
    
    # === Target vs Price Momentum Divergence ===
    if all(c in df.columns for c in ["price_target", "last_price", "price_target_3m_ago", "price_3m_ago"]):
        current_ratio = _safe_div(
            df["price_target"].astype(float),
            df["last_price"].astype(float)
        )
        prior_ratio = _safe_div(
            df["price_target_3m_ago"].astype(float),
            df["price_3m_ago"].astype(float)
        )
        result["pt_vs_price_momentum"] = _safe_pct_change(current_ratio, prior_ratio)
    
    # === Skewness Trend (Mean vs Median) ===
    if all(c in df.columns for c in ["price_target", "price_target_median", 
                                      "price_target_3m_ago", "price_target_median_3m_ago"]):
        current_skew = df["price_target"].astype(float) - df["price_target_median"].astype(float)
        prior_skew = df["price_target_3m_ago"].astype(float) - df["price_target_median_3m_ago"].astype(float)
        result["pt_skew_trend"] = current_skew - prior_skew
    
    logger.info("Engineered price target dynamics features (15 features)")
    return result


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Calculate percentage change with safe division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (current - previous) / previous.abs().replace(0, pd.NA)
    return result.astype("Float64")

# ... existing code ...
```

---

### Task 2: Update `temporal.py` - Add Fiscal Calendar Features

```python
# ... existing code ...

def engineer_fiscal_calendar_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Engineer fiscal calendar-aware temporal features.
    
    Features computed (8):
    - fiscal_year_progress: Position within fiscal year (0-1)
    - days_to_quarter_end: Days until fiscal quarter end
    - fiscal_half: Fiscal half indicator (1 or 2)
    - reporting_lag_zscore: Normalized reporting lag
    - late_reporter_flag: Late reporting flag (>60 days)
    - days_since_fy_end: Days since fiscal year end
    - days_to_next_fy_end: Days until next fiscal year end
    - earnings_imminent: Within 14 days of earnings
    - pre_earnings_window: Within 30-day pre-earnings window
    
    Args:
        df: Input DataFrame with fiscal date columns
        reference_date: Reference date for calculations (defaults to now)
        
    Returns:
        DataFrame with fiscal calendar features added
    """
    result = df.copy()
    
    if reference_date is None:
        effective_ref_date = pd.Timestamp.now().normalize()
    else:
        effective_ref_date = pd.Timestamp(reference_date).normalize()
    
    # === Fiscal Position Features ===
    if "fiscal_month" in df.columns:
        fiscal_month = df["fiscal_month"].astype(float)
        
        # Position within fiscal year (0-1 scale)
        result["fiscal_year_progress"] = (fiscal_month / 12.0).astype("Float64")
        
        # Days to quarter end (approximate)
        result["days_to_quarter_end"] = (
            (3 - ((fiscal_month - 1) % 3)) * 30
        ).astype("Int64")
        
        # Fiscal half indicator
        result["fiscal_half"] = (
            (fiscal_month > 6).astype(int) + 1
        ).astype("Int64")
    
    # === Reporting Lag Analysis ===
    if "reporting_lag" in df.columns:
        reporting_lag = df["reporting_lag"].astype(float)
        median_lag = reporting_lag.median()
        std_lag = reporting_lag.std()
        
        if std_lag > 0:
            result["reporting_lag_zscore"] = (
                (reporting_lag - median_lag) / std_lag
            ).astype("Float64")
        else:
            result["reporting_lag_zscore"] = pd.Series(0.0, index=df.index, dtype="Float64")
        
        # Late reporter flag (>60 days is concerning)
        result["late_reporter_flag"] = (reporting_lag > 60).astype("boolean")
    
    # === FY End Timing Features ===
    if "fy_end_date" in df.columns:
        fy_end = pd.to_datetime(df["fy_end_date"], errors="coerce")
        
        # Days since FY end
        result["days_since_fy_end"] = (
            (effective_ref_date - fy_end).dt.days
        ).astype("Int64")
        
        # Days to next FY end
        if "next_fy_end_date" in df.columns:
            next_fy = pd.to_datetime(df["next_fy_end_date"], errors="coerce")
            result["days_to_next_fy_end"] = (
                (next_fy - effective_ref_date).dt.days
            ).astype("Int64")
    
    # === Earnings Timing Features ===
    if "next_earnings" in df.columns:
        next_earn = pd.to_datetime(df["next_earnings"], errors="coerce")
        days_to_earnings = (next_earn - effective_ref_date).dt.days
        
        # Earnings imminent flag (within 2 weeks)
        result["earnings_imminent"] = (
            (days_to_earnings <= 14) & (days_to_earnings >= 0)
        ).astype("boolean")
        
        # Pre-earnings window (30 days before - blackout period proxy)
        result["pre_earnings_window"] = (
            (days_to_earnings <= 30) & (days_to_earnings > 0)
        ).astype("boolean")
    
    logger.info("Engineered fiscal calendar features (8 features)")
    return result


def engineer_dividend_timing_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Engineer dividend timing and cycle features.
    
    Features computed (8):
    - days_to_dividend_ex_date: Days until ex-dividend date
    - days_to_dividend_record_date: Days until record date
    - days_to_dividend_payable_date: Days until payable date
    - approaching_ex_date: Within 7 days of ex-date
    - recently_ex_dividend: Went ex within last 7 days
    - dividend_cycle_days: Days in dividend cycle
    - dividend_cycle_position: Position within cycle (0-1)
    - dividend_announcement_recency: Days since announcement
    
    Args:
        df: Input DataFrame with dividend date columns
        reference_date: Reference date for calculations
        
    Returns:
        DataFrame with dividend timing features added
    """
    result = df.copy()
    
    if reference_date is None:
        effective_ref_date = pd.Timestamp.now().normalize()
    else:
        effective_ref_date = pd.Timestamp(reference_date).normalize()
    
    # === Days to Key Dividend Dates ===
    date_mappings = {
        "dividend_record_ex_date": "days_to_dividend_ex_date",
        "dividend_record_record_date": "days_to_dividend_record_date",
        "dividend_record_payable_date": "days_to_dividend_payable_date",
    }
    
    for source_col, target_col in date_mappings.items():
        if source_col in df.columns:
            date_val = pd.to_datetime(df[source_col], errors="coerce")
            result[target_col] = (date_val - effective_ref_date).dt.days.astype("Float64")
    
    # === Ex-Date Proximity Indicators ===
    if "dividend_record_ex_date" in df.columns:
        ex_date = pd.to_datetime(df["dividend_record_ex_date"], errors="coerce")
        days_to_ex = (ex_date - effective_ref_date).dt.days
        
        # Approaching ex-date (within 7 days, future)
        result["approaching_ex_date"] = (
            (days_to_ex > 0) & (days_to_ex <= 7)
        ).astype("boolean")
        
        # Recently went ex-dividend (within last 7 days)
        result["recently_ex_dividend"] = (
            (days_to_ex >= -7) & (days_to_ex < 0)
        ).astype("boolean")
    
    # === Dividend Cycle Position ===
    if "dividend_record_frequency" in df.columns:
        freq_map = {
            "Monthly": 30,
            "Quarterly": 90,
            "Semi-Annual": 180,
            "Semi-Annually": 180,
            "Annual": 365,
            "Annually": 365,
            "n/a": None,
        }
        result["dividend_cycle_days"] = (
            df["dividend_record_frequency"].map(freq_map)
        ).astype("Int64")
        
        # Cycle position (0-1)
        if "days_to_dividend_ex_date" in result.columns:
            cycle_days = result["dividend_cycle_days"].astype(float)
            days_to_ex = result["days_to_dividend_ex_date"].clip(lower=0)
            result["dividend_cycle_position"] = (
                (cycle_days - days_to_ex) / cycle_days.replace(0, pd.NA)
            ).astype("Float64")
    
    # === Announcement Recency ===
    if "dividend_record_announce_date" in df.columns:
        announce_date = pd.to_datetime(df["dividend_record_announce_date"], errors="coerce")
        result["dividend_announcement_recency"] = (
            (effective_ref_date - announce_date).dt.days
        ).astype("Float64")
    
    logger.info("Engineered dividend timing features (8 features)")
    return result

# ... existing code ...
```

---

### Task 3: Update `earnings.py` - Add EPS Trajectory Features

```python
# ... existing code ...

def engineer_eps_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer EPS trajectory features from historical data.
    
    Features computed (10):
    - eps_quarterly_trend: Quarterly EPS trend slope (normalized)
    - eps_quarterly_volatility: Quarterly EPS coefficient of variation
    - eps_yoy_quarterly_growth: Year-over-year quarterly EPS growth
    - eps_qoq_growth: Quarter-over-quarter EPS growth
    - eps_positive_streak: Count of positive EPS quarters
    - eps_cagr_5y: 5-year EPS compound annual growth rate
    - eps_cagr_3y: 3-year EPS compound annual growth rate
    - eps_annual_trend: Annual EPS trend slope (normalized)
    - eps_vs_5y_avg: Current EPS vs 5-year average
    - eps_growth_acceleration: Growth acceleration (3Y - 5Y CAGR)
    
    Args:
        df: Input DataFrame with EPS historical columns
        
    Returns:
        DataFrame with EPS trajectory features added
    """
    result = df.copy()
    
    # === Quarterly EPS Analysis ===
    eps_q_cols = [
        "net_eps_basic_fq", "net_eps_basic_1fqfq", "net_eps_basic_2fqfq",
        "net_eps_basic_3fqfq", "net_eps_basic_4fqfq"
    ]
    
    available_q_cols = [c for c in eps_q_cols if c in df.columns]
    if len(available_q_cols) >= 2:
        eps_q_matrix = df[available_q_cols].astype(float).values
        
        # Quarterly trend slope
        result["eps_quarterly_trend"] = _calculate_normalized_trend(eps_q_matrix)
        
        # Quarterly volatility (CV)
        result["eps_quarterly_volatility"] = _calculate_cv(eps_q_matrix)
        
        # Positive quarters count
        result["eps_positive_streak"] = pd.Series(
            (eps_q_matrix > 0).sum(axis=1), 
            index=df.index, 
            dtype="Int64"
        )
    
    # YoY quarterly growth (current Q vs same Q last year)
    if "net_eps_basic_fq" in df.columns and "net_eps_basic_4fqfq" in df.columns:
        result["eps_yoy_quarterly_growth"] = _safe_pct_change(
            df["net_eps_basic_fq"].astype(float),
            df["net_eps_basic_4fqfq"].astype(float)
        )
    
    # QoQ growth
    if "net_eps_basic_fq" in df.columns and "net_eps_basic_1fqfq" in df.columns:
        result["eps_qoq_growth"] = _safe_pct_change(
            df["net_eps_basic_fq"].astype(float),
            df["net_eps_basic_1fqfq"].astype(float)
        )
    
    # === Annual EPS Analysis ===
    eps_y_cols = [
        "net_eps_basic_fy", "net_eps_basic_1fy", "net_eps_basic_2fy",
        "net_eps_basic_3fy", "net_eps_basic_4fy", "net_eps_basic_5fy"
    ]
    
    available_y_cols = [c for c in eps_y_cols if c in df.columns]
    if len(available_y_cols) >= 3:
        eps_y_matrix = df[available_y_cols].astype(float).values
        
        # Annual trend
        result["eps_annual_trend"] = _calculate_normalized_trend(eps_y_matrix)
        
        # CAGR calculations
        if "net_eps_basic_fy" in df.columns and "net_eps_basic_5fy" in df.columns:
            result["eps_cagr_5y"] = _calculate_cagr(
                df["net_eps_basic_fy"].astype(float),
                df["net_eps_basic_5fy"].astype(float),
                5
            )
        
        if "net_eps_basic_fy" in df.columns and "net_eps_basic_3fy" in df.columns:
            result["eps_cagr_3y"] = _calculate_cagr(
                df["net_eps_basic_fy"].astype(float),
                df["net_eps_basic_3fy"].astype(float),
                3
            )
        
        # Current vs 5Y average
        eps_5y_avg = np.nanmean(eps_y_matrix, axis=1)
        if "net_eps_basic_fy" in df.columns:
            result["eps_vs_5y_avg"] = (
                df["net_eps_basic_fy"].astype(float) / 
                pd.Series(eps_5y_avg, index=df.index).replace(0, pd.NA) - 1
            ).astype("Float64")
    
    # === Growth Acceleration ===
    if "eps_cagr_3y" in result.columns and "eps_cagr_5y" in result.columns:
        result["eps_growth_acceleration"] = (
            result["eps_cagr_3y"] - result["eps_cagr_5y"]
        )
    
    logger.info("Engineered EPS trajectory features (10 features)")
    return result


def _calculate_normalized_trend(matrix: np.ndarray) -> pd.Series:
    """Calculate normalized trend slope across time periods."""
    n_periods = matrix.shape[1]
    x = np.arange(n_periods)
    
    slopes = []
    for row in matrix:
        valid_mask = ~np.isnan(row)
        if valid_mask.sum() < 2:
            slopes.append(np.nan)
            continue
        y = row[valid_mask]
        x_valid = x[valid_mask]
        slope = np.polyfit(x_valid, y, 1)[0]
        mean_abs = np.abs(y).mean()
        slopes.append(slope / mean_abs if mean_abs > 0 else 0)
    
    return pd.Series(slopes, dtype="Float64")


def _calculate_cv(matrix: np.ndarray) -> pd.Series:
    """Calculate coefficient of variation for each row."""
    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.nanmean(matrix, axis=1)
        stds = np.nanstd(matrix, axis=1)
        cv = np.abs(stds / means)
    return pd.Series(cv, dtype="Float64")


def _calculate_cagr(end_val: pd.Series, start_val: pd.Series, years: int) -> pd.Series:
    """Calculate compound annual growth rate."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = end_val / start_val.replace(0, pd.NA)
        cagr = np.sign(ratio) * (np.abs(ratio) ** (1/years)) - 1
    return cagr.astype("Float64")


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Calculate percentage change with safe division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (current - previous) / previous.abs().replace(0, pd.NA)
    return result.astype("Float64")

# ... existing code ...
```

---

### Task 4: Update `leverage.py` - Add Cash Flow Temporal Features

```python
# ... existing code ...

def engineer_cashflow_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer cash flow temporal pattern features.
    
    Features computed (12):
    - fcf_quarterly_trend: FCF trend across last 5 quarters
    - fcf_quarterly_volatility: FCF coefficient of variation
    - fcf_positive_ratio: Ratio of positive FCF quarters
    - cfo_quarterly_trend: CFO trend across quarters
    - cfo_yoy_quarterly: CFO year-over-year quarterly growth
    - investment_intensity_trend: CFI investment intensity trend
    - cfo_5y_trend: CFO 5-year trend slope
    - cfo_5y_stability: CFO 5-year stability (1 - CV)
    - cfo_margin_current: Current CFO margin
    - cfo_margin_trend: CFO margin trend
    - acquisition_activity_trend: Cash acquisition trend
    - acquisition_quarters_active: Active acquisition quarter count
    
    Args:
        df: Input DataFrame with cash flow historical columns
        
    Returns:
        DataFrame with cash flow temporal features added
    """
    result = df.copy()
    
    # === Quarterly FCF Analysis ===
    fcf_cols = ["fcf_fq", "fcf_1fqfq", "fcf_2fqfq", "fcf_3fqfq", "fcf_4fqfq"]
    available_fcf = [c for c in fcf_cols if c in df.columns]
    
    if len(available_fcf) >= 2:
        fcf_matrix = df[available_fcf].astype(float).values
        
        result["fcf_quarterly_trend"] = _calculate_trend_slope(fcf_matrix)
        result["fcf_quarterly_volatility"] = _calculate_cv(fcf_matrix)
        result["fcf_positive_ratio"] = pd.Series(
            (fcf_matrix > 0).sum(axis=1) / len(available_fcf),
            index=df.index,
            dtype="Float64"
        )
    
    # === Quarterly CFO Analysis ===
    cfo_cols = ["cfo_fq", "cfo_1fqfq", "cfo_2fqfq", "cfo_3fqfq", "cfo_4fqfq"]
    available_cfo = [c for c in cfo_cols if c in df.columns]
    
    if len(available_cfo) >= 2:
        cfo_matrix = df[available_cfo].astype(float).values
        result["cfo_quarterly_trend"] = _calculate_trend_slope(cfo_matrix)
        
        # YoY quarterly CFO
        if "cfo_fq" in df.columns and "cfo_4fqfq" in df.columns:
            result["cfo_yoy_quarterly"] = _safe_pct_change(
                df["cfo_fq"].astype(float),
                df["cfo_4fqfq"].astype(float)
            )
    
    # === CFI Investment Pattern ===
    cfi_cols = ["cfi_fq", "cfi_1fqfq", "cfi_2fqfq", "cfi_3fqfq", "cfi_4fqfq"]
    available_cfi = [c for c in cfi_cols if c in df.columns]
    
    if len(available_cfi) >= 2:
        cfi_matrix = df[available_cfi].astype(float).values
        # CFI is typically negative; invert for investment intensity
        result["investment_intensity_trend"] = _calculate_trend_slope(-cfi_matrix)
    
    # === Multi-Year CFO Patterns ===
    cfo_annual = ["cfo_fy", "cfo_1fy", "cfo_2fy", "cfo_3fy", "cfo_4fy"]
    available_annual = [c for c in cfo_annual if c in df.columns]
    
    if len(available_annual) >= 3:
        cfo_annual_matrix = df[available_annual].astype(float).values
        result["cfo_5y_trend"] = _calculate_trend_slope(cfo_annual_matrix)
        cv = _calculate_cv(cfo_annual_matrix)
        result["cfo_5y_stability"] = (1 - cv.clip(0, 1)).astype("Float64")
    
    # === CFO Margin Analysis ===
    if "cfo_ltm" in df.columns and "total_revenues_ltm" in df.columns:
        result["cfo_margin_current"] = _safe_div(
            df["cfo_ltm"].astype(float),
            df["total_revenues_ltm"].astype(float)
        )
    
    if "cfo_1fy" in df.columns and "total_revenues_1fy" in df.columns:
        cfo_margin_1fy = _safe_div(
            df["cfo_1fy"].astype(float),
            df["total_revenues_1fy"].astype(float)
        )
        if "cfo_margin_current" in result.columns:
            result["cfo_margin_trend"] = result["cfo_margin_current"] - cfo_margin_1fy
    
    # === Acquisition Activity ===
    acq_cols = [
        "cash_acquisitions_fq", "cash_acquisitions_1fqfq",
        "cash_acquisitions_2fqfq", "cash_acquisitions_3fqfq", "cash_acquisitions_4fqfq"
    ]
    available_acq = [c for c in acq_cols if c in df.columns]
    
    if len(available_acq) >= 2:
        acq_matrix = df[available_acq].astype(float).values
        result["acquisition_activity_trend"] = _calculate_trend_slope(acq_matrix)
        result["acquisition_quarters_active"] = pd.Series(
            (acq_matrix != 0).sum(axis=1),
            index=df.index,
            dtype="Int64"
        )
    
    logger.info("Engineered cash flow temporal features (12 features)")
    return result


def _calculate_trend_slope(matrix: np.ndarray) -> pd.Series:
    """Calculate normalized trend slope."""
    n = matrix.shape[1]
    x = np.arange(n)
    
    slopes = []
    for row in matrix:
        valid = ~np.isnan(row)
        if valid.sum() < 2:
            slopes.append(np.nan)
            continue
        coef = np.polyfit(x[valid], row[valid], 1)
        mean_abs = np.abs(row[valid]).mean()
        slopes.append(coef[0] / mean_abs if mean_abs > 0 else 0)
    
    return pd.Series(slopes, dtype="Float64")


def _calculate_cv(matrix: np.ndarray) -> pd.Series:
    """Calculate coefficient of variation."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return pd.Series(
            np.abs(np.nanstd(matrix, axis=1) / np.nanmean(matrix, axis=1)),
            dtype="Float64"
        )


def _safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Safe percentage change calculation."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return ((current - previous) / previous.abs().replace(0, pd.NA)).astype("Float64")

# ... existing code ...
```

---

### Task 5: Update `__init__.py` - Registry Updates

```python
# ... existing code ...
"""
Advanced feature engineering - modular implementation.

Phase 9.3 Feature Engineering Registry (v1.14)
Total: 349 features across 21 categories

Categories (Updated):
- Momentum & Technical (25): EMA crossovers, RSI, 52W High/Low, price momentum
- Valuation Ratios (25): P/E, P/B, EV/EBITDA, EV/Sales, PEG, valuation trends
- Profitability (16): Operating margin, net margin, ROE, ROA, ROIC
- Quality & Risk (18): Altman Z-Score, Piotroski F-Score, accruals ratio
- Cash Flow (17): FCF yield, OCF/Sales, cash conversion, temporal patterns  # +12
- Capital Allocation (23): Buyback yield, dividend coverage, payout ratios
- Analyst Sentiment (25): Rating changes, target revisions, PT dynamics  # +15
- Market Sentiment (5): Relative strength, volume trends, beta stability
- Leverage & Liquidity (9): Debt ratios, current ratio, interest coverage
- Temporal Patterns (25): Seasonality, fiscal calendar, quarter-end  # +8
- Composite Scores (5): Combined quality, value, momentum scores
- Growth Metrics (9): Revenue growth, EBITDA growth, earnings CAGR
- Efficiency Ratios (4): Asset turnover, inventory turnover, receivables days
- Employee Productivity (21): Revenue per employee, productivity trends
- Balance Sheet Dynamics (9): Working capital trends, asset quality
- Revenue Forecasting (9): Analyst estimate spreads, revision momentum
- Earnings Quality (43): Estimated vs. Actual, GAAP vs. Adjusted, EPS trajectory  # +10
- Technical Analysis (15): RSI, 52-week range, volume momentum
- Valuation Timeseries (16): Multi-period valuation trends, mean reversion
- Dividend Reliability (20): Consistency, coverage, dividend timing  # +8
- Employment Dynamics (10): Workforce volatility, hiring intensity
"""

# ... existing imports ...

from .sentiment import (
    engineer_analyst_quality_features,
    engineer_market_sentiment_features,
    engineer_price_target_dynamics,  # NEW
)
from .temporal import (
    engineer_temporal_features,
    engineer_fiscal_calendar_features,  # NEW
    engineer_dividend_timing_features,  # NEW
)
from .leverage import (
    engineer_leverage_ratios,
    engineer_liquidity_ratios,
    engineer_efficiency_ratios,
    engineer_balance_sheet_trends,
    engineer_cashflow_temporal_features,  # NEW
)
from .earnings import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics,
    engineer_eps_trajectory_features,  # NEW
)

# ... existing code ...

# Feature Registry for Auto-discovery (Phase 9.3 v1.14)
FEATURE_REGISTRY = {
    # ... existing entries ...
    
    # NEW: Price Target Dynamics (extends Analyst Sentiment)
    "price_target_dynamics": {
        "function": engineer_price_target_dynamics,
        "category": "Analyst Sentiment",
        "feature_count": 15,
    },
    
    # NEW: Cash Flow Temporal (extends Cash Flow)
    "cashflow_temporal": {
        "function": engineer_cashflow_temporal_features,
        "category": "Cash Flow",
        "feature_count": 12,
    },
    
    # NEW: EPS Trajectory (extends Earnings Quality)
    "eps_trajectory": {
        "function": engineer_eps_trajectory_features,
        "category": "Earnings Quality",
        "feature_count": 10,
    },
    
    # NEW: Fiscal Calendar (extends Temporal Patterns)
    "fiscal_calendar": {
        "function": engineer_fiscal_calendar_features,
        "category": "Temporal Patterns",
        "feature_count": 8,
    },
    
    # NEW: Dividend Timing (extends Dividend Reliability)
    "dividend_timing": {
        "function": engineer_dividend_timing_features,
        "category": "Dividend Reliability",
        "feature_count": 8,
    },
    
    # ... existing entries ...
}

# ... existing code ...

__all__ = [
    # ... existing exports ...
    # NEW exports
    "engineer_price_target_dynamics",
    "engineer_cashflow_temporal_features",
    "engineer_eps_trajectory_features",
    "engineer_fiscal_calendar_features",
    "engineer_dividend_timing_features",
]
```

---

## Actionable Implementation Checklist

### Phase 1: Schema Updates

- [x] **Task 1.1**: Add 53 new feature definitions to `COLUMN_SCHEMA` in `schema.py`
- [x] **Task 1.2**: Update `PHASE93_FEATURE_CATEGORIES` with new features in existing categories
- [x] **Task 1.3**: Update module docstring to reflect v1.14 and 349 total features
- [x] **Task 1.4**: Run schema validation tests

### Phase 2: Module Implementation

- [x] **Task 2.1**: Implement `engineer_price_target_dynamics()` in `sentiment.py`
- [x] **Task 2.2**: Implement `engineer_fiscal_calendar_features()` in `temporal.py`
- [x] **Task 2.3**: Implement `engineer_dividend_timing_features()` in `temporal.py`
- [x] **Task 2.4**: Implement `engineer_eps_trajectory_features()` in `earnings.py`
- [x] **Task 2.5**: Implement `engineer_cashflow_temporal_features()` in `leverage.py`

### Phase 3: Registry Integration

- [x] **Task 3.1**: Update `FEATURE_REGISTRY` in `__init__.py` with 5 new entries
- [x] **Task 3.2**: Update `__all__` exports in `__init__.py`
- [x] **Task 3.3**: Update module docstring with v1.14 counts

### Phase 4: Testing & Validation

- [x] **Task 4.1**: Create unit tests for each new function
- [x] **Task 4.2**: Verify feature counts match schema definitions
- [x] **Task 4.3**: Run integration tests with sample data
- [x] **Task 4.4**: Validate feature output dtypes match schema

### Phase 5: Documentation

- [x] **Task 5.1**: Update `CHANGELOG.md` with v1.14 enhancements
- [x] **Task 5.2**: Update any relevant README documentation and code guidelines