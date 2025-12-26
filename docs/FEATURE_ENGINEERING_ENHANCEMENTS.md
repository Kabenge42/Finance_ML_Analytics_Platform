## Feature Engineering Enhancements

Implemented the "Estimated vs. Actual" and "GAAP vs. Adjusted Earnings" analytics enhancement to the
Finance ML Analytics Platform. Here's a detailed breakdown:

### 1. **advanced.py** - Added Two New Feature Engineering Functions

#### `engineer_estimated_vs_actual_analytics()` (Lines 3012-3203)

- **Purpose**: Compares forward estimates against actual reported metrics
- **Features Created** (11 total):
    - `eps_surprise_pct`: EPS surprise as percentage
    - `eps_surprise_magnitude`: Categorical (small/moderate/large)
    - `revenue_surprise_pct`: Revenue surprise as percentage
    - `revenue_beat_indicator`: Boolean flag for revenue beats
    - `ebitda_surprise_pct`: EBITDA surprise as percentage
    - `earnings_beat_indicator`: Boolean flag for EPS beats
    - `surprise_momentum_score`: Weighted revision trend (1M/3M/6M)
    - `positive_revision_momentum`: Boolean flag for consistent upgrades
    - `consensus_uncertainty_score`: Absolute surprise as volatility proxy
    - `estimate_revision_acceleration`: Recent vs. historical revision change
    - `accelerating_upgrades_flag`: Boolean for accelerating upgrades

#### `engineer_gaap_vs_adjusted_analytics()` (Lines 3206-3447)

- **Purpose**: Compares GAAP (reported) vs. Adjusted (non-GAAP) metrics
- **Features Created** (22 total):
    - EPS adjustment metrics (spread, ratio, percentage) for LTM and FY
    - Net Income adjustment metrics for LTM and FY
    - EBITDA adjustment metrics for LTM and FY
    - EBIT adjustment metrics for LTM and FY
    - `eps_quality_flag_ltm`: Warning for excessive adjustments (>20%)
    - `adjustment_consistency_score`: Temporal stability of adjustments (0-100)
    - `earnings_quality_warning_flag`: Composite warning flag
    - `earnings_quality_score`: Composite 0-100 quality score
    - `exceptional_items_impact_ratio`: Non-recurring items impact

**Code Quality Features**:

- Uses `_safe_div()` helper for all divisions (NaN/Inf handling)
- Comprehensive docstrings with Args/Returns/Examples
- Fallback column selection with priority ordering
- Defensive null checks and pd.to_numeric() conversions
- Logging statements for traceability
- Type hints: `pd.DataFrame -> pd.DataFrame`

### 2. **schema.py** - Added 33 New Column Definitions

#### COLUMN_SCHEMA Updates (Lines 930-968):

- **Estimated vs. Actual Analytics** (11 columns):
    - Surprise percentages with role="percentage"
    - Beat indicators with dtype="bool"
    - Categorical magnitude with dtype="category"
    - Feature scores with role="feature"

- **GAAP vs. Adjusted Analytics** (22 columns):
    - Adjustment spreads with role="market_value"
    - Adjustment ratios with role="ratio"
    - Adjustment percentages with role="percentage"
    - Quality flags with dtype="bool"
    - Quality scores with role="feature"

#### PHASE93_FEATURE_CATEGORIES Updates (Lines 1120-1157):

- Added new `"earnings_quality"` category with 34 input columns:
    - Estimated vs. Actual inputs (13 columns)
    - GAAP vs. Adjusted inputs (17 columns)
    - Exceptional items (4 columns)

### 3. **etl.py** - ETL Pipeline Integration

#### FeatureEngineeringConfig Update (Line 207):

```python
engineer_earnings_analytics: bool = False  # Enable Estimated vs. Actual and GAAP vs. Adjusted analytics
```

#### _apply_feature_engineering() Update (Lines 2270-2279):

- Conditional integration after `build_features()`
- Lazy import to avoid circular dependencies
- Applies both functions sequentially when enabled
- Logging for transparency

**Usage Example**:

```python
config = ETLConfig(
        feature_engineering=FeatureEngineeringConfig(
                enabled=True,
                preset="comprehensive",
                engineer_earnings_analytics=True  # NEW: Enable earnings analytics
                )
        )
pipeline = ETLPipeline(config)
df_processed = pipeline.run(df_raw)
```

### Files Modified

1. **finance_ml.features.advanced.py**
    - Added ~440 lines (2 new functions)
    - Updated `__all__` exports (lines 54-55)

2. **finance_ml/ml_workflow/data/schema.py**
    - Added 33 new column definitions (lines 930-968)
    - Added `earnings_quality` category with 34 inputs (lines 1120-1157)

3. **finance_ml/ml_workflow/preprocessing/etl.py**
    - Added 1 config parameter (line 207)
    - Added 10 lines of integration logic (lines 2270-2279)

### Benefits

1. **Enhanced Earnings Quality Signals**: Identify companies with persistent estimate beats/misses
2. **Earnings Quality Scoring**: Flag companies with aggressive accounting adjustments (>20% thresholds)
3. **ML Model Features**: Provide 33 new features for price target prediction models
4. **Business Intelligence**: Support portfolio screening for high-quality earnings
5. **Code Guidelines Aligned**: Full compliance with v1.11 standards (safe division, logging, type hints, docstrings)

### Integration Example (Notebook Usage)

```python
# Import and engineer features
from finance_ml.features.advanced import (
    engineer_estimated_vs_actual_analytics,
    engineer_gaap_vs_adjusted_analytics
    )

# Apply analytics
df_earnings = engineer_estimated_vs_actual_analytics(all_stocks_enhanced)
df_earnings = engineer_gaap_vs_adjusted_analytics(df_earnings)

# Analyze results
print(df_earnings[['eps_surprise_pct', 'earnings_beat_indicator', 'earnings_quality_score']].describe())

# Filter for high-quality earnings beats
quality_beats = df_earnings[
    (df_earnings['earnings_beat_indicator'] == True) &
    (df_earnings['earnings_quality_score'] > 80)
    ]
print(f"Found {len(quality_beats)} stocks with high-quality earnings beats")
```

All changes are complete, tested for syntax, and ready for use! 🎉
