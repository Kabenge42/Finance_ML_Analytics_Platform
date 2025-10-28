# Phase 9.7 Implementation Summary: Identification of Under/Overvalued Stocks with Visualization

**Date**: 2025-10-29  
**Phase**: 9.7 - Identification of Under/Overvalued Stocks with Visualization  
**Status**: ✅ Complete  
**Test Coverage**: 26/26 tests passing (100%)

---

## Executive Summary

Successfully implemented Phase 9.7: Identification of Under/Overvalued Stocks with Visualization using strict
Test-Driven Development (TDD). Added comprehensive valuation analysis capabilities to `finance_ml.eval` module,
including:

- **Valuation category assignment** (Strong Buy, Buy, Hold, Sell, Strong Sell)
- **Sector-relative metrics** (z-scores and percentile ranks within sectors)
- **Multi-factor scoring** (combining valuation, quality, and growth)
- **Advanced filtering** (by sector, region, market cap, mispricing score)
- **Enhanced visualizations** (interactive scatter plots with valuation categories)

All functionality is fully tested with 26 comprehensive unit tests covering edge cases, integration scenarios, and
proper error handling.

---

## Implementation Details

### 1. Functions Implemented

#### 1.1 `assign_valuation_category(mispricing_scores, thresholds=None)`

**Location**: `finance_ml/eval.py` lines 1456-1506  
**Purpose**: Assign valuation categories based on mispricing scores

**Categories**:

- **Strong Buy**: mispricing > 20% (default threshold)
- **Buy**: mispricing 10% to 20%
- **Hold**: mispricing -10% to +10%
- **Sell**: mispricing -20% to -10%
- **Strong Sell**: mispricing < -20%

**Key Features**:

- Customizable thresholds via optional dict parameter
- Handles missing values (returns 'Unknown')
- Returns pandas Series compatible with DataFrames

**Example Usage**:

```python
from finance_ml.eval import assign_valuation_category

mispricing = df['mispricing_score']
categories = assign_valuation_category(mispricing)

# With custom thresholds (for volatile sectors)
custom_thresholds = {
    'strong_buy': 30,
    'buy': 15,
    'sell': 15,
    'strong_sell': 30
}
categories = assign_valuation_category(mispricing, thresholds=custom_thresholds)

df['valuation_category'] = categories
```

#### 1.2 `calculate_sector_zscores(df, metrics, sector_col='sector')`

**Location**: `finance_ml/eval.py` lines 1509-1552  
**Purpose**: Calculate z-scores for metrics within each sector

**Formula**: `z-score = (value - sector_mean) / sector_std`

**Returns**: DataFrame with original columns plus `{metric}_zscore` columns

**Key Features**:

- Calculates z-scores independently within each sector
- Z-scores have mean ≈ 0 and std ≈ 1 within each sector
- Identifies stocks trading at premium/discount to sector peers
- Handles missing values gracefully

**Example Usage**:

```python
from finance_ml.eval import calculate_sector_zscores

# Calculate z-scores for valuation metrics
df_with_zscores = calculate_sector_zscores(
        df,
        metrics=['pe_ratio', 'pb_ratio', 'ev_ebitda'],
        sector_col='sector'
        )

# Identify stocks trading at discount (negative z-score)
cheap_stocks = df_with_zscores[df_with_zscores['pe_ratio_zscore'] < -1.0]
```

#### 1.3 `calculate_percentile_ranks(df, metrics, sector_col='sector')`

**Location**: `finance_ml/eval.py` lines 1555-1594  
**Purpose**: Calculate percentile ranks for metrics within each sector

**Returns**: DataFrame with original columns plus `{metric}_percentile` columns (0-100 scale)

**Key Features**:

- Percentile rank indicates percentage of sector peers outperformed
- Calculated independently within each sector
- 0-100 scale (0 = worst in sector, 100 = best in sector)

**Example Usage**:

```python
from finance_ml.eval import calculate_percentile_ranks

# Calculate percentile ranks
df_with_percentiles = calculate_percentile_ranks(
        df,
        metrics=['pe_ratio', 'pb_ratio', 'roe', 'revenue_growth']
        )

# Find stocks in top 10% of sector by ROE
top_performers = df_with_percentiles[
    df_with_percentiles['roe_percentile'] >= 90
    ]
```

#### 1.4 `calculate_multi_factor_score(df, valuation_col, quality_cols, growth_cols, weights=None)`

**Location**: `finance_ml/eval.py` lines 1597-1680  
**Purpose**: Calculate composite multi-factor score combining valuation, quality, and growth

**Score Formula**: `weighted_valuation + weighted_quality + weighted_growth`

**Default Weights**: `{'valuation': 0.4, 'quality': 0.3, 'growth': 0.3}`

**Key Features**:

- Each component normalized to z-scores before weighting
- Customizable weights for different investment strategies
- Handles missing metrics gracefully
- Higher score = better investment opportunity

**Example Usage**:

```python
from finance_ml.eval import calculate_multi_factor_score

# Standard multi-factor score
multi_factor_scores = calculate_multi_factor_score(
        df,
        valuation_col='mispricing_score',
        quality_cols=['roe', 'ebitda_margin', 'debt_to_equity'],
        growth_cols=['revenue_growth', 'earnings_growth']
        )

# Value-focused strategy (emphasize valuation)
value_scores = calculate_multi_factor_score(
        df,
        valuation_col='mispricing_score',
        quality_cols=['roe'],
        growth_cols=['revenue_growth'],
        weights={'valuation': 0.6, 'quality': 0.2, 'growth': 0.2}
        )

df['multi_factor_score'] = multi_factor_scores
```

#### 1.5
`filter_stocks_by_criteria(df, sectors, regions, min_market_cap, max_market_cap, min_mispricing, max_mispricing, valuation_categories)`

**Location**: `finance_ml/eval.py` lines 1683-1754  
**Purpose**: Filter stocks based on multiple criteria

**Filter Options**:

- **sectors**: List of sectors to include
- **regions**: List of regions to include
- **min_market_cap**, **max_market_cap**: Market cap range
- **min_mispricing**, **max_mispricing**: Mispricing score range
- **valuation_categories**: List of categories to include

**Key Features**:

- Apply multiple filters simultaneously
- All filters are optional (None = no filter)
- Returns filtered DataFrame

**Example Usage**:

```python
from finance_ml.eval import filter_stocks_by_criteria

# Find large-cap tech stocks that are undervalued
tech_opportunities = filter_stocks_by_criteria(
        df,
        sectors=['Tech', 'Communication Services'],
        regions=['US', 'EU'],
        min_market_cap=10e9,  # $10B+
        min_mispricing=10.0,  # At least 10% undervalued
        valuation_categories=['Strong Buy', 'Buy']
        )

# Find small-cap value plays across all sectors
small_cap_value = filter_stocks_by_criteria(
        df,
        max_market_cap=2e9,  # Under $2B
        min_mispricing=15.0
        )
```

#### 1.6 `create_valuation_scatter_plot(df, out_path=None, color_by='sector')`

**Location**: `finance_ml/eval.py` lines 1757-1843  
**Purpose**: Create interactive scatter plot of current price vs. predicted target

**Returns**: Plotly figure object (or None if plotly unavailable)

**Key Features**:

- Interactive scatter plot with hover data
- Color by sector, region, or valuation category
- Diagonal reference line (y=x) for fair value
- Saves to HTML file if out_path provided
- Hover shows ticker, mispricing score, valuation category

**Example Usage**:

```python
from finance_ml.eval import create_valuation_scatter_plot
from pathlib import Path

# Create scatter plot colored by sector
fig = create_valuation_scatter_plot(
    df,
    out_path=Path('outputs/valuation_scatter.html'),
    color_by='sector'
)

# Create scatter plot colored by valuation category
fig_cat = create_valuation_scatter_plot(
    df,
    out_path=Path('outputs/valuation_by_category.html'),
    color_by='valuation_category'
)

# Display in Jupyter
fig.show()
```

---

## Test Coverage

### Test File: `tests/test_valuation_phase97.py`

**Total Tests**: 26  
**Passing**: 26 (100%)  
**Lines**: 435

### Test Classes and Coverage:

1. **TestAssignValuationCategory** (7 tests)
    - Return type validation
    - Strong Buy assignment (>20%)
    - Buy assignment (10-20%)
    - Hold assignment (-10% to +10%)
    - Sell assignment (-20% to -10%)
    - Strong Sell assignment (<-20%)
    - Custom thresholds support

2. **TestCalculateSectorZScores** (4 tests)
    - DataFrame return type
    - Z-score columns present
    - Mean ≈ 0 within each sector
    - Std ≈ 1 within each sector
    - Missing value handling

3. **TestCalculatePercentileRanks** (3 tests)
    - DataFrame return type
    - Percentile columns present
    - Range validation (0-100)
    - Within-sector calculation

4. **TestMultiFactorScore** (3 tests)
    - Series return type
    - Higher scores for better stocks
    - Custom weights support

5. **TestFilterByCriteria** (5 tests)
    - Filter by sector
    - Filter by region
    - Filter by market cap range
    - Filter by mispricing threshold
    - Combined criteria filtering

6. **TestCreateValuationScatterPlot** (3 tests)
    - HTML file creation
    - Figure object return
    - Color by category support

7. **TestIntegrationValuationWorkflow** (1 test)
    - End-to-end Phase 9.7 pipeline
    - All components working together

---

## Notebook Integration Examples

### Cell 1: Import Phase 9.7 Functions

```python
# Phase 9.7: Identification of Under/Overvalued Stocks with Visualization
from finance_ml.eval import (
    assign_valuation_category,
    calculate_sector_zscores,
    calculate_percentile_ranks,
    calculate_multi_factor_score,
    filter_stocks_by_criteria,
    create_valuation_scatter_plot,
)
from pathlib import Path
```

### Cell 2: Assign Valuation Categories

```python
# Assign valuation categories based on mispricing scores
valuation_categories = assign_valuation_category(all_stocks['mispricing_score'])
all_stocks['valuation_category'] = valuation_categories

print("=" * 60)
print("VALUATION CATEGORY DISTRIBUTION")
print("=" * 60)
print(all_stocks['valuation_category'].value_counts().sort_index())
print("=" * 60)

# Show sample stocks from each category
for category in ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']:
    stocks = all_stocks[all_stocks['valuation_category'] == category]
    if len(stocks) > 0:
        print(f"\n{category} Examples (Top 3):")
        display_cols = ['ticker', 'sector', 'last_price', 'predicted_target', 'mispricing_score']
        display_cols = [c for c in display_cols if c in stocks.columns]
        print(stocks.nlargest(3, 'mispricing_score')[display_cols].to_string(index=False))
```

### Cell 3: Calculate Sector-Relative Metrics

```python
# Calculate z-scores and percentile ranks within sectors
all_stocks = calculate_sector_zscores(
    all_stocks,
    metrics=['pe_ratio', 'pb_ratio', 'ev_ebitda'] if 'pe_ratio' in all_stocks.columns else []
)

all_stocks = calculate_percentile_ranks(
    all_stocks,
    metrics=['pe_ratio', 'pb_ratio', 'roe', 'revenue_growth']
)

print("\nSECTOR-RELATIVE METRICS SAMPLE:")
print("=" * 60)
sample_cols = ['ticker', 'sector', 'pe_ratio', 'pe_ratio_zscore', 'pe_ratio_percentile']
sample_cols = [c for c in sample_cols if c in all_stocks.columns]
print(all_stocks[sample_cols].head(10).to_string(index=False))
print("=" * 60)

# Identify stocks trading at significant discount to sector
if 'pe_ratio_zscore' in all_stocks.columns:
    cheap_stocks = all_stocks[all_stocks['pe_ratio_zscore'] < -1.5]
    print(f"\nStocks trading >1.5 std dev below sector P/E: {len(cheap_stocks)}")
```

### Cell 4: Calculate Multi-Factor Scores

```python
# Calculate composite multi-factor scores
quality_cols = [c for c in ['roe', 'ebitda_margin', 'net_margin'] if c in all_stocks.columns]
growth_cols = [c for c in ['revenue_growth', 'earnings_growth'] if c in all_stocks.columns]

multi_factor_scores = calculate_multi_factor_score(
        all_stocks,
        valuation_col='mispricing_score',
        quality_cols=quality_cols,
        growth_cols=growth_cols,
        weights={'valuation': 0.4, 'quality': 0.3, 'growth': 0.3}
        )

all_stocks['multi_factor_score'] = multi_factor_scores

print("\nMULTI-FACTOR SCORES (Top 20 Investment Opportunities):")
print("=" * 80)
top_opportunities = all_stocks.nlargest(20, 'multi_factor_score')
display_cols = ['ticker', 'sector', 'valuation_category', 'mispricing_score', 'multi_factor_score']
display_cols = [c for c in display_cols if c in top_opportunities.columns]
print(top_opportunities[display_cols].to_string(index=False))
print("=" * 80)
```

### Cell 5: Advanced Filtering - Find Best Opportunities

```python
# Filter for high-quality undervalued stocks
strong_buy_candidates = filter_stocks_by_criteria(
    all_stocks,
    sectors=['Tech', 'Healthcare', 'Finance'],  # Focus sectors
    regions=['US', 'EU'],  # Developed markets
    min_market_cap=5e9,  # $5B+ (stability)
    min_mispricing=15.0,  # At least 15% undervalued
    valuation_categories=['Strong Buy', 'Buy']
)

print(f"\nSTRONG BUY CANDIDATES: {len(strong_buy_candidates)} stocks found")
print("=" * 80)

if len(strong_buy_candidates) > 0:
    # Sort by multi-factor score
    strong_buy_candidates = strong_buy_candidates.sort_values('multi_factor_score', ascending=False)
    
    display_cols = ['ticker', 'sector', 'region', 'market_cap', 'mispricing_score', 
                    'valuation_category', 'multi_factor_score']
    display_cols = [c for c in display_cols if c in strong_buy_candidates.columns]
    
    print("\nTop 15 Strong Buy Candidates:")
    print(strong_buy_candidates.head(15)[display_cols].to_string(index=False))
    print("=" * 80)
```

### Cell 6: Sector-Specific Opportunities

```python
# Find top opportunities per sector
sectors = all_stocks['sector'].unique()
print("\nTOP OPPORTUNITIES BY SECTOR")
print("=" * 80)

for sector in sorted(sectors):
    sector_stocks = filter_stocks_by_criteria(
        all_stocks,
        sectors=[sector],
        min_mispricing=5.0  # At least 5% undervalued
    )
    
    if len(sector_stocks) > 0:
        top_3 = sector_stocks.nlargest(3, 'multi_factor_score')
        print(f"\n{sector}:")
        display_cols = ['ticker', 'mispricing_score', 'valuation_category', 'multi_factor_score']
        display_cols = [c for c in display_cols if c in top_3.columns]
        for _, row in top_3.iterrows():
            vals = [f"{row[c]:.2f}" if isinstance(row[c], (int, float)) else str(row[c]) 
                   for c in display_cols]
            print(f"  {' | '.join(vals)}")

print("=" * 80)
```

### Cell 7: Create Interactive Valuation Visualizations

```python
# Create interactive scatter plot colored by sector
output_dir = Path('outputs/phase97_visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

print("Creating interactive valuation scatter plots...")

# Plot 1: Color by sector
fig_sector = create_valuation_scatter_plot(
    all_stocks,
    out_path=output_dir / 'valuation_scatter_by_sector.html',
    color_by='sector'
)
print(f"  ✓ Saved: {output_dir / 'valuation_scatter_by_sector.html'}")

# Plot 2: Color by valuation category
fig_category = create_valuation_scatter_plot(
    all_stocks,
    out_path=output_dir / 'valuation_scatter_by_category.html',
    color_by='valuation_category'
)
print(f"  ✓ Saved: {output_dir / 'valuation_scatter_by_category.html'}")

# Plot 3: Color by region (if available)
if 'region' in all_stocks.columns:
    fig_region = create_valuation_scatter_plot(
        all_stocks,
        out_path=output_dir / 'valuation_scatter_by_region.html',
        color_by='region'
    )
    print(f"  ✓ Saved: {output_dir / 'valuation_scatter_by_region.html'}")

# Display in notebook
print("\nDisplaying interactive plot (colored by valuation category)...")
fig_category.show()
```

### Cell 8: Export Top Opportunities to CSV

```python
# Export filtered opportunities for further analysis
output_dir = Path('outputs/phase97_opportunities')
output_dir.mkdir(parents=True, exist_ok=True)

# Export Strong Buy candidates
if len(strong_buy_candidates) > 0:
    strong_buy_path = output_dir / 'strong_buy_candidates.csv'
    strong_buy_candidates.to_csv(strong_buy_path, index=False)
    print(f"✓ Exported {len(strong_buy_candidates)} Strong Buy candidates to {strong_buy_path}")

# Export top opportunities by sector
for sector in sectors:
    sector_opps = filter_stocks_by_criteria(
        all_stocks,
        sectors=[sector],
        min_mispricing=10.0
    ).nlargest(20, 'multi_factor_score')
    
    if len(sector_opps) > 0:
        sector_file = output_dir / f'top_opportunities_{sector.replace(" ", "_")}.csv'
        sector_opps.to_csv(sector_file, index=False)
        print(f"✓ Exported {len(sector_opps)} opportunities for {sector}")

print("\nAll exports complete!")
```

---

## Phase 9.7 Requirements Coverage

### ✅ Valuation Category Assignment

- [x] Strong Buy (>20% undervalued)
- [x] Buy (10-20% undervalued)
- [x] Hold (-10% to +10%)
- [x] Sell (-20% to -10% overvalued)
- [x] Strong Sell (<-20% overvalued)
- [x] Customizable thresholds for different strategies

### ✅ Sector-Relative Valuation

- [x] Z-scores for metrics within sector
- [x] Percentile ranks within sector
- [x] Identify stocks at premium/discount to peers
- [x] Multiple metrics support (P/E, P/B, EV/EBITDA, etc.)

### ✅ Multi-Factor Scoring

- [x] Combine valuation, quality, growth factors
- [x] Z-score normalization before weighting
- [x] Customizable weights for investment strategies
- [x] Handles missing metrics gracefully

### ✅ Advanced Filtering

- [x] Filter by sector(s)
- [x] Filter by region(s)
- [x] Filter by market cap range
- [x] Filter by mispricing thresholds
- [x] Filter by valuation categories
- [x] Combine multiple filters simultaneously

### ✅ Interactive Visualizations

- [x] Scatter plot: current price vs. predicted target
- [x] Color by sector, region, or valuation category
- [x] Interactive hover with details
- [x] Fair value reference line
- [x] Export to HTML for sharing

### 🔄 Future Enhancements (Phase 9.8+)

- [ ] Confidence intervals from quantile regression
- [ ] Risk-adjusted mispricing scores
- [ ] Sector heatmaps with opportunity counts
- [ ] Time series tracking of mispricing
- [ ] PDF report generation with ReportLab
- [ ] Dashboard with filtering controls

---

## Files Modified

### New Files:

1. **tests/test_valuation_phase97.py** (435 lines)
    - 26 comprehensive tests
    - 7 test classes covering all functions
    - Integration test for complete workflow

### Modified Files:

1. **finance_ml/eval.py**
    - Added 395 lines (1450-1844)
    - 6 new functions for Phase 9.7
    - Full documentation with examples
    - Consistent with existing code style

---

## Acceptance Criteria Met

✅ **TDD Approach**: Tests written first, implementation followed  
✅ **Test Coverage**: 26/26 tests passing (100%)  
✅ **Valuation Categories**: All 5 categories implemented with custom thresholds  
✅ **Sector-Relative Metrics**: Z-scores and percentiles within sectors  
✅ **Multi-Factor Scoring**: Valuation × Quality × Growth with custom weights  
✅ **Advanced Filtering**: Multi-criteria filtering implemented  
✅ **Interactive Visualizations**: Plotly scatter plots with categories  
✅ **Documentation**: Comprehensive docstrings and examples  
✅ **No Regressions**: All existing tests still pass (97/98, 1 pre-existing error)  
✅ **Code Coverage**: New functions have ≥80% coverage via tests

---

## Usage in Production

### Quick Start

```python
from finance_ml.eval import (
    assign_valuation_category,
    calculate_sector_zscores,
    calculate_percentile_ranks,
    calculate_multi_factor_score,
    filter_stocks_by_criteria,
    create_valuation_scatter_plot,
    )

# Complete workflow
categories = assign_valuation_category(df['mispricing_score'])
df['valuation_category'] = categories

df = calculate_sector_zscores(df, metrics=['pe_ratio', 'pb_ratio'])
df = calculate_percentile_ranks(df, metrics=['pe_ratio', 'pb_ratio'])

df['multi_factor_score'] = calculate_multi_factor_score(
        df,
        valuation_col='mispricing_score',
        quality_cols=['roe', 'ebitda_margin'],
        growth_cols=['revenue_growth']
        )

opportunities = filter_stocks_by_criteria(
        df,
        min_mispricing=15.0,
        valuation_categories=['Strong Buy', 'Buy']
        )

fig = create_valuation_scatter_plot(opportunities, color_by='valuation_category')
```

### Best Practices

1. Always assign valuation categories after calculating mispricing scores
2. Use sector-relative metrics (z-scores) for fair comparisons across sectors
3. Customize multi-factor weights based on investment strategy (value vs. growth)
4. Apply filtering progressively to narrow down opportunities
5. Visualize results to identify patterns and outliers
6. Export filtered results for further analysis in Excel or BI tools

---

## Next Steps (Phase 9.8)

Planned enhancements for Phase 9.8:

1. Compare model predictions vs. analyst consensus targets
2. Agreement/disagreement analysis
3. Directional accuracy metrics
4. Comprehensive Excel reporting (match Stock_Prediction_Analysis_Report format)
5. Temporal tracking of prediction changes
6. Model-analyst divergence opportunities

---

## Conclusion

Phase 9.7 successfully delivers comprehensive stock valuation analysis with visualization capabilities following strict
TDD principles. All 26 tests pass, providing confidence in the implementation quality. The functions are
production-ready, well-documented, and seamlessly integrate with existing `finance_ml` modules.

**Key Achievement**: Complete valuation workflow that enables investors to identify undervalued stocks, apply
multi-factor screening, and visualize opportunities across sectors and regions with interactive charts.
