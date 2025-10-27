# Column Mapping Summary

## Overview

The `finance_ml/data.py` module has been updated to ensure complete consistency between the PostgreSQL database schema (
`equities` table) and the column names used throughout the notebook and package.

## Changes Made

### 1. Enhanced `normalize_columns()` Function

The function now includes:

- **Full Schema Mapping**: Complete mapping of all 234 database columns from their original names (with spaces and
  special characters) to Python-friendly normalized names
- **Column Aliases**: Automatic creation of commonly used generic column names that map to their most relevant variants

### 2. Database Schema Column Mappings

Key column mappings from database to normalized Python names:

| Database Column                 | Normalized Name               | Notes                                |
|---------------------------------|-------------------------------|--------------------------------------|
| `"Ticker"`                      | `ticker`                      | Stock ticker symbol                  |
| `"Sector"`                      | `sector`                      | Industry sector                      |
| `"Last Price"`                  | `last_price`                  | Current stock price                  |
| `"Market Cap"`                  | `market_cap`                  | Market capitalization                |
| `"Price Target"`                | `price_target`                | Analyst price target                 |
| `"P/E (LTM)"`                   | `p_e_ltm`                     | Price-to-Earnings (Last 12 Months)   |
| `"P/E (NTM)"`                   | `p_e_ntm`                     | Price-to-Earnings (Next 12 Months)   |
| `"Total Revenues (LTM)"`        | `total_revenues_ltm`          | Revenue Last 12 Months               |
| `"EBITDA (LTM)"`                | `ebitda_ltm`                  | EBITDA Last 12 Months                |
| `"Net Income - (IS) (LTM)"`     | `net_income_is_ltm`           | Net Income from Income Statement LTM |
| `"P/B (LTM)"`                   | `p_b_ltm`                     | Price-to-Book ratio LTM              |
| `"Gross Profit Margin % (LTM)"` | `gross_profit_margin_pct_ltm` | Gross profit margin percentage       |

### 3. Column Aliases for Generic Names

To maintain backward compatibility with existing notebook code, the following aliases are automatically created:

| Alias Name     | Maps To                       | Description                                    |
|----------------|-------------------------------|------------------------------------------------|
| `p_e`          | `p_e_ltm`                     | Default P/E ratio (uses LTM)                   |
| `revenue`      | `total_revenues_ltm`          | Default revenue (uses LTM)                     |
| `ebitda`       | `ebitda_ltm`                  | Default EBITDA (uses LTM)                      |
| `net_income`   | `net_income_is_ltm`           | Default net income (uses Income Statement LTM) |
| `p_b`          | `p_b_ltm`                     | Default Price-to-Book (uses LTM)               |
| `gross_margin` | `gross_profit_margin_pct_ltm` | Default gross margin (uses LTM %)              |

**Note**: `ev_ebitda` (Enterprise Value to EBITDA) is not directly available and would need to be calculated from
`enterprise_value` and `ebitda_ltm`.

## Time Period Suffixes

The database includes multiple time periods for most financial metrics:

- **LTM** - Last Twelve Months (trailing 12 months)
- **NTM** - Next Twelve Months (forward 12 months)
- **FY** - Fiscal Year
- **FQ** - Fiscal Quarter
- **-1FY** - Previous Fiscal Year
- **5YAVGFQ** - 5-Year Average (Fiscal Quarter)
- **5YAVGLTM** - 5-Year Average (Last Twelve Months)
- **5YAVGFY** - 5-Year Average (Fiscal Year)

By default, aliases use **LTM** (Last Twelve Months) as it represents the most recent actual performance.

## Usage in Notebook

The notebook (`ml_finance_model_main.ipynb`) can now use either:

1. **Generic alias names** (recommended for simplicity):
   ```python
   all_stocks['p_e']  # Automatically uses p_e_ltm
   all_stocks['revenue']  # Automatically uses total_revenues_ltm
   ```

2. **Specific normalized names** (for explicit time period control):
   ```python
   all_stocks['p_e_ntm']  # Forward P/E
   all_stocks['total_revenues_fy']  # Fiscal year revenue
   ```

3. **Original database column names** (converted automatically):
    - When loaded from database: `"P/E (LTM)"` → `p_e_ltm`
    - When loaded from CSV: columns normalized automatically

## Data Loading Consistency

The `load_from_db()` and `load_from_csv()` functions both call `normalize_columns()` to ensure:

1. All column names are lowercase with underscores
2. Schema mappings are applied consistently
3. Column aliases are created automatically
4. Data from any source (DB or CSV) has identical column names

## Benefits

1. **Schema Consistency**: Database schema drives column naming
2. **Backward Compatibility**: Existing notebook code continues to work
3. **Flexibility**: Can use generic or specific column names
4. **Maintainability**: Single source of truth for column mappings
5. **Clear Semantics**: Column names clearly indicate their source and time period

## Verification

To verify column consistency, check:

```python
# List all available columns
print(all_stocks.columns.tolist())

# Check if generic aliases exist
print('p_e' in all_stocks.columns)  # Should be True
print('revenue' in all_stocks.columns)  # Should be True

# Check actual source columns
print('p_e_ltm' in all_stocks.columns)  # Should be True
print('total_revenues_ltm' in all_stocks.columns)  # Should be True
```

## Future Enhancements

Potential future additions:

- Calculated columns (e.g., `ev_ebitda` from `enterprise_value` / `ebitda_ltm`)
- Additional aliases based on usage patterns
- Column validation to ensure required financial metrics are present
- Automatic detection of best available time period (fallback logic)
