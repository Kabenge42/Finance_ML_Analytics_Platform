# Helper Functions Schema Qualification Fix

## Issue Summary

PostgreSQL functions in `feature_registry.sql` were failing with error:

```
ERROR: function calc_change_ratio(numeric, numeric) does not exist
Hint: No function matches the given name and argument types. You might need to add explicit type casts.
```

This error occurred during function inlining when PostgreSQL tried to resolve helper function calls within SQL
functions.

## Root Cause Analysis

### The Problem

The helper functions (`calc_change_ratio`, `pct_change`, `clamp_score`, `ema_crossover_signal`) were defined in the
`public` schema but were being called **without schema qualification** in SQL functions throughout
`feature_registry.sql`.

### Why It Failed

1. **Function Inlining Context**: When PostgreSQL inlines SQL functions (like `calc_valuation_timeseries_features`), it
   needs to resolve all function references at definition time.

2. **Search Path Dependency**: Unqualified function calls rely on the `search_path` setting. If the `public` schema is
   not in the search path or is not the first schema searched, PostgreSQL cannot find the helper functions.

3. **Inconsistent Qualification**: Some functions like `safe_divide` were already qualified as `public.safe_divide`, but
   the other helper functions were not, creating an inconsistent pattern.

### Example of the Issue

**Before (BROKEN):**

```sql
CREATE OR REPLACE FUNCTION calc_valuation_timeseries_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                ..
                .
            )
AS
$$
SELECT "ISIN"                                                                      AS isin,
       calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (-1FYLTM)"::NUMERIC) AS ev_sales_trend_1y,
       -- ... more unqualified calls
       public.safe_divide("P/B (LTM)"::NUMERIC, "P/B (5YAVG)"::NUMERIC)            AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;
```

**After (FIXED):**

```sql
CREATE OR REPLACE FUNCTION calc_valuation_timeseries_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                ..
                .
            )
AS
$$
SELECT "ISIN"                                                                             AS isin,
       public.calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (-1FYLTM)"::NUMERIC) AS ev_sales_trend_1y,
       -- ... all calls now qualified
       public.safe_divide("P/B (LTM)"::NUMERIC, "P/B (5YAVG)"::NUMERIC)                   AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;
```

## Solution Implemented

### Changes Made

1. **Schema Qualification**: Added `public.` prefix to all 97 helper function calls in `feature_registry.sql`:
    - `calc_change_ratio(...)` → `public.calc_change_ratio(...)`
    - `pct_change(...)` → `public.pct_change(...)`
    - `clamp_score(...)` → `public.clamp_score(...)`
    - `ema_crossover_signal(...)` → `public.ema_crossover_signal(...)`

2. **Consistency**: Now all helper functions are consistently qualified with their schema name, matching the pattern
   already used for `public.safe_divide()`.

### Functions Fixed

The following helper functions are now properly qualified throughout the file:

- `public.calc_change_ratio(current_val NUMERIC, previous_val NUMERIC)` - 40+ calls
- `public.pct_change(current_val NUMERIC, previous_val NUMERIC)` - 30+ calls
- `public.clamp_score(val NUMERIC, min_val NUMERIC, max_val NUMERIC)` - 10+ calls
- `public.ema_crossover_signal(fast_ema NUMERIC, slow_ema NUMERIC)` - 5+ calls
- `public.safe_divide(numerator NUMERIC, denominator NUMERIC)` - already qualified

## Why This Solution is Robust and Long-Term

### 1. **Search Path Independence**

The solution eliminates dependency on PostgreSQL's `search_path` configuration. Functions will resolve correctly
regardless of:

- Database connection settings
- User-specific search path configurations
- Schema precedence changes

### 2. **Explicit and Clear**

Schema qualification makes it immediately clear where each function is defined, improving code readability and
maintainability.

### 3. **Prevents Future Issues**

- New functions added to the file will follow the established pattern
- Developers will see the consistent qualification pattern and replicate it
- No ambiguity about which schema's function is being called

### 4. **Best Practice Alignment**

This follows PostgreSQL best practices for production code:

- Always qualify function calls in stored procedures/functions
- Avoid implicit schema resolution in critical code
- Make dependencies explicit

## Prevention Guidelines

### For Future Development

When adding new SQL functions to `feature_registry.sql`:

1. **Always qualify helper function calls** with `public.` prefix
2. **Always qualify table references** with full schema path (e.g., `postgres.public.equities`)
3. **Test functions** in environments with different search_path settings
4. **Review consistency** - if one function is qualified, all should be

### Code Review Checklist

- [ ] All helper function calls include schema qualification
- [ ] All table references include full schema path
- [ ] No reliance on implicit search_path resolution
- [ ] Consistent qualification pattern throughout the file

## Related Files

- `feature_registry.sql` - Main file with all fixes applied
- `create_helper_functions.sql` - Defines the helper functions
- `public/public/calc_change_ratio.sql` - Individual function definition
- `public/public/pct_change.sql` - Individual function definition
- `public/public/safe_divide.sql` - Individual function definition

## Testing Recommendations

After applying this fix:

1. **Execute the SQL file** in a clean database to verify no errors
2. **Test with different search_path** settings:
   ```sql
   SET search_path TO pg_catalog;
   SELECT * FROM calc_valuation_timeseries_features();
   ```
3. **Verify materialized view creation** succeeds:
   ```sql
   REFRESH MATERIALIZED VIEW mv_all_stock_features;
   ```

## Impact

- **97 function calls** updated with schema qualification
- **Zero breaking changes** - purely additive qualification
- **Backward compatible** - existing queries continue to work
- **Performance neutral** - schema qualification has no runtime overhead

## Date Applied

2026-01-29

## Author

Junie (Autonomous Code Fix)
