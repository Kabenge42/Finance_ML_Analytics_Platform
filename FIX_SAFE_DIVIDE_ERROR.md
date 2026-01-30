# Fix for PostgreSQL Error: function safe_divide(numeric, numeric) does not exist

## Problem Description

When executing SQL queries that use the `calc_valuation_features` function, PostgreSQL throws the following error:

```
[42883] ERROR: function safe_divide(numeric, numeric) does not exist
  Hint: No function matches the given name and argument types. You might need to add explicit type casts.
  Where: SQL function "calc_valuation_features" during inlining
```

## Root Cause

The error occurs because:

1. The `calc_valuation_features` function (and other feature calculation functions) depend on helper functions like
   `safe_divide`, `pct_change`, `calc_change_ratio`, etc.
2. When PostgreSQL tries to inline these functions for optimization, it needs the helper functions to already exist in
   the database.
3. If the helper functions are not created or were dropped, the inlining process fails with this error.

## Solution

### Option 1: Execute Helper Functions Script (Recommended)

Run the standalone helper functions script **before** executing the main feature registry:

```powershell
# Connect to your PostgreSQL database and execute:
psql -U your_username -d your_database -f create_helper_functions.sql
```

Or using a PostgreSQL client (pgAdmin, DBeaver, etc.):

1. Open `create_helper_functions.sql`
2. Execute the entire script
3. Verify the success message appears

### Option 2: Execute Feature Registry in Correct Order

If you're running `feature_registry.sql`, ensure it's executed completely from the beginning. The helper functions are
defined at the top of the file (lines 17-73), so they must be created before the feature calculation functions.

```powershell
# Execute the complete feature_registry.sql file:
psql -U your_username -d your_database -f feature_registry.sql
```

### Option 3: Manual Function Creation

If you need to quickly fix the issue, execute this SQL directly in your database:

```sql
CREATE OR REPLACE FUNCTION safe_divide(
    numerator   NUMERIC,
    denominator NUMERIC
)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
    LANGUAGE SQL
AS
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;
```

## Verification

After creating the helper functions, verify they exist:

```sql
-- Check if safe_divide exists
SELECT proname, pronargs, proargtypes
FROM pg_proc
WHERE proname = 'safe_divide';

-- Test the function
SELECT safe_divide(10, 2); -- Should return 5
SELECT safe_divide(10, 0); -- Should return NULL (safe division by zero)
```

## Prevention

To prevent this issue in the future:

1. **Always execute `create_helper_functions.sql` first** before running any feature calculation queries
2. **Don't drop helper functions** unless you're rebuilding the entire schema
3. **Use transaction blocks** when executing large SQL scripts to ensure atomicity:

   ```sql
   BEGIN;
   i create_helper_functions.sql
   i feature_registry.sql
   COMMIT;
   ```

## Files Involved

- `create_helper_functions.sql` - Standalone script for helper functions (NEW)
- `feature_registry.sql` - Main feature registry (contains helper functions at the top)
- `calc_valuation_features` - Function that uses `safe_divide` (line 97-131 in feature_registry.sql)

## Additional Helper Functions

The following helper functions are also created and may be required by other feature calculations:

- `pct_change(numeric, numeric)` - Calculates percentage change
- `calc_change_ratio(numeric, numeric)` - Calculates change ratio without percentage
- `clamp_score(numeric, numeric, numeric)` - Constrains values between min and max
- `ema_crossover_signal(numeric, numeric)` - Calculates EMA crossover signals

All of these are included in `create_helper_functions.sql`.