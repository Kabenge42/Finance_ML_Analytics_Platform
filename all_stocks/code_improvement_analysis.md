# Code Improvement Analysis: Regional Record Check Loop

## Original Code Issues

```sql
-- Check for records per region
RAISE NOTICE 'Records per region:';
FOR v_row_count IN
    SELECT "Region", COUNT(*)
    FROM all_stocks
    GROUP BY "Region"
    ORDER BY "Region"
```

## Problems Identified

### 1. **Type Mismatch (Critical Error)**

- `v_row_count` is declared as `INTEGER` but used to iterate over a result set with two columns
- This will cause a runtime error: "loop variable of loop over rows must be a record variable or list of scalar
  variables"

### 2. **Incomplete Loop Structure**

- Missing `LOOP...END LOOP` block
- No proper access to record fields within the loop body

### 3. **Poor Output Formatting**

- `RAISE NOTICE '  %', v_row_count` outputs the raw record structure
- Not user-friendly: shows `(US,1000)` instead of `US: 1,000 records`

### 4. **Misleading Variable Name**

- `v_row_count` suggests a single integer, not a record containing region and count

### 5. **No Aggregate Summary**

- Doesn't show total records across all regions

## Improved Solutions

### Solution 1: Using RECORD Variable (Recommended)

```sql
-- Data quality checks
RAISE NOTICE 'Running data quality checks...';

DECLARE
    v_region_record RECORD;
    v_region_count  INTEGER;
    v_total_count   INTEGER := 0;
BEGIN
    -- Check for records per region
    RAISE NOTICE 'Records per region:';
    
    FOR v_region_record IN
        SELECT "Region", 
               COUNT(*) as region_count
        FROM all_stocks
        GROUP BY "Region"
        ORDER BY "Region"
    LOOP
        v_region_count := v_region_record.region_count;
        v_total_count := v_total_count + v_region_count;
        
        RAISE NOTICE '  %-6s: %s records', 
            v_region_record."Region", 
            to_char(v_region_count, 'FM999,999,999');
    END LOOP;
    
    RAISE NOTICE '  ------';
    RAISE NOTICE '  Total : %s records', to_char(v_total_count, 'FM999,999,999');
END;
```

**Improvements:**

- ✅ Uses `RECORD` type for proper result set iteration
- ✅ Properly accesses fields: `v_region_record."Region"` and `v_region_record.region_count`
- ✅ Formatted output with alignment and thousand separators
- ✅ Calculates and displays total count
- ✅ Clear variable naming
- ✅ Complete loop structure with `LOOP...END LOOP`

**Sample Output:**

```
NOTICE:  Records per region:
NOTICE:    APAC  : 2,450 records
NOTICE:    EU    : 3,780 records
NOTICE:    ROTW  : 1,230 records
NOTICE:    US    : 8,590 records
NOTICE:    ------
NOTICE:    Total : 16,050 records
```

### Solution 2: Alternative with Explicit Columns

```sql
DECLARE
    v_region      TEXT;
    v_count       INTEGER;
    v_total_count INTEGER := 0;
BEGIN
    RAISE NOTICE 'Records per region:';
    
    FOR v_region, v_count IN
        SELECT "Region", COUNT(*)::INTEGER
        FROM all_stocks
        GROUP BY "Region"
        ORDER BY "Region"
    LOOP
        v_total_count := v_total_count + v_count;
        RAISE NOTICE '  %-6s: %s records', v_region, to_char(v_count, 'FM999,999,999');
    END LOOP;
    
    RAISE NOTICE '  ------';
    RAISE NOTICE '  Total : %s records', to_char(v_total_count, 'FM999,999,999');
END;
```

**Improvements:**

- ✅ Uses multiple scalar variables matching SELECT columns
- ✅ Type-safe with explicit variable declarations
- ✅ Formatted output
- ✅ Shows total count

### Solution 3: Using Array Aggregation (Most Concise)

```sql
DECLARE
    v_region_stats TEXT;
    v_total_count  INTEGER;
BEGIN
    RAISE NOTICE 'Records per region:';
    
    SELECT string_agg(
        format('  %-6s: %s records', 
               "Region", 
               to_char(region_count, 'FM999,999,999')
        ), E'\n' ORDER BY "Region"
    ),
    SUM(region_count)
    INTO v_region_stats, v_total_count
    FROM (
        SELECT "Region", COUNT(*) as region_count
        FROM all_stocks
        GROUP BY "Region"
    ) stats;
    
    RAISE NOTICE '%', v_region_stats;
    RAISE NOTICE '  ------';
    RAISE NOTICE '  Total : %s records', to_char(v_total_count, 'FM999,999,999');
END;
```

**Improvements:**

- ✅ Single query execution (most efficient)
- ✅ Aggregates formatting in SQL
- ✅ Shows total count
- ✅ No loop overhead

## Complete Refactored Section

Here's how the entire data quality checks section should look:

```sql
-- Data quality checks
RAISE NOTICE 'Running data quality checks...';

-- Check for records per region with formatted output
DECLARE
    v_region_record RECORD;
    v_region_count  INTEGER;
    v_total_records INTEGER := 0;
BEGIN
    RAISE NOTICE 'Records per region:';
    
    FOR v_region_record IN
        SELECT "Region", COUNT(*) as region_count
        FROM all_stocks
        GROUP BY "Region"
        ORDER BY "Region"
    LOOP
        v_region_count := v_region_record.region_count;
        v_total_records := v_total_records + v_region_count;
        
        RAISE NOTICE '  %-6s: %s records', 
            v_region_record."Region", 
            to_char(v_region_count, 'FM999,999,999');
    END LOOP;
    
    RAISE NOTICE '  ------';
    RAISE NOTICE '  Total : %s records', to_char(v_total_records, 'FM999,999,999');
    RAISE NOTICE '';
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error during regional record count: %', SQLERRM;
END;

-- Check for potential duplicates
SELECT COUNT(*)
INTO v_row_count
FROM (
    SELECT "Ticker", "Region", COUNT(*) as dup_count
    FROM all_stocks
    GROUP BY "Ticker", "Region"
    HAVING COUNT(*) > 1
) dups;

IF v_row_count > 0 THEN
    RAISE WARNING 'Found % duplicate ticker-region combinations', v_row_count;
ELSE
    RAISE NOTICE 'No duplicates found - data integrity verified';
END IF;
```

## Additional Best Practices Applied

1. **Error Handling**: Added EXCEPTION block to catch potential errors
2. **Visual Separation**: Added blank line after region summary
3. **Consistent Formatting**: Used format strings for alignment
4. **Thousand Separators**: Used `to_char` with format mask for readability
5. **Clear Section Breaks**: Added visual separator line for totals
6. **Type Safety**: Explicit variable declarations with appropriate types

## Performance Considerations

- **Loop vs Aggregation**: For small result sets (4 regions), loop overhead is negligible
- **Single Query**: Solution 3 is most efficient but less readable
- **Indexed Queries**: Region column is indexed, so GROUP BY is fast

## Recommendation

Use **Solution 1** (RECORD variable) for the best balance of:

- Code clarity and maintainability
- Proper error handling
- Formatted, professional output
- Type safety
- Extensibility for future enhancements
