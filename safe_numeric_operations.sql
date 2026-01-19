-- Add numeric overload for safe_div
CREATE OR REPLACE FUNCTION safe_div(numerator numeric, denominator numeric)
    RETURNS numeric
    LANGUAGE sql
    IMMUTABLE
AS
$$
SELECT CASE
           WHEN denominator IS NULL OR denominator = 0 THEN NULL
           ELSE numerator / denominator
           END;
$$;

-- Add numeric overload for safe_pct (if needed)
CREATE OR REPLACE FUNCTION safe_pct(numerator numeric, denominator numeric)
    RETURNS numeric
    LANGUAGE sql
    IMMUTABLE
AS
$$
SELECT CASE
           WHEN denominator IS NULL OR denominator = 0 THEN NULL
           ELSE (numerator / denominator) * 100
           END;
$$;


-- ... existing code ...

-- Add numeric overload for yoy_growth (if needed)
DROP FUNCTION IF EXISTS yoy_growth(numeric, numeric);
CREATE OR REPLACE FUNCTION yoy_growth(current_value numeric, prior_value numeric)
    RETURNS numeric
    LANGUAGE sql
    IMMUTABLE
AS
$$
SELECT CASE
           WHEN prior_value IS NULL OR prior_value = 0 THEN NULL
           ELSE ((current_value - prior_value) / prior_value) * 100
           END;
$$;;
