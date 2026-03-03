create function safe_power(base numeric, exponent numeric) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT CASE
           WHEN base IS NULL OR exponent IS NULL THEN NULL
           WHEN base = 0 THEN 0
           WHEN base > 0 THEN POWER(base, exponent)
           -- Negative base with integer exponent: compute safely
           WHEN FLOOR(exponent) = exponent THEN -POWER(-base, exponent)
           -- Negative base with non-integer exponent: undefined in reals
           ELSE NULL
           END AS result;
$$;

alter function safe_power(numeric, numeric) owner to postgres;

