create function safe_divide(numerator numeric, denominator numeric) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;

alter function safe_divide(unknown, unknown) owner to postgres;

