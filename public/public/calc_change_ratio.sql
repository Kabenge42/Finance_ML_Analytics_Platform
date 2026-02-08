create function calc_change_ratio(current_val numeric, previous_val numeric) returns numeric
    immutable
    parallel safe
    language sql
as
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$;

alter function calc_change_ratio(unknown, unknown) owner to postgres;

