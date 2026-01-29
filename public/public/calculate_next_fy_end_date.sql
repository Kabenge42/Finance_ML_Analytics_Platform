create function calculate_next_fy_end_date(fy_end_date date) returns date
    immutable
    strict
    language plpgsql
as
$$
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN (fy_end_date + INTERVAL '1 year')::DATE;
END;
$$;

alter function calculate_next_fy_end_date(date) owner to postgres;

