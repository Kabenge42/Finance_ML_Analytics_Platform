create function months_to_frequency(interval_months integer) returns text
    immutable
    language plpgsql
as
$$
BEGIN
    RETURN CASE
               WHEN interval_months <= 3 THEN 'Quarterly'
               WHEN interval_months <= 6 THEN 'Semi-Annually'
               ELSE 'Annually'
        END;
END;
$$;

alter function months_to_frequency(integer) owner to postgres;

