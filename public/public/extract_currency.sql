create function extract_currency(d1_value text) returns text
    immutable
    language plpgsql
as
$$
                BEGIN RETURN REGEXP_REPLACE(TRIM(d1_value), '[0-9]+[A-Za-z]*$', ''); END;
                $$;

alter function extract_currency(text) owner to postgres;

