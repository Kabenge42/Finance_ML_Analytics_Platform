create function extract_unit(d1_value text) returns numeric
    immutable
    language plpgsql
as
$$
                DECLARE num_part TEXT;
                BEGIN
                    num_part := SUBSTRING(TRIM(d1_value) FROM '([0-9]+)');
                    IF num_part IS NULL OR num_part = '' THEN RETURN NULL; END IF;
                    RETURN num_part::NUMERIC;
                END; $$;

alter function extract_unit(text) owner to postgres;

