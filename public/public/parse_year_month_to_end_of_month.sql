create function parse_year_month_to_end_of_month(date_text text) returns date
    immutable
    language plpgsql
as
$$
                DECLARE year_val INTEGER; month_val INTEGER;
                BEGIN
                    IF date_text IS NULL OR TRIM(date_text) = '' THEN RETURN NULL; END IF;
                    IF date_text !~ '^\d{4}-\d{2}$' THEN RETURN NULL; END IF;
                    year_val  := SPLIT_PART(date_text, '-', 1)::INTEGER;
                    month_val := SPLIT_PART(date_text, '-', 2)::INTEGER;
                    IF year_val < 1900 OR year_val > 2100 OR month_val < 1 OR month_val > 12 THEN RETURN NULL; END IF;
                    RETURN (MAKE_DATE(year_val, month_val, 1) + INTERVAL '1 month - 1 day')::DATE;
                END; $$;

alter function parse_year_month_to_end_of_month(text) owner to postgres;

