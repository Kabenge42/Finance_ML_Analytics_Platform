create function parse_fiscal_year_end_date(fy_end_text text) returns date
    immutable
    strict
    language plpgsql
as
$$
DECLARE
    month_name TEXT;
    year_text  TEXT;
    month_num  INTEGER;
    year_value INTEGER;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    fy_end_text := TRIM(fy_end_text);
    month_name := SPLIT_PART(fy_end_text, ' ', 1);
    year_text := SPLIT_PART(fy_end_text, ' ', 2);

    -- Validate year format and range
    IF year_text !~ '^\d{4}$' THEN
        RETURN NULL;
    END IF;

    year_value := year_text::INTEGER;
    IF year_value < 1900 OR year_value > 2100 THEN
        RETURN NULL;
    END IF;

    month_num := month_abbrev_to_number(month_name);
    IF month_num IS NULL THEN
        RETURN NULL;
    END IF;

    RETURN (MAKE_DATE(year_value, month_num, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$;

alter function parse_fiscal_year_end_date(text) owner to postgres;

