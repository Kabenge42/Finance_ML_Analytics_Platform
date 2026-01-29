create function derive_earnings_report_frequency(income_statement_report_date date, fy_end_date date) returns text
    immutable
    language plpgsql
as
$$
DECLARE
    months_diff INTEGER;
BEGIN
    IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN
        RETURN 'Quarterly';
    END IF;

    months_diff := ABS(
            (EXTRACT(YEAR FROM income_statement_report_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
                + (EXTRACT(MONTH FROM income_statement_report_date) - EXTRACT(MONTH FROM fy_end_date))
                   )::INTEGER;

    -- Normalize to 1-12 range
    months_diff := COALESCE(NULLIF(months_diff % 12, 0), 12);

    -- Determine frequency: check if months align with semi-annual or quarterly
    RETURN CASE
               WHEN months_diff IN (6, 12) THEN 'Semi-Annually'
               ELSE 'Quarterly'
        END;
END;
$$;

alter function derive_earnings_report_frequency(date, date) owner to postgres;

