create function get_expected_reporting_lag_days(earnings_report_frequency text) returns integer
    immutable
    language plpgsql
as
$$
BEGIN
    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
               WHEN 'QUARTERLY' THEN 45
               WHEN 'SEMI-ANNUALLY' THEN 60
               WHEN 'SEMI-ANNUAL' THEN 60
               WHEN 'ANNUALLY' THEN 90
               WHEN 'ANNUAL' THEN 90
               ELSE 45
        END;
END;
$$;

alter function get_expected_reporting_lag_days(text) owner to postgres;

