create function calculate_expected_report_date(period_end_date date, earnings_report_frequency text) returns date
    immutable
    language plpgsql
as
$$
BEGIN
    IF period_end_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN period_end_date + (get_expected_reporting_lag_days(earnings_report_frequency) || ' days')::INTERVAL;
END;
$$;

alter function calculate_expected_report_date(date, text) owner to postgres;

