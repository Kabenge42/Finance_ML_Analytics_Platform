create function calculate_reporting_interval(earnings_report_frequency text) returns integer
    immutable
    language plpgsql
as
$$
BEGIN
    RETURN frequency_to_months(earnings_report_frequency);
END;
$$;

alter function calculate_reporting_interval(text) owner to postgres;

