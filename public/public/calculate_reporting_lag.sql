create function calculate_reporting_lag(next_earnings date, income_statement_report_date date) returns integer
    immutable
    language plpgsql
as
$$
BEGIN
    IF next_earnings IS NULL OR income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN next_earnings - income_statement_report_date;
END;
$$;

alter function calculate_reporting_lag(date, date) owner to postgres;

create function calculate_reporting_lag(next_earnings             date, income_statement_report_date date,
                                        earnings_report_frequency text DEFAULT 'Quarterly'::text) returns integer
    immutable
    language plpgsql
as
$$
BEGIN
    IF next_earnings IS NULL OR income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN next_earnings - income_statement_report_date;
END;
$$;

alter function calculate_reporting_lag(date, date, text) owner to postgres;

