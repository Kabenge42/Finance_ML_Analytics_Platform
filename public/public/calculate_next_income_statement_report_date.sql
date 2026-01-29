create function calculate_next_income_statement_report_date(income_statement_report_date date, earnings_report_frequency text) returns date
    immutable
    language plpgsql
as
$$
BEGIN
    IF income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN (income_statement_report_date +
            (frequency_to_months(earnings_report_frequency) || ' months')::INTERVAL)::DATE;
END;
$$;

alter function calculate_next_income_statement_report_date(date, text) owner to postgres;

