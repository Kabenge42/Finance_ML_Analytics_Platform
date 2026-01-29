create function validate_fiscal_dates(fy_end_date date, report_date date, reference_date date DEFAULT CURRENT_DATE)
    returns TABLE
            (
                issue    text,
                severity text
            )
    immutable
    language plpgsql
as
$$
BEGIN
    IF fy_end_date > reference_date THEN
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT as fy_end_future, 'WARNING'::TEXT as fy_end_warning;
    END IF;

    IF report_date IS NOT NULL AND report_date < fy_end_date - INTERVAL '1 year' THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT as report_date_predates,
                            'ERROR'::TEXT                            as report_date_error;
    END IF;

    IF report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT as report_date_future,
                            'WARNING'::TEXT                      as report_date_warning;
    END IF;

    IF fy_end_date != (DATE_TRUNC('month', fy_end_date) + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT as fy_end_ldm, 'INFO'::TEXT as fy_end_info;
    END IF;
END;
$$;

alter function validate_fiscal_dates(date, date, date) owner to postgres;

