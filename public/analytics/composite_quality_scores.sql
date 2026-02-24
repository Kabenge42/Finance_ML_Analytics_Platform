create table analytics.composite_quality_scores
(
    sector                            text,
    earnings_report_frequency         text,
    isin                              text,
    next_fy_end_date                  date,
    region                            text,
    name                              text,
    next_earnings_status              text,
    industry                          text,
    income_statement_report_date      date,
    next_income_statement_report_date date,
    style_class                       text,
    ticker                            text,
    dividend_record_record_date       date,
    reference_date                    date,
    country                           text,
    last_updated                      date,
    reporting_interval                double precision,
    fy_end_date                       date,
    fy_end                            text,
    next_earnings                     date,
    unit                              text,
    exchange                          text,
    next_earnings_report              text,
    next_fiscal_quarter               text,
    next_earnings_when                text,
    dividend_record_announce_date     date,
    dividend_record_payable_date      date,
    trading_country                   text,
    dividend_record_ex_date           date,
    size_class                        text,
    dividend_record_frequency         text,
    composite_score                   double precision
);

alter table analytics.composite_quality_scores
    owner to postgres;

