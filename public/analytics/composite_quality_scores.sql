create table analytics.composite_quality_scores
(
    dividend_record_announce_date     date,
    ticker                            text,
    exchange                          text,
    income_statement_report_date      date,
    reference_date                    date,
    dividend_record_ex_date           date,
    dividend_record_record_date       date,
    trading_country                   text,
    industry                          text,
    style_class                       text,
    unit                              text,
    isin                              text,
    sector                            text,
    size_class                        text,
    reporting_interval                double precision,
    name                              text,
    country                           text,
    next_earnings                     date,
    dividend_record_frequency         text,
    next_earnings_report              text,
    next_earnings_status              text,
    next_fy_end_date                  date,
    next_fiscal_quarter               text,
    earnings_report_frequency         text,
    last_updated                      date,
    fy_end                            text,
    region                            text,
    next_income_statement_report_date date,
    fy_end_date                       date,
    next_earnings_when                text,
    dividend_record_payable_date      date,
    composite_score                   double precision
);

alter table analytics.composite_quality_scores
    owner to postgres;

