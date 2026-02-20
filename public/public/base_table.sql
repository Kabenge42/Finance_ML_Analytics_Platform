create table base_table
(
    isin                              text not null
        constraint base_table_pk
            primary key,
    ticker                            text,
    name                              text,
    region                            text,
    country                           text,
    trading_country                   text,
    exchange                          text,
    sector                            text,
    industry                          text,
    dividend_record_frequency         text,
    earnings_report_frequency         text,
    fy_end                            text,
    next_earnings_report              text,
    next_earnings_status              text,
    next_earnings_when                text,
    next_fiscal_quarter               text,
    reporting_interval                integer,
    size_class                        text,
    style_class                       text,
    unit                              text,
    dividend_record_announce_date     date,
    dividend_record_ex_date           date,
    dividend_record_payable_date      date,
    dividend_record_record_date       date,
    fy_end_date                       date,
    income_statement_report_date      date,
    last_updated                      date,
    next_earnings                     date,
    next_fy_end_date                  date,
    next_income_statement_report_date date,
    reference_date                    date
);

alter table base_table
    owner to postgres;

create index base_table_geography_uindex
    on base_table (region, country, trading_country, exchange);

