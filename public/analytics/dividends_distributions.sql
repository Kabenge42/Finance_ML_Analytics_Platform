create table analytics.dividends_distributions
(
    dividend_streak             text,
    dividend_yield_ltm          text,
    dividend_yield_ntm          text,
    dividend_payout_ratio       text,
    fcf_dividend_coverage       text,
    buyback_yield               text,
    total_shareholder_yield     text,
    dividend_growth_expectation text,
    days_since_ex_date          text,
    days_to_payment             text,
    dividend_announced_flag     text,
    ex_date_approaching_flag    text,
    dividend_frequency_score    text,
    dividend_consistency        text,
    recent_dividend_change      text,
    dividend_yield_vs_5y_avg    text,
    div_yield_ltm               text,
    div_yield_ntm               text,
    div_yield_ind               text,
    div_yield_1fy_ind           text,
    div_yield_5y_avg            text,
    div_yield_vs_5y_avg         text,
    div_yield_growth_expected   text,
    dividend_streak_comp        text,
    high_yield_flag             text,
    sustainable_dividend_flag   text
);

alter table analytics.dividends_distributions
    owner to postgres;

