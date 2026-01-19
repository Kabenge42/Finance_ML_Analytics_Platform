create function calc_dividend_timing()
    returns TABLE
            (
                ticker                   text,
                days_since_ex_date       integer,
                days_to_payment          integer,
                dividend_announced_flag  integer,
                ex_date_approaching_flag integer,
                dividend_frequency_score integer,
                dividend_consistency     numeric,
                recent_dividend_change   numeric,
                dividend_yield_vs_5y_avg numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                   AS ticker,
       -- Days Since Ex-Dividend Date
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,

       -- Days to Next Payment
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,

       -- Dividend Announced Flag (recent announcement within 30 days)
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,

       -- Ex-Date Approaching Flag (within next 14 days)
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                    AS ex_date_approaching_flag,

       -- Dividend Frequency Score (Quarterly=4, Semi-Annual=2, Annual=1, etc.)
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly' THEN 4
           WHEN 'Semi-Annual' THEN 2
           WHEN 'Annual' THEN 1
           WHEN 'Monthly' THEN 12
           ELSE 0
           END                                                    AS dividend_frequency_score,

       -- Dividend Consistency (streak / 10, capped at 1.0)
       LEAST(1.0, "Dividend Streak"::NUMERIC / 10.0)              AS dividend_consistency,

       -- Recent Dividend Change (current vs previous year indicated yield)
       CASE
           WHEN "Div Yield (-1FYInd)" > 0
               THEN ("Div Yield (Ind)" - "Div Yield (-1FYInd)") /
                    NULLIF("Div Yield (-1FYInd)", 0) * 100
           END                                                    AS recent_dividend_change,

       -- Dividend Yield vs 5Y Average
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg

FROM postgres.public.equities;
$$;

alter function calc_dividend_timing() owner to postgres;

