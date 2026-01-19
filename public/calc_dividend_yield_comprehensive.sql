create function calc_dividend_yield_comprehensive()
    returns TABLE
            (
                ticker                 text,
                div_yield_ind          numeric,
                div_yield_ltm          numeric,
                div_yield_ttm          numeric,
                div_yield_ntm          numeric,
                div_yield_5yavg        numeric,
                div_yield_ind_1fy      numeric,
                div_yield_ind_2fy      numeric,
                div_yield_ind_3fy      numeric,
                div_yield_ind_4fy      numeric,
                div_yield_ind_5fy      numeric,
                dividend_streak        integer,
                dividend_per_share_ltm numeric,
                dividend_amount        numeric,
                common_dividends_ltm   numeric,
                common_dividends_fy    numeric,
                buyback_yield          numeric,
                div_yield_change_1y    numeric,
                div_yield_change_2y    numeric,
                div_yield_change_3y    numeric,
                div_yield_change_5y    numeric,
                dividend_growth_ntm    numeric
            )
    language sql
as
$$
SELECT "Ticker"                                  AS ticker,
       -- Current Yields
       "Div Yield (Ind)"                         AS div_yield_ind,
       "Div Yield (LTM)"                         AS div_yield_ltm,
       "Div Yield (TTM)"                         AS div_yield_ttm,
       "Div Yield (NTM)"                         AS div_yield_ntm,
       "Div Yield (5YAVGLTM)"                    AS div_yield_5yavg,
       -- Historical Indicated Yields
       "Div Yield (-1FYInd)"                     AS div_yield_ind_1fy,
       "Div Yield (-2FYInd)"                     AS div_yield_ind_2fy,
       "Div Yield (-3FYInd)"                     AS div_yield_ind_3fy,
       "Div Yield (-4FYInd)"                     AS div_yield_ind_4fy,
       "Div Yield (-5FYInd)"                     AS div_yield_ind_5fy,
       -- Dividend Analytics
       "Dividend Streak"::INTEGER                AS dividend_streak,
       "Dividend Per Share (LTM)"                AS dividend_per_share_ltm,
       "Dividend Record (Amount)"                AS dividend_amount,
       "Common Dividends Paid (LTM)"             AS common_dividends_ltm,
       "Common Dividends Paid (FY)"              AS common_dividends_fy,
       "Buyback Yield (LTM)"                     AS buyback_yield,
       -- Yield Trends
       "Div Yield (Ind)" - "Div Yield (-1FYInd)" AS div_yield_change_1y,
       "Div Yield (Ind)" - "Div Yield (-2FYInd)" AS div_yield_change_2y,
       "Div Yield (Ind)" - "Div Yield (-3FYInd)" AS div_yield_change_3y,
       "Div Yield (Ind)" - "Div Yield (-5FYInd)" AS div_yield_change_5y,
       "Div Yield (NTM)" - "Div Yield (LTM)"     AS dividend_growth_ntm
FROM postgres.public.equities;
$$;

alter function calc_dividend_yield_comprehensive() owner to postgres;

