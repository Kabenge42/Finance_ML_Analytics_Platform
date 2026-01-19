create function calc_sentiment_features()
    returns TABLE
            (
                ticker                    text,
                analyst_bullish_pct       numeric,
                analyst_bearish_pct       numeric,
                analyst_conviction        numeric,
                upside_potential          numeric,
                price_target_spread_pct   numeric,
                price_target_revision_1m  numeric,
                price_target_revision_3m  numeric,
                eps_revision_momentum     numeric,
                analyst_rating_normalized numeric,
                analyst_coverage_quality  numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                 AS ticker,
       -- Analyst Distribution
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Strong Buys Ratings" + "# Buys Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bullish_pct,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Sell Ratings" + "# Strong Sell Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bearish_pct,
       -- Analyst Conviction (difference between bullish and bearish)
       ABS(
               CASE
                   WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                         "# Sell Ratings" + "# Strong Sell Ratings") > 0
                       THEN (("# Strong Buys Ratings" + "# Buys Ratings") -
                             ("# Sell Ratings" + "# Strong Sell Ratings")) /
                            NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                                   "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
                   END
       )                                                                        AS analyst_conviction,
       -- Upside Potential (NULLIF handles zero division)
       ("Price Target - Median" - "Last Price") / NULLIF("Last Price", 0) * 100 AS upside_potential,
       -- Price Target Spread
       ("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0) *
       100                                                                      AS price_target_spread_pct,
       -- Price Target Revisions
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0)                                       AS price_target_revision_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0)                                       AS price_target_revision_3m,
       -- EPS Revision Momentum (weighted average)
       COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
       COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
       COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1Y)", 0) *
       0.10                                                                     AS eps_revision_momentum,
       -- Analyst Rating Normalized (1-5 to 0-100)
       ("Analyst Rating" - 1) * 25                                              AS analyst_rating_normalized,
       -- Coverage Quality (LN(1 + x) is always > 0 for positive Market Cap)
       "Price Target - #" / NULLIF(LN(1 + "Market Cap"), 0)                     AS analyst_coverage_quality
FROM postgres.public.equities;
$$;

alter function calc_sentiment_features() owner to postgres;

