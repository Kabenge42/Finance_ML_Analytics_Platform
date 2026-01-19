create function calc_all_enhanced_features()
    returns TABLE
            (
                ticker                       text,
                name                         text,
                sector                       text,
                industry                     text,
                revenue_fq_vs_5yavg          numeric,
                revenue_qoq_growth           numeric,
                revenue_seasonality_factor   numeric,
                sga_to_revenue_fy            numeric,
                sga_trend_yoy                numeric,
                marketing_to_revenue_fy      numeric,
                operating_leverage_score     numeric,
                cost_efficiency_trend        numeric,
                price_to_tbv                 numeric,
                tbv_per_share                numeric,
                tangible_equity_ratio        numeric,
                tbv_margin_of_safety         numeric,
                net_interest_income          numeric,
                interest_coverage_ebitda     numeric,
                financial_income_quality     numeric,
                interest_burden_ratio        numeric,
                price_momentum_3y            numeric,
                price_momentum_5y            numeric,
                momentum_acceleration_3y     numeric,
                long_term_trend_score        numeric,
                momentum_consistency         numeric,
                secular_trend_flag           integer,
                beta_2y                      numeric,
                beta_trend_short             numeric,
                beta_trend_long              numeric,
                beta_stability               numeric,
                beta_regime_change           integer,
                systematic_risk_score        numeric,
                defensive_stock_flag         integer,
                high_beta_flag               integer,
                estimate_skew_fy1e           numeric,
                revenue_consensus_confidence numeric,
                upside_to_consensus          numeric,
                total_unusual_items          numeric,
                unusual_to_net_income_ratio  numeric,
                clean_earnings_flag          integer,
                recurring_unusual_flag       integer,
                earnings_noise_score         numeric,
                quality_adjusted_ni          numeric,
                pe_forward_discount          numeric,
                peg_adjusted                 numeric,
                peg_forward                  numeric,
                earnings_yield               numeric,
                fcf_yield                    numeric,
                shareholder_yield_total      numeric,
                valuation_composite_score    numeric,
                defensive_interval           numeric,
                working_capital_turnover     numeric,
                liquidity_score              numeric,
                working_capital_efficiency   numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT
    -- Identity
    "Ticker"                                                                      AS ticker,
    "Name"                                                                        AS name,
    "Sector"                                                                      AS sector,
    "Industry"                                                                    AS industry,

    -- Revenue Quarterly Features (from calc_revenue_quarterly_features)
    "Total Revenues (FQ)" / NULLIF("Total Revenues (5YAVGFQ)", 0)                 AS revenue_fq_vs_5yavg,
    ("Total Revenues (FQ)" * 4 - "Total Revenues (LTM)") /
    NULLIF("Total Revenues (LTM)", 0) * 100                                       AS revenue_qoq_growth,
    "Total Revenues (FQ)" / NULLIF("Total Revenues (LTM)" / 4, 0)                 AS revenue_seasonality_factor,

    -- Cost Structure Features (from calc_cost_structure_features)
    "Selling General & Admin Expenses/Total (FY)" /
    NULLIF("Total Revenues (FY)", 0) * 100                                        AS sga_to_revenue_fy,
    ("Selling General & Admin Expenses/Total (FY)" -
     "Selling General & Admin Expenses/Total (-1FY)") /
    NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0) * 100         AS sga_trend_yoy,
    "Marketing Expenses (FY)" / NULLIF("Total Revenues (FY)", 0) * 100            AS marketing_to_revenue_fy,
    CASE
        WHEN ABS("Total Revenues (-1FY)") > 0 AND ABS("Total Operating Expenses (LTM)") > 0
            THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0)) /
                 NULLIF((("Selling General & Admin Expenses/Total (FY)" -
                          "Selling General & Admin Expenses/Total (-1FY)") /
                         NULLIF(ABS("Selling General & Admin Expenses/Total (-1FY)"), 0)), 0)
        END                                                                       AS operating_leverage_score,
    ("Selling General & Admin Expenses/Total (-1FY)" / NULLIF("Total Revenues (-1FY)", 0)) -
    ("Selling General & Admin Expenses/Total (FY)" / NULLIF("Total Revenues (FY)", 0))
                                                                                  AS cost_efficiency_trend,

    -- Tangible Book Features (from calc_tangible_book_features)
    "Last Price" / NULLIF("TBV (LTM)" / NULLIF("Shrs Out", 0), 0)                 AS price_to_tbv,
    "TBV (LTM)" / NULLIF("Shrs Out", 0)                                           AS tbv_per_share,
    "TBV (LTM)" / NULLIF("Total Equity (LTM)", 0)                                 AS tangible_equity_ratio,
    ("TBV (LTM)" - "Market Cap") / NULLIF("TBV (LTM)", 0) * 100                   AS tbv_margin_of_safety,

    -- Interest Income Features (from calc_interest_income_features)
    "Interest Income On Investments (LTM)" - "Interest Expense/Total (LTM)"       AS net_interest_income,
    "EBITDA (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage_ebitda,
    "Operating Income (LTM)" /
    NULLIF("Operating Income (LTM)" + "Interest Income On Investments (LTM)", 0)  AS financial_income_quality,
    "Interest Expense/Total (LTM)" / NULLIF("EBIT (LTM)", 0)                      AS interest_burden_ratio,

    -- Long-Term Momentum Features (from calc_long_term_momentum_features)
    ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 100         AS price_momentum_3y,
    ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 100         AS price_momentum_5y,
    (POWER("Last Price" / NULLIF("Price (3Y Ago)", 0), 1.0 / 3.0) - 1) -
    (POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1)            AS momentum_acceleration_3y,
    (("Last Price" - "Price (1Y Ago)") / NULLIF("Price (1Y Ago)", 0) * 0.5 +
     ("Last Price" - "Price (3Y Ago)") / NULLIF("Price (3Y Ago)", 0) * 0.3 +
     ("Last Price" - "Price (5Y Ago)") / NULLIF("Price (5Y Ago)", 0) * 0.2) * 100 AS long_term_trend_score,
    CASE
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") AND ("Last Price" > "Price (3Y Ago)") THEN 1.0
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") AND
             ("Last Price" > "Price (1Y Ago)") THEN 0.75
        WHEN ("Last Price" > "Price (1M Ago)") AND ("Last Price" > "Price (3M Ago)") THEN 0.5
        WHEN ("Last Price" > "Price (1M Ago)") THEN 0.25
        ELSE 0
        END                                                                       AS momentum_consistency,
    CASE
        WHEN POWER("Last Price" / NULLIF("Price (5Y Ago)", 0), 1.0 / 5.0) - 1 > 0.10 THEN 1
        ELSE 0
        END                                                                       AS secular_trend_flag,

    -- Beta Risk Features (from calc_beta_risk_features)
    "Beta (2Y)"                                                                   AS beta_2y,
    "Beta (1Y)" - "Beta (2Y)"                                                     AS beta_trend_short,
    "Beta (2Y)" - "Beta (5Y)"                                                     AS beta_trend_long,
    CASE
        WHEN GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
             LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") > 0
            THEN 1.0 / (GREATEST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)") -
                        LEAST("Beta (1Y)", "Beta (2Y)", "Beta (5Y)"))
        ELSE 1.0
        END                                                                       AS beta_stability,
    CASE WHEN ABS("Beta (1Y)" - "Beta (5Y)") > 0.3 THEN 1 ELSE 0 END              AS beta_regime_change,
    "Beta (1Y)" * 0.5 + "Beta (2Y)" * 0.3 + "Beta (5Y)" * 0.2                     AS systematic_risk_score,
    CASE
        WHEN "Beta (1Y)" < 0.8 AND "Beta (2Y)" < 0.8 AND "Beta (5Y)" < 0.8 THEN 1
        ELSE 0
        END                                                                       AS defensive_stock_flag,
    CASE WHEN "Beta (1Y)" > 1.3 AND "Beta (2Y)" > 1.3 THEN 1 ELSE 0 END           AS high_beta_flag,

    -- Revenue Estimate Consensus (from calc_revenue_estimate_consensus)
    ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
    NULLIF("Revenues - Est Med (FY1E)", 0) * 100                                  AS estimate_skew_fy1e,
    GREATEST(0, LEAST(100,
                      100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                NULLIF("Revenues - Est Med (FY1E)", 0) * 100)))   AS revenue_consensus_confidence,
    ("Revenues - Est Med (FY1E)" - "Total Revenues (LTM)") /
    NULLIF("Total Revenues (LTM)", 0) * 100                                       AS upside_to_consensus,

    -- Unusual Items Features (from calc_unusual_items_features)
    ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
    ABS("Restructuring Charges (LTM)") + ABS("Merger & Restructuring Charges (LTM)") +
    ABS("Gain (Loss) On Sale Of Assets (LTM)") +
    ABS("Other Unusual Items/Total (LTM)")                                        AS total_unusual_items,
    (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
     ABS("Restructuring Charges (LTM)") + ABS("Other Unusual Items/Total (LTM)")) /
    NULLIF(ABS("Net Income - (IS) (LTM)"), 0)                                     AS unusual_to_net_income_ratio,
    CASE
        WHEN ABS("Impairment of Goodwill (LTM)") < 1 AND ABS("Asset Writedown (LTM)") < 1 AND
             ABS("Restructuring Charges (LTM)") < 1 AND ABS("Other Unusual Items/Total (LTM)") < 1 THEN 1
        ELSE 0
        END                                                                       AS clean_earnings_flag,
    CASE
        WHEN (CASE WHEN ABS("Impairment of Goodwill (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Impairment of Goodwill (-1FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (FY)") > 0 THEN 1 ELSE 0 END +
              CASE WHEN ABS("Restructuring Charges (-1FY)") > 0 THEN 1 ELSE 0 END) >= 3 THEN 1
        ELSE 0
        END                                                                       AS recurring_unusual_flag,
    LEAST(100, (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                ABS("Restructuring Charges (LTM)") + ABS("Other Unusual Items/Total (LTM)")) /
               NULLIF(ABS("Net Income - (IS) (LTM)"), 0) * 100)                   AS earnings_noise_score,
    "Net Income - (IS) (LTM)" + "Impairment of Goodwill (LTM)" +
    "Asset Writedown (LTM)" + "Restructuring Charges (LTM)"                       AS quality_adjusted_ni,

    -- Enhanced Valuation Ratios (from calc_enhanced_valuation_ratios)
    ("P/E (LTM)" - "Forward P/E") / NULLIF("P/E (LTM)", 0) * 100                  AS pe_forward_discount,
    CASE
        WHEN "Total Revenues/CAGR (5Y FY)" > 0
            THEN "P/E (LTM)" / NULLIF("Total Revenues/CAGR (5Y FY)", 0)
        END                                                                       AS peg_adjusted,
    CASE
        WHEN "Revenues - Est YoY % (FY1E)" > 0
            THEN "Forward P/E" / NULLIF("Revenues - Est YoY % (FY1E)", 0)
        END                                                                       AS peg_forward,
    CASE WHEN "P/E (LTM)" > 0 THEN 1.0 / "P/E (LTM)" * 100 END                    AS earnings_yield,
    "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                                   AS fcf_yield,
    COALESCE("Div Yield (LTM)", 0) + COALESCE("Buyback Yield (LTM)", 0)           AS shareholder_yield_total,
    GREATEST(0, LEAST(100,
                      50 -
                      (CASE WHEN "P/E (LTM)" < 15 THEN 10 WHEN "P/E (LTM)" < 25 THEN 0 ELSE -10 END) -
                      (CASE WHEN "PEG Ratio" < 1 THEN 15 WHEN "PEG Ratio" < 2 THEN 5 ELSE -5 END) -
                      (CASE
                           WHEN "EV/EBITDA (LTM)" < 10 THEN 10
                           WHEN "EV/EBITDA (LTM)" < 15 THEN 0
                           ELSE -10 END) +
                      (CASE
                           WHEN "FCF (LTM)" / NULLIF("Market Cap", 0) * 100 > 5 THEN 15
                           ELSE 0 END)))                                          AS valuation_composite_score,

    -- Working Capital Deep Features (from calc_working_capital_deep_features)
    ("Total Current Assets (LTM)" / NULLIF("Total Operating Expenses (LTM)" / 365, 0))
                                                                                  AS defensive_interval,
    "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turnover,
    GREATEST(0, LEAST(100,
                      (CASE
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 2
                               THEN 40
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1.5
                               THEN 30
                           WHEN "Total Current Assets (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 1
                               THEN 20
                           ELSE 0 END) +
                      (CASE
                           WHEN ("Total Current Assets (LTM)" - "Inventory (LTM)") /
                                NULLIF("Total Current Liabilities (LTM)", 0) >= 1 THEN 30
                           ELSE 15 END) +
                      (CASE
                           WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.5
                               THEN 30
                           WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) >= 0.2
                               THEN 15
                           ELSE 0 END)))                                          AS liquidity_score,
    "Total Revenues (LTM)" / NULLIF(ABS("Working Capital (LTM)"), 0)              AS working_capital_efficiency

FROM postgres.public.equities;
$$;

alter function calc_all_enhanced_features() owner to postgres;

