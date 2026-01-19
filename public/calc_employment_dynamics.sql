create function calc_employment_dynamics()
    returns TABLE
            (
                ticker                    text,
                fte_growth_2y_pct         numeric,
                fte_acceleration          numeric,
                workforce_volatility      numeric,
                hiring_intensity          numeric,
                productivity_trend        numeric,
                headcount_vs_revenue      numeric,
                workforce_efficiency_gain numeric,
                layoff_risk_flag          integer,
                rapid_hiring_flag         integer,
                sustainable_growth_flag   integer
            )
    language sql
as
$$
SELECT "Ticker"                                           AS ticker,
       -- FTE Growth 2Y %
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                            AS fte_growth_2y_pct,

       -- FTE Acceleration (1Y growth vs 3Y CAGR)
       CASE
           WHEN "Full Time Employees (-1FY)" > 0 AND "Full Time Employees (-3FY)" > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) -
                    (POWER("Full Time Employees (FY)" / NULLIF("Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)
           END * 100                                      AS fte_acceleration,

       -- Workforce Volatility (std dev of growth rates, simplified)
       ABS(("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
           NULLIF("Full Time Employees (-1FY)", 0) -
           ("Full Time Employees (-1FY)" - "Full Time Employees (-2FY)") /
           NULLIF("Full Time Employees (-2FY)", 0)) * 100 AS workforce_volatility,

       -- Hiring Intensity (FTE growth / Revenue growth)
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) /
                    NULLIF((("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                            NULLIF(ABS("Total Revenues (-1FY)"), 0)), 0)
           END                                            AS hiring_intensity,

       -- Productivity Trend (Revenue per employee growth)
       CASE
           WHEN "Full Time Employees (FY)" > 0 AND "Full Time Employees (-1FY)" > 0
               THEN (("Total Revenues (FY)" / "Full Time Employees (FY)") -
                     ("Total Revenues (-1FY)" / "Full Time Employees (-1FY)")) /
                    NULLIF(ABS("Total Revenues (-1FY)" / "Full Time Employees (-1FY)"), 0) * 100
           END                                            AS productivity_trend,

       -- Headcount vs Revenue Growth Alignment
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100)    AS headcount_vs_revenue,

       -- Workforce Efficiency Gain (revenue growth > headcount growth)
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0) -
                     ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) * 100
           ELSE 0
           END                                            AS workforce_efficiency_gain,

       -- Layoff Risk Flag (declining headcount + declining revenue)
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                            AS layoff_risk_flag,

       -- Rapid Hiring Flag (headcount growth > 20%)
       CASE
           WHEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0
           END                                            AS rapid_hiring_flag,

       -- Sustainable Growth Flag (revenue growth > headcount growth > 0)
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               AND ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0
           END                                            AS sustainable_growth_flag

FROM postgres.public.equities;
$$;

alter function calc_employment_dynamics() owner to postgres;

