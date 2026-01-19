create function calc_employment_features()
    returns TABLE
            (
                ticker               text,
                revenue_per_employee numeric,
                profit_per_employee  numeric,
                ebitda_per_employee  numeric,
                assets_per_employee  numeric,
                fte_growth_1y_pct    numeric,
                fte_growth_3y_pct    numeric,
                workforce_stability  numeric
            )
    language sql
as
$$
SELECT "Ticker" AS ticker,
       -- Revenue per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS revenue_per_employee,
       -- Profit per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Normalized Net Income (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS profit_per_employee,
       -- EBITDA per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS ebitda_per_employee,
       -- Assets per Employee
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END  AS assets_per_employee,
       -- FTE Growth 1Y
       CASE
           WHEN "Full Time Employees (-1FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                    NULLIF("Full Time Employees (-1FY)", 0) * 100
           END  AS fte_growth_1y_pct,
       -- FTE Growth 3Y
       CASE
           WHEN "Full Time Employees (-3FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
                    NULLIF("Full Time Employees (-3FY)", 0) * 100
           END  AS fte_growth_3y_pct,
       -- Workforce Stability (vs 5Y avg)
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0
               THEN "Full Time Employees (FY)" / NULLIF("Avg Employees (5YAVGFY)", 0)
           END  AS workforce_stability
FROM postgres.public.equities;
$$;

alter function calc_employment_features() owner to postgres;

