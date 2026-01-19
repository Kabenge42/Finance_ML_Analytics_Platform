create function calc_gaap_revision_features()
    returns TABLE
            (
                ticker                       text,
                gaap_revision_momentum       numeric,
                gaap_revision_1m             numeric,
                gaap_revision_3m             numeric,
                gaap_revision_6m             numeric,
                gaap_revision_1y             numeric,
                gaap_vs_norm_revision_spread numeric,
                gaap_revision_acceleration   numeric,
                gaap_positive_revision_flag  integer,
                revision_quality_divergence  numeric
            )
    language sql
as
$$
SELECT "Ticker"                                                                    AS ticker,
       -- GAAP EPS Revision Momentum (weighted average)
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15                    AS gaap_revision_momentum,

       -- Individual GAAP Revisions
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                        AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                        AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                        AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                        AS gaap_revision_1y,

       -- GAAP vs Normalized Revision Spread (quality signal)
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"      AS gaap_vs_norm_revision_spread,

       -- GAAP Revision Acceleration (1M vs 6M)
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 6M)" AS gaap_revision_acceleration,

       -- GAAP Positive Revision Flag (all periods positive)
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS gaap_positive_revision_flag,

       -- Revision Quality Divergence (GAAP vs Adjusted moving differently)
       ABS(("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
           ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)"))
                                                                                   AS revision_quality_divergence

FROM postgres.public.equities;
$$;

alter function calc_gaap_revision_features() owner to postgres;

