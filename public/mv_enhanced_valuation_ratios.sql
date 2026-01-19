create materialized view mv_enhanced_valuation_ratios as
SELECT ticker,
       forward_pe,
       trailing_pe,
       pe_forward_discount,
       peg_ratio,
       peg_adjusted,
       peg_forward,
       pe_to_growth,
       ev_to_fcf,
       earnings_yield,
       fcf_yield,
       shareholder_yield_total,
       valuation_composite_score
FROM v_enhanced_valuation_ratios;

alter materialized view mv_enhanced_valuation_ratios owner to postgres;

create index idx_mv_val_ticker
    on mv_enhanced_valuation_ratios (ticker);

create index idx_mv_val_composite
    on mv_enhanced_valuation_ratios (valuation_composite_score);

