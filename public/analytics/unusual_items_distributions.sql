create table analytics.unusual_items_distributions
(
    other_unusual_items_ltm   text,
    impairment_goodwill_ltm   text,
    asset_writedown_ltm       text,
    restructuring_charges_ltm text,
    total_unusual_items       text,
    unusual_items_to_revenue  text,
    unusual_items_to_ebitda   text,
    has_unusual_items_flag    text,
    earnings_quality_impact   text
);

alter table analytics.unusual_items_distributions
    owner to postgres;

