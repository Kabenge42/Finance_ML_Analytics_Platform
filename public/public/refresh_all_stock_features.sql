create function refresh_all_stock_features() returns void
    language plpgsql
as
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
END;
$$;

alter function refresh_all_stock_features() owner to postgres;

