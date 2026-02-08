create table analytics.calculated_features_registry
(
    feature_key        text not null
        primary key,
    feature_alias      text not null,
    category           text not null,
    source_function    text
        constraint feature_registry_metadata_function_name_fk
            references analytics.feature_registry_metadata,
    description        text,
    source_columns     text[],
    primary_source_col text,
    calculation_type   text,
    data_type          text,
    updated_at         timestamp default CURRENT_TIMESTAMP
);

alter table analytics.calculated_features_registry
    owner to postgres;

create index idx_calc_features_category
    on analytics.calculated_features_registry (category);

create index idx_calc_features_source_fn
    on analytics.calculated_features_registry (source_function);

create index idx_calc_features_primary_col
    on analytics.calculated_features_registry (primary_source_col);

