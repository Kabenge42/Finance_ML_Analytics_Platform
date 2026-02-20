create table calculated_features_registry
(
    feature_key        text not null
        primary key,
    feature_alias      text not null,
    category           text not null,
    source_function    text
        constraint feature_registry_metadata_function_name_fk
            references feature_registry_metadata,
    description        text,
    source_columns     text[],
    primary_source_col text,
    calculation_type   text,
    data_type          text,
    updated_at         timestamp default CURRENT_TIMESTAMP
);

alter table calculated_features_registry
    owner to postgres;

create index idx_calc_features_category
    on calculated_features_registry (category);

create index idx_calc_features_source_fn
    on calculated_features_registry (source_function);

create index idx_calc_features_primary_col
    on calculated_features_registry (primary_source_col);

