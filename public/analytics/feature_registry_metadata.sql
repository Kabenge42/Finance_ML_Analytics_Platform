create table analytics.feature_registry_metadata
(
    function_name     text not null
        primary key,
    category          text not null,
    feature_count     integer,
    description       text,
    python_equivalent text,
    updated_at        timestamp default CURRENT_TIMESTAMP
);

alter table analytics.feature_registry_metadata
    owner to postgres;

create index idx_feature_registry_category
    on analytics.feature_registry_metadata (category);

create index idx_feature_registry_python_equiv
    on analytics.feature_registry_metadata (python_equivalent);

