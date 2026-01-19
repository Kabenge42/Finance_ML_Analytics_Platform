create table feature_registry_metadata
(
    function_name     text not null
        primary key,
    category          text not null,
    feature_count     integer,
    description       text,
    python_equivalent text
);

alter table feature_registry_metadata
    owner to postgres;

