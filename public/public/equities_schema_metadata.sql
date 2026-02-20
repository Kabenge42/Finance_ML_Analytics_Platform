create table equities_schema_metadata
(
    column_name    text                        not null
        primary key,
    column_alias   text      default '_'::text not null,
    role           text                        not null,
    column_count   integer,
    description    text,
    ddl_equivalent text,
    updated_at     timestamp default CURRENT_TIMESTAMP
);

comment on table equities_schema_metadata is 'Metadata table documenting all columns in the equities table with their roles, aliases, and DDL definitions';

alter table equities_schema_metadata
    owner to postgres;

create index idx_equities_schema_metadata_role
    on equities_schema_metadata (role);

create index idx_equities_schema_metadata_ddl
    on equities_schema_metadata (ddl_equivalent);

