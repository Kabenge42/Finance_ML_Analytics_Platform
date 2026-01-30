create table equities_master_table
(
    column_name    text                          not null,
    role           text                          not null,
    column_count   integer,
    description    text,
    ddl_equivalent text,
    updated_at     timestamp default CURRENT_TIMESTAMP,
    column_alias   text      default 'n/a'::text not null
);

comment on table equities_master_table is 'Metadata table documenting all columns in the equities table with their roles, aliases, and DDL definitions';

alter table equities_master_table
    owner to postgres;

create index idx_equities_master_table_role
    on equities_master_table (role);

create index idx_equities_master_table_ddl
    on equities_master_table (ddl_equivalent);

create index equities_master_table_column_alias_index
    on equities_master_table (column_alias) include (role, column_name);

