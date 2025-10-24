create table alembic_version
(
    version_num VARCHAR(32) not null
);

create unique index sqlite_autoindex_alembic_version_1
    on alembic_version (version_num collate BINARY);

