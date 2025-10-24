create table projects
(
    id                  VARCHAR              not null,
    data                TEXT,
    owner               TEXT,
    name                TEXT,
    default_environment VARCHAR default ''   not null,
    tags                TEXT    default '[]' not null
);

create unique index sqlite_autoindex_projects_1
    on projects (id collate BINARY);

create unique index sqlite_autoindex_projects_2
    on projects (owner collate BINARY, name collate BINARY);

