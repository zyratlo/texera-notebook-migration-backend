-- ============================================
-- 1. Connect to the texera_db database
-- ============================================
\c texera_db

SET search_path TO texera_db;

-- ============================================
-- 2. Create the tables to store wid, mapping, and notebook
-- ============================================

BEGIN;

CREATE TABLE notebook_migration_notebook_data (
    wid         INT     NOT NULL    PRIMARY KEY,
    notebook    JSONB   NOT NULL,
    FOREIGN KEY (wid) REFERENCES workflow(wid) ON DELETE CASCADE
);

CREATE TABLE notebook_migration_mapping_data (
    wid         INT     NOT NULL,
    version     INT     NOT NULL,
    mapping     JSONB   NOT NULL,
    PRIMARY KEY (wid, version),
    FOREIGN KEY (wid) REFERENCES workflow(wid) ON DELETE CASCADE
);

COMMIT;