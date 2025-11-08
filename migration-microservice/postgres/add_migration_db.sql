-- ============================================
-- 1. Create the notebook migration database
-- ============================================
DROP DATABASE IF EXISTS texera_notebook_migration_db;
CREATE DATABASE texera_notebook_migration_db;
\c texera_notebook_migration_db;

-- ============================================
-- 2. Create the table to store wid, mapping, and notebook
-- ============================================
CREATE SCHEMA texera_notebook_migration_db;
SET search_path TO texera_notebook_migration_db;

BEGIN;

CREATE TABLE migration_data (
    wid INT PRIMARY KEY NOT NULL,
    mapping JSONB NOT NULL,
    notebook JSONB NOT NULL
);

COMMIT;