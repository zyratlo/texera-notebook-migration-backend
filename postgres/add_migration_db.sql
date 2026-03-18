-- ============================================
-- 1. Connect to the texera_db database
-- ============================================
\c texera_db

SET search_path TO texera_db;

-- ============================================
-- 2. Delete tables if they already exist
-- ============================================

BEGIN;

DROP TABLE IF EXISTS notebook CASCADE;
DROP TABLE IF EXISTS workflow_notebook_mapping CASCADE;

-- ============================================
-- 3. Create the tables to store notebook and mapping
-- ============================================

CREATE TABLE notebook (
    nid         SERIAL  NOT NULL PRIMARY KEY,
    wid         INT     NOT NULL,
    notebook    JSONB   NOT NULL,
    FOREIGN KEY (wid) REFERENCES workflow(wid) ON DELETE CASCADE
);

CREATE TABLE workflow_notebook_mapping (
    wid         INT     NOT NULL,
    vid         INT     NOT NULL,
    nid         INT     NOT NULL,
    mapping     JSONB   NOT NULL,
    PRIMARY KEY (wid, vid, nid),
    FOREIGN KEY (wid) REFERENCES workflow(wid) ON DELETE CASCADE,
    FOREIGN KEY (vid) REFERENCES workflow_version(vid) ON DELETE CASCADE,
    FOREIGN KEY (nid) REFERENCES notebook(nid) ON DELETE CASCADE
);

COMMIT;