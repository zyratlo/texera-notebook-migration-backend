-- ============================================
-- 1. Connect to the texera_db database
-- ============================================
\c texera_db

SET search_path TO texera_db;

-- ============================================
-- 2. Create the tables to store wid, mapping, and notebook
-- ============================================

BEGIN;

CREATE TABLE notebook (
    wid         INT     NOT NULL,
    nid         SERIAL  NOT NULL,
    notebook    JSONB   NOT NULL,
    PRIMARY KEY (wid, nid),
    FOREIGN KEY (wid) REFERENCES workflow(wid) ON DELETE CASCADE
);

CREATE TABLE workflow_notebook_mapping (
    wid         INT     NOT NULL,
    vid         INT     NOT NULL,
    nid         INT     NOT NULL,
    mapping     JSONB   NOT NULL,
    PRIMARY KEY (vid, nid),
    FOREIGN KEY (vid) REFERENCES workflow_version(vid) ON DELETE CASCADE,
    FOREIGN KEY (wid, nid) REFERENCES notebook(wid, nid) ON DELETE CASCADE
);

COMMIT;