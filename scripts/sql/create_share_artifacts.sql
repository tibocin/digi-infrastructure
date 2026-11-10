-- -----------------------------------------------------------------------------
-- File: digi-infrastructure/scripts/sql/create_share_artifacts.sql
-- Purpose: Durable ShareArtifact table for published/draft/revoked resumes.
-- Related: digi-core share_artifacts/orm.py, tkt:cvpunk-unify-artifact-store
-- Tags: #operations #postgres #share-artifacts #resume
-- -----------------------------------------------------------------------------
-- Why: InMemoryShareArtifactStore loses published resumes on container restart.
-- Host: digi-postgres · database digi_core only · do not touch cvpunk.
-- Apply:
--   docker exec -i digi-postgres psql -U digi -d digi_core -v ON_ERROR_STOP=1 \
--     < scripts/sql/create_share_artifacts.sql

BEGIN;

CREATE TABLE IF NOT EXISTS share_artifacts (
    share_slug       VARCHAR(128) PRIMARY KEY,
    role_key         VARCHAR(128) NOT NULL DEFAULT '',
    status           VARCHAR(32)  NOT NULL,
    title            VARCHAR(256) NOT NULL DEFAULT '',
    snapshot_markdown TEXT        NOT NULL DEFAULT '',
    snapshot_json    JSON         NOT NULL DEFAULT '{}'::json,
    chat_excerpt     TEXT         NOT NULL DEFAULT '',
    content_sha256   VARCHAR(64)  NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    published_at     TIMESTAMPTZ  NULL,
    revoked_at       TIMESTAMPTZ  NULL
);

CREATE INDEX IF NOT EXISTS ix_share_artifacts_status
    ON share_artifacts (status);

CREATE INDEX IF NOT EXISTS ix_share_artifacts_updated_at
    ON share_artifacts (updated_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE share_artifacts TO digi_core_user;

COMMIT;
