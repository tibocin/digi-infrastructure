-- -----------------------------------------------------------------------------
-- File: scripts/sql/fix_query_responses_array_columns.sql
-- Purpose: Convert digi_core.query_responses array-shaped JSON columns to native arrays.
-- Related: digi-core QueryResponse model, tkt:query-responses-array-columns
-- Tags: #postgres #digi-core #query_responses #hotfix #schema
-- -----------------------------------------------------------------------------
-- Apply ONLY against database digi_core on digi-postgres.
-- Do NOT run against cvpunk (or any other app database).
-- Do NOT run alembic upgrade head for stamp a1e2b828c295 (not in digi-core checkout).
--
-- Expected after commit (information_schema / format_type):
--   sources_used, chroma_document_ids, neo4j_node_ids → character varying[]
--   query_embedding → double precision[]
--   response_metadata stays json
--
-- Example (secrets never printed):
--   docker exec -i digi-postgres psql -U digi -d digi_core -v ON_ERROR_STOP=1 \
--     < scripts/sql/fix_query_responses_array_columns.sql
-- -----------------------------------------------------------------------------

BEGIN;

-- 1) Abort if any non-null value in the four columns is not a JSON array.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM query_responses
    WHERE (sources_used IS NOT NULL AND json_typeof(sources_used) <> 'array')
       OR (chroma_document_ids IS NOT NULL AND json_typeof(chroma_document_ids) <> 'array')
       OR (neo4j_node_ids IS NOT NULL AND json_typeof(neo4j_node_ids) <> 'array')
       OR (query_embedding IS NOT NULL AND json_typeof(query_embedding) <> 'array')
  ) THEN
    RAISE EXCEPTION
      'fix_query_responses_array_columns: abort — non-null non-array JSON in sources_used / chroma_document_ids / neo4j_node_ids / query_embedding';
  END IF;
END $$;

-- 2) String id columns: json arrays → character varying[] (NULLs stay NULL).
ALTER TABLE query_responses
  ALTER COLUMN sources_used TYPE character varying[]
  USING (
    CASE
      WHEN sources_used IS NULL THEN NULL
      ELSE ARRAY(SELECT json_array_elements_text(sources_used))
    END
  );

ALTER TABLE query_responses
  ALTER COLUMN chroma_document_ids TYPE character varying[]
  USING (
    CASE
      WHEN chroma_document_ids IS NULL THEN NULL
      ELSE ARRAY(SELECT json_array_elements_text(chroma_document_ids))
    END
  );

ALTER TABLE query_responses
  ALTER COLUMN neo4j_node_ids TYPE character varying[]
  USING (
    CASE
      WHEN neo4j_node_ids IS NULL THEN NULL
      ELSE ARRAY(SELECT json_array_elements_text(neo4j_node_ids))
    END
  );

-- 3) query_embedding: json array of numbers → double precision[].
ALTER TABLE query_responses
  ALTER COLUMN query_embedding TYPE double precision[]
  USING (
    CASE
      WHEN query_embedding IS NULL THEN NULL
      ELSE ARRAY(
        SELECT json_array_elements_text(query_embedding)::double precision
      )
    END
  );

-- 4) response_metadata remains json (no change).

COMMIT;
