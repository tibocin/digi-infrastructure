<!--
File: scripts/sql/README.md
Purpose: Tracked one-shot Postgres hotfixes for shared digi-postgres app DBs.
Related: docs/SCHEMA_MANAGEMENT.md, tkt:query-responses-array-columns
Tags: #sql #postgres #hotfix
-->

# scripts/sql

Tracked SQL hotfixes (not dumps). Dumps stay gitignored via `*.sql`; this
directory is allow-listed in `.gitignore`.

| Script | Database | Notes |
|--------|----------|-------|
| `fix_query_responses_array_columns.sql` | `digi_core` only | Converts json-shaped array columns to native arrays. Never `cvpunk`. |
| `create_share_artifacts.sql` | `digi_core` only | Durable published/draft/revoked resumes for `/resumes` and `/r/{slug}`. |

Apply example:

```bash
docker exec -i digi-postgres psql -U digi -d digi_core -v ON_ERROR_STOP=1 \
  < scripts/sql/fix_query_responses_array_columns.sql
```
