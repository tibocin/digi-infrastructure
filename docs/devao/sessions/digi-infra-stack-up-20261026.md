<!--
File: docs/devao/sessions/digi-infra-stack-up-20261026.md
Purpose: Tracked brief for tkt:infra-stack-up private compose session
Related: docs/PRIVATE_STACK.md, .devao/chalkboard (gitignored)
Tags: #devao #session #infra-stack-up
-->

# Session digi-infra-stack-up-20261026

- **Goal:** Ship `tkt:infra-stack-up` — PCS + Postgres, Neo4j, Qdrant, Redis healthy on private compose; ports unpublished; no secrets/origin in git.
- **Pack:** dossierium (Digi product frame); ChangeIntent `target_system=git`.
- **Non-goals:** ask.tibocin.xyz cutover; `tkt:hosted-showcase`.
- **Command:** `make private-up` → `make private-health` → `make assert-private-ports` (see docs/PRIVATE_STACK.md).
