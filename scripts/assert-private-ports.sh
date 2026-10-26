#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# File: scripts/assert-private-ports.sh
# Purpose: Fail if PCS or data-store ports are published on the host
# Related: docker-compose.yml, docker-compose.prod.yml, docs/PRIVATE_STACK.md
# Tags: #security #private-network #infra-stack-up
# -----------------------------------------------------------------------------
# Why: public hosts must not expose Postgres/Neo4j/Qdrant/Redis/PCS.
# Side effects: none (read-only inspect). Output: exit 0/1.
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${COMPOSE_FILE:-}" ]]; then
  # Caller set COMPOSE_FILE (colon-separated); do not force -f flags.
  dc() { docker compose "$@"; }
else
  dc() { docker compose -f docker-compose.yml -f docker-compose.prod.yml "$@"; }
fi

SERVICES=(postgres neo4j qdrant redis pcs)

echo "Asserting private ports..."

failures=0
for svc in "${SERVICES[@]}"; do
  bindings="$(dc ps "$svc" --format '{{.Ports}}' 2>/dev/null || true)"
  bindings="$(echo "$bindings" | tr -d '\r' | head -n 1)"
  if echo "$bindings" | grep -Eq '[0-9]+->'; then
    echo "FAIL: $svc publishes host ports: $bindings"
    failures=$((failures + 1))
    continue
  fi
  echo "OK: $svc unpublished"
done

if [[ "$failures" -ne 0 ]]; then
  echo "Private port assertion failed ($failures service(s))."
  exit 1
fi

echo "All PCS and data-store ports are unpublished on the host."
exit 0
