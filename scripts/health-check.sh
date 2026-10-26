#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# File: scripts/health-check.sh
# Purpose: Health-check PCS + data stores via private compose network
# Related: docker-compose.yml, docs/PRIVATE_STACK.md, tkt:infra-stack-up
# Tags: health-check, monitoring, infrastructure, private-network
# -----------------------------------------------------------------------------
# Why: private mode has no host port publish; use compose exec, not localhost.
# Side effects: none. Does not print secret values or origin addresses.
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${COMPOSE_FILE:-}" ]]; then
  dc() { docker compose "$@"; }
else
  # Default: private stack (skip auto-loaded override).
  dc() { docker compose -f docker-compose.yml -f docker-compose.prod.yml "$@"; }
fi

# Why: compose up -d returns before healthchecks finish (40s start_period on neo4j/pcs).
retry() {
  local description="$1"
  shift
  local max_wait="${HEALTH_WAIT_SECONDS:-90}"
  local interval="${HEALTH_RETRY_INTERVAL:-3}"
  local start
  start=$(date +%s)
  while true; do
    if "$@"; then
      return 0
    fi
    if (( $(date +%s) - start >= max_wait )); then
      echo "${description}: not ready after ${max_wait}s" >&2
      "$@"
      return 1
    fi
    sleep "$interval"
  done
}

check_redis() {
  dc exec -T redis redis-cli ping | grep -qi PONG
}

echo "Checking infrastructure health (compose exec)..."

echo "Container status..."
dc ps

echo "PostgreSQL..."
retry "PostgreSQL" dc exec -T postgres pg_isready -U "${POSTGRES_USER:-digi}"

echo "Neo4j..."
retry "Neo4j" dc exec -T neo4j wget -q -O /dev/null http://localhost:7474

echo "Qdrant..."
retry "Qdrant" dc exec -T qdrant sh -c 'nc -z localhost 6333'

echo "Redis..."
retry "Redis" check_redis

echo "PCS..."
retry "PCS" dc exec -T pcs curl -sf "http://localhost:8000/api/v1/health/" >/dev/null

echo "Infrastructure health check completed."
