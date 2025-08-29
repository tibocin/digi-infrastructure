#!/bin/bash
# -----------------------------------------------------------------------------
# File: scripts/health-check.sh
# Purpose: Infrastructure health check script
# Related: docker-compose.yml
# Tags: health-check, monitoring, infrastructure
# -----------------------------------------------------------------------------

set -e

# Get Qdrant API key from .env file
if [ -f .env ]; then
    QDRANT_API_KEY=$(grep '^QDRANT_API_KEY=' .env | cut -d'=' -f2)
    export QDRANT_API_KEY
fi

echo "🔍 Checking infrastructure health..."

# Check if containers are running
echo "📦 Checking container status..."
docker compose ps

# Check PostgreSQL
echo "🐘 Checking PostgreSQL..."
docker compose exec -T postgres pg_isready -U digi

# Check Neo4j
echo "🕸️  Checking Neo4j..."
curl -f http://localhost:7474/browser/ || echo "Neo4j not accessible"

# Check Qdrant
echo "🔍 Checking Qdrant..."
curl -H "api-key: ${QDRANT_API_KEY:-}" -f http://localhost:6333/collections || echo "Qdrant not accessible"

# Check Redis
echo "🔴 Checking Redis..."
docker compose exec -T redis redis-cli ping

# Check Prometheus
echo "📊 Checking Prometheus..."
curl -f http://localhost:9090/-/healthy || echo "Prometheus not accessible"

# Check Grafana
echo "📈 Checking Grafana..."
curl -f http://localhost:3000/api/health || echo "Grafana not accessible"

echo "✅ Infrastructure health check completed"


