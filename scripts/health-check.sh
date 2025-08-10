#!/bin/bash
# -----------------------------------------------------------------------------
# File: scripts/health-check.sh
# Purpose: Infrastructure health check script
# Related: docker-compose.yml
# Tags: health-check, monitoring, infrastructure
# -----------------------------------------------------------------------------

set -e

echo "🔍 Checking infrastructure health..."

# Check if containers are running
echo "📦 Checking container status..."
docker-compose ps

# Check PostgreSQL
echo "🐘 Checking PostgreSQL..."
docker-compose exec -T postgres pg_isready -U digi

# Check Neo4j
echo "🕸️  Checking Neo4j..."
curl -f http://localhost:7474/browser/ || echo "Neo4j not accessible"

# Check ChromaDB
echo "🔍 Checking ChromaDB..."
curl -f http://localhost:8001/api/v1/heartbeat || echo "ChromaDB not accessible"

# Check Redis
echo "🔴 Checking Redis..."
docker-compose exec -T redis redis-cli ping

# Check Prometheus
echo "📊 Checking Prometheus..."
curl -f http://localhost:9090/-/healthy || echo "Prometheus not accessible"

# Check Grafana
echo "📈 Checking Grafana..."
curl -f http://localhost:3000/api/health || echo "Grafana not accessible"

echo "✅ Infrastructure health check completed"


