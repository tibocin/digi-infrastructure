# Infrastructure Repository Design

## Overview

The `digi-infrastructure` repository contains only the shared database infrastructure, monitoring, and backup services. This provides a clean separation between infrastructure and application concerns.

## Repository Structure

```
digi-infrastructure/
├── docker-compose.yml              # Database containers only
├── docker-compose.override.yml     # Development overrides
├── docker-compose.prod.yml         # Production configuration
├── scripts/
│   ├── init-multiple-databases.sh  # PostgreSQL multi-db setup
│   ├── health-check.sh            # Infrastructure health checks
│   └── backup.sh                  # Backup orchestration
├── monitoring/
│   ├── prometheus.yml             # Prometheus configuration
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   ├── dashboards/
│   │   │   │   ├── dashboards.yml
│   │   │   │   └── digi-infrastructure.json
│   │   │   └── datasources/
│   │   │       └── prometheus.yml
│   │   └── dashboards/
│   └── alerts/
│       └── alert-rules.yml
├── backup/
│   ├── backup-config.yml          # Backup configuration
│   └── restore.sh                 # Restore scripts
├── config/
│   ├── env.sample                 # Environment template
│   └── env.prod.sample           # Production environment
├── docs/
│   ├── DEPLOYMENT.md             # Deployment guide
│   ├── MONITORING.md             # Monitoring setup
│   └── BACKUP_RESTORE.md         # Backup procedures
├── .env.example                  # Environment variables template
├── .gitignore
├── README.md
└── Makefile                      # Common operations
```

## Docker Compose Configuration

### docker-compose.yml (Core Infrastructure)

```yaml
version: '3.8'

networks:
  digi-net:
    driver: bridge

volumes:
  pg_data:
  neo4j_data:
  qdrant_data:
  qdrant_snapshots:
  redis_data:
  prometheus_data:
  grafana_data:
  backup_cache:

services:
  # PostgreSQL with multi-database support
  postgres:
    image: postgres:15
    container_name: digi-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-digi}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-digi}
      POSTGRES_DB: ${POSTGRES_DB:-digi}
      POSTGRES_MULTIPLE_DATABASES: ${POSTGRES_MULTIPLE_DATABASES:-digi_core,lernmi,beep_boop}
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./scripts/init-multiple-databases.sh:/docker-entrypoint-initdb.d/init-multiple-databases.sh:ro
    networks:
      - digi-net
    ports:
      - "5432:5432"  # Expose for external connections

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5.18
    container_name: digi-neo4j
    restart: unless-stopped
    environment:
      NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD:-digi}"
      NEO4J_dbms_default__database=neo4j
    volumes:
      - neo4j_data:/data
    networks:
      - digi-net
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: digi-qdrant
    restart: unless-stopped
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__STORAGE__STORAGE_PATH: /qdrant/storage
      QDRANT__STORAGE__SNAPSHOTS_PATH: /qdrant/snapshots
    volumes:
      - qdrant_data:/qdrant/storage
      - qdrant_snapshots:/qdrant/snapshots
    networks:
      - digi-net
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API

  # Redis Cache
  redis:
    image: redis:7
    container_name: digi-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - digi-net
    ports:
      - "6379:6379"

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    container_name: digi-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - prometheus_data:/prometheus
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/etc/prometheus/console_libraries"
      - "--web.console.templates=/etc/prometheus/consoles"
      - "--storage.tsdb.retention.time=200h"
      - "--web.enable-lifecycle"
    networks:
      - digi-net

  grafana:
    image: grafana/grafana:latest
    container_name: digi-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_DISABLE_GRAVATAR=true
      - GF_SECURITY_COOKIE_SECURE=true
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    networks:
      - digi-net

  # Backup Service
  backup-sidecar:
    image: restic/restic:latest
    container_name: digi-backup
    depends_on:
      - postgres
      - neo4j
      - qdrant
      - redis
    environment:
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      RESTIC_REPOSITORY: ${RESTIC_REPOSITORY}
      RESTIC_PASSWORD: ${RESTIC_PASSWORD}
    volumes:
      - pg_data:/backup/pg_data:ro
      - neo4j_data:/backup/neo4j_data:ro
      - qdrant_data:/backup/qdrant_data:ro
      - redis_data:/backup/redis_data:ro
      - backup_cache:/root/.cache/restic
    entrypoint: ["/bin/sh", "-c"]
    command: |
      "echo '0 3 * * * restic backup /backup && restic forget --prune --keep-daily 7 --keep-weekly 4 --keep-monthly 6' > /etc/crontabs/root && crond -f -d 8"
    networks:
      - digi-net
```

### docker-compose.override.yml (Development)

```yaml
version: "3.8"

services:
  postgres:
    ports:
      - "5432:5432" # Expose for local development
    environment:
      POSTGRES_MULTIPLE_DATABASES: "digi_core,lernmi,beep_boop,test"

  neo4j:
    ports:
      - "7474:7474" # Browser interface
      - "7687:7687" # Bolt protocol

  qdrant:
    ports:
      - "6333:6333" # HTTP API
      - "6334:6334" # gRPC API

  redis:
    ports:
      - "6379:6379" # Redis CLI

  grafana:
    ports:
      - "3000:3000" # Web interface
```

## Environment Configuration

### .env.example

```bash
# Infrastructure Configuration
POSTGRES_USER=digi
POSTGRES_PASSWORD=change_me
POSTGRES_DB=digi
POSTGRES_MULTIPLE_DATABASES=digi_core,lernmi,beep_boop

# Neo4j Configuration
NEO4J_PASSWORD=change_me

# Monitoring Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change_me

# Backup Configuration
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=your_aws_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
RESTIC_REPOSITORY=s3:https://s3.us-east-2.amazonaws.com/digi-backup
RESTIC_PASSWORD=your_restic_password

# Optional: Qdrant Configuration
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_TELEMETRY_DISABLED=true
```

## Health Check Script

### scripts/health-check.sh

```bash
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

# Check Qdrant
echo "🔍 Checking Qdrant..."
curl -f http://localhost:6333/collections || echo "Qdrant not accessible"

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
```

## Makefile for Common Operations

### Makefile

```makefile
# -----------------------------------------------------------------------------
# File: Makefile
# Purpose: Common infrastructure operations
# Related: docker-compose.yml
# Tags: make, operations, infrastructure
# -----------------------------------------------------------------------------

.PHONY: help up down restart logs ps health backup restore clean

help: ## Show this help message
	@echo "Digi Infrastructure Management"
	@echo "============================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Start infrastructure services
	docker-compose up -d

down: ## Stop infrastructure services
	docker-compose down

restart: ## Restart infrastructure services
	docker-compose restart

logs: ## Show logs from all services
	docker-compose logs -f

ps: ## Show running containers
	docker-compose ps

health: ## Run health checks
	./scripts/health-check.sh

backup: ## Create backup
	docker-compose exec backup-sidecar restic backup /backup

restore: ## Restore from backup (latest)
	docker-compose exec backup-sidecar restic restore latest --target /restore

clean: ## Remove all containers and volumes
	docker-compose down -v
	docker system prune -f

init: ## Initialize databases for all apps
	docker-compose exec postgres psql -U digi -d digi -c "SELECT 1;"

status: ## Show detailed status
	@echo "=== Container Status ==="
	@docker-compose ps
	@echo ""
	@echo "=== Database Status ==="
	@docker-compose exec -T postgres psql -U digi -d digi -c "\l" || echo "PostgreSQL not accessible"
	@echo ""
	@echo "=== Network Status ==="
	@docker network ls | grep digi || echo "No digi networks found"
```

## Deployment Guide

### docs/DEPLOYMENT.md

````markdown
# Infrastructure Deployment Guide

## Prerequisites

1. Docker and Docker Compose installed
2. At least 8GB RAM available
3. 50GB+ disk space
4. AWS S3 bucket for backups (optional)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/digi-infrastructure.git
   cd digi-infrastructure
   ```
````

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

3. **Start infrastructure**

   ```bash
   make up
   ```

4. **Verify health**
   ```bash
   make health
   ```

## Production Deployment

1. **Use production compose file**

   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

2. **Configure monitoring**

   - Set up Grafana dashboards
   - Configure alerting rules
   - Set up log aggregation

3. **Configure backups**
   - Set AWS credentials
   - Test backup and restore procedures

## Connecting Apps

Apps connect to infrastructure using external links:

```yaml
# In app's docker-compose.yml
services:
  my-app:
    external_links:
      - digi-infrastructure_postgres_1:postgres
      - digi-infrastructure_neo4j_1:neo4j
      - digi-infrastructure_qdrant_1:qdrant
      - digi-infrastructure_redis_1:redis
    networks:
      - digi-net
```

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474
- **Qdrant**: http://localhost:6333

## Troubleshooting

1. **Check container status**: `make ps`
2. **View logs**: `make logs`
3. **Health check**: `make health`
4. **Restart services**: `make restart`

```

## Benefits of This Approach

### ✅ **Clear Separation of Concerns**
- Infrastructure managed separately from applications
- Dedicated team can manage infrastructure
- Applications focus on business logic

### ✅ **Independent Scaling**
- Scale infrastructure independently of apps
- Add new database instances without app changes
- Upgrade database versions independently

### ✅ **Simplified App Development**
- Apps only need to connect to existing infrastructure
- No need to manage database containers in app repos
- Consistent infrastructure across all apps

### ✅ **Centralized Monitoring**
- Single monitoring setup for all infrastructure
- Unified backup and restore procedures
- Centralized logging and alerting

### ✅ **Cost Efficiency**
- Shared infrastructure reduces resource usage
- Single backup strategy for all databases
- Unified monitoring and maintenance

## Next Steps

1. **Create the infrastructure repository** with the structure above
2. **Move current database containers** to the new repo
3. **Update app repositories** to use external links
4. **Test the multi-app setup** with the new structure
5. **Deploy and monitor** the new architecture

This approach gives you the flexibility of independent apps while maintaining the efficiency of shared infrastructure!
```
