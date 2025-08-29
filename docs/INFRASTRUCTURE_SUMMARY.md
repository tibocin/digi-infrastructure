# Digi Infrastructure Repository Summary

## Overview

This repository contains the shared database infrastructure for the Digi ecosystem. It provides PostgreSQL, Neo4j, Qdrant, Redis, and monitoring services that multiple applications can connect to.

## Repository Structure

```
digi-infrastructure/
├── README.md                           # Main documentation
├── Makefile                           # Common operations
├── docker-compose.yml                 # Core infrastructure services
├── docker-compose.override.yml        # Development overrides
├── .gitignore                         # Git ignore rules
├── config/
│   └── env.example                   # Environment template
├── scripts/
│   ├── init-multiple-databases.sh    # PostgreSQL multi-db setup
│   └── health-check.sh               # Infrastructure health checks
├── monitoring/
│   ├── prometheus.yml                # Prometheus configuration
│   └── grafana/
│       └── provisioning/
│           ├── dashboards/
│           │   └── dashboards.yml    # Grafana dashboard config
│           └── datasources/
│               └── prometheus.yml    # Grafana datasource config
├── docs/
│   ├── DEPLOYMENT.md                 # Deployment guide
│   ├── SCHEMA_MANAGEMENT.md          # Schema management guide
│   ├── INFRASTRUCTURE_REPOSITORY.md  # Repository design
│   └── MULTI_APP_DEPLOYMENT.md       # Multi-app strategy
└── backup/                           # Backup configuration (future)
```

## Services Provided

### Database Services

- **PostgreSQL**: Multi-database setup for different apps
- **Neo4j**: Graph database with multi-database support
- **Qdrant**: Vector database with multi-collection support
- **Redis**: Cache with multi-database support

### Monitoring Services

- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

### Backup Services

- **Restic**: Automated backups to S3

## Multi-App Support

This infrastructure supports multiple applications:

- **digi_core**: Main RAG application
- **lernmi**: Learning and evaluation system
- **beep_boop**: Bot application

Each app gets its own:

- PostgreSQL database
- Neo4j database
- Qdrant collection
- Redis database number

## Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/digi-infrastructure.git
cd digi-infrastructure
make setup

# Start services
make up

# Check health
make health

# View monitoring
make monitor
```

## Connecting Applications

Apps connect using external links:

```yaml
# In your app's docker-compose.yml
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

## Key Features

### ✅ **Shared Infrastructure**

- Single set of database containers
- Shared monitoring and backup
- Resource efficiency

### ✅ **Multi-App Isolation**

- Separate databases per app
- Independent schema management
- App-specific credentials

### ✅ **Easy Management**

- Simple Makefile commands
- Health check scripts
- Comprehensive documentation

### ✅ **Production Ready**

- Monitoring stack included
- Automated backups
- Security best practices

## Environment Configuration

Key environment variables:

```bash
# Multi-app databases
POSTGRES_MULTIPLE_DATABASES=digi_core,lernmi,beep_boop

# App-specific connections
DIGI_CORE_DB_URL=postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core
LERNMI_DB_URL=postgresql://lernmi_user:lernmi_pass@postgres:5432/lernmi
BEEP_BOOP_DB_URL=postgresql://beep_boop_user:beep_boop_pass@postgres:5432/beep_boop
```

## Common Commands

```bash
make up          # Start all services
make down        # Stop all services
make health      # Check service health
make logs        # View service logs
make backup      # Create backup
make restore     # Restore from backup
make setup       # Initial setup
make monitor     # Open monitoring interfaces
```

## Monitoring Access

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474
- **Qdrant**: http://localhost:6333

## Next Steps

1. **Deploy infrastructure**: `make setup`
2. **Create app repositories**: Each app gets its own repo
3. **Connect apps**: Use external links in app docker-compose.yml
4. **Monitor and maintain**: Use provided monitoring tools

This infrastructure provides a solid foundation for the multi-app Digi ecosystem!
