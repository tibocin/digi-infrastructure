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

2. **Configure environment**

   ```bash
   cp config/env.example .env
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

## Common Commands

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Health check
make health

# Show status
make status

# Create backup
make backup

# Restore from backup
make restore

# Setup (first time)
make setup
```

## Environment Configuration

Key environment variables:

```bash
# PostgreSQL
POSTGRES_USER=digi
POSTGRES_PASSWORD=your_secure_password
POSTGRES_MULTIPLE_DATABASES=digi_core,lernmi,beep_boop,stackr,revao,satsflow,cvpunk_jbhunter

# Neo4j
NEO4J_PASSWORD=your_secure_password

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_secure_password

# Backup (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
RESTIC_REPOSITORY=s3:https://s3.region.amazonaws.com/bucket
RESTIC_PASSWORD=your_restic_password
```

## Security Considerations

1. **Change default passwords** in `.env`
2. **Use strong passwords** for all services
3. **Configure SSL/TLS** for production
4. **Set up firewall rules** to restrict access
5. **Regular security updates** for all images

## Backup and Restore

### Creating Backups

```bash
# Manual backup
make backup

# Check backup status
docker-compose exec backup-sidecar restic snapshots
```

### Restoring from Backup

```bash
# List available snapshots
docker-compose exec backup-sidecar restic snapshots

# Restore latest
make restore

# Restore specific snapshot
docker-compose exec backup-sidecar restic restore <snapshot-id> --target /restore
```

## Scaling Considerations

1. **Resource limits**: Monitor CPU/memory usage
2. **Storage**: Ensure adequate disk space
3. **Network**: Consider bandwidth requirements
4. **Backup storage**: Plan for backup retention

## Maintenance

### Regular Tasks

1. **Update images**: `make update`
2. **Check health**: `make health`
3. **Review logs**: `make logs`
4. **Monitor metrics**: Access Grafana dashboard

### Troubleshooting

1. **Service not starting**: Check logs with `make logs`
2. **Database connection issues**: Verify credentials in `.env`
3. **Performance issues**: Monitor resource usage
4. **Backup failures**: Check AWS credentials and permissions
