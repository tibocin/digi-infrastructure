# Digi-Infrastructure Port Management System

## Overview

The Digi-Infrastructure Port Management System provides centralized port allocation and conflict resolution for all services in the digi ecosystem. This system prevents port conflicts between digi-core, PCS, and other services while maintaining a clear allocation strategy.

## Port Allocation Strategy

### Port Ranges

- **1000-1999**: Core infrastructure services (PostgreSQL, Neo4j, Redis, Qdrant)
- **2000-2999**: Application services (digi-core, PCS, LernMI, Beep-Boop)
- **3000-3999**: Monitoring and observability (Grafana, Prometheus, Jaeger)
- **4000-4999**: Development and testing services
- **5000-5999**: Reserved for future services
- **6000-6999**: Backup and maintenance services
- **7000-7999**: Reserved for high-priority services
- **8000-8999**: Reserved for user-facing applications

### Current Allocations

#### Infrastructure Services

- PostgreSQL: 5432 (primary), 5433 (test)
- Neo4j: 7474 (browser), 7687 (bolt)
- Redis: 6379
- Qdrant: 6333

#### Application Services

- **digi-core**: 8000 (primary application)
- **PCS API**: 8002 (planned)
- **PCS Web**: 8003 (planned)
- **LernMI**: 8004 (planned)
- **Beep-Boop**: 8005 (planned)

#### Monitoring Services

- Grafana: 3000
- Prometheus: 9090

## Files

### `port-mapping.yml`

Central configuration file defining all port allocations, service dependencies, and conflict resolution strategies.

### `port-validator.py`

Python script that validates port allocations and detects conflicts across all services.

## Usage

### 1. Check Current Port Status

```bash
cd digi-infrastructure
python port-validator.py
```

### 2. Add New Service

1. Edit `port-mapping.yml`
2. Add service configuration under appropriate category
3. Run port validation to ensure no conflicts
4. Update docker-compose files if needed

### 3. Resolve Port Conflicts

The port validator will automatically detect conflicts and provide resolution plans:

- **External Port Conflicts**: Multiple services using same external port
- **Docker Port Conflicts**: Running containers conflicting with planned allocations
- **System Port Conflicts**: Services using system-reserved ports
- **Port Availability**: Ports already in use by other processes

## Conflict Resolution Examples

### Example 1: digi-core vs PCS on Port 8000

**Problem**: Both digi-core and PCS want to use port 8000

**Solution**:

- digi-core keeps port 8000 (primary application)
- PCS moves to port 8002 (planned)

### Example 2: PostgreSQL System Port

**Problem**: PostgreSQL uses port 5432 (system-reserved)

**Solution**:

- PostgreSQL keeps port 5432 (standard database port)
- Other services avoid this port range

## Best Practices

1. **Always run port validation** before starting new services
2. **Use port ranges** to organize services logically
3. **Document port changes** in commit messages
4. **Test port availability** before deployment
5. **Coordinate port changes** across team members

## Integration with Docker Compose

The port mapping system integrates with Docker Compose files:

```yaml
services:
  digi-core-app:
    ports:
      - "8000:8000" # From port-mapping.yml

  pcs-api:
    ports:
      - "8002:8000" # From port-mapping.yml
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**

   - Check running containers: `docker ps`
   - Check system processes: `netstat -tulpn | grep :PORT`
   - Use port validator to identify conflicts

2. **Service Won't Start**

   - Verify port is available
   - Check port mapping configuration
   - Ensure no conflicting services

3. **Port Conflicts After Updates**
   - Run port validation
   - Update port mappings if needed
   - Restart affected services

### Debug Commands

```bash
# Check all port allocations
python port-validator.py

# Check specific port usage
netstat -tulpn | grep :8000

# Check Docker container ports
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Check service health
curl http://localhost:8000/health
```

## Future Enhancements

- [ ] Web-based port management interface
- [ ] Automatic port conflict resolution
- [ ] Integration with service discovery
- [ ] Port allocation API
- [ ] Real-time port monitoring

## Contributing

When adding new services or changing port allocations:

1. Update `port-mapping.yml`
2. Run port validation
3. Update documentation
4. Test changes
5. Submit pull request with clear description

## Support

For port management issues:

1. Check this README
2. Run port validation
3. Review conflict resolution plans
4. Contact infrastructure team if needed
