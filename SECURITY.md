# Security Documentation

## Critical Security Notice

**ALL PREVIOUSLY COMMITTED CREDENTIALS HAVE BEEN COMPROMISED AND MUST BE ROTATED IMMEDIATELY.**

## Exposed Credentials (IMMEDIATE ACTION REQUIRED)

The following credentials were previously hardcoded in version control and are now compromised:

### Database Credentials
- `PCS_DB_PASSWORD`: `pNcq1IDDdt8xfZHTq_C5txmqCa3Jc7Oz59DBG5INcCY`
- `pcs_password` (PCS service)
- `neo4j_password` (PCS service)

### Security Keys
- `PCS_SECURITY_SECRET_KEY`: `CygmYZFhKsOnSQ-JH842eZCd8-bxQ-KEZrpd-rtmZZA`
- `PCS_SECURITY_JWT_SECRET_KEY`: `LsvWuM9LiAMS4oylq1Po76Cg8gDFAexmIOACFI3sfzA`

### Admin Passwords
- `PGADMIN_DEFAULT_PASSWORD`: `admin123`

## Immediate Actions Required

### 1. Rotate All Exposed Credentials
```bash
# Generate new secure passwords
openssl rand -base64 32

# Generate new security keys
openssl rand -hex 32
```

### 2. Update Environment Variables
- Copy `env.example` to `.env`
- Fill in new secure values
- **NEVER commit `.env` files**

### 3. Restart All Services
```bash
docker compose down
docker compose up -d
```

## Security Measures Implemented

### 1. Environment Variable Usage
- All sensitive data now uses environment variables
- No hardcoded credentials in version control
- Example configuration in `env.example`

### 2. Git Ignore Protection
- `.env` files are ignored by Git
- Environment-specific files are excluded
- Backup and log files are protected

### 3. Credential Requirements
- Minimum 16 character passwords
- Mix of character types required
- Unique values per environment
- Regular rotation (90 days)

## Environment Variable Reference

### Required Variables
```bash
# Database
POSTGRES_PASSWORD=secure_password_here
PCS_DB_PASSWORD=secure_pcs_password_here
NEO4J_PASSWORD=secure_neo4j_password_here

# Security
PCS_SECURITY_SECRET_KEY=64_char_secret_here
PCS_SECURITY_JWT_SECRET_KEY=64_char_jwt_secret_here

# Admin
GRAFANA_ADMIN_PASSWORD=secure_grafana_password_here
PGADMIN_DEFAULT_PASSWORD=secure_pgadmin_password_here
```

### Optional Variables
```bash
# AWS Backup
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
RESTIC_REPOSITORY=your_repo
RESTIC_PASSWORD=secure_restic_password_here
```

## Best Practices

### 1. Credential Management
- Use password managers or secrets management systems
- Generate unique credentials per environment
- Implement credential rotation policies
- Monitor for credential exposure

### 2. Environment Separation
- Different credentials for dev/staging/prod
- Isolated database instances per environment
- Environment-specific configuration files

### 3. Access Control
- Limit database access to application containers only
- Use network isolation (Docker networks)
- Implement proper authentication and authorization
- Regular access reviews

### 4. Monitoring and Alerting
- Monitor for unauthorized access attempts
- Log all authentication events
- Set up alerts for suspicious activity
- Regular security audits

## Emergency Procedures

### If Credentials Are Compromised
1. **Immediate**: Rotate all affected credentials
2. **Investigation**: Determine scope of compromise
3. **Notification**: Alert security team and stakeholders
4. **Recovery**: Restore from secure backups if necessary
5. **Post-mortem**: Document lessons learned

### Contact Information
- **Security Team**: [Contact Information]
- **Emergency**: [Emergency Contact]
- **Escalation**: [Escalation Path]

## Compliance Notes

- All credentials must meet minimum security requirements
- Regular security assessments are required
- Credential exposure must be reported within 24 hours
- Audit logs must be maintained for compliance

## Related Documentation

- [Environment Setup](env.example)
- [Docker Compose Configuration](docker-compose.yml)
- [Infrastructure Overview](INFRASTRUCTURE_REPOSITORY.md)
- [Deployment Guide](DEPLOYMENT.md)
