# Database Migrations

This document describes the database migration system for the PCS (Prompt Context System) using Alembic.

## Overview

The PCS system uses Alembic for database schema management and migrations. This provides:

- Version-controlled database schema changes
- Automated migration generation
- Rollback capabilities
- Cross-environment consistency

## Quick Start

### 1. Check Migration Status

```bash
python scripts/db.py current
```

### 2. View Migration History

```bash
python scripts/db.py history
```

### 3. Run Migrations

```bash
python scripts/db.py migrate
```

### 4. Create New Migration

```bash
python scripts/db.py create-migration "Add user preferences table"
```

## Migration Commands

### Using the Database Script

```bash
# Check database connection
python scripts/db.py check

# Run all pending migrations
python scripts/db.py migrate

# Create a new migration
python scripts/db.py create-migration "Description of changes"

# Rollback to specific revision
python scripts/db.py rollback --revision 9a39eb7fd78f

# View current revision
python scripts/db.py current

# View migration history
python scripts/db.py history

# Reset database (drop all tables and recreate)
python scripts/db.py reset
```

### Using Alembic Directly

```bash
# Initialize alembic (already done)
alembic init alembic

# Create new migration
alembic revision -m "Description"

# Run migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# Rollback to specific revision
alembic downgrade 9a39eb7fd78f

# View current status
alembic current

# View history
alembic history
```

## Migration Files

### Location

Migrations are stored in `alembic/versions/` directory.

### Naming Convention

- Format: `{revision_id}_{description}.py`
- Example: `9a39eb7fd78f_initial_database_schema.py`

### Structure

Each migration file contains:

- `upgrade()`: Apply the migration
- `downgrade()`: Rollback the migration
- Metadata (revision ID, dependencies, etc.)

## Current Schema

### Tables

#### contexts

- `id` (UUID, Primary Key)
- `name` (String, Unique)
- `description` (Text, Optional)
- `context_data` (JSON, Optional)
- `is_active` (Boolean, Default: True)
- `created_at` (DateTime, Auto)
- `updated_at` (DateTime, Auto)

#### conversations

- `id` (UUID, Primary Key)
- `title` (String)
- `context_id` (UUID, Foreign Key to contexts.id)
- `conversation_data` (JSON, Optional)
- `is_active` (Boolean, Default: True)
- `created_at` (DateTime, Auto)
- `updated_at` (DateTime, Auto)

#### prompts

- `id` (UUID, Primary Key)
- `name` (String, Unique)
- `content` (Text)
- `prompt_type` (String)
- `context_id` (UUID, Foreign Key to contexts.id)
- `is_active` (Boolean, Default: True)
- `created_at` (DateTime, Auto)
- `updated_at` (DateTime, Auto)

### Indexes

- `ix_contexts_name` on contexts.name
- `ix_conversations_context_id` on conversations.context_id
- `ix_prompts_name` on prompts.name
- `ix_prompts_context_id` on prompts.context_id

## Configuration

### alembic.ini

- Database URL configuration
- Script location settings
- Logging configuration

### env.py

- Environment-specific configuration
- Model metadata integration
- Database connection handling

## Best Practices

### 1. Migration Naming

- Use descriptive names that explain the change
- Include the table/feature being modified
- Example: "Add user preferences table" not "Update schema"

### 2. Testing Migrations

- Always test migrations in development first
- Test both upgrade and downgrade paths
- Verify data integrity after migration

### 3. Rollback Strategy

- Keep migrations small and focused
- Ensure downgrade operations are safe
- Test rollback scenarios

### 4. Data Migration

- Use separate migrations for schema and data changes
- Include data validation in migrations
- Handle large datasets carefully

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors when running migrations:

```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the database script
python scripts/db.py migrate
```

#### Database Connection Issues

```bash
# Check connection
python scripts/db.py check

# Verify environment variables
echo $DATABASE_URL
```

#### Migration Conflicts

```bash
# View current status
python scripts/db.py current

# Check for conflicts
alembic check

# Resolve conflicts manually if needed
```

### Recovery

#### Reset Database

```bash
# Complete reset (WARNING: This will delete all data)
python scripts/db.py reset
```

#### Manual Rollback

```bash
# Rollback to specific revision
python scripts/db.py rollback --revision <revision_id>
```

## Development Workflow

### 1. Schema Changes

1. Modify models in `src/pcs/models/`
2. Create migration: `python scripts/db.py create-migration "Description"`
3. Review generated migration file
4. Test migration: `python scripts/db.py migrate`
5. Verify schema changes

### 2. Testing

1. Run migrations in test environment
2. Verify application functionality
3. Test rollback scenarios
4. Update tests if needed

### 3. Deployment

1. Backup production database
2. Run migrations: `python scripts/db.py migrate`
3. Verify application functionality
4. Monitor for issues

## Environment Variables

### Required

- `DATABASE_URL`: PostgreSQL connection string

### Optional

- `ALEMBIC_CONFIG`: Path to alembic.ini (defaults to ./alembic.ini)

## Related Documentation

- [Database Configuration](../core/config.py)
- [Data Models](../models/)
- [Database Connection](../core/database.py)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
