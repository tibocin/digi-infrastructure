# Schema Management in Multi-App Architecture

## Overview

In the shared infrastructure + independent apps approach, **each app manages its own schema and migrations** within its own database. This provides isolation, independence, and flexibility.

## Repository Structure

```
digi-infrastructure/          # Shared database infrastructure
├── docker-compose.yml       # Database containers only
├── scripts/
│   └── init-multiple-databases.sh
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── backup/
│   └── backup-config.yml
└── README.md

digi-core/                   # digi-core app
├── docker-compose.yml       # App container + external links
├── src/
│   ├── alembic/            # digi-core migrations
│   │   ├── versions/
│   │   └── env.py
│   └── app/
│       ├── models.py        # digi-core models
│       └── database.py
├── alembic.ini
└── README.md

lernmi/                      # lernmi app
├── docker-compose.yml       # App container + external links
├── src/
│   ├── alembic/            # lernmi migrations
│   │   ├── versions/
│   │   └── env.py
│   └── app/
│       ├── models.py        # lernmi models
│       └── database.py
├── alembic.ini
└── README.md

beep-boop/                   # beep-boop app
├── docker-compose.yml       # App container + external links
├── src/
│   ├── alembic/            # beep-boop migrations
│   │   ├── versions/
│   │   └── env.py
│   └── app/
│       ├── models.py        # beep-boop models
│       └── database.py
├── alembic.ini
└── README.md
```

## How Schema Management Works

### 1. Database Isolation

Each app gets its own database within the shared PostgreSQL instance:

```sql
-- digi-core database
CREATE DATABASE digi_core;
CREATE USER digi_core_user WITH PASSWORD 'digi_core_pass';
GRANT ALL PRIVILEGES ON DATABASE digi_core TO digi_core_user;

-- lernmi database
CREATE DATABASE lernmi;
CREATE USER lernmi_user WITH PASSWORD 'lernmi_pass';
GRANT ALL PRIVILEGES ON DATABASE lernmi TO lernmi_user;

-- beep-boop database
CREATE DATABASE beep_boop;
CREATE USER beep_boop_user WITH PASSWORD 'beep_boop_pass';
GRANT ALL PRIVILEGES ON DATABASE beep_boop TO beep_boop_user;
```

### 2. Independent Migrations

Each app has its own Alembic setup:

```python
# digi-core/alembic/env.py
def get_url():
    return os.getenv("DIGI_CORE_DB_URL", "postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core")

# lernmi/alembic/env.py
def get_url():
    return os.getenv("LERNMI_DB_URL", "postgresql://lernmi_user:lernmi_pass@postgres:5432/lernmi")

# beep-boop/alembic/env.py
def get_url():
    return os.getenv("BEEP_BOOP_DB_URL", "postgresql://beep_boop_user:beep_boop_pass@postgres:5432/beep_boop")
```

### 3. Schema Independence

Each app can have completely different schemas:

```python
# digi-core/src/app/models.py
class QueryLog(Base):
    __tablename__ = "query_logs"
    # digi-core specific schema

class ReasoningContext(Base):
    __tablename__ = "reasoning_contexts"
    # digi-core specific schema

# lernmi/src/app/models.py
class Evaluation(Base):
    __tablename__ = "evaluations"
    # lernmi specific schema

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    # lernmi specific schema

# beep-boop/src/app/models.py
class Conversation(Base):
    __tablename__ = "conversations"
    # beep-boop specific schema

class BotConfig(Base):
    __tablename__ = "bot_configs"
    # beep-boop specific schema
```

## Migration Workflow

### 1. Development Workflow

```bash
# Working on digi-core
cd digi-core
alembic revision --autogenerate -m "Add new table"
alembic upgrade head

# Working on lernmi
cd lernmi
alembic revision --autogenerate -m "Add evaluation metrics"
alembic upgrade head

# Working on beep-boop
cd beep-boop
alembic revision --autogenerate -m "Add conversation history"
alembic upgrade head
```

### 2. Deployment Workflow

```bash
# Deploy infrastructure first
cd digi-infrastructure
docker-compose up -d

# Deploy apps independently
cd ../digi-core
docker-compose up -d

cd ../lernmi
docker-compose up -d

cd ../beep-boop
docker-compose up -d
```

## Infrastructure Repository

### digi-infrastructure/docker-compose.yml

```yaml
version: "3.8"
services:
  postgres:
    image: postgres:15
    container_name: digi-postgres
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

  neo4j:
    image: neo4j:5.18
    container_name: digi-neo4j
    environment:
      NEO4J_AUTH: "neo4j/${NEO4J_PASSWORD:-digi}"
    volumes:
      - neo4j_data:/data
    networks:
      - digi-net

  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: digi-qdrant
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

  redis:
    image: redis:7
    container_name: digi-redis
    volumes:
      - redis_data:/data
    networks:
      - digi-net

  # Monitoring and backup services...
```

## App Repository Structure

### digi-core/docker-compose.yml

```yaml
version: "3.8"
services:
  digi-core-app:
    build: .
    container_name: digi-core-app
    environment:
      - DIGI_CORE_DB_URL=postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core
    external_links:
      - digi-infrastructure_postgres_1:postgres
      - digi-infrastructure_neo4j_1:neo4j
      - digi-infrastructure_qdrant_1:qdrant
      - digi-infrastructure_redis_1:redis
    networks:
      - digi-net

networks:
  digi-net:
    external: true
```

### digi-core/alembic.ini

```ini
[alembic]
script_location = src/alembic
sqlalchemy.url = postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core
```

## Schema Evolution Examples

### Example 1: Adding New Features

```python
# digi-core: Add new table for file processing
class FileProcessingStatus(Base):
    __tablename__ = "file_processing_status"
    # ... schema

# lernmi: Add new evaluation metrics
class EvaluationMetrics(Base):
    __tablename__ = "evaluation_metrics"
    # ... schema

# beep-boop: Add conversation analytics
class ConversationAnalytics(Base):
    __tablename__ = "conversation_analytics"
    # ... schema
```

### Example 2: Schema Changes

```python
# digi-core: Add new column to existing table
# Migration: add_column_to_query_logs.py
def upgrade():
    op.add_column('query_logs', sa.Column('processing_time_ms', sa.Integer()))

# lernmi: Modify evaluation table
# Migration: modify_evaluations_table.py
def upgrade():
    op.alter_column('evaluations', 'score', type_=sa.Float())

# beep-boop: Add index for performance
# Migration: add_conversation_index.py
def upgrade():
    op.create_index('idx_conversation_timestamp', 'conversations', ['timestamp'])
```

## Cross-App Data Sharing

### Option 1: API-Based Sharing

```python
# lernmi app calls digi-core API
import requests

def get_digi_core_data():
    response = requests.get(
        "http://digi-core-app:8000/api/data",
        headers={"X-Digi-Key": "lernmi-api-key"}
    )
    return response.json()
```

### Option 2: Shared Schema (Advanced)

If apps need to share data, create a shared schema:

```sql
-- Create shared schema in each database
CREATE SCHEMA shared;

-- Shared tables in each app's database
CREATE TABLE shared.user_profiles (
    id UUID PRIMARY KEY,
    name TEXT,
    preferences JSONB
);
```

## Migration Best Practices

### 1. Independent Development

```bash
# Each app team works independently
cd digi-core
alembic revision --autogenerate -m "Add user preferences"
alembic upgrade head

cd ../lernmi
alembic revision --autogenerate -m "Add learning metrics"
alembic upgrade head
```

### 2. Deployment Coordination

```bash
# Deploy infrastructure first
cd digi-infrastructure
docker-compose up -d

# Deploy apps in order (if dependencies exist)
cd ../digi-core
docker-compose up -d

cd ../lernmi  # May depend on digi-core
docker-compose up -d

cd ../beep-boop  # May depend on both
docker-compose up -d
```

### 3. Rollback Strategy

```bash
# Rollback specific app
cd digi-core
alembic downgrade -1

# Or rollback to specific version
alembic downgrade a1b2c3d4e5f6
```

## Monitoring Schema Changes

### 1. Migration Tracking

```python
# Track migration status per app
@app.get("/health/migrations")
async def migration_status():
    return {
        "digi_core": check_migration_status("digi_core"),
        "lernmi": check_migration_status("lernmi"),
        "beep_boop": check_migration_status("beep_boop")
    }
```

### 2. Schema Validation

```python
# Validate schemas on startup
def validate_schema(app_name: str):
    engine = get_app_engine(app_name)
    inspector = inspect(engine)

    # Check required tables exist
    required_tables = get_required_tables(app_name)
    existing_tables = inspector.get_table_names()

    missing_tables = set(required_tables) - set(existing_tables)
    if missing_tables:
        raise SchemaValidationError(f"Missing tables: {missing_tables}")
```

## Benefits of This Approach

### ✅ **Schema Independence**

- Each app can evolve its schema independently
- No coordination needed between app teams
- Different apps can use different SQLAlchemy versions

### ✅ **Deployment Flexibility**

- Deploy apps independently
- Rollback specific apps without affecting others
- Different deployment schedules per app

### ✅ **Team Autonomy**

- Each team owns their schema and migrations
- No cross-team coordination for schema changes
- Independent development cycles

### ✅ **Data Isolation**

- Complete data separation between apps
- No risk of schema conflicts
- App-specific optimizations possible

### ✅ **Scalability**

- Scale apps independently
- Different performance requirements per app
- Independent backup and restore strategies

## Conclusion

The **independent schema management** approach provides maximum flexibility and team autonomy while maintaining the benefits of shared infrastructure. Each app can evolve independently while sharing the same database instances, monitoring, and backup infrastructure.
