# Multi-App Database Deployment Strategy

## Overview

This document explains how multiple applications can share the same database instances (PostgreSQL, Neo4j, Qdrant, Redis) while maintaining data isolation and security.

## Current Architecture

Your current setup uses a **single-tenant architecture** where:

- One `digi-core-app` container
- One set of database containers (postgres, neo4j, qdrant, redis)
- All data belongs to the `digi_core` application

## Multi-App Architecture Options

### Option 1: Shared Instances with Database Isolation (Recommended)

**Principle**: Multiple apps share the same database containers but use different databases/collections within those containers.

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Infrastructure                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  PostgreSQL │  │    Neo4j    │  │    Qdrant   │      │
│  │   Container │  │  Container  │  │  Container  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
│         │                │                │              │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐      │
│  │ digi_core   │  │ digi_core   │  │ digi_core   │      │
│  │ lernmi      │  │ lernmi      │  │ lernmi      │      │
│  │ beep_boop   │  │ beep_boop   │  │ beep_boop   │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Option 2: Separate Instances per App

**Principle**: Each app gets its own database containers.

```
┌─────────────────────────────────────────────────────────────┐
│                    App-Specific Infrastructure            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ digi-core   │  │   lernmi    │  │  beep_boop  │      │
│  │ PostgreSQL  │  │ PostgreSQL  │  │ PostgreSQL  │      │
│  │   Neo4j     │  │   Neo4j     │  │   Neo4j     │      │
│  │   Qdrant    │  │   Qdrant    │  │   Qdrant    │      │
│  │    Redis    │  │    Redis    │  │    Redis    │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Recommended Approach: Option 1 (Shared Instances)

### Why This Approach?

1. **Resource Efficiency**: Single database instances reduce overhead
2. **Simplified Management**: One set of containers to maintain
3. **Shared Infrastructure**: Common backup, monitoring, security
4. **Cost Effective**: Lower resource requirements
5. **Cross-App Analytics**: Easy to query across apps when needed

### How It Works

#### 1. PostgreSQL Multi-Database Setup

```yaml
# docker-compose.yml
services:
  postgres:
    environment:
      POSTGRES_MULTIPLE_DATABASES: "digi_core,lernmi,beep_boop"
    volumes:
      - ./scripts/init-multiple-databases.sh:/docker-entrypoint-initdb.d/init-multiple-databases.sh:ro
```

**Database Structure**:

- `digi_core` database → digi-core app data
- `lernmi` database → lernmi app data
- `beep_boop` database → beep-boop app data

**Connection URLs**:

```bash
# digi-core app
DIGI_CORE_DB_URL="postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core"

# lernmi app
LERNMI_DB_URL="postgresql://lernmi_user:lernmi_pass@postgres:5432/lernmi"

# beep-boop app
BEEP_BOOP_DB_URL="postgresql://beep_boop_user:beep_boop_pass@postgres:5432/beep_boop"
```

#### 2. Neo4j Multi-Database Setup

```yaml
# Environment variables
NEO4J_DATABASE_DIGI_CORE="digi_core"
NEO4J_DATABASE_LERNMI="lernmi"
NEO4J_DATABASE_BEEP_BOOP="beep_boop"
```

**Database Structure**:

- `digi_core` database → digi-core graph data
- `lernmi` database → lernmi graph data
- `beep_boop` database → beep-boop graph data

#### 3. Qdrant Multi-Collection Setup

```yaml
# Environment variables
QDRANT_COLLECTION_DIGI_CORE="digi_core_knowledge"
QDRANT_COLLECTION_LERNMI="lernmi_knowledge"
QDRANT_COLLECTION_BEEP_BOOP="beep_boop_knowledge"
```

**Collection Structure**:

- `digi_core_knowledge` collection → digi-core embeddings
- `lernmi_knowledge` collection → lernmi embeddings
- `beep_boop_knowledge` collection → beep-boop embeddings

#### 4. Redis Multi-Database Setup

```yaml
# Environment variables
REDIS_DB_DIGI_CORE=0
REDIS_DB_LERNMI=1
REDIS_DB_BEEP_BOOP=2
```

**Database Structure**:

- Database 0 → digi-core cache/sessions
- Database 1 → lernmi cache/sessions
- Database 2 → beep-boop cache/sessions

## Deployment Strategies

### Strategy 1: Single Docker Compose (Current)

**All apps in one docker-compose.yml**:

```yaml
services:
  # Shared database infrastructure
  postgres:
    # ... configuration
  neo4j:
    # ... configuration
  qdrant:
    # ... configuration
  redis:
    # ... configuration

  # App containers
  digi-core-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration

  lernmi-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration

  beep-boop-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration
```

**Pros**:

- ✅ Simple deployment
- ✅ Shared networking
- ✅ Single backup strategy
- ✅ Unified monitoring

**Cons**:

- ❌ All apps must be deployed together
- ❌ Can't scale apps independently
- ❌ Single point of failure

### Strategy 2: Separate Docker Compose Files

**Each app has its own docker-compose.yml**:

```yaml
# digi-core/docker-compose.yml
services:
  digi-core-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration

# lernmi/docker-compose.yml
services:
  lernmi-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration

# beep-boop/docker-compose.yml
services:
  beep-boop-app:
    depends_on: [postgres, neo4j, qdrant, redis]
    # ... configuration
```

**Pros**:

- ✅ Independent deployment
- ✅ Independent scaling
- ✅ App-specific configuration
- ✅ Isolated failures

**Cons**:

- ❌ More complex orchestration
- ❌ Duplicate database connections
- ❌ More resource usage

### Strategy 3: Hybrid Approach (Recommended)

**Shared infrastructure + independent apps**:

```yaml
# infrastructure/docker-compose.yml (shared databases)
services:
  postgres:
    # ... configuration
  neo4j:
    # ... configuration
  qdrant:
    # ... configuration
  redis:
    # ... configuration

# digi-core/docker-compose.yml
services:
  digi-core-app:
    external_links:
      - infrastructure_postgres_1:postgres
      - infrastructure_neo4j_1:neo4j
      - infrastructure_qdrant_1:qdrant
      - infrastructure_redis_1:redis
    # ... configuration
```

## Implementation Steps

### Step 1: Update Database Initialization

1. **PostgreSQL**: Use the `init-multiple-databases.sh` script
2. **Neo4j**: Create databases on first connection
3. **Qdrant**: Use different collection names
4. **Redis**: Use different database numbers

### Step 2: Update Application Code

1. **Database Connections**: Use app-specific connection strings
2. **RAG Clients**: Use app-specific collections/databases
3. **Health Checks**: Monitor all app connections

### Step 3: Update Environment Configuration

```bash
# .env
POSTGRES_MULTIPLE_DATABASES="digi_core,lernmi,beep_boop"

# App-specific connections
DIGI_CORE_DB_URL="postgresql://digi_core_user:digi_core_pass@postgres:5432/digi_core"
LERNMI_DB_URL="postgresql://lernmi_user:lernmi_pass@postgres:5432/lernmi"
BEEP_BOOP_DB_URL="postgresql://beep_boop_user:beep_boop_pass@postgres:5432/beep_boop"
```

### Step 4: Update Monitoring

1. **Health Checks**: Monitor all app database connections
2. **Metrics**: Track per-app database usage
3. **Logging**: Separate logs by app

## Security Considerations

### 1. Database Isolation

- **PostgreSQL**: Separate databases with different users
- **Neo4j**: Separate databases with role-based access
- **Qdrant**: Separate collections with metadata filtering
- **Redis**: Separate database numbers

### 2. Network Security

- All apps share the same Docker network
- Internal communication only
- No direct external access to databases

### 3. Access Control

- App-specific database users
- Read-only access where possible
- API key-based authentication

## Backup Strategy

### Current Backup Approach

```yaml
backup-sidecar:
  depends_on:
    - postgres
    - neo4j
    - qdrant
    - redis
  volumes:
    - pg_data:/backup/pg_data:ro
    - neo4j_data:/backup/neo4j_data:ro
    - qdrant_data:/backup/qdrant_data:ro
    - redis_data:/backup/redis_data:ro
```

### Multi-App Backup Considerations

1. **PostgreSQL**: Backup all databases
2. **Neo4j**: Backup all databases
3. **Qdrant**: Backup all collections
4. **Redis**: Backup all databases

## Monitoring and Observability

### Health Checks

```python
# Check all app database connections
@app.get("/health/databases")
async def database_health_check():
    return {
        "digi_core": check_app_connections("digi_core"),
        "lernmi": check_app_connections("lernmi"),
        "beep_boop": check_app_connections("beep_boop")
    }
```

### Metrics

- Per-app database connection counts
- Per-app query performance
- Per-app storage usage
- Cross-app analytics

## Qdrant Integration for Downstream Apps

### Connection Configuration

Each app connects to the shared Qdrant instance using environment variables:

```bash
# App-specific Qdrant configuration
QDRANT_HOST=qdrant  # Docker service name
QDRANT_PORT=6333    # HTTP API port
QDRANT_GRPC_PORT=6334  # gRPC port (optional)
QDRANT_API_KEY=your_app_api_key  # For authentication
QDRANT_COLLECTION_PREFIX=app_name  # For collection naming
```

### Collection Naming Convention

Apps use tenant-specific collection names to ensure data isolation:

```typescript
// Collection naming pattern
const collectionName = `${process.env.QDRANT_COLLECTION_PREFIX}_${collectionType}`;

// Examples:
// - digi_core_knowledge
// - lernmi_memories  
// - beep_boop_conversations
// - devao_analytics
```

### Multi-Tenant Data Isolation

```typescript
// Example: Store document with tenant isolation
const document = {
  id: "doc_123",
  vector: embedding,
  payload: {
    content: "Document content",
    tenant_id: process.env.APP_TENANT_ID,  // Critical for isolation
    app_name: process.env.APP_NAME,
    created_at: new Date().toISOString(),
    metadata: {
      type: "knowledge",
      category: "technical"
    }
  }
};

// Search with tenant filter
const results = await qdrantClient.search(
  collectionName: "app_knowledge",
  queryVector: queryEmbedding,
  filter: {
    must: [
      { key: "tenant_id", match: { value: process.env.APP_TENANT_ID } }
    ]
  }
);
```

### Health Check Implementation

```typescript
// Qdrant health check for apps
async function checkQdrantHealth(): Promise<HealthStatus> {
  try {
    const collections = await qdrantClient.getCollections();
    const appCollection = collections.find(c => 
      c.name.startsWith(process.env.QDRANT_COLLECTION_PREFIX)
    );
    
    if (appCollection) {
      return { 
        status: "healthy", 
        message: `Qdrant connection OK, collection: ${appCollection.name}` 
      };
    } else {
      return { 
        status: "degraded", 
        message: "Qdrant connected but app collection not found" 
      };
    }
  } catch (error) {
    return { 
      status: "unhealthy", 
      message: `Qdrant error: ${error.message}` 
    };
  }
}
```

## Migration Strategy

### Phase 1: Prepare Infrastructure

1. Update database initialization scripts
2. Add multi-app client code
3. Update environment configuration

### Phase 2: Deploy New Apps

1. Deploy lernmi app with new database connections
2. Deploy beep-boop app with new database connections
3. Test cross-app functionality

### Phase 3: Monitor and Optimize

1. Monitor performance across apps
2. Optimize database connections
3. Implement cross-app analytics

## Conclusion

The **shared instances with database isolation** approach provides the best balance of:

- Resource efficiency
- Data isolation
- Simplified management
- Cost effectiveness

This strategy allows you to run multiple apps while maintaining the benefits of shared infrastructure and the security of data isolation.
