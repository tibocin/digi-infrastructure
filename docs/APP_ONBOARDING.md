# App Onboarding & Initialization Guide

## Overview

This guide provides step-by-step instructions for onboarding new applications into the Digi ecosystem. The onboarding process ensures that applications are properly integrated with the shared infrastructure, Prompt and Context Service (PCS), and follow established patterns for consistency and maintainability.

## Onboarding Process Overview

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   App Discovery │  │   Registration  │  │   Integration   │  │   Validation    │
│   & Planning    │  │   & Setup       │  │   & Testing     │  │   & Deployment  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Phase 1: App Discovery & Planning

### 1.1 App Assessment

Before onboarding, assess your application's requirements:

- [ ] **Infrastructure Needs**: Database types, storage requirements, compute resources
- [ ] **AI/ML Capabilities**: Prompt requirements, context management needs
- [ ] **Integration Points**: APIs, webhooks, event systems
- [ ] **Performance Requirements**: Response times, throughput, scalability needs
- [ ] **Security Requirements**: Authentication, authorization, data privacy

### 1.2 Architecture Review

Review your application architecture against Digi ecosystem patterns:

```yaml
# Recommended App Architecture
app_structure:
  - api/ # REST/GraphQL API endpoints
  - services/ # Business logic services
  - models/ # Data models and schemas
  - utils/ # Utility functions
  - config/ # Configuration management
  - tests/ # Test suites
  - docs/ # Application documentation
  - docker/ # Container configuration
```

### 1.3 Resource Planning

Estimate resource requirements:

```yaml
# Resource Estimation Template
resources:
  database:
    postgresql: "Estimated GB storage"
    neo4j: "Estimated GB storage"
    qdrant: "Estimated GB storage"
    redis: "Estimated GB storage"

  compute:
    cpu: "CPU cores needed"
    memory: "RAM requirements"
    storage: "Local storage needs"

  network:
    bandwidth: "Expected traffic"
    latency: "Response time requirements"
```

## Phase 2: Registration & Setup

### 2.1 Infrastructure Repository Setup

1. **Clone the infrastructure repository**

   ```bash
   git clone https://github.com/your-org/digi-infrastructure.git
   cd digi-infrastructure
   ```

2. **Configure environment**

   ```bash
   cp config/env.example .env
   # Edit .env with your specific values
   ```

3. **Start shared infrastructure**
   ```bash
   make up
   make health
   ```

### 2.2 App Registration with PCS

1. **Generate app credentials**

   ```bash
   # Generate unique app identifier
   APP_ID=$(uuidgen)
   APP_KEY=$(openssl rand -hex 32)

   echo "APP_ID=$APP_ID"
   echo "APP_KEY=$APP_KEY"
   ```

2. **Register app with PCS**

   ```bash
   curl -X POST http://localhost:8000/api/v1/apps \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "app_name": "your_app_name",
       "app_id": "'$APP_ID'",
       "app_key": "'$APP_KEY'",
       "description": "Your app description",
       "permissions": ["prompt_read", "context_write", "conversation_create"],
       "rate_limits": {
         "requests_per_minute": 1000,
         "requests_per_hour": 50000
       },
       "webhook_urls": {
         "prompt_updates": "https://your-app.com/webhooks/prompt-updates",
         "context_changes": "https://your-app.com/webhooks/context-changes"
       }
     }'
   ```

3. **Verify registration**
   ```bash
   curl -X GET http://localhost:8000/api/v1/apps/$APP_ID \
     -H "Authorization: Bearer $APP_KEY"
   ```

### 2.3 Database Schema Setup

1. **Create app-specific databases**

   ```bash
   # Connect to PostgreSQL
   docker-compose exec postgres psql -U digi -d digi

   # Create app database
   CREATE DATABASE your_app_name;

   # Create app user
   CREATE USER your_app_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE your_app_name TO your_app_user;
   ```

2. **Initialize app schemas**

   ```sql
   -- Connect to your app database
   \c your_app_name

   -- Create app-specific tables
   CREATE TABLE app_users (
       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       username VARCHAR(255) UNIQUE NOT NULL,
       email VARCHAR(255) UNIQUE NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );

   -- Add more tables as needed
   ```

3. **Set up Neo4j labels and constraints**

   ```cypher
   // Connect to Neo4j
   // Browser: http://localhost:7474

   // Create app-specific labels
   CREATE CONSTRAINT app_user_id IF NOT EXISTS FOR (u:AppUser) REQUIRE u.id IS UNIQUE;

   // Create indexes
   CREATE INDEX app_user_email IF NOT EXISTS FOR (u:AppUser) ON (u.email);
   ```

### 2.4 Configuration Management

1. **Create app configuration file**

   ```yaml
   # config/apps/your_app_name.yml
   app:
     name: "your_app_name"
     version: "1.0.0"
     environment: "development"

   database:
     postgresql:
       host: "localhost"
       port: 5432
       database: "your_app_name"
       username: "your_app_user"
       password: "${POSTGRES_PASSWORD}"

     neo4j:
       host: "localhost"
       port: 7687
       username: "neo4j"
       password: "${NEO4J_PASSWORD}"

     redis:
       host: "localhost"
       port: 6379
       password: "${REDIS_PASSWORD}"

   pcs:
     base_url: "http://localhost:8000"
     app_id: "${PCS_APP_ID}"
     app_key: "${PCS_APP_KEY}"
     timeout: 30000

   monitoring:
     prometheus_port: 9090
     health_check_interval: 30
   ```

2. **Update environment variables**
   ```bash
   # Add to your .env file
   echo "PCS_APP_ID=$APP_ID" >> .env
   echo "PCS_APP_KEY=$APP_KEY" >> .env
   echo "YOUR_APP_POSTGRES_PASSWORD=secure_password" >> .env
   ```

### 2.5 Bootstrap System & Idempotent Prompt Management

The bootstrap system is a critical component that ensures all PCS-enabled applications start with the necessary foundational prompts and data structures. This system must be **completely idempotent** - it should be safe to run multiple times without errors or duplicate data.

#### Core Principles

##### 1. Idempotent Operations

- **Database Schema:** Check if tables exist before creation
- **Prompt Templates:** Verify existence before insertion, handle updates gracefully
- **System Metadata:** Maintain initialization state without duplication
- **Data Seeding:** Skip if already present, update if changed

##### 2. Prompt Template Management

The bootstrap system creates foundational prompts that serve as the "operating system" for semantic intelligence:

**Required Core Prompts:**

- `query_understanding` - Analyzes user intent and information needs
- `knowledge_retrieval` - Finds relevant knowledge from stored data
- `response_generation` - Generates contextual, helpful responses
- `feedback_learning` - Learns from user feedback to improve

##### 3. Implementation Pattern

```python
async def _create_initial_prompts(self):
    """Create foundational prompts idempotently."""
    logger.info("🧠 Creating/updating initial prompts...")

    try:
        initial_prompts = self._get_initial_prompt_templates()

        for prompt_data in initial_prompts:
            # Check if prompt exists
            existing_prompt = self.prompt_manager.get_prompt_template(prompt_data['name'])

            if existing_prompt:
                # Update if content changed
                if self._prompt_needs_update(existing_prompt, prompt_data):
                    self.prompt_manager.update_prompt_template(prompt_data['name'], prompt_data)
                    logger.info(f"🔄 Updated prompt: {prompt_data['name']}")
                else:
                    logger.info(f"✅ Prompt already exists: {prompt_data['name']}")
            else:
                # Create new prompt
                self.prompt_manager.create_prompt_template(prompt_data)
                logger.info(f"✅ Created prompt: {prompt_data['name']}")

        logger.info("✅ All initial prompts processed idempotently")

    except Exception as e:
        logger.error(f"Failed to process initial prompts: {e}")
        raise
```

##### 4. Bootstrap State Management

The system maintains state through the `system_metadata` table:

```sql
-- Key bootstrap state indicators
initialized: 'true'/'false'     -- Application initialization status
version: '1.0.0'               -- Current application version
bootstrap_date: timestamp      -- Date of last successful bootstrap
prompt_count: integer          -- Number of foundational prompts
```

##### 5. Error Handling and Recovery

- **Graceful Degradation:** Continue operation even if some prompts fail
- **Partial Success Handling:** Track what succeeded vs. failed
- **Rollback Capability:** Revert to previous state if critical failures occur
- **Health Checks:** Verify bootstrap state on startup

#### Usage in PCS Repositories

All repositories using PCS should implement this bootstrap pattern:

1. **Check Initialization Status** - Verify if app is already bootstrapped
2. **Create/Update Prompts** - Handle foundational prompts idempotently
3. **Seed Required Data** - Initialize core data structures
4. **Set Initialization Flag** - Mark bootstrap as complete
5. **Health Monitoring** - Track bootstrap state and prompt health

#### Testing Bootstrap Idempotency

```bash
# Test multiple bootstrap runs
docker restart <app-container>
# Should show: "✅ Application already initialized, skipping bootstrap"

# Force re-bootstrap
docker exec <app-container> python -c "from app.bootstrap import DigiCoreBootstrap; DigiCoreBootstrap().force_rebootstrap()"
# Should handle updates gracefully without duplicates
```

#### Best Practices

- **Always check existence** before creating resources
- **Use upsert patterns** for data that may change
- **Maintain audit trails** of bootstrap operations
- **Implement health checks** for bootstrap state
- **Handle version conflicts** gracefully
- **Provide clear logging** for debugging bootstrap issues

## Phase 3: Integration & Testing

### 3.1 PCS SDK Integration

1. **Install SDK**

   ```bash
   # TypeScript/JavaScript
   npm install @digi/pcs-sdk

   # Python
   pip install digi-pcs-sdk

   # Go
   go get github.com/digi/pcs-sdk
   ```

2. **Initialize SDK in your app**

   ```typescript
   // TypeScript/JavaScript
   import { PCSSDK } from "@digi/pcs-sdk";

   const pcs = new PCSSDK({
     baseUrl: process.env.PCS_BASE_URL,
     apiKey: process.env.PCS_APP_KEY,
     appId: process.env.PCS_APP_ID,
     timeout: 30000,
   });

   // Test connection
   try {
     await pcs.health.check();
     console.log("PCS connection successful");
   } catch (error) {
     console.error("PCS connection failed:", error);
   }
   ```

3. **Create app-specific prompts**

   ```typescript
   // Define your app's prompt templates
   const appPrompts = [
     {
       name: "user_welcome",
       template:
         "Welcome to {{app_name}}, {{user_name}}! How can I help you today?",
       variables: ["app_name", "user_name"],
     },
     {
       name: "feature_explanation",
       template: "Let me explain {{feature_name}} to you. {{context}}",
       variables: ["feature_name", "context"],
     },
   ];

   // Register prompts with PCS
   for (const prompt of appPrompts) {
     await pcs.prompts.create(prompt);
   }
   ```

### 3.2 Database Integration

1. **Set up database connections**

   ```typescript
   // PostgreSQL connection
   import { Pool } from "pg";

   const postgresPool = new Pool({
     host: process.env.POSTGRES_HOST,
     port: parseInt(process.env.POSTGRES_PORT || "5432"),
     database: process.env.POSTGRES_DATABASE,
     user: process.env.POSTGRES_USER,
     password: process.env.POSTGRES_PASSWORD,
     max: 20,
     idleTimeoutMillis: 30000,
     connectionTimeoutMillis: 2000,
   });

   // Neo4j connection
   import neo4j from "neo4j-driver";

   const neo4jDriver = neo4j.driver(
     process.env.NEO4J_URI || "bolt://localhost:7687",
     neo4j.auth.basic(
       process.env.NEO4J_USER || "neo4j",
       process.env.NEO4J_PASSWORD || "digi"
     )
   );

   // Redis connection
   import Redis from "ioredis";

   const redis = new Redis({
     host: process.env.REDIS_HOST || "localhost",
     port: parseInt(process.env.REDIS_PORT || "6379"),
     password: process.env.REDIS_PASSWORD,
     retryDelayOnFailover: 100,
     maxRetriesPerRequest: 3,
   });

   // Qdrant connection (using our HTTP wrapper)
   import { QdrantHTTPClient } from "@pcs/typescript-sdk";

   const qdrantClient = new QdrantHTTPClient({
     host: process.env.QDRANT_HOST || "localhost",
     port: parseInt(process.env.QDRANT_PORT || "6333"),
     apiKey: process.env.QDRANT_API_KEY,
     timeout: 30000,
     maxRetries: 3,
     retryDelay: 1000,
   });
   ```

2. **Create data access layer**

   ```typescript
   // Example data service
   class UserService {
     async createUser(userData: CreateUserRequest): Promise<User> {
       const client = await postgresPool.connect();
       try {
         const result = await client.query(
           "INSERT INTO app_users (username, email) VALUES ($1, $2) RETURNING *",
           [userData.username, userData.email]
         );
         return result.rows[0];
       } finally {
         client.release();
       }
     }

     async getUserById(id: string): Promise<User | null> {
       const client = await postgresPool.connect();
       try {
         const result = await client.query(
           "SELECT * FROM app_users WHERE id = $1",
           [id]
         );
         return result.rows[0] || null;
       } finally {
         client.release();
       }
     }
   }

   // Qdrant vector service example
   class VectorService {
     async storeDocument(document: VectorDocument): Promise<string> {
       try {
         const result = await qdrantClient.upsert(
           collectionName: "app_knowledge",
           points: [{
             id: document.id,
             vector: document.embedding,
             payload: {
               content: document.content,
               metadata: document.metadata,
               tenant_id: process.env.APP_TENANT_ID,
               created_at: new Date().toISOString()
             }
           }]
         );
         return document.id;
       } catch (error) {
         throw new Error(`Failed to store document: ${error.message}`);
       }
     }

     async searchSimilar(query: string, limit: number = 10): Promise<SearchResult[]> {
       try {
         // First get embedding from PCS or local embedding service
         const embedding = await this.getEmbedding(query);

         const results = await qdrantClient.search(
           collectionName: "app_knowledge",
           queryVector: embedding,
           limit: limit,
           filter: {
             must: [
               { key: "tenant_id", match: { value: process.env.APP_TENANT_ID } }
             ]
           }
         );

         return results.map(result => ({
           id: result.id,
           score: result.score,
           content: result.payload.content,
           metadata: result.payload.metadata
         }));
       } catch (error) {
         throw new Error(`Search failed: ${error.message}`);
       }
     }
   }
   ```

### 3.3 Health Checks & Monitoring

1. **Implement health check endpoints**

   ```typescript
   // Health check service
   class HealthCheckService {
     async checkDatabase(): Promise<HealthStatus> {
       try {
         await postgresPool.query("SELECT 1");
         return { status: "healthy", message: "Database connection OK" };
       } catch (error) {
         return {
           status: "unhealthy",
           message: `Database error: ${error.message}`,
         };
       }
     }

     async checkPCS(): Promise<HealthStatus> {
       try {
         await pcs.health.check();
         return { status: "healthy", message: "PCS connection OK" };
       } catch (error) {
         return { status: "unhealthy", message: `PCS error: ${error.message}` };
       }
     }

     async checkRedis(): Promise<HealthStatus> {
       try {
         await redis.ping();
         return { status: "healthy", message: "Redis connection OK" };
       } catch (error) {
         return {
           status: "unhealthy",
           message: `Redis error: ${error.message}`,
         };
       }
     }
   }

   // Health check endpoint
   app.get("/health", async (req, res) => {
     const healthChecks = await Promise.all([
       healthCheckService.checkDatabase(),
       healthCheckService.checkPCS(),
       healthCheckService.checkRedis(),
     ]);

     const overallStatus = healthChecks.every(
       (check) => check.status === "healthy"
     )
       ? "healthy"
       : "unhealthy";

     res.status(overallStatus === "healthy" ? 200 : 503).json({
       status: overallStatus,
       timestamp: new Date().toISOString(),
       checks: healthChecks,
     });
   });
   ```

2. **Set up Prometheus metrics**

   ```typescript
   import prometheus from "prom-client";

   // Define metrics
   const httpRequestDuration = new prometheus.Histogram({
     name: "http_request_duration_seconds",
     help: "Duration of HTTP requests in seconds",
     labelNames: ["method", "route", "status_code"],
   });

   const httpRequestsTotal = new prometheus.Counter({
     name: "http_requests_total",
     help: "Total number of HTTP requests",
     labelNames: ["method", "route", "status_code"],
   });

   // Metrics endpoint
   app.get("/metrics", async (req, res) => {
     res.set("Content-Type", prometheus.register.contentType);
     res.end(await prometheus.register.metrics());
   });
   ```

## Phase 4: Validation & Deployment

### 4.1 Testing

1. **Unit tests**

   ```bash
   # Run unit tests
   npm test

   # Run with coverage
   npm run test:coverage
   ```

2. **Integration tests**

   ```bash
   # Test database connections
   npm run test:integration

   # Test PCS integration
   npm run test:pcs
   ```

3. **End-to-end tests**
   ```bash
   # Test complete workflows
   npm run test:e2e
   ```

### 4.2 Performance Testing

1. **Load testing**

   ```bash
   # Install artillery for load testing
   npm install -g artillery

   # Run load test
   artillery run load-test.yml
   ```

2. **Database performance**

   ```sql
   -- Test query performance
   EXPLAIN ANALYZE SELECT * FROM app_users WHERE email = 'test@example.com';

   -- Check index usage
   SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   WHERE schemaname = 'public';
   ```

### 4.3 Security Validation

1. **Security scan**

   ```bash
   # Run security audit
   npm audit

   # Check for vulnerabilities
   npm audit --audit-level moderate
   ```

2. **Permission validation**
   ```bash
   # Verify PCS permissions
   curl -X GET http://localhost:8000/api/v1/apps/$APP_ID/permissions \
     -H "Authorization: Bearer $APP_KEY"
   ```

### 4.4 Deployment

1. **Create deployment configuration**

   ```yaml
   # docker-compose.yml for your app
   version: "3.8"

   services:
     your-app:
       build: .
       container_name: your-app
       environment:
         - NODE_ENV=production
         - PCS_BASE_URL=${PCS_BASE_URL}
         - PCS_APP_ID=${PCS_APP_ID}
         - PCS_APP_KEY=${PCS_APP_KEY}
       ports:
         - "3000:3000"
       networks:
         - digi-net
       depends_on:
         - postgres
         - neo4j
         - redis

   networks:
     digi-net:
       external: true
   ```

2. **Deploy to staging**

   ```bash
   # Deploy to staging environment
   docker-compose -f docker-compose.staging.yml up -d

   # Run health checks
   curl http://staging.your-app.com/health
   ```

3. **Deploy to production**

   ```bash
   # Deploy to production
   docker-compose -f docker-compose.prod.yml up -d

   # Verify deployment
   curl http://your-app.com/health
   ```

## Post-Onboarding

### 5.1 Monitoring & Alerting

1. **Set up Grafana dashboards**

   - Import app-specific dashboards
   - Configure alerting rules
   - Set up notification channels

2. **Configure log aggregation**
   ```yaml
   # Logging configuration
   logging:
     level: "info"
     format: "json"
     outputs:
       - console
       - file
       - syslog
   ```

### 5.2 Maintenance & Updates

1. **Regular health checks**

   ```bash
   # Automated health checks
   crontab -e

   # Add: */5 * * * * curl -f http://your-app.com/health || echo "App unhealthy"
   ```

2. **Backup verification**

   ```bash
   # Test backup restoration
   make backup-test
   ```

3. **Performance monitoring**
   ```bash
   # Monitor performance metrics
   curl http://your-app.com/metrics | grep your_app
   ```

## Troubleshooting

### Common Issues

1. **PCS Connection Failed**

   - Verify app credentials
   - Check network connectivity
   - Validate API endpoints

2. **Database Connection Issues**

   - Verify database credentials
   - Check container status
   - Validate network configuration

3. **Performance Problems**
   - Check resource usage
   - Analyze query performance
   - Review monitoring metrics

### Support Resources

- **Documentation**: [Dynamic Prompting Architecture](DYNAMIC_PROMPTING_ARCHITECTURE.md)
- **PCS SDK Reference**: [PCS SDK Reference](PCS_SDK_REFERENCE.md)
- **Infrastructure Guide**: [Infrastructure Repository Design](INFRASTRUCTURE_REPOSITORY.md)
- **Community**: [Digi Ecosystem Discord/Slack]

## Onboarding Checklist

### Pre-Onboarding

- [ ] App requirements assessed
- [ ] Architecture reviewed
- [ ] Resource requirements estimated
- [ ] Team access granted to infrastructure repo

### Infrastructure Setup

- [ ] Infrastructure repository cloned and configured
- [ ] Shared services started and verified
- [ ] App-specific databases created
- [ ] Database schemas initialized

### PCS Integration

- [ ] App registered with PCS
- [ ] SDK installed and configured
- [ ] Initial prompts created
- [ ] Connection tested

### Bootstrap System

- [ ] Implement idempotent bootstrap logic
- [ ] Create foundational prompt templates
- [ ] Set up system metadata tracking
- [ ] Test bootstrap idempotency (multiple runs)
- [ ] Implement bootstrap health checks
- [ ] Test force re-bootstrap functionality

### Development & Testing

- [ ] Database connections implemented
- [ ] Health checks implemented
- [ ] Unit tests written and passing
- [ ] Integration tests passing
- [ ] Performance tests completed

### Deployment

- [ ] Staging deployment successful
- [ ] Production deployment successful
- [ ] Monitoring configured
- [ ] Alerting set up
- [ ] Documentation updated

### Post-Deployment

- [ ] Performance baseline established
- [ ] Backup procedures tested
- [ ] Team training completed
- [ ] Support procedures documented

## Next Steps

After successful onboarding:

1. **Explore advanced PCS features**
2. **Implement context-aware prompting**
3. **Set up automated testing pipelines**
4. **Contribute to ecosystem improvements**
5. **Share best practices with other teams**

## Related Documentation

- [Dynamic Prompting Architecture](DYNAMIC_PROMPTING_ARCHITORY.md)
- [PCS SDK Reference](PCS_SDK_REFERENCE.md)
- [Infrastructure Repository Design](INFRASTRUCTURE_REPOSITORY.md)
- [Schema Management](SCHEMA_MANAGEMENT.md)
- [Multi-App Deployment Strategy](MULTI_APP_DEPLOYMENT.md)
- [Bootstrap System Implementation](../src/app/bootstrap.py) - Working example of idempotent bootstrap
