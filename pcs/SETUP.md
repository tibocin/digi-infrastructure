# PCS Setup Guide

## Current Implementation Status

✅ **COMPLETED STEPS:**

- **Step 1**: Project Initialization - Complete project structure with UV and modern Python tooling
- **Step 2**: Core Configuration System - Type-safe configuration with Pydantic v2
- **Step 3**: Database Connection Management - Async database connections with SQLAlchemy 2.0
- **Step 4**: Data Models Foundation - Base models with relationships
- **Step 5**: FastAPI Application Setup - Application factory with middleware

🚧 **NEXT STEPS:**

- **Step 6**: Authentication & Security - JWT and API key authentication
- **Step 7**: Basic Health Check Endpoints - Multi-level health checks (partially done)
- **Step 8**: Logging & Metrics Setup - Structured logging with Prometheus metrics
- **Step 9**: Database Migrations Setup - Alembic configuration
- **Step 10**: Development Environment Setup - Docker compose for development

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to PCS directory
cd pcs

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, set these values:
# PCS_SECURITY_SECRET_KEY=your-secret-key-here
# PCS_SECURITY_JWT_SECRET_KEY=your-jwt-secret-key-here
```

### 3. Test the Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Test imports
uv run python -c "
import sys
sys.path.insert(0, 'src')
from pcs.core import config
print('✅ PCS imports working!')
"

# Run tests
uv run pytest tests/unit/test_config.py -v
```

### 4. Start Development Server

```bash
# Set required environment variables
export PCS_SECURITY_SECRET_KEY="development_secret_key_change_in_production"
export PCS_SECURITY_JWT_SECRET_KEY="development_jwt_secret_key_change_in_production"

# Start the server
uv run python src/pcs/main.py
```

### 5. Verify Health Endpoints

```bash
# Test basic health
curl http://localhost:8000/api/v1/health/

# Test root endpoint
curl http://localhost:8000/
```

## Integration with Digi Infrastructure

The PCS service is designed to integrate seamlessly with the digi-infrastructure:

### Database Integration

- **PostgreSQL**: Primary data storage
- **Redis**: Caching and session management
- **ChromaDB**: Vector embeddings for semantic search
- **Neo4j**: Graph relationships between contexts

### Service Discovery

PCS connects to digi-infrastructure services via:

```yaml
external_links:
  - digi-infrastructure_postgres_1:postgres
  - digi-infrastructure_redis_1:redis
  - digi-infrastructure_chroma_1:chroma
  - digi-infrastructure_neo4j_1:neo4j
```

### Health Monitoring

- Basic health: `GET /api/v1/health/`
- Detailed health: `GET /api/v1/health/detailed`
- Readiness probe: `GET /api/v1/health/readiness`
- Liveness probe: `GET /api/v1/health/liveness`

## Architecture Overview

```
PCS Application
├── FastAPI Server (Port 8000)
├── SQLAlchemy ORM (Async)
├── Pydantic Settings
├── Custom Middleware Stack
├── Health Check Endpoints
└── Future: Authentication, Logging, Metrics
```

## Database Models

- **BaseModel**: Common fields (id, created_at, updated_at)
- **PromptTemplate**: Template definitions and versions
- **Context**: Hierarchical context management
- **Conversation**: Chat history and state
- **Relationships**: Context interconnections

## Development Workflow

1. Make changes to source code
2. Run tests: `uv run pytest`
3. Check code quality: `uv run black src tests && uv run ruff check src tests`
4. Test manually via health endpoints
5. Commit changes

## Docker Deployment

```bash
# Build and run with Docker
docker-compose up --build

# Or integrate with digi-infrastructure
cd ../  # Back to digi-infrastructure root
docker-compose -f docker-compose.yml -f pcs/docker-compose.yml up
```

## Next Implementation Steps

1. **JWT Authentication**: Implement secure token-based auth
2. **API Endpoints**: Create CRUD endpoints for prompts, contexts, conversations
3. **Template Engine**: Dynamic prompt generation with variables
4. **Context Engine**: Intelligent context selection and ranking
5. **Vector Search**: Semantic search via ChromaDB integration
6. **Monitoring**: Prometheus metrics and structured logging
7. **SDK**: Client libraries for easy integration

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `uv sync --extra dev` to install dependencies
2. **Config Validation**: Ensure secret keys are at least 16 characters
3. **Database Connection**: Verify digi-infrastructure services are running
4. **Port Conflicts**: Check that port 8000 is available

### Debug Mode

```bash
export PCS_DEBUG=true
export PCS_ENVIRONMENT=development
uv run python src/pcs/main.py
```

This enables:

- Detailed error messages
- Auto-reload on code changes
- OpenAPI documentation at `/docs`
- Verbose logging
