# Prompt and Context Service (PCS)

**Filepath:** `pcs/README.md`  
**Purpose:** Production-ready autonomous coding agent service with advanced context management  
**Related Components:** Digi Infrastructure, Database Services, Template Engine, SDK  
**Tags:** pcs, autonomous-agent, prompt-engineering, context-management, fastapi

## Overview

The Prompt and Context Service (PCS) is a sophisticated autonomous coding agent system that provides intelligent prompt engineering, dynamic context management, and automated code generation capabilities. Built as part of the Digi Infrastructure ecosystem, PCS serves as the foundation for AI-driven development workflows.

## Key Features

- **🤖 Autonomous Code Generation**: AI-powered coding assistant with context awareness
- **📋 Advanced Prompt Engineering**: Dynamic prompt templates with variable substitution
- **🗄️ Intelligent Context Management**: Hierarchical context relationships and caching
- **🔄 Conversation Management**: Persistent conversation history with role-based messaging
- **🛡️ Enterprise Security**: JWT authentication, API keys, and role-based access control
- **📊 Production Monitoring**: Comprehensive health checks, metrics, and logging
- **🔌 SDK Integration**: Easy integration with existing applications via REST API

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PCS Architecture                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   FastAPI   │  │ SQLAlchemy  │  │   Redis     │          │
│  │   Server    │  │ ORM Layer   │  │   Cache     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │ Prompt      │  │ PostgreSQL  │  │ ChromaDB    │          │
│  │ Templates   │  │ Database    │  │ Vectors     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │ Context     │  │ Conversation│  │ Knowledge   │          │
│  │ Management  │  │ History     │  │ Base        │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+ with uv package manager
- Access to Digi Infrastructure services (PostgreSQL, Redis, ChromaDB)
- Docker (for development environment)

### Installation

1. **Set up the development environment**

   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Navigate to PCS directory
   cd pcs

   # Install dependencies
   uv sync --extra dev
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Initialize the database**

   ```bash
   uv run alembic upgrade head
   ```

4. **Start the development server**

   ```bash
   uv run pcs-server
   ```

5. **Verify installation**

   ```bash
   curl http://localhost:8000/api/v1/health/
   ```

## API Reference

### Authentication

```bash
# Obtain JWT token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/prompts/
```

### Core Endpoints

| Endpoint                 | Method    | Purpose                 |
| ------------------------ | --------- | ----------------------- |
| `/api/v1/prompts/`       | GET, POST | Manage prompt templates |
| `/api/v1/contexts/`      | GET, POST | Context management      |
| `/api/v1/conversations/` | GET, POST | Conversation handling   |
| `/api/v1/health/`        | GET       | Health monitoring       |

### Example Usage

```python
import httpx

# Create a prompt template
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/prompts/",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "code_review",
            "description": "Code review assistant",
            "template": "Please review this code: {{code}}",
            "variables": ["code"]
        }
    )
```

## Configuration

PCS uses environment variables for configuration:

```bash
# Core Settings
PCS_ENVIRONMENT=development
PCS_DEBUG=true
PCS_SECRET_KEY=your-secret-key

# Database
PCS_DB_HOST=localhost
PCS_DB_PORT=5432
PCS_DB_USER=pcs_user
PCS_DB_PASSWORD=pcs_password
PCS_DB_NAME=pcs_dev

# Redis Cache
PCS_REDIS_HOST=localhost
PCS_REDIS_PORT=6379
PCS_REDIS_DB=0

# Security
PCS_JWT_SECRET_KEY=jwt-secret-key
PCS_JWT_ALGORITHM=HS256
PCS_JWT_EXPIRE_MINUTES=30
```

## Development

### Project Structure

```
pcs/
├── src/pcs/                    # Main application code
│   ├── api/                   # FastAPI routes and dependencies
│   ├── core/                  # Core functionality (config, database, exceptions)
│   ├── models/                # SQLAlchemy data models
│   ├── services/              # Business logic layer
│   ├── repositories/          # Data access layer
│   └── utils/                 # Utility functions
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test fixtures
├── alembic/                   # Database migrations
├── docs/                      # Documentation
└── requirements/              # Dependency files
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
```

### Code Quality

```bash
# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run mypy src

# Run all quality checks
uv run pre-commit run --all-files
```

## Deployment

### Docker

```bash
# Build image
docker build -t pcs:latest .

# Run container
docker run -p 8000:8000 \
  --env-file .env \
  pcs:latest
```

### Production

```bash
# Install production dependencies
uv sync --extra prod

# Run with Gunicorn
uv run gunicorn pcs.main:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

## Integration with Digi Infrastructure

PCS is designed to integrate seamlessly with the Digi Infrastructure ecosystem:

```yaml
# docker-compose.yml integration
services:
  pcs:
    build: ./pcs
    external_links:
      - digi-infrastructure_postgres_1:postgres
      - digi-infrastructure_redis_1:redis
      - digi-infrastructure_chroma_1:chroma
    networks:
      - digi-net
```

## SDK Usage

```python
from pcs_sdk import PCSClient

# Initialize client
client = PCSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Generate code with context
result = await client.generate_code(
    prompt="Create a REST API endpoint",
    context={
        "framework": "FastAPI",
        "database": "PostgreSQL"
    }
)
```

## Monitoring

- **Health Checks**: `/api/v1/health/detailed`
- **Metrics**: Prometheus metrics at `/metrics`
- **Logs**: Structured JSON logs with correlation IDs
- **Tracing**: OpenTelemetry integration for distributed tracing

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `uv run pre-commit run --all-files`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Submit a pull request

## Security

- JWT-based authentication with configurable expiration
- API key support for service-to-service communication
- Role-based access control (RBAC)
- Input validation and sanitization
- Rate limiting and request throttling
- Security headers and CORS configuration

## Performance

- Async/await throughout for high concurrency
- Redis caching for frequently accessed data
- Connection pooling for database operations
- Vector database optimization for semantic search
- Prometheus metrics for performance monitoring

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Support

- **Documentation**: [https://docs.digiinfra.com/pcs](https://docs.digiinfra.com/pcs)
- **Issues**: [GitHub Issues](https://github.com/your-org/digi-infrastructure/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/digi-infrastructure/discussions)
- **Slack**: #pcs-support (internal)
