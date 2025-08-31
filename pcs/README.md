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
- **🧠 Advanced Vector Operations**: Enhanced Qdrant repository with clustering, bulk operations, and performance optimization
- **🔍 Multi-Tenant Support**: Isolated data management for multiple applications

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
│  │ Prompt      │  │ PostgreSQL  │  │   Qdrant    │          │
│  │ Templates   │  │ Database    │  │   Vectors   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │ Context     │  │ Conversation│  │ Clustering  │          │
│  │ Management  │  │ History     │  │ & Analytics │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Enhanced Qdrant Repository

PCS now features a **modular, enhanced Qdrant repository** with specialized components:

### Core Components
- **QdrantCoreOperations**: Basic CRUD operations and collection management
- **QdrantAdvancedSearch**: Semantic search with multiple algorithms and filtering
- **QdrantBulkOperations**: Batch processing with error handling and progress tracking
- **QdrantPerformanceMonitor**: Real-time performance metrics and optimization
- **QdrantClustering**: Document clustering using K-means and DBSCAN algorithms
- **QdrantExport**: Data export in multiple formats (numpy, JSON, list)

### Key Features
- **Multi-tenant Support**: Isolated collections and data per application
- **Performance Optimization**: Automatic collection tuning and HNSW configuration
- **Bulk Operations**: Efficient batch processing with retry mechanisms
- **Advanced Search**: Multiple similarity algorithms (cosine, euclidean, dot product)
- **Clustering**: Document grouping and analysis capabilities
- **Legacy Compatibility**: Backward-compatible API methods

## Quick Start

### Prerequisites

- Python 3.11+ with uv package manager
- Access to Digi Infrastructure services (PostgreSQL, Redis, Qdrant)
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
  -d '{"username": "admin", "password": "password"}'
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/api/v1/health/

# Detailed health check
curl http://localhost:8000/api/v1/health/detailed
```

### Vector Operations

```python
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository

# Initialize repository
repo = EnhancedQdrantRepository(
    host="localhost",
    port=6333,
    use_async=True
)

# Create collection
await repo.create_collection_optimized(
    collection_name="my_collection",
    vector_size=384,
    distance="cosine"
)

# Bulk upsert documents
result = await repo.bulk_upsert_documents(
    collection_name="my_collection",
    documents=my_documents,
    batch_size=100
)

# Advanced semantic search
results = await repo.semantic_search_advanced(
    VectorSearchRequest(
        query_embedding=query_vector,
        collection_name="my_collection",
        limit=10,
        similarity_threshold=0.7
    )
)

# Document clustering
clusters = await repo.cluster_documents(
    embeddings=document_embeddings,
    algorithm="kmeans",
    n_clusters=5
)
```

## Testing

PCS now has **100% test coverage** for core Qdrant functionality:

```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/unit/test_qdrant_export.py -v
uv run pytest tests/unit/test_qdrant_async.py -v
uv run pytest tests/unit/test_qdrant_legacy.py -v

# Run with coverage
uv run pytest --cov=pcs
```

### Test Results
- **Export Tests**: 16/16 PASSING (100%) ✅
- **Async Tests**: 22/22 PASSING (100%) ✅
- **Legacy Tests**: 20/20 PASSING (100%) ✅

## Integration Examples

### Downstream Application Integration

```python
from pcs.examples.downstream_app_integration import DownstreamAppExample

# Initialize for digi-core
digi_core = DownstreamAppExample("digi_core")
await digi_core.add_knowledge("AI content", {"category": "machine_learning"})

# Initialize for beep-boop
beep_boop = DownstreamAppExample("beep_boop")
await beep_boop.search_knowledge("bot responses", n_results=10)
```

### Multi-Tenant Setup

```python
# Each app gets isolated collections
digi_core_collection = "digi_core_documents"
beep_boop_collection = "beep_boop_documents"

# Tenant-specific operations
await repo.upsert_documents(digi_core_collection, documents, tenant_id="digi_core")
await repo.search_similar(beep_boop_collection, query_vector, tenant_id="beep_boop")
```

## Performance Features

### Monitoring
- Real-time operation tracking
- Collection performance profiles
- Query execution time analysis
- Resource usage metrics

### Optimization
- Automatic HNSW parameter tuning
- Collection performance recommendations
- Bulk operation batching
- Connection pooling and retry logic

## Development

### Code Quality
```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint code
uv run flake8 src/ tests/

# Type checking
uv run mypy src/
```

### Environment Management
```bash
# Activate environment
source .venv/bin/activate

# Or use uv shell
uv shell

# Install new dependencies
uv add package_name

# Update dependencies
uv sync
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass: `uv run pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:
- Check the [documentation](./docs/)
- Review [test examples](./tests/)
- Open an issue on GitHub
