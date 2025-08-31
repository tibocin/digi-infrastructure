# PCS Setup Guide

## Current Implementation Status

✅ **COMPLETED STEPS:**

- **Step 1**: Project Initialization - Complete project structure with UV and modern Python tooling
- **Step 2**: Core Configuration System - Type-safe configuration with Pydantic v2
- **Step 3**: Database Connection Management - Async database connections with SQLAlchemy 2.0
- **Step 4**: Data Models Foundation - Base models with relationships
- **Step 5**: FastAPI Application Setup - Application factory with middleware
- **Step 6**: Enhanced Qdrant Repository - Modular repository with specialized components
- **Step 7**: Comprehensive Testing - 100% test coverage for core Qdrant functionality
- **Step 8**: Multi-Tenant Support - Isolated data management for multiple applications
- **Step 9**: Performance Monitoring - Real-time metrics and optimization
- **Step 10**: Clustering Capabilities - K-means and DBSCAN document clustering

🚧 **NEXT STEPS:**

- **Step 11**: Authentication & Security - JWT and API key authentication
- **Step 12**: Advanced Health Check Endpoints - Multi-level health checks
- **Step 13**: Logging & Metrics Setup - Structured logging with Prometheus metrics
- **Step 14**: Database Migrations Setup - Alembic configuration
- **Step 15**: Development Environment Setup - Docker compose for development

## Enhanced Qdrant Repository Features

The PCS now includes a **production-ready, enhanced Qdrant repository** with:

### Core Components
- **QdrantCoreOperations**: Basic CRUD operations and collection management
- **QdrantAdvancedSearch**: Semantic search with multiple algorithms and filtering
- **QdrantBulkOperations**: Batch processing with error handling and progress tracking
- **QdrantPerformanceMonitor**: Real-time performance metrics and optimization
- **QdrantClustering**: Document clustering using K-means and DBSCAN algorithms
- **QdrantExport**: Data export in multiple formats (numpy, JSON, list)

### Key Capabilities
- **Multi-tenant Support**: Isolated collections per application (digi-core, beep-boop, etc.)
- **Performance Optimization**: Automatic collection tuning and HNSW configuration
- **Bulk Operations**: Efficient batch processing with retry mechanisms
- **Advanced Search**: Multiple similarity algorithms (cosine, euclidean, dot product)
- **Clustering**: Document grouping and analysis capabilities
- **Legacy Compatibility**: Backward-compatible API methods for existing integrations

## Test Coverage Status

✅ **100% Test Coverage Achieved:**

- **Export Functionality**: 16/16 tests passing - NumPy array handling, format conversion, tenant filtering
- **Async Operations**: 22/22 tests passing - Mock configurations, method signatures, async/await patterns
- **Legacy Compatibility**: 20/20 tests passing - Backward compatibility, parameter handling, method delegation

All tests use modern testing practices with proper mocking, error handling, and edge case coverage.

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
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
print('✅ PCS imports working!')
print('✅ Enhanced Qdrant Repository available!')
"

# Run tests to verify functionality
uv run pytest tests/unit/test_qdrant_export.py -v
uv run pytest tests/unit/test_qdrant_async.py -v
uv run pytest tests/unit/test_qdrant_legacy.py -v
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

## Enhanced Qdrant Repository Usage

### Basic Setup

```python
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository

# Initialize repository
repo = EnhancedQdrantRepository(
    host="localhost",
    port=6333,
    use_async=True
)
```

### Collection Management

```python
# Create optimized collection
success = await repo.create_collection_optimized(
    collection_name="my_collection",
    vector_size=384,
    distance="cosine"
)

# Get collection statistics
stats = await repo.get_collection_statistics("my_collection", tenant_id="digi_core")
```

### Document Operations

```python
# Bulk upsert with progress tracking
result = await repo.bulk_upsert_documents(
    collection_name="my_collection",
    documents=my_documents,
    batch_size=100,
    tenant_id="digi_core"
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
```

### Clustering and Analytics

```python
# Document clustering
clusters = await repo.cluster_documents(
    embeddings=document_embeddings,
    algorithm="kmeans",
    n_clusters=5
)

# Performance monitoring
performance = repo.get_performance_summary()
optimization_recommendations = repo.get_optimization_recommendations("my_collection")
```

## Integration with Digi Infrastructure

The PCS service is designed to integrate seamlessly with the digi-infrastructure:

### Database Integration

- **PostgreSQL**: Primary data storage with multi-tenant support
- **Redis**: Caching and session management
- **Qdrant**: Enhanced vector database with clustering and analytics
- **Neo4j**: Graph relationships between contexts

### Multi-App Support

PCS supports multiple applications with isolated data:

```python
# Each app gets its own collections
digi_core_collection = "digi_core_documents"
beep_boop_collection = "beep_boop_documents"

# Tenant-specific operations
await repo.upsert_documents(digi_core_collection, documents, tenant_id="digi_core")
await repo.search_similar(beep_boop_collection, query_vector, tenant_id="beep_boop")
```

### Service Discovery

PCS connects to digi-infrastructure services via:
- **Environment Variables**: Service URLs and credentials
- **Docker Networks**: Container-to-container communication
- **Health Checks**: Service availability monitoring

## Development Workflow

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

# Run tests
uv run pytest
```

### Testing Strategy

- **Unit Tests**: Mocked dependencies for fast execution
- **Integration Tests**: Real service interactions
- **Performance Tests**: Load testing and optimization
- **Coverage Reports**: Ensure comprehensive testing

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

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Test Failures**: Check mock configurations and dependencies
3. **Connection Issues**: Verify service URLs and network connectivity
4. **Performance Issues**: Monitor metrics and use optimization recommendations

### Debug Mode

```bash
# Enable debug logging
export PCS_DEBUG=true
export PCS_LOG_LEVEL=debug

# Run with verbose output
uv run python src/pcs/main.py --debug
```

## Next Steps

With the enhanced Qdrant repository and comprehensive testing in place, the next priorities are:

1. **Authentication System**: JWT tokens and API key management
2. **Advanced Health Checks**: Service dependency monitoring
3. **Production Monitoring**: Prometheus metrics and Grafana dashboards
4. **Deployment**: Docker containers and orchestration
5. **Documentation**: API reference and integration guides

The foundation is now solid for building production-ready AI-powered applications with robust vector database operations.
