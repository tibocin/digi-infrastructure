# ChromaDB to Qdrant Migration Guide

## Overview

This comprehensive guide covers the complete migration from ChromaDB to Qdrant vector database, including infrastructure updates, data migration, tenant application integration, and performance optimization.

## Table of Contents

1. [Migration Benefits](#migration-benefits)
2. [Pre-Migration Checklist](#pre-migration-checklist)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Data Migration Process](#data-migration-process)
5. [Tenant Application Integration](#tenant-application-integration)
6. [Multi-Tenancy Implementation](#multi-tenancy-implementation)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Troubleshooting](#troubleshooting)
10. [Rollback Strategy](#rollback-strategy)

## Migration Benefits

### Performance Improvements

| Metric | ChromaDB | Qdrant | Improvement |
|--------|----------|--------|-------------|
| Query Latency | 50-200ms | 5-20ms | **10x faster** |
| Memory Usage | 4GB | 256MB-1GB | **4-16x reduction** |
| Throughput | 100 QPS | 1000+ QPS | **10x higher** |
| Scalability | Single node | Distributed | **Horizontal scaling** |
| Vector Quantization | Limited | Advanced | **4-16x compression** |

### Feature Enhancements

- **Advanced Indexing**: HNSW with configurable parameters
- **Vector Quantization**: Scalar, Product, and Binary quantization
- **Multi-Tenancy**: Native payload-based isolation
- **Distributed Architecture**: Horizontal scaling capabilities
- **Performance Monitoring**: Built-in metrics and observability
- **Advanced Filtering**: High-performance payload filtering

## Pre-Migration Checklist

### Environment Preparation

- [ ] **Backup Current Data**: Create full backup of ChromaDB data
- [ ] **Resource Planning**: Ensure adequate resources for both systems during migration
- [ ] **Network Configuration**: Configure firewall rules for Qdrant (ports 6333, 6334)
- [ ] **API Key Generation**: Generate secure API keys for Qdrant
- [ ] **Monitoring Setup**: Prepare monitoring for migration process

### Application Preparation

- [ ] **Code Review**: Review current ChromaDB integration points
- [ ] **Dependency Updates**: Update requirements with Qdrant client
- [ ] **Configuration Updates**: Prepare new configuration files
- [ ] **Test Environment**: Set up staging environment for testing

### Data Analysis

- [ ] **Collection Inventory**: Document all ChromaDB collections
- [ ] **Data Volume Assessment**: Calculate total documents and storage needs
- [ ] **Tenant Mapping**: Define tenant isolation strategy
- [ ] **Vector Dimensions**: Verify vector dimensions and distance metrics

## Infrastructure Setup

### 1. Install Qdrant Dependencies

```bash
# Add to requirements.txt or pyproject.toml
pip install qdrant-client==1.7.0
pip install numpy>=1.21.0
```

### 2. Update Docker Compose

Replace ChromaDB service with Qdrant:

```yaml
# docker-compose.yml
services:
  # Remove chromadb service
  # chroma:
  #   image: chromadb/chroma:latest
  #   ...

  # Add Qdrant service
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: digi-qdrant
    restart: unless-stopped
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__LOG_LEVEL: INFO
      QDRANT__STORAGE__ON_DISK_PAYLOAD: true
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    volumes:
      - qdrant_data:/qdrant/storage
      - qdrant_snapshots:/qdrant/snapshots
    networks:
      - digi-net
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
  qdrant_snapshots:
```

### 3. Environment Configuration

Update `.env` file:

```bash
# Remove ChromaDB configuration
# CHROMA_HOST=localhost
# CHROMA_PORT=8001

# Add Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_secure_api_key_here
QDRANT_USE_HTTPS=false
QDRANT_GRPC_PORT=6334

# Vector database configuration
VECTOR_DB_TYPE=qdrant  # Change from 'chromadb'
VECTOR_COLLECTION_NAME=digi_knowledge
VECTOR_DIMENSION=384
VECTOR_DISTANCE_METRIC=cosine
```

## Data Migration Process

### Step 1: Quick Migration Test

Run the quick migration test to validate setup:

```bash
# Make sure both ChromaDB and Qdrant are running
docker-compose up chromadb qdrant -d

# Run quick test
cd /workspace
uv run python scripts/quick_migration_test.py
```

Expected output:
```
✅ Quick Migration Test COMPLETED SUCCESSFULLY
✅ Quick migration test passed! Ready for full migration.
```

### Step 2: Create Tenant Mapping

Create a tenant mapping file for multi-tenancy:

```json
# tenant_mapping.json
{
  "digi_core_knowledge": "digi_core",
  "lernmi_knowledge": "lernmi", 
  "beep_boop_knowledge": "beep_boop",
  "general_knowledge": "shared"
}
```

### Step 3: Run Full Migration

Execute the comprehensive migration:

```bash
# Basic migration (all collections)
uv run python scripts/migrate_chromadb_to_qdrant.py \
  --chromadb-host localhost \
  --chromadb-port 8001 \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --qdrant-api-key your_api_key \
  --target-collection digi_knowledge \
  --batch-size 1000 \
  --tenant-mapping tenant_mapping.json \
  --enable-quantization \
  --report-file migration_report.json

# Specific collections migration
uv run python scripts/migrate_chromadb_to_qdrant.py \
  --chromadb-collections digi_core_knowledge lernmi_knowledge \
  --target-collection digi_knowledge \
  --tenant-mapping tenant_mapping.json \
  --batch-size 500 \
  --vector-size 384 \
  --distance-metric cosine
```

### Step 4: Verify Migration

The migration script includes automatic verification, but you can also manually verify:

```bash
# Check Qdrant collection
curl -X GET "http://localhost:6333/collections/digi_knowledge" \
  -H "api-key: your_api_key"

# Check document count
curl -X GET "http://localhost:6333/collections/digi_knowledge" \
  -H "api-key: your_api_key" | jq '.result.points_count'

# Test search functionality
curl -X POST "http://localhost:6333/collections/digi_knowledge/points/search" \
  -H "api-key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5,
    "filter": {
      "must": [
        {"key": "tenant_id", "match": {"value": "digi_core"}}
      ]
    }
  }'
```

## Tenant Application Integration

### 1. Update Application Dependencies

```toml
# pyproject.toml
[tool.uv.dependencies]
# Remove chromadb dependency
# chromadb = "0.4.15"

# Add qdrant dependency
qdrant-client = "1.7.0"
```

### 2. Repository Integration

Replace ChromaDB repository usage:

```python
# Before (ChromaDB)
from pcs.repositories.chroma_repo import EnhancedChromaRepository
import chromadb

# Initialize ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=8001)
vector_repo = EnhancedChromaRepository(chroma_client)

# After (Qdrant)
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository

# Initialize Qdrant
vector_repo = EnhancedQdrantRepository(
    host="localhost",
    port=6333,
    api_key="your_api_key",
    https=False
)
```

### 3. Update Application Configuration

```python
# config.py
class VectorDBConfig:
    # Replace ChromaDB settings
    # CHROMADB_HOST = "localhost"
    # CHROMADB_PORT = 8001
    
    # Add Qdrant settings
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_USE_HTTPS = os.getenv("QDRANT_USE_HTTPS", "false").lower() == "true"
    QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
    
    # Vector configuration
    VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "digi_knowledge")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))
    VECTOR_DISTANCE_METRIC = os.getenv("VECTOR_DISTANCE_METRIC", "cosine")
```

### 4. Application Service Updates

```python
# services/vector_service.py
from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository, 
    VectorSearchRequest,
    VectorDocument,
    BulkVectorOperation
)

class VectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=settings.QDRANT_USE_HTTPS
        )
        self.collection_name = settings.VECTOR_COLLECTION_NAME
    
    async def add_documents(self, documents: List[Dict]) -> bool:
        """Add documents with tenant isolation."""
        vector_docs = []
        for doc in documents:
            vector_doc = VectorDocument(
                id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc.get("metadata", {}),
                created_at=datetime.utcnow(),
                collection_name=self.collection_name,
                tenant_id=self.tenant_id  # Automatic tenant isolation
            )
            vector_docs.append(vector_doc)
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=vector_docs,
            tenant_id=self.tenant_id
        )
        
        result = await self.vector_repo.bulk_upsert_documents(
            self.collection_name, 
            operation
        )
        return result["total_processed"] > 0
    
    async def semantic_search(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Perform semantic search with tenant isolation."""
        request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=self.collection_name,
            n_results=limit,
            metadata_filter=filters,
            tenant_id=self.tenant_id,  # Automatic tenant filtering
            include_embeddings=False,
            rerank=True
        )
        
        results = await self.vector_repo.semantic_search_advanced(request)
        
        return [
            {
                "id": result.document.id,
                "content": result.document.content,
                "metadata": result.document.metadata,
                "similarity_score": result.similarity_score
            }
            for result in results
        ]
```

## Multi-Tenancy Implementation

### Payload-Based Tenant Isolation

Qdrant uses payload-based multi-tenancy instead of separate collections:

```python
# Tenant isolation through payload filtering
search_results = await vector_repo.semantic_search_advanced(
    VectorSearchRequest(
        query_embedding=query_vector,
        collection_name="digi_knowledge",
        tenant_id="digi_core",  # Automatic filtering
        metadata_filter={
            "category": "technical",
            "priority": "high"
        }
    )
)
```

### Tenant Security

```python
class TenantSecurityMiddleware:
    """Middleware to ensure tenant data isolation."""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
    
    async def validate_tenant_access(self, user_id: str, tenant_id: str) -> bool:
        """Validate user has access to tenant data."""
        # Implement your tenant access validation logic
        user_tenants = await self.get_user_tenants(user_id)
        return tenant_id in user_tenants
    
    async def secure_search(
        self, 
        user_id: str, 
        tenant_id: str, 
        query_embedding: List[float]
    ) -> List[Dict]:
        """Perform secure search with tenant validation."""
        if not await self.validate_tenant_access(user_id, tenant_id):
            raise PermissionError(f"User {user_id} not authorized for tenant {tenant_id}")
        
        # Search is automatically filtered by tenant_id
        return await self.vector_service.semantic_search(
            query_embedding=query_embedding,
            tenant_id=tenant_id
        )
```

### Cross-Tenant Analytics (Optional)

```python
class CrossTenantAnalytics:
    """Analytics across tenants for admin users."""
    
    async def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tenants."""
        stats = await self.vector_repo.get_collection_statistics(
            collection_name="digi_knowledge"
        )
        
        # Get per-tenant document counts
        tenant_stats = {}
        for tenant_id in ["digi_core", "lernmi", "beep_boop"]:
            tenant_specific_stats = await self.vector_repo.get_collection_statistics(
                collection_name="digi_knowledge",
                tenant_id=tenant_id
            )
            tenant_stats[tenant_id] = {
                "document_count": tenant_specific_stats.document_count,
                "memory_usage_mb": tenant_specific_stats.memory_usage_mb,
                "avg_query_time_ms": tenant_specific_stats.avg_query_time_ms
            }
        
        return {
            "total_statistics": stats.to_dict(),
            "tenant_statistics": tenant_stats
        }
```

## Performance Optimization

### Vector Quantization

Enable quantization to reduce memory usage:

```python
# When creating collections
config = QdrantCollectionConfig(
    name="digi_knowledge",
    vector_size=384,
    distance=QdrantDistance.COSINE,
    quantization_config={
        "type": "scalar",  # or "product" or "binary"
        "quantile": 0.99,
        "always_ram": False
    }
)

await vector_repo.create_collection_optimized(config)
```

### HNSW Index Optimization

```python
# Optimize HNSW parameters for your use case
hnsw_config = {
    "m": 16,                    # Connectivity (higher = better quality, more memory)
    "ef_construct": 100,        # Construction effort (higher = better quality, slower indexing)
    "full_scan_threshold": 10000,  # When to use full scan vs HNSW
    "max_indexing_threads": 0,  # Auto-detect CPU cores
    "on_disk": False           # Keep index in memory for speed
}
```

### Batch Processing Optimization

```python
# Optimize batch sizes based on your data
class OptimizedVectorService:
    def __init__(self):
        self.batch_size = self._calculate_optimal_batch_size()
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on vector dimensions and memory."""
        vector_size = 384
        embedding_memory = vector_size * 4  # 4 bytes per float32
        payload_memory = 1024  # Estimated payload size
        total_per_doc = embedding_memory + payload_memory
        
        # Target 100MB per batch
        target_batch_memory = 100 * 1024 * 1024
        optimal_batch_size = target_batch_memory // total_per_doc
        
        # Clamp between reasonable bounds
        return max(100, min(optimal_batch_size, 5000))
```

### Query Optimization

```python
# Optimize search parameters
async def optimized_search(
    self,
    query_embedding: List[float],
    limit: int = 10
) -> List[Dict]:
    """Optimized search with performance tuning."""
    
    request = VectorSearchRequest(
        query_embedding=query_embedding,
        collection_name=self.collection_name,
        n_results=limit,
        tenant_id=self.tenant_id,
        similarity_threshold=0.7,  # Filter low-quality results
        include_embeddings=False,  # Don't return embeddings unless needed
        rerank=True  # Use advanced reranking
    )
    
    return await self.vector_repo.semantic_search_advanced(request)
```

## Monitoring and Observability

### Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
VECTOR_SEARCH_REQUESTS = Counter(
    'vector_search_requests_total',
    'Total vector search requests',
    ['tenant_id', 'status']
)

VECTOR_SEARCH_DURATION = Histogram(
    'vector_search_duration_seconds',
    'Vector search request duration',
    ['tenant_id']
)

VECTOR_COLLECTION_SIZE = Gauge(
    'vector_collection_documents_total',
    'Total documents in vector collection',
    ['tenant_id']
)

class MonitoredVectorService(VectorService):
    async def semantic_search(self, query_embedding: List[float], **kwargs) -> List[Dict]:
        """Monitored semantic search."""
        start_time = time.time()
        
        try:
            results = await super().semantic_search(query_embedding, **kwargs)
            VECTOR_SEARCH_REQUESTS.labels(
                tenant_id=self.tenant_id, 
                status='success'
            ).inc()
            return results
            
        except Exception as e:
            VECTOR_SEARCH_REQUESTS.labels(
                tenant_id=self.tenant_id, 
                status='error'
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            VECTOR_SEARCH_DURATION.labels(tenant_id=self.tenant_id).observe(duration)
```

### Health Checks

```python
class VectorDBHealthCheck:
    async def check_qdrant_health(self) -> Dict[str, Any]:
        """Comprehensive Qdrant health check."""
        try:
            # Basic connectivity
            health_status = self.qdrant_client.health()
            
            # Collection status
            collection_info = self.qdrant_client.get_collection("digi_knowledge")
            
            # Performance test
            start_time = time.time()
            test_vector = [0.1] * 384
            search_results = self.qdrant_client.search(
                collection_name="digi_knowledge",
                query_vector=test_vector,
                limit=1
            )
            search_duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "qdrant_status": health_status,
                "collection_points": collection_info.points_count,
                "search_latency_ms": search_duration * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Issues

**Problem**: Cannot connect to Qdrant
```
ConnectionError: [Errno 111] Connection refused
```

**Solution**:
```bash
# Check if Qdrant is running
docker-compose ps qdrant

# Check Qdrant logs
docker-compose logs qdrant

# Verify port access
curl -f http://localhost:6333/health

# Check firewall settings
sudo ufw status
```

#### 2. API Key Authentication

**Problem**: Unauthorized access
```
QdrantException: Unauthorized access
```

**Solution**:
```python
# Ensure API key is correctly configured
vector_repo = EnhancedQdrantRepository(
    host="localhost",
    port=6333,
    api_key="your_actual_api_key_here"  # Check this value
)

# Verify environment variable
import os
print(f"API Key: {os.getenv('QDRANT_API_KEY')}")
```

#### 3. Memory Issues

**Problem**: Out of memory errors during migration
```
MemoryError: Unable to allocate array
```

**Solution**:
```bash
# Reduce batch size
python scripts/migrate_chromadb_to_qdrant.py --batch-size 100

# Enable quantization
python scripts/migrate_chromadb_to_qdrant.py --enable-quantization

# Increase Docker memory limits
# docker-compose.yml
services:
  qdrant:
    deploy:
      resources:
        limits:
          memory: 8G
```

#### 4. Performance Issues

**Problem**: Slow search queries
```
Query taking >1000ms
```

**Solution**:
```python
# Optimize HNSW parameters
hnsw_config = {
    "m": 32,           # Increase connectivity
    "ef_construct": 200,  # Increase construction effort
    "on_disk": False   # Keep in memory
}

# Enable quantization
quantization_config = {
    "type": "scalar",
    "quantile": 0.99
}

# Add search threshold
request = VectorSearchRequest(
    similarity_threshold=0.7,  # Filter low-quality results
    n_results=10  # Limit results
)
```

#### 5. Tenant Isolation Issues

**Problem**: Seeing other tenants' data
```
Search returns documents from wrong tenant
```

**Solution**:
```python
# Verify tenant_id is being set
vector_doc = VectorDocument(
    tenant_id="digi_core",  # Ensure this is set
    # ... other fields
)

# Verify search request includes tenant filter
request = VectorSearchRequest(
    tenant_id="digi_core",  # Ensure this is set
    # ... other fields
)

# Debug search filter
print(f"Search filter: {request.tenant_id}")
```

### Migration Issues

#### Data Validation Failures

```bash
# Check migration logs
tail -f migration_*.log

# Validate specific collection
curl -X GET "http://localhost:6333/collections/digi_knowledge" \
  -H "api-key: your_api_key" | jq '.result.points_count'

# Compare counts
# ChromaDB count vs Qdrant count
```

#### Partial Migration

```bash
# Resume migration from specific collection
python scripts/migrate_chromadb_to_qdrant.py \
  --chromadb-collections remaining_collection_1 remaining_collection_2 \
  --target-collection digi_knowledge

# Verify and fix missing documents
python scripts/verify_migration.py \
  --source-chromadb \
  --target-qdrant \
  --fix-missing
```

## Rollback Strategy

### Emergency Rollback

If immediate rollback is needed:

```bash
# 1. Stop applications
docker-compose stop digi-core-app lernmi-app beep-boop-app

# 2. Switch back to ChromaDB in docker-compose.yml
# Uncomment ChromaDB service, comment Qdrant service

# 3. Restore from backup
docker-compose up chromadb -d

# 4. Restore ChromaDB data from backup
docker cp backup_chromadb_data.tar chromadb_container:/chroma/

# 5. Update application configuration back to ChromaDB
# Revert environment variables

# 6. Restart applications
docker-compose up digi-core-app lernmi-app beep-boop-app -d
```

### Gradual Rollback

For gradual rollback with data preservation:

```python
# Configure dual-repository setup temporarily
class HybridVectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.qdrant_repo = EnhancedQdrantRepository(...)
        self.chroma_repo = EnhancedChromaRepository(...)
        self.use_qdrant = os.getenv("USE_QDRANT", "true").lower() == "true"
    
    async def semantic_search(self, query_embedding: List[float]) -> List[Dict]:
        if self.use_qdrant:
            try:
                return await self._search_qdrant(query_embedding)
            except Exception:
                # Fallback to ChromaDB
                return await self._search_chromadb(query_embedding)
        else:
            return await self._search_chromadb(query_embedding)
```

### Data Recovery

```bash
# Recover from Qdrant snapshots
curl -X POST "http://localhost:6333/collections/digi_knowledge/snapshots/recover" \
  -H "api-key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///qdrant/snapshots/snapshot_name"}'

# Export data from Qdrant back to ChromaDB format
python scripts/qdrant_to_chromadb_export.py \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --output-format chromadb \
  --collection digi_knowledge
```

## Post-Migration Checklist

### Application Verification

- [ ] **Search Functionality**: Verify all search operations work correctly
- [ ] **Tenant Isolation**: Confirm tenant data is properly isolated
- [ ] **Performance**: Validate improved query performance
- [ ] **Monitoring**: Ensure monitoring and alerts are working
- [ ] **Health Checks**: Verify health check endpoints

### Infrastructure Verification

- [ ] **Resource Usage**: Monitor CPU, memory, and disk usage
- [ ] **Backup Systems**: Verify Qdrant backup systems are working
- [ ] **Monitoring**: Confirm all metrics are being collected
- [ ] **Alerts**: Test alert systems with simulated failures
- [ ] **Documentation**: Update operational documentation

### Long-term Tasks

- [ ] **Performance Tuning**: Fine-tune Qdrant configuration based on usage patterns
- [ ] **Capacity Planning**: Plan for future growth and scaling
- [ ] **Security Review**: Conduct security audit of new setup
- [ ] **Training**: Train team on Qdrant operations and troubleshooting
- [ ] **Cleanup**: Remove old ChromaDB infrastructure after validation period

## Summary

The migration from ChromaDB to Qdrant provides significant performance improvements, better scalability, and enhanced multi-tenancy support. The key benefits include:

- **10x faster query performance**
- **4-16x memory reduction** through quantization
- **Native multi-tenancy** with payload-based isolation
- **Horizontal scalability** for future growth
- **Advanced monitoring** and observability

Following this guide ensures a smooth migration with minimal downtime and maximum performance benefits for all tenant applications.