# Tenant Application Qdrant Integration Guide

## Overview

This guide provides comprehensive instructions for tenant applications (digi-core, lernmi, beep-boop) to integrate with the new Qdrant vector database service, including multi-tenancy implementation, performance optimization, and best practices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Interface](#repository-interface)
3. [Multi-Tenant Implementation](#multi-tenant-implementation)
4. [Code Examples](#code-examples)
5. [Performance Best Practices](#performance-best-practices)
6. [Error Handling](#error-handling)
7. [Testing Integration](#testing-integration)
8. [Monitoring and Observability](#monitoring-and-observability)

## Quick Start

### 1. Update Dependencies

```toml
# pyproject.toml
[tool.uv.dependencies]
# Remove ChromaDB
# chromadb = "0.4.15"

# Add Qdrant
qdrant-client = "1.7.0"
numpy = ">=1.21.0"
```

### 2. Environment Configuration

```bash
# .env
# Vector Database Configuration
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_secure_api_key
QDRANT_USE_HTTPS=false
QDRANT_GRPC_PORT=6334

# Collection Configuration
VECTOR_COLLECTION_NAME=digi_knowledge
VECTOR_DIMENSION=384
VECTOR_DISTANCE_METRIC=cosine

# Tenant Configuration
TENANT_ID=digi_core  # or lernmi, beep_boop
```

### 3. Basic Integration

```python
# services/vector_service.py
from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository,
    VectorSearchRequest,
    VectorDocument
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
```

## Repository Interface

### Complete API Reference

The `EnhancedQdrantRepository` provides a comprehensive interface that maintains backward compatibility with ChromaDB while adding advanced Qdrant features.

#### Core Methods

```python
# Collection Management
await vector_repo.create_collection(name, metadata, vector_size, distance)
await vector_repo.get_collection(name)
await vector_repo.get_or_create_collection(name, metadata, vector_size, distance)
await vector_repo.delete_collection(name)

# Document Operations
await vector_repo.add_documents(collection_name, documents, ids, metadatas, embeddings, tenant_id)
await vector_repo.update_documents(collection_name, ids, documents, metadatas, embeddings, tenant_id)
await vector_repo.delete_documents(collection_name, ids, where, where_document, tenant_id)
await vector_repo.get_documents(collection_name, ids, where, limit, offset, tenant_id)
await vector_repo.count_documents(collection_name, tenant_id)

# Search Operations
await vector_repo.similarity_search(collection_name, query_embedding, n_results, threshold, metadata_filter, tenant_id)
await vector_repo.query_documents(collection_name, query_embeddings, n_results, where, tenant_id)

# Advanced Operations
await vector_repo.semantic_search_advanced(request)
await vector_repo.find_similar_documents(collection_name, target_embedding, similarity_threshold, max_results, metadata_filter, tenant_id)
await vector_repo.cluster_documents(collection_name, n_clusters, algorithm, metadata_filter, tenant_id)
await vector_repo.bulk_upsert_documents(collection_name, operation)
```

#### Advanced Qdrant Features

```python
# Collection Optimization
await vector_repo.create_collection_optimized(config, metadata)
await vector_repo.optimize_collection_performance(collection_name)
await vector_repo.get_collection_statistics(collection_name, tenant_id)

# Data Export/Import
await vector_repo.export_embeddings(collection_name, format_type, include_metadata, tenant_id)
```

## Multi-Tenant Implementation

### Automatic Tenant Isolation

Qdrant uses payload-based multi-tenancy, which provides better performance than separate collections:

```python
class TenantAwareVectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
    
    async def add_document(self, doc_id: str, content: str, embedding: List[float], metadata: Dict = None) -> bool:
        """Add document with automatic tenant isolation."""
        return await self.vector_repo.add_documents(
            collection_name=settings.VECTOR_COLLECTION_NAME,
            documents=[content],
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata or {}],
            tenant_id=self.tenant_id  # Automatic tenant filtering
        )
    
    async def search(self, query_embedding: List[float], limit: int = 10, filters: Dict = None) -> List[Dict]:
        """Search with automatic tenant isolation."""
        return await self.vector_repo.similarity_search(
            collection_name=settings.VECTOR_COLLECTION_NAME,
            query_embedding=query_embedding,
            n_results=limit,
            metadata_filter=filters,
            tenant_id=self.tenant_id  # Automatic tenant filtering
        )
```

### Tenant Security Layer

```python
from functools import wraps
from typing import Callable, Any

def tenant_secured(func: Callable) -> Callable:
    """Decorator to ensure tenant isolation in vector operations."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Ensure tenant_id is always set
        if not hasattr(self, 'tenant_id') or not self.tenant_id:
            raise ValueError("Tenant ID must be set for secure operations")
        
        # Add tenant_id to kwargs if not present
        if 'tenant_id' not in kwargs:
            kwargs['tenant_id'] = self.tenant_id
        
        return await func(self, *args, **kwargs)
    return wrapper

class SecureVectorService(TenantAwareVectorService):
    @tenant_secured
    async def search(self, query_embedding: List[float], **kwargs) -> List[Dict]:
        return await super().search(query_embedding, **kwargs)
    
    @tenant_secured
    async def add_document(self, doc_id: str, content: str, embedding: List[float], **kwargs) -> bool:
        return await super().add_document(doc_id, content, embedding, **kwargs)
```

## Code Examples

### 1. Knowledge Base Integration

```python
# knowledge_service.py
from typing import List, Dict, Optional
from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository,
    VectorSearchRequest,
    VectorDocument,
    BulkVectorOperation
)

class KnowledgeService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.VECTOR_COLLECTION_NAME
    
    async def add_knowledge_article(
        self,
        article_id: str,
        title: str,
        content: str,
        embedding: List[float],
        category: str,
        tags: List[str] = None
    ) -> bool:
        """Add knowledge article with rich metadata."""
        metadata = {
            "title": title,
            "category": category,
            "tags": tags or [],
            "type": "knowledge_article",
            "word_count": len(content.split()),
            "created_by": self.tenant_id
        }
        
        vector_doc = VectorDocument(
            id=article_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            created_at=datetime.utcnow(),
            collection_name=self.collection_name,
            tenant_id=self.tenant_id
        )
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=[vector_doc],
            tenant_id=self.tenant_id
        )
        
        result = await self.vector_repo.bulk_upsert_documents(
            self.collection_name,
            operation
        )
        
        return result["total_processed"] > 0
    
    async def search_knowledge(
        self,
        query_embedding: List[float],
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search knowledge base with advanced filtering."""
        filters = {"type": "knowledge_article"}
        
        if category:
            filters["category"] = category
        
        if tags:
            filters["tags"] = {"any": tags}  # Match any of the provided tags
        
        request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=self.collection_name,
            n_results=limit,
            similarity_threshold=min_similarity,
            metadata_filter=filters,
            tenant_id=self.tenant_id,
            include_embeddings=False,
            rerank=True
        )
        
        results = await self.vector_repo.semantic_search_advanced(request)
        
        return [
            {
                "id": result.document.id,
                "title": result.document.metadata.get("title", ""),
                "content": result.document.content,
                "category": result.document.metadata.get("category", ""),
                "tags": result.document.metadata.get("tags", []),
                "similarity_score": result.similarity_score,
                "word_count": result.document.metadata.get("word_count", 0)
            }
            for result in results
        ]
    
    async def get_related_articles(
        self,
        article_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find articles related to a specific article."""
        # Get the target article's embedding
        documents = await self.vector_repo.get_documents(
            collection_name=self.collection_name,
            ids=[article_id],
            tenant_id=self.tenant_id
        )
        
        if not documents["embeddings"]:
            return []
        
        target_embedding = documents["embeddings"][0]
        
        # Find similar articles
        similar_results = await self.vector_repo.find_similar_documents(
            collection_name=self.collection_name,
            target_embedding=target_embedding,
            similarity_threshold=0.6,
            max_results=limit + 1,  # +1 to exclude the original article
            metadata_filter={"type": "knowledge_article"},
            tenant_id=self.tenant_id
        )
        
        # Filter out the original article
        related_articles = [
            {
                "id": result.document.id,
                "title": result.document.metadata.get("title", ""),
                "similarity_score": result.similarity_score
            }
            for result in similar_results
            if result.document.id != article_id
        ]
        
        return related_articles[:limit]
```

### 2. Chat/RAG Integration

```python
# chat_service.py
class ChatRAGService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.knowledge_service = KnowledgeService(tenant_id)
    
    async def get_context_for_query(
        self,
        query_embedding: List[float],
        max_context_length: int = 4000,
        min_similarity: float = 0.75
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for RAG pipeline."""
        relevant_docs = await self.knowledge_service.search_knowledge(
            query_embedding=query_embedding,
            limit=20,  # Get more candidates
            min_similarity=min_similarity
        )
        
        # Select context within token limit
        selected_context = []
        current_length = 0
        
        for doc in relevant_docs:
            doc_length = len(doc["content"])
            if current_length + doc_length <= max_context_length:
                selected_context.append(doc)
                current_length += doc_length
            else:
                break
        
        return selected_context
    
    async def add_conversation_context(
        self,
        conversation_id: str,
        message: str,
        embedding: List[float],
        role: str = "user"
    ) -> bool:
        """Add conversation messages as searchable context."""
        metadata = {
            "type": "conversation",
            "conversation_id": conversation_id,
            "role": role,
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": self.tenant_id
        }
        
        vector_doc = VectorDocument(
            id=f"conv_{conversation_id}_{uuid.uuid4()}",
            content=message,
            embedding=embedding,
            metadata=metadata,
            created_at=datetime.utcnow(),
            collection_name=settings.VECTOR_COLLECTION_NAME,
            tenant_id=self.tenant_id
        )
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=[vector_doc],
            tenant_id=self.tenant_id
        )
        
        result = await self.vector_repo.bulk_upsert_documents(
            settings.VECTOR_COLLECTION_NAME,
            operation
        )
        
        return result["total_processed"] > 0
```

### 3. Document Management Integration

```python
# document_service.py
class DocumentService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
    
    async def index_document(
        self,
        document_id: str,
        document_chunks: List[Dict[str, Any]]  # {"text": str, "embedding": List[float], "metadata": Dict}
    ) -> Dict[str, Any]:
        """Index a document as multiple chunks."""
        vector_docs = []
        
        for i, chunk in enumerate(document_chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            
            metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "type": "document_chunk",
                **chunk.get("metadata", {})
            }
            
            vector_doc = VectorDocument(
                id=chunk_id,
                content=chunk["text"],
                embedding=chunk["embedding"],
                metadata=metadata,
                created_at=datetime.utcnow(),
                collection_name=settings.VECTOR_COLLECTION_NAME,
                tenant_id=self.tenant_id
            )
            vector_docs.append(vector_doc)
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=vector_docs,
            batch_size=100,
            tenant_id=self.tenant_id
        )
        
        result = await self.vector_repo.bulk_upsert_documents(
            settings.VECTOR_COLLECTION_NAME,
            operation
        )
        
        return {
            "document_id": document_id,
            "chunks_indexed": len(vector_docs),
            "chunks_processed": result["total_processed"],
            "processing_time": result["execution_time_seconds"]
        }
    
    async def search_documents(
        self,
        query_embedding: List[float],
        document_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search across document chunks."""
        filters = {"type": "document_chunk"}
        
        if document_types:
            filters["document_type"] = {"any": document_types}
        
        request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            n_results=limit,
            metadata_filter=filters,
            tenant_id=self.tenant_id,
            similarity_threshold=0.7,
            rerank=True
        )
        
        results = await self.vector_repo.semantic_search_advanced(request)
        
        # Group results by document
        documents = {}
        for result in results:
            doc_id = result.document.metadata["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "chunks": [],
                    "max_similarity": 0
                }
            
            documents[doc_id]["chunks"].append({
                "chunk_id": result.document.id,
                "content": result.document.content,
                "similarity_score": result.similarity_score,
                "chunk_index": result.document.metadata.get("chunk_index", 0)
            })
            
            documents[doc_id]["max_similarity"] = max(
                documents[doc_id]["max_similarity"],
                result.similarity_score
            )
        
        # Sort by best similarity and return top documents
        sorted_documents = sorted(
            documents.values(),
            key=lambda x: x["max_similarity"],
            reverse=True
        )
        
        return sorted_documents
```

## Performance Best Practices

### 1. Batch Operations

```python
class PerformanceOptimizedVectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.batch_size = self._calculate_optimal_batch_size()
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on vector dimensions."""
        vector_size = settings.VECTOR_DIMENSION
        # Estimate memory per document (embedding + payload)
        memory_per_doc = (vector_size * 4) + 1024  # 4 bytes per float + 1KB payload
        # Target 50MB per batch
        target_memory = 50 * 1024 * 1024
        optimal_size = target_memory // memory_per_doc
        return max(100, min(optimal_size, 2000))
    
    async def bulk_add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple documents efficiently."""
        vector_docs = []
        
        for doc in documents:
            vector_doc = VectorDocument(
                id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc.get("metadata", {}),
                created_at=datetime.utcnow(),
                collection_name=settings.VECTOR_COLLECTION_NAME,
                tenant_id=self.tenant_id
            )
            vector_docs.append(vector_doc)
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=vector_docs,
            batch_size=self.batch_size,
            tenant_id=self.tenant_id
        )
        
        return await self.vector_repo.bulk_upsert_documents(
            settings.VECTOR_COLLECTION_NAME,
            operation
        )
```

### 2. Search Optimization

```python
class OptimizedSearchService:
    async def efficient_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """Optimized search with best practices."""
        request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            n_results=limit,
            similarity_threshold=0.7,  # Filter low-quality results early
            metadata_filter=filters,
            tenant_id=self.tenant_id,
            include_embeddings=False,  # Don't return embeddings unless needed
            rerank=True  # Use advanced reranking for better results
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
    
    async def search_with_pagination(
        self,
        query_embedding: List[float],
        page: int = 1,
        page_size: int = 20,
        filters: Dict = None
    ) -> Dict[str, Any]:
        """Search with pagination support."""
        offset = (page - 1) * page_size
        
        request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            n_results=page_size,
            offset=offset,
            metadata_filter=filters,
            tenant_id=self.tenant_id,
            similarity_threshold=0.5
        )
        
        results = await self.vector_repo.semantic_search_advanced(request)
        
        return {
            "results": [result.to_dict() for result in results],
            "page": page,
            "page_size": page_size,
            "total_results": len(results),
            "has_more": len(results) == page_size
        }
```

### 3. Caching Layer

```python
import redis
import json
import hashlib
from typing import Optional

class CachedVectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        self.cache_ttl = 3600  # 1 hour
    
    def _get_cache_key(self, query_embedding: List[float], filters: Dict = None) -> str:
        """Generate cache key for search query."""
        cache_data = {
            "tenant_id": self.tenant_id,
            "embedding": query_embedding[:10],  # Use first 10 dimensions for key
            "filters": filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"vector_search:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def cached_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Dict = None
    ) -> List[Dict]:
        """Search with Redis caching."""
        cache_key = self._get_cache_key(query_embedding, filters)
        
        # Try to get from cache
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Perform search
        results = await self.efficient_search(query_embedding, limit, filters)
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(results)
        )
        
        return results
```

## Error Handling

### Comprehensive Error Handling

```python
import logging
from typing import Optional
from enum import Enum

class VectorOperationError(Exception):
    """Base exception for vector operations."""
    pass

class TenantIsolationError(VectorOperationError):
    """Exception for tenant isolation violations."""
    pass

class VectorSearchError(VectorOperationError):
    """Exception for search operation failures."""
    pass

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RobustVectorService:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.logger = logging.getLogger(f"vector_service.{tenant_id}")
        self.vector_repo = EnhancedQdrantRepository(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY
        )
    
    async def safe_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Dict = None,
        max_retries: int = 3
    ) -> Optional[List[Dict]]:
        """Search with comprehensive error handling and retries."""
        for attempt in range(max_retries):
            try:
                # Validate inputs
                if not query_embedding:
                    raise VectorSearchError("Query embedding cannot be empty")
                
                if len(query_embedding) != settings.VECTOR_DIMENSION:
                    raise VectorSearchError(
                        f"Query embedding dimension {len(query_embedding)} "
                        f"doesn't match expected {settings.VECTOR_DIMENSION}"
                    )
                
                # Validate tenant isolation
                if not self.tenant_id:
                    raise TenantIsolationError("Tenant ID is required for secure operations")
                
                # Perform search
                results = await self.vector_repo.similarity_search(
                    collection_name=settings.VECTOR_COLLECTION_NAME,
                    query_embedding=query_embedding,
                    n_results=limit,
                    metadata_filter=filters,
                    tenant_id=self.tenant_id
                )
                
                self.logger.info(
                    f"Search successful: {len(results)} results for tenant {self.tenant_id}"
                )
                
                return results
                
            except TenantIsolationError as e:
                # Critical error - don't retry
                self.logger.error(f"Tenant isolation error: {str(e)}")
                self._record_error(str(e), ErrorSeverity.CRITICAL)
                raise
                
            except VectorSearchError as e:
                # Input validation error - don't retry
                self.logger.error(f"Vector search error: {str(e)}")
                self._record_error(str(e), ErrorSeverity.MEDIUM)
                return None
                
            except Exception as e:
                # Network/service error - retry
                self.logger.warning(
                    f"Search attempt {attempt + 1} failed: {str(e)}"
                )
                
                if attempt == max_retries - 1:
                    self.logger.error(f"Search failed after {max_retries} attempts")
                    self._record_error(str(e), ErrorSeverity.HIGH)
                    return None
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return None
    
    def _record_error(self, error_message: str, severity: ErrorSeverity):
        """Record error for monitoring and alerting."""
        error_data = {
            "tenant_id": self.tenant_id,
            "error_message": error_message,
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "vector_service"
        }
        
        # Log to structured logger
        self.logger.error("Vector service error", extra=error_data)
        
        # Send to monitoring system (implement based on your monitoring setup)
        # self.monitoring_client.record_error(error_data)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for vector service."""
        try:
            # Test basic connectivity
            test_embedding = [0.1] * settings.VECTOR_DIMENSION
            
            start_time = time.time()
            results = await self.safe_search(
                query_embedding=test_embedding,
                limit=1
            )
            search_time = time.time() - start_time
            
            return {
                "status": "healthy" if results is not None else "degraded",
                "tenant_id": self.tenant_id,
                "search_latency_ms": search_time * 1000,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "tenant_id": self.tenant_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Testing Integration

### Unit Tests

```python
# test_vector_service.py
import pytest
from unittest.mock import AsyncMock, Mock
from your_app.services.vector_service import VectorService

@pytest.fixture
def mock_vector_repo():
    repo = AsyncMock()
    repo.similarity_search.return_value = [
        {
            "id": "test_doc_1",
            "document": "Test document content",
            "similarity": 0.85,
            "metadata": {"type": "test"}
        }
    ]
    return repo

@pytest.fixture
def vector_service(mock_vector_repo):
    service = VectorService("test_tenant")
    service.vector_repo = mock_vector_repo
    return service

@pytest.mark.asyncio
async def test_search_with_tenant_isolation(vector_service, mock_vector_repo):
    """Test that search operations include tenant isolation."""
    query_embedding = [0.1] * 384
    
    results = await vector_service.search(query_embedding, limit=10)
    
    # Verify tenant_id was passed to repository
    mock_vector_repo.similarity_search.assert_called_once()
    call_args = mock_vector_repo.similarity_search.call_args
    assert call_args.kwargs['tenant_id'] == "test_tenant"
    
    # Verify results
    assert len(results) == 1
    assert results[0]["id"] == "test_doc_1"

@pytest.mark.asyncio
async def test_add_document_with_metadata(vector_service, mock_vector_repo):
    """Test adding document with proper metadata."""
    mock_vector_repo.add_documents.return_value = True
    
    result = await vector_service.add_document(
        doc_id="test_doc",
        content="Test content",
        embedding=[0.1] * 384,
        metadata={"category": "test"}
    )
    
    assert result is True
    mock_vector_repo.add_documents.assert_called_once()
    call_args = mock_vector_repo.add_documents.call_args
    assert call_args.kwargs['tenant_id'] == "test_tenant"
```

### Integration Tests

```python
# test_integration.py
import pytest
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository

@pytest.mark.integration
class TestQdrantIntegration:
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test environment."""
        self.tenant_id = "test_tenant"
        self.vector_repo = EnhancedQdrantRepository(
            host="localhost",
            port=6333,
            api_key=None  # No API key for testing
        )
        self.collection_name = "test_collection"
        
        # Create test collection
        await self.vector_repo.create_collection(
            name=self.collection_name,
            vector_size=384,
            distance="cosine"
        )
        
        yield
        
        # Cleanup
        await self.vector_repo.delete_collection(self.collection_name)
    
    @pytest.mark.asyncio
    async def test_full_document_lifecycle(self):
        """Test complete document lifecycle."""
        # Add document
        doc_id = "test_doc_1"
        content = "This is a test document for integration testing."
        embedding = [0.1] * 384
        metadata = {"type": "test", "category": "integration"}
        
        success = await self.vector_repo.add_documents(
            collection_name=self.collection_name,
            documents=[content],
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            tenant_id=self.tenant_id
        )
        assert success
        
        # Search for document
        search_results = await self.vector_repo.similarity_search(
            collection_name=self.collection_name,
            query_embedding=embedding,
            n_results=5,
            tenant_id=self.tenant_id
        )
        
        assert len(search_results) >= 1
        assert search_results[0]["id"] == doc_id
        assert search_results[0]["similarity"] > 0.9  # Should be very similar
        
        # Update document
        new_content = "This is an updated test document."
        new_metadata = {"type": "test", "category": "updated"}
        
        update_success = await self.vector_repo.update_documents(
            collection_name=self.collection_name,
            ids=[doc_id],
            documents=[new_content],
            metadatas=[new_metadata],
            tenant_id=self.tenant_id
        )
        assert update_success
        
        # Verify update
        documents = await self.vector_repo.get_documents(
            collection_name=self.collection_name,
            ids=[doc_id],
            tenant_id=self.tenant_id
        )
        
        assert documents["documents"][0] == new_content
        assert documents["metadatas"][0]["category"] == "updated"
        
        # Delete document
        delete_success = await self.vector_repo.delete_documents(
            collection_name=self.collection_name,
            ids=[doc_id],
            tenant_id=self.tenant_id
        )
        assert delete_success
        
        # Verify deletion
        final_documents = await self.vector_repo.get_documents(
            collection_name=self.collection_name,
            ids=[doc_id],
            tenant_id=self.tenant_id
        )
        
        assert len(final_documents["ids"]) == 0
    
    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        """Test that tenant isolation works correctly."""
        # Add documents for different tenants
        tenant1_doc = {
            "id": "tenant1_doc",
            "content": "Document for tenant 1",
            "embedding": [0.1] * 384,
            "metadata": {"owner": "tenant1"}
        }
        
        tenant2_doc = {
            "id": "tenant2_doc", 
            "content": "Document for tenant 2",
            "embedding": [0.2] * 384,
            "metadata": {"owner": "tenant2"}
        }
        
        # Add documents for both tenants
        await self.vector_repo.add_documents(
            collection_name=self.collection_name,
            documents=[tenant1_doc["content"]],
            ids=[tenant1_doc["id"]],
            embeddings=[tenant1_doc["embedding"]],
            metadatas=[tenant1_doc["metadata"]],
            tenant_id="tenant1"
        )
        
        await self.vector_repo.add_documents(
            collection_name=self.collection_name,
            documents=[tenant2_doc["content"]],
            ids=[tenant2_doc["id"]],
            embeddings=[tenant2_doc["embedding"]],
            metadatas=[tenant2_doc["metadata"]],
            tenant_id="tenant2"
        )
        
        # Search as tenant1 - should only see tenant1 docs
        tenant1_results = await self.vector_repo.similarity_search(
            collection_name=self.collection_name,
            query_embedding=[0.1] * 384,
            n_results=10,
            tenant_id="tenant1"
        )
        
        tenant1_ids = [result["id"] for result in tenant1_results]
        assert "tenant1_doc" in tenant1_ids
        assert "tenant2_doc" not in tenant1_ids
        
        # Search as tenant2 - should only see tenant2 docs
        tenant2_results = await self.vector_repo.similarity_search(
            collection_name=self.collection_name,
            query_embedding=[0.2] * 384,
            n_results=10,
            tenant_id="tenant2"
        )
        
        tenant2_ids = [result["id"] for result in tenant2_results]
        assert "tenant2_doc" in tenant2_ids
        assert "tenant1_doc" not in tenant2_ids
```

## Monitoring and Observability

### Application Metrics

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Define metrics
VECTOR_OPERATIONS = Counter(
    'vector_operations_total',
    'Total vector database operations',
    ['tenant_id', 'operation', 'status']
)

VECTOR_OPERATION_DURATION = Histogram(
    'vector_operation_duration_seconds',
    'Duration of vector database operations',
    ['tenant_id', 'operation']
)

VECTOR_COLLECTION_SIZE = Gauge(
    'vector_collection_documents',
    'Number of documents in vector collection',
    ['tenant_id']
)

VECTOR_SEARCH_RESULTS = Histogram(
    'vector_search_results_count',
    'Number of results returned by vector search',
    ['tenant_id']
)

VECTOR_SERVICE_INFO = Info(
    'vector_service_info',
    'Information about vector service configuration'
)

class MonitoredVectorService(VectorService):
    def __init__(self, tenant_id: str):
        super().__init__(tenant_id)
        
        # Set service info
        VECTOR_SERVICE_INFO.info({
            'version': '1.0.0',
            'vector_db': 'qdrant',
            'tenant_id': tenant_id
        })
    
    async def search(self, query_embedding: List[float], **kwargs) -> List[Dict]:
        """Monitored search operation."""
        start_time = time.time()
        
        try:
            results = await super().search(query_embedding, **kwargs)
            
            # Record success metrics
            VECTOR_OPERATIONS.labels(
                tenant_id=self.tenant_id,
                operation='search',
                status='success'
            ).inc()
            
            VECTOR_SEARCH_RESULTS.labels(
                tenant_id=self.tenant_id
            ).observe(len(results))
            
            return results
            
        except Exception as e:
            # Record error metrics
            VECTOR_OPERATIONS.labels(
                tenant_id=self.tenant_id,
                operation='search', 
                status='error'
            ).inc()
            raise
            
        finally:
            # Record duration
            duration = time.time() - start_time
            VECTOR_OPERATION_DURATION.labels(
                tenant_id=self.tenant_id,
                operation='search'
            ).observe(duration)
    
    async def update_collection_size_metric(self):
        """Update collection size metric."""
        try:
            count = await self.vector_repo.count_documents(
                collection_name=settings.VECTOR_COLLECTION_NAME,
                tenant_id=self.tenant_id
            )
            
            VECTOR_COLLECTION_SIZE.labels(
                tenant_id=self.tenant_id
            ).set(count)
            
        except Exception as e:
            self.logger.error(f"Failed to update collection size metric: {e}")
```

### Health Checks

```python
# health_checks.py
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/health/vector-db")
async def vector_db_health() -> Dict[str, Any]:
    """Health check endpoint for vector database."""
    health_results = {}
    
    # Check each tenant's vector service
    for tenant_id in ["digi_core", "lernmi", "beep_boop"]:
        try:
            vector_service = MonitoredVectorService(tenant_id)
            tenant_health = await vector_service.health_check()
            health_results[tenant_id] = tenant_health
            
        except Exception as e:
            health_results[tenant_id] = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Overall health status
    all_healthy = all(
        result["status"] == "healthy" 
        for result in health_results.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "vector_db": "qdrant",
        "tenants": health_results,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/vector-db/{tenant_id}")
async def tenant_vector_db_health(tenant_id: str) -> Dict[str, Any]:
    """Health check for specific tenant."""
    vector_service = MonitoredVectorService(tenant_id)
    return await vector_service.health_check()
```

This comprehensive integration guide provides tenant applications with all the necessary tools and patterns to successfully integrate with the new Qdrant vector database while maintaining security, performance, and observability.
