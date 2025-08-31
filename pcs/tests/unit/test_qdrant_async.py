"""
Filepath: pcs/tests/unit/test_qdrant_async.py
Purpose: Async functionality tests for Qdrant repository
Related Components: Async operations, async client, async methods
Tags: testing, async, qdrant, repository
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, UTC

from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
from pcs.repositories.qdrant_types import (
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorIndexType,
    VectorCollectionStats,
    BulkVectorOperation,
    VectorSearchRequest
)

# Create alias for backward compatibility in tests
QdrantRepository = EnhancedQdrantRepository
EnhancedQdrantHTTPRepository = EnhancedQdrantRepository


class TestAsyncRepository:
    """Test suite for async repository functionality."""
    
    @pytest.mark.asyncio
    async def test_async_repository_initialization(self, async_repository):
        """Test async repository initialization."""
        assert async_repository.use_async is True
        assert async_repository.client is not None
    
    @pytest.mark.asyncio
    async def test_async_create_collection(self, async_repository):
        """Test async collection creation."""
        result = await async_repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384
        )
        assert result is True
        async_repository.client.create_collection_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_upsert_documents(self, async_repository, sample_vector_documents):
        """Test async document upsert."""
        result = await async_repository.upsert_documents(
            "test_collection",
            sample_vector_documents
        )
        
        assert result is not None
        async_repository.client.upsert_points.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_search_similar(self, async_repository):
        """Test async similarity search."""
        results = await async_repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_async_semantic_search(self, async_repository):
        """Test async semantic search."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        results = await async_repository.semantic_search_advanced(request)
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
    
    @pytest.mark.asyncio
    async def test_async_find_similar_documents(self, async_repository):
        """Test async find similar documents."""
        results = await async_repository.find_similar_documents(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_async_get_collection_statistics(self, async_repository):
        """Test async collection statistics."""
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            stats = await async_repository.get_collection_statistics("test_collection")
        
        assert isinstance(stats, VectorCollectionStats)
        assert stats.points_count == 100
    
    @pytest.mark.asyncio
    async def test_async_export_embeddings(self, async_repository):
        """Test async embedding export."""
        # Mock scroll results
        async_repository.client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await async_repository.export_embeddings("test_collection")
        
        assert result["collection_name"] == "test_collection"
        assert len(result["embeddings"]) == 1
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]


class TestAsyncClientOperations:
    """Test suite for async client operations."""
    
    @pytest.mark.asyncio
    async def test_async_client_search_points(self, async_repository):
        """Test async client search_points method."""
        # Configure mock to return search results
        async_repository.client.search_points_async.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        results = await async_repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
    
    @pytest.mark.asyncio
    async def test_async_client_upsert_points(self, async_repository, sample_vector_documents):
        """Test async client upsert_points method."""
        result = await async_repository.upsert_documents(
            "test_collection",
            sample_vector_documents
        )
        
        assert result is not None
        async_repository.client.upsert_points.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_client_create_collection(self, async_repository):
        """Test async client create_collection method."""
        result = await async_repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384
        )
        
        assert result is True
        async_repository.client.create_collection_async.assert_called_once()


class TestAsyncErrorHandling:
    """Test suite for async error handling."""
    
    @pytest.mark.asyncio
    async def test_async_search_error_handling(self, async_repository):
        """Test async search error handling."""
        async_repository.client.search_points_async.side_effect = Exception("Search error")
        
        with pytest.raises(Exception, match="Search error"):
            await async_repository.search_similar(
                collection_name="test_collection",
                query_embedding=[0.1, 0.2, 0.3]
            )
    
    @pytest.mark.asyncio
    async def test_async_upsert_error_handling(self, async_repository, sample_vector_documents):
        """Test async upsert error handling."""
        async_repository.client.upsert_points.side_effect = Exception("Upsert error")
        
        with pytest.raises(Exception, match="Upsert error"):
            await async_repository.upsert_documents(
                "test_collection",
                sample_vector_documents
            )
    
    @pytest.mark.asyncio
    async def test_async_create_collection_error_handling(self, async_repository):
        """Test async create collection error handling."""
        async_repository.client.create_collection_async.side_effect = Exception("Create error")
        
        with pytest.raises(Exception, match="Create error"):
            await async_repository.create_collection_optimized(
                collection_name="test_collection",
                vector_size=384
            )


class TestAsyncPerformanceMonitoring:
    """Test suite for async performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_async_performance_monitoring(self, async_repository, sample_vector_documents):
        """Test async performance monitoring."""
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor') as mock_monitor:
            await async_repository.upsert_documents(
                "test_collection",
                sample_vector_documents
            )
            
            # Verify performance monitoring was used
            mock_monitor.assert_called()
    
    @pytest.mark.asyncio
    async def test_async_query_performance_tracking(self, async_repository):
        """Test async query performance tracking."""
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            await async_repository.search_similar(
                collection_name="test_collection",
                query_embedding=[0.1, 0.2, 0.3]
            )
            
            # Performance metrics should be updated
            assert hasattr(async_repository, 'performance_metrics')


class TestAsyncMultiTenancy:
    """Test suite for async multi-tenancy functionality."""
    
    @pytest.mark.asyncio
    async def test_async_tenant_filtering(self, async_repository):
        """Test async tenant filtering in search."""
        # Configure mock to return search results
        async_repository.client.search_points_async.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        results = await async_repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_async_tenant_statistics(self, async_repository):
        """Test async tenant-specific statistics."""
        # Mock scroll for tenant counting
        async_repository.client.scroll.return_value = ([
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant2"})
        ], None)
        
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            stats = await async_repository.get_collection_statistics(
                "test_collection",
                tenant_id="tenant1"
            )
        
        assert stats.points_count == 2  # Only tenant1 documents


class TestAsyncBulkOperations:
    """Test suite for async bulk operations."""
    
    @pytest.mark.asyncio
    async def test_async_bulk_upsert(self, async_repository, sample_vector_documents):
        """Test async bulk upsert operations."""
        operation = BulkVectorOperation(
            operation_type="insert",
            collection_name="test_collection",
            documents=sample_vector_documents,
            batch_size=2,
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            result = await async_repository.bulk_upsert_documents("test_collection", operation)
        
        assert result["total_processed"] == 3
        assert result["batch_count"] == 2
        assert result["execution_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_async_bulk_upsert_error_handling(self, async_repository, sample_vector_documents):
        """Test async bulk upsert error handling."""
        operation = BulkVectorOperation(
            operation_type="insert",
            collection_name="test_collection",
            documents=sample_vector_documents,
            batch_size=2,
            tenant_id="tenant1"
        )
        
        # Mock error on upsert
        async_repository.client.upsert_points.side_effect = Exception("Bulk upsert error")
        
        with pytest.raises(Exception, match="Bulk upsert error"):
            await async_repository.bulk_upsert_documents("test_collection", operation)


class TestAsyncMethodCompatibility:
    """Test suite for async method compatibility."""
    
    def test_async_method_availability(self, async_repository):
        """Test that all async methods are available."""
        assert hasattr(async_repository, 'search_similar')
        assert hasattr(async_repository, 'upsert_documents')
        assert hasattr(async_repository, 'create_collection_optimized')
        assert hasattr(async_repository, 'get_collection_statistics')
        assert hasattr(async_repository, 'export_embeddings')
        assert hasattr(async_repository, 'bulk_upsert_documents')
    
    def test_async_method_callable(self, async_repository):
        """Test that all async methods are callable."""
        assert callable(async_repository.search_similar)
        assert callable(async_repository.upsert_documents)
        assert callable(async_repository.create_collection_optimized)
        assert callable(async_repository.get_collection_statistics)
        assert callable(async_repository.export_embeddings)
        assert callable(async_repository.bulk_upsert_documents)
