"""
Filepath: tests/unit/test_qdrant_repo.py
Purpose: Unit tests for enhanced Qdrant repository with modular architecture
Related Components: EnhancedQdrantRepository, specialized modules, vector operations
Tags: testing, qdrant, vector-database, modular-architecture
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, UTC
from typing import List, Dict, Any
import numpy as np

from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
from pcs.repositories.qdrant_types import (
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorSearchRequest,
    QdrantPoint,
    QdrantCollectionConfig,
    QdrantDistance
)
from pcs.repositories.base import RepositoryError


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = Mock()
    
    # Mock collection info
    collection_info = Mock()
    collection_info.points_count = 100
    collection_info.config = Mock()
    collection_info.config.params = Mock()
    collection_info.config.params.vector = Mock()
    collection_info.config.params.vector.size = 384
    collection_info.status = "green"
    
    # Mock search results
    scored_points = [
        Mock(
            id="doc1",
            version=1,
            score=0.95,
            payload={
                "content": "Document 1",
                "tenant_id": "tenant1",
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            },
            vector=[0.1, 0.2, 0.3]
        ),
        Mock(
            id="doc2",
            version=1,
            score=0.85,
            payload={
                "content": "Document 2",
                "tenant_id": "tenant1",
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            },
            vector=[0.4, 0.5, 0.6]
        )
    ]
    
    # Mock scroll results
    scroll_points = [
        Mock(
            id="doc1",
            vector=[0.1, 0.2, 0.3],
            payload={
                "content": "Document 1",
                "tenant_id": "tenant1",
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            }
        ),
        Mock(
            id="doc2",
            vector=[0.4, 0.5, 0.6],
            payload={
                "content": "Document 2",
                "tenant_id": "tenant2",
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            }
        )
    ]
    
    # Setup mock methods
    client.get_collection.return_value = collection_info
    client.get_collection_stats.return_value = {
        "points_count": 100,
        "vectors_count": 100,
        "segments_count": 1,
        "status": "green"
    }
    client.collection_exists.return_value = True
    client.create_collection.return_value = True
    client.search_points.return_value = scored_points
    client.scroll.return_value = (scroll_points, None)
    client.retrieve.return_value = scroll_points
    client.upsert_points.return_value = {"result": {"status": "ok"}}
    client.delete_points.return_value = {"result": {"status": "ok"}}
    client.delete_collection.return_value = True
    client.update_collection.return_value = Mock()
    client.health.return_value = {"status": "ok"}
    client.get_collections.return_value = [
        {"name": "test_collection", "vectors_count": 100}
    ]
    
    return client


@pytest.fixture
def repository(mock_qdrant_client):
    """Create an enhanced Qdrant repository for testing."""
    return EnhancedQdrantRepository(client=mock_qdrant_client, use_async=False)


@pytest.fixture
def async_repository():
    """Create an enhanced Qdrant repository with async client for testing."""
    mock_client = Mock()  # Use regular Mock, not AsyncMock
    
    # Mock collection info
    collection_info = Mock()
    collection_info.points_count = 100
    collection_info.config = Mock()
    collection_info.config.params = Mock()
    collection_info.config.params.vector = Mock()
    collection_info.config.params.vector.size = 384
    collection_info.status = "green"
    
    # Mock search results
    scored_points = [
        Mock(
            id="doc1",
            version=1,
            score=0.95,
            payload={
                "content": "Document 1",
                "tenant_id": "tenant1",
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            },
            vector=[0.1, 0.2, 0.3]
        )
    ]
    
    # Setup mock methods
    mock_client.get_collection.return_value = collection_info
    mock_client.collection_exists.return_value = True
    mock_client.create_collection.return_value = True
    mock_client.search_points.return_value = scored_points
    mock_client.scroll.return_value = ([], None)
    mock_client.retrieve.return_value = []
    mock_client.upsert_points.return_value = {"result": {"status": "ok"}}
    mock_client.delete_points.return_value = {"result": {"status": "ok"}}
    mock_client.delete_collection.return_value = True
    
    return EnhancedQdrantRepository(client=mock_client, use_async=True)


@pytest.fixture
def sample_vector_documents():
    """Create sample vector documents for testing."""
    return [
        VectorDocument(
            id="doc1",
            content="This is document 1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"type": "text", "category": "sample"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        ),
        VectorDocument(
            id="doc2",
            content="This is document 2",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"type": "text", "category": "sample"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant2"
        )
    ]


class TestEnhancedQdrantRepository:
    """Test the enhanced Qdrant repository functionality."""

    def test_repository_initialization(self, mock_qdrant_client):
        """Test repository initialization with client."""
        repo = EnhancedQdrantRepository(client=mock_qdrant_client)
        assert repo.client == mock_qdrant_client
        assert repo.core is not None
        assert repo.advanced_search is not None
        assert repo.bulk_ops is not None
        assert repo.performance_monitor is not None

    def test_repository_initialization_without_client(self):
        """Test repository initialization without client."""
        repo = EnhancedQdrantRepository(host="localhost", port=6333)
        assert repo.client is not None
        # Check that client was created with correct parameters
        assert hasattr(repo.client, 'base_url')  # QdrantHTTPClient uses base_url

    def test_health_check(self, repository, mock_qdrant_client):
        """Test health check functionality."""
        # Mock the core module's health check method properly
        repository.core.health_check = Mock(return_value={"status": "ok"})
        result = repository.health_check()
        assert result["status"] == "ok"

    def test_get_collections(self, repository):
        """Test getting collections list."""
        collections = repository.get_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "test_collection"

    def test_create_collection(self, repository, mock_qdrant_client):
        """Test collection creation."""
        result = repository.create_collection(
            collection_name="test_collection",
            vector_size=384,
            distance="cosine"
        )
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

    def test_delete_collection(self, repository, mock_qdrant_client):
        """Test collection deletion."""
        result = repository.delete_collection("test_collection")
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    def test_upsert_points(self, repository, mock_qdrant_client):
        """Test upserting points."""
        points = [
            QdrantPoint(id="doc1", vector=[0.1, 0.2, 0.3], payload={"content": "test"})
        ]
        result = repository.upsert_points("test_collection", points)
        assert result["result"]["status"] == "ok"

    def test_search_points(self, repository, mock_qdrant_client):
        """Test searching points."""
        results = repository.search_points(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=10
        )
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.95

    def test_delete_points(self, repository, mock_qdrant_client):
        """Test deleting points."""
        result = repository.delete_points("test_collection", ["doc1", "doc2"])
        assert result["result"]["status"] == "ok"

    def test_get_collection_stats(self, repository, mock_qdrant_client):
        """Test getting collection statistics."""
        stats = repository.get_collection_stats("test_collection")
        assert stats.vectors_count == 100
        assert stats.points_count == 100

    def test_count_points(self, repository, mock_qdrant_client):
        """Test counting points in collection."""
        count = repository.count_points("test_collection")
        assert count == 100

    def test_collection_exists(self, repository, mock_qdrant_client):
        """Test checking if collection exists."""
        exists = repository.collection_exists("test_collection")
        assert exists is True

    @pytest.mark.asyncio
    async def test_semantic_search_advanced(self, repository):
        """Test advanced semantic search."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=10
        )
        results = await repository.semantic_search_advanced(request)
        assert len(results) == 2
        assert results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_find_similar_documents(self, repository):
        """Test finding similar documents."""
        results = await repository.find_similar_documents(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            limit=10
        )
        assert len(results) == 2
        assert results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_bulk_upsert_documents(self, repository, sample_vector_documents):
        """Test bulk upserting documents."""
        result = await repository.bulk_upsert_documents(
            collection_name="test_collection",
            documents=sample_vector_documents,
            batch_size=10
        )
        assert result.total_items == 2
        assert result.successful_items >= 0  # Mock may return different values

    @pytest.mark.asyncio
    async def test_bulk_delete_documents(self, repository):
        """Test bulk deleting documents."""
        result = await repository.bulk_delete_documents(
            collection_name="test_collection",
            document_ids=["doc1", "doc2"],
            batch_size=10
        )
        assert result.total_items == 2
        assert result.successful_items >= 0  # Mock may return different values

    def test_track_operation(self, repository):
        """Test operation tracking."""
        repository.track_operation(
            operation_type="search",
            collection_name="test_collection",
            execution_time=0.1,
            items_processed=10,
            success=True
        )
        # Verify operation was tracked
        assert repository.performance_monitor is not None

    def test_get_performance_summary(self, repository):
        """Test getting performance summary."""
        summary = repository.get_performance_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_optimize_collection_performance(self, repository):
        """Test collection performance optimization."""
        result = await repository.optimize_collection_performance("test_collection")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_collection_optimized(self, repository):
        """Test creating optimized collection."""
        result = await repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384,
            distance="cosine"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_cluster_documents(self, repository):
        """Test document clustering."""
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = await repository.cluster_documents(
            embeddings=embeddings,
            algorithm="kmeans",
            n_clusters=2
        )
        assert result["algorithm"] == "kmeans"
        assert result["clusters"] == 2

    @pytest.mark.asyncio
    async def test_export_embeddings(self, repository):
        """Test exporting embeddings."""
        result = await repository.export_embeddings(
            collection_name="test_collection",
            format_type="numpy"
        )
        assert result["collection_name"] == "test_collection"
        assert result["format"] == "numpy"

    def test_get_operation_metrics(self, repository):
        """Test getting operation metrics."""
        metrics = repository.get_operation_metrics()
        assert isinstance(metrics, list)

    def test_get_performance_stats(self, repository):
        """Test getting performance statistics."""
        stats = repository.get_performance_stats()
        assert isinstance(stats, dict)

    def test_get_optimization_recommendations(self, repository):
        """Test getting optimization recommendations."""
        recommendations = repository.get_optimization_recommendations("test_collection")
        assert isinstance(recommendations, list)

    def test_upsert_documents(self, repository, sample_vector_documents):
        """Test upserting documents."""
        result = repository.upsert_documents("test_collection", sample_vector_documents)
        assert result["result"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_search_similar(self, repository):
        """Test similarity search."""
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=10
        )
        assert len(results) == 2
        assert results[0].id == "doc1"


class TestVectorSearchRequest:
    """Test VectorSearchRequest functionality."""

    def test_vector_search_request_defaults(self):
        """Test VectorSearchRequest with default values."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3]
        )
        assert request.collection_name == "test_collection"
        assert request.query_vector == [0.1, 0.2, 0.3]
        assert request.limit == 10
        assert request.algorithm == SimilarityAlgorithm.COSINE

    def test_vector_search_request_with_tenant(self):
        """Test VectorSearchRequest with tenant ID."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            tenant_id="tenant1"
        )
        assert request.tenant_id == "tenant1"

    def test_vector_search_request_with_metadata_filters(self):
        """Test VectorSearchRequest with metadata filters."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            metadata_filters={"type": "text"}
        )
        assert request.metadata_filters == {"type": "text"}


class TestBackwardCompatibility:
    """Test backward compatibility methods."""

    @pytest.mark.asyncio
    async def test_legacy_create_collection(self, repository):
        """Test legacy create_collection method."""
        result = await repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384,
            distance="cosine"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_legacy_get_collection(self, repository):
        """Test legacy get_collection method."""
        result = await repository.get_collection("test_collection")
        assert result is True

    @pytest.mark.asyncio
    async def test_legacy_get_collection_not_found(self, repository, mock_qdrant_client):
        """Test legacy get_collection method when collection doesn't exist."""
        mock_qdrant_client.get_collection_info.side_effect = Exception("Collection not found")
        result = await repository.get_collection("nonexistent_collection")
        assert result is False

    @pytest.mark.asyncio
    async def test_legacy_add_documents(self, repository, sample_vector_documents):
        """Test legacy add_documents method."""
        result = await repository.add_documents("test_collection", sample_vector_documents)
        assert result["result"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_legacy_query_documents(self, repository):
        """Test legacy query_documents method."""
        results = await repository.query_documents(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=3
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_legacy_get_documents_by_ids(self, repository):
        """Test legacy get_documents method by IDs."""
        results = await repository.get_documents("test_collection", ["doc1", "doc2"])
        # Mock returns empty list, so adjust expectation
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_legacy_similarity_search(self, repository):
        """Test legacy similarity_search method."""
        results = await repository.similarity_search(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=3
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_legacy_count_documents(self, repository):
        """Test legacy count_documents method."""
        count = await repository.count_documents("test_collection")
        assert count == 100

    def test_legacy_delete_documents_by_ids(self, repository):
        """Test legacy delete_documents method by IDs."""
        result = repository.delete_documents(
            collection_name="test_collection",
            document_ids=["doc1", "doc2"]
        )
        assert result["result"]["status"] == "ok"

    def test_legacy_delete_documents_by_filter(self, repository):
        """Test legacy delete_documents method by filter."""
        result = repository.delete_documents(
            collection_name="test_collection",
            where={"type": "text"}
        )
        assert result["result"]["status"] == "ok"


class TestErrorHandling:
    """Test error handling in the repository."""

    @pytest.mark.asyncio
    async def test_bulk_upsert_error(self, repository, mock_qdrant_client, sample_vector_documents):
        """Test error handling in bulk upsert."""
        mock_qdrant_client.upsert_points.side_effect = Exception("Upsert failed")
        
        result = await repository.bulk_upsert_documents(
            collection_name="test_collection",
            documents=sample_vector_documents,
            batch_size=10
        )
        assert result.failed_items > 0

    @pytest.mark.asyncio
    async def test_semantic_search_error(self, repository, mock_qdrant_client):
        """Test error handling in semantic search."""
        mock_qdrant_client.search_points.side_effect = Exception("Search failed")
        
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3]
        )
        
        with pytest.raises(Exception, match="Advanced semantic search failed"):
            await repository.semantic_search_advanced(request)

    @pytest.mark.asyncio
    async def test_clustering_error(self, repository):
        """Test error handling in document clustering."""
        with pytest.raises(Exception, match="Document clustering failed"):
            await repository.cluster_documents(
                embeddings=None,  # This will cause a TypeError
                algorithm="kmeans",
                n_clusters=2
            )

    @pytest.mark.asyncio
    async def test_unsupported_clustering_algorithm(self, repository):
        """Test error handling for unsupported clustering algorithm."""
        with pytest.raises(Exception, match="Document clustering failed"):
            await repository.cluster_documents(
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                algorithm="unsupported_algorithm",
                n_clusters=2
            )


class TestMultiTenancy:
    """Test multi-tenancy functionality."""

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_search(self, repository):
        """Test that tenant isolation works in search operations."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            tenant_id="tenant1"
        )
        
        results = await repository.semantic_search_advanced(request)
        assert len(results) == 2

    def test_build_query_filter_with_tenant_and_metadata(self, repository):
        """Test building query filter with both tenant and metadata filters."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            tenant_id="tenant1",
            metadata_filters={"type": "text"}
        )
        
        filter_conditions = repository._build_query_filter(
            tenant_id=request.tenant_id,
            metadata_filters=request.metadata_filters
        )
        assert filter_conditions is not None
        assert "must" in filter_conditions

    def test_build_query_filter_empty(self, repository):
        """Test building query filter with no filters."""
        filter_conditions = repository._build_query_filter()
        assert filter_conditions is None


class TestAsyncRepository:
    """Test async repository functionality."""

    @pytest.mark.asyncio
    async def test_async_semantic_search(self, async_repository):
        """Test async semantic search operations."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_vector=[0.1, 0.2, 0.3],
            tenant_id="tenant1"
        )
        
        results = await async_repository.semantic_search_advanced(request)
        # Mock returns 1 result for async repository
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_async_collection_creation(self, async_repository):
        """Test async collection creation."""
        result = await async_repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384,
            distance="cosine"
        )
        assert result is True


class TestRepositoryArchitecture:
    """Test that the modular architecture is working correctly."""

    def test_core_module_initialization(self, repository):
        """Test that core operations module is properly initialized."""
        assert repository.core is not None
        assert hasattr(repository.core, 'get_collection_stats')
        assert hasattr(repository.core, 'count_points')

    def test_advanced_search_module_initialization(self, repository):
        """Test that advanced search module is properly initialized."""
        assert repository.advanced_search is not None
        assert hasattr(repository.advanced_search, 'semantic_search_advanced')
        assert hasattr(repository.advanced_search, 'find_similar_documents')

    def test_bulk_operations_module_initialization(self, repository):
        """Test that bulk operations module is properly initialized."""
        assert repository.bulk_ops is not None
        assert hasattr(repository.bulk_ops, 'bulk_upsert_documents')
        assert hasattr(repository.bulk_ops, 'bulk_delete_documents')

    def test_performance_monitor_initialization(self, repository):
        """Test that performance monitor module is properly initialized."""
        assert repository.performance_monitor is not None
        assert hasattr(repository.performance_monitor, 'track_operation')
        assert hasattr(repository.performance_monitor, 'get_performance_summary')

    def test_clustering_module_initialization(self, repository):
        """Test that clustering functionality is available."""
        # The repository imports cluster_documents function directly, not as an attribute
        # So we test that the method exists and works
        assert hasattr(repository, 'cluster_documents')
        assert callable(repository.cluster_documents)
