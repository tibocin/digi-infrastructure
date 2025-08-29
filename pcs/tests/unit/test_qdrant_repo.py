"""
Filepath: tests/unit/test_qdrant_repo.py
Purpose: Unit tests for enhanced Qdrant repository implementation with multi-tenancy
Related Components: EnhancedQdrantRepository, VectorDocument, SimilarityResult, BulkVectorOperation, multi-tenancy
Tags: testing, qdrant, vector-database, semantic-search, similarity-algorithms, multi-tenant
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, UTC, timedelta
from uuid import uuid4
from typing import List, Dict, Any
import numpy as np

# Use our custom types instead of external qdrant_client
from pcs.repositories.qdrant_http_client import (
    QdrantDistance, QdrantCollectionConfig, QdrantPoint, QdrantSearchResult
)

# Import missing types
from qdrant_client.models import Distance, PointStruct

from pcs.repositories.qdrant_http_repo import (
    EnhancedQdrantHTTPRepository,
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorIndexType,
    VectorCollectionStats,
    BulkVectorOperation,
    VectorSearchRequest
)

# Create alias for backward compatibility in tests
QdrantRepository = EnhancedQdrantHTTPRepository
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
    collection_info.config.params.hnsw_config = Mock()
    collection_info.config.params.hnsw_config.m = 16
    collection_info.status = "green"  # Use string instead of enum
    
    # Mock search results
    scored_points = [
        Mock(
            id="doc1",
            version=1,
            score=0.95,
            payload={
                "content": "Document 1",
                "tenant_id": "tenant1",
                "created_at": datetime.utcnow().isoformat(),
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
                "created_at": datetime.utcnow().isoformat(),
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
                "created_at": datetime.utcnow().isoformat(),
                "type": "text"
            }
        ),
        Mock(
            id="doc2",
            vector=[0.4, 0.5, 0.6],
            payload={
                "content": "Document 2",
                "tenant_id": "tenant2",
                "created_at": datetime.utcnow().isoformat(),
                "type": "text"
            }
        )
    ]
    
    # Setup mock methods
    client.get_collection.return_value = collection_info
    client.collection_exists.return_value = True
    client.create_collection.return_value = True
    client.search.return_value = scored_points
    client.scroll.return_value = (scroll_points, None)
    client.retrieve.return_value = scroll_points
    client.upsert.return_value = Mock()
    client.delete.return_value = Mock()
    client.delete_collection.return_value = True
    client.update_collection.return_value = Mock()
    
    return client


@pytest.fixture
def repository(mock_qdrant_client):
    """Create an enhanced Qdrant repository for testing."""
    return EnhancedQdrantHTTPRepository(client=mock_qdrant_client, use_async=False)


@pytest.fixture
def async_repository():
    """Create an enhanced Qdrant repository with async client for testing."""
    mock_client = AsyncMock()
    
    # Mock collection info
    collection_info = Mock()
    collection_info.points_count = 100
    collection_info.config = Mock()
    collection_info.config.params = Mock()
    collection_info.config.params.vector = Mock()
    collection_info.config.params.vector.size = 384
    collection_info.config.params.hnsw_config = Mock()
    collection_info.config.params.hnsw_config.m = 16
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
                "created_at": datetime.utcnow().isoformat(),
                "type": "text"
            },
            vector=[0.1, 0.2, 0.3]
        )
    ]
    
    # Setup async mock methods
    mock_client.get_collection.return_value = collection_info
    mock_client.collection_exists.return_value = True
    mock_client.create_collection.return_value = True
    mock_client.create_collection_async.return_value = True
    mock_client.search.return_value = scored_points
    mock_client.scroll.return_value = ([], None)
    mock_client.retrieve.return_value = []
    mock_client.upsert.return_value = Mock()
    mock_client.delete.return_value = Mock()
    mock_client.delete_collection.return_value = True
    
    return EnhancedQdrantHTTPRepository(client=mock_client, use_async=True)


@pytest.fixture
def sample_vector_documents():
    """Create sample vector documents for testing."""
    return [
        VectorDocument(
            id="doc1",
            content="This is document 1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"type": "text", "category": "sample"},
            created_at=datetime.utcnow(),
            collection_name="test_collection",
            tenant_id="tenant1"
        ),
        VectorDocument(
            id="doc2",
            content="This is document 2",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"type": "text", "category": "example"},
            created_at=datetime.utcnow(),
            collection_name="test_collection",
            tenant_id="tenant1"
        ),
        VectorDocument(
            id="doc3",
            content="This is document 3",
            embedding=[0.9, 1.0, 1.1, 1.2],
            metadata={"type": "text", "category": "demo"},
            created_at=datetime.utcnow(),
            collection_name="test_collection",
            tenant_id="tenant2"
        )
    ]


class TestEnhancedQdrantRepository:
    """Test suite for enhanced Qdrant repository functionality."""

    def test_initialization_with_client(self, mock_qdrant_client):
        """Test repository initialization with existing client."""
        repo = EnhancedQdrantHTTPRepository(client=mock_qdrant_client)
        assert repo.client == mock_qdrant_client
        assert repo._query_metrics == []
        assert repo._collection_cache == {}
        assert repo._is_async is False

    def test_initialization_with_parameters(self):
        """Test repository initialization with connection parameters."""
        with patch('pcs.repositories.qdrant_http_repo.QdrantHTTPClient') as mock_client_class:
            repo = EnhancedQdrantHTTPRepository(
                host="localhost",
                port=6333,
                api_key="test_key"
            )
            
            mock_client_class.assert_called_once_with(
                host="localhost",
                port=6333,
                api_key="test_key",
                timeout=30.0,
                max_retries=3,
                retry_delay=1.0
            )

    @pytest.mark.asyncio
    async def test_create_collection_optimized(self, repository, mock_qdrant_client):
        """Test optimized collection creation with advanced configuration."""
        config = QdrantCollectionConfig(
            name="test_collection",
            vector_size=384,
            distance=QdrantDistance.COSINE,
            hnsw_config={"m": 16, "ef_construct": 100},
            optimizers_config={"deleted_threshold": 0.2}
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.create_collection_optimized(
                collection_name=config.name,
                vector_size=config.vector_size,
                distance=config.distance.value,
                hnsw_config=config.hnsw_config,
                optimizers_config=config.optimizers_config
            )
        
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
        assert "test_collection" in repository._collection_cache

    @pytest.mark.asyncio
    async def test_bulk_upsert_documents(self, repository, mock_qdrant_client, sample_vector_documents):
        """Test bulk upsert operations with batch processing."""
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=2,
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.bulk_upsert_documents("test_collection", operation)
        
        assert result["total_processed"] == 3
        assert result["batch_count"] == 2  # 2 docs in first batch, 1 in second
        assert result["execution_time_seconds"] > 0
        assert mock_qdrant_client.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_semantic_search_advanced(self, repository, mock_qdrant_client):
        """Test advanced semantic search with multi-tenancy and reranking."""
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3],
            collection_name="test_collection",
            n_results=5,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            rerank=True,
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            results = await repository.semantic_search_advanced(request)
        
        assert len(results) == 2  # Based on mock data
        assert all(isinstance(result, SimilarityResult) for result in results)
        assert all(result.similarity_score >= request.similarity_threshold for result in results)
        
        # Verify tenant filtering was applied
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert "filter" in call_kwargs

    @pytest.mark.asyncio
    async def test_find_similar_documents_with_tenant(self, repository, mock_qdrant_client):
        """Test finding similar documents with tenant isolation."""
        target_embedding = [0.1, 0.2, 0.3, 0.4]
        
        results = await repository.find_similar_documents(
            collection_name="test_collection",
            target_embedding=target_embedding,
            similarity_threshold=0.7,
            max_results=3,
            tenant_id="tenant1"
        )
        
        assert isinstance(results, list)
        mock_qdrant_client.search.assert_called()

    @pytest.mark.asyncio
    async def test_cluster_documents_kmeans(self, repository, mock_qdrant_client):
        """Test document clustering with K-means algorithm and tenant filtering."""
        # Mock scroll to return points with embeddings
        scroll_points = [
            Mock(
                id="doc1",
                vector=[0.1, 0.2],
                payload={"content": "Doc 1", "tenant_id": "tenant1", "type": "text"}
            ),
            Mock(
                id="doc2",
                vector=[0.3, 0.4],
                payload={"content": "Doc 2", "tenant_id": "tenant1", "type": "text"}
            ),
            Mock(
                id="doc3",
                vector=[0.5, 0.6],
                payload={"content": "Doc 3", "tenant_id": "tenant1", "type": "text"}
            )
        ]
        mock_qdrant_client.scroll.return_value = (scroll_points, None)
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.cluster_documents(
                collection_name="test_collection",
                n_clusters=2,
                algorithm="kmeans",
                tenant_id="tenant1"
            )
        
        assert "clusters" in result
        assert "statistics" in result
        assert result["algorithm"] == "kmeans"
        assert result["statistics"]["total_documents"] == 3

    @pytest.mark.asyncio
    async def test_cluster_documents_empty_collection(self, repository, mock_qdrant_client):
        """Test clustering with empty collection."""
        mock_qdrant_client.scroll.return_value = ([], None)
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.cluster_documents("test_collection", n_clusters=2)
        
        assert result["clusters"] == []
        assert result["statistics"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_get_collection_statistics(self, repository, mock_qdrant_client):
        """Test getting comprehensive collection statistics."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            stats = await repository.get_collection_statistics("test_collection")
        
        assert isinstance(stats, VectorCollectionStats)
        assert stats.name == "test_collection"
        assert stats.document_count == 100
        assert stats.dimension == 384
        assert stats.memory_usage_mb > 0

    @pytest.mark.asyncio
    async def test_get_collection_statistics_with_tenant(self, repository, mock_qdrant_client):
        """Test getting tenant-specific collection statistics."""
        # Mock scroll for tenant counting
        scroll_points = [
            Mock(id="doc1", payload={"tenant_id": "tenant1"}),
            Mock(id="doc2", payload={"tenant_id": "tenant1"})
        ]
        mock_qdrant_client.scroll.return_value = (scroll_points, None)
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            stats = await repository.get_collection_statistics("test_collection", tenant_id="tenant1")
        
        assert isinstance(stats, VectorCollectionStats)
        assert stats.document_count == 2  # Tenant-specific count

    @pytest.mark.asyncio
    async def test_optimize_collection_performance(self, repository, mock_qdrant_client):
        """Test collection performance optimization."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.optimize_collection_performance("test_collection")
        
        assert "before_optimization" in result
        assert "optimizations_applied" in result
        assert "performance_improvements" in result
        mock_qdrant_client.update_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_embeddings_numpy(self, repository, mock_qdrant_client):
        """Test exporting embeddings in numpy format."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection_name="test_collection",
                format_type="numpy",
                include_metadata=True
            )
        
        assert result["collection_name"] == "test_collection"
        assert result["format"] == "numpy"
        assert result["document_count"] == 2
        assert isinstance(result["embeddings"], np.ndarray)
        assert "documents" in result
        assert "metadatas" in result

    @pytest.mark.asyncio
    async def test_export_embeddings_json(self, repository, mock_qdrant_client):
        """Test exporting embeddings in JSON format."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection_name="test_collection",
                format_type="json",
                include_metadata=True
            )
        
        assert result["format"] == "json"
        assert "data" in result
        assert len(result["data"]) == 2
        assert all("embedding" in item for item in result["data"])

    @pytest.mark.asyncio
    async def test_export_embeddings_with_tenant(self, repository, mock_qdrant_client):
        """Test exporting embeddings with tenant filtering."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection_name="test_collection",
                format_type="json",
                tenant_id="tenant1"
            )
        
        assert result["tenant_id"] == "tenant1"
        mock_qdrant_client.scroll.assert_called()

    def test_calculate_similarity_cosine(self, repository):
        """Test similarity calculation with cosine algorithm."""
        # Qdrant returns similarity scores directly for cosine
        similarity = repository._calculate_similarity(0.8, SimilarityAlgorithm.COSINE)
        assert similarity == 0.8

    def test_calculate_similarity_euclidean(self, repository):
        """Test similarity calculation with Euclidean algorithm."""
        similarity = repository._calculate_similarity(1.0, SimilarityAlgorithm.EUCLIDEAN)
        assert similarity == 0.5  # 1 / (1 + 1.0)

    def test_calculate_similarity_manhattan(self, repository):
        """Test similarity calculation with Manhattan algorithm."""
        similarity = repository._calculate_similarity(3.0, SimilarityAlgorithm.MANHATTAN)
        assert similarity == 0.25  # 1 / (1 + 3.0)

    @pytest.mark.asyncio
    async def test_rerank_results(self, repository):
        """Test result reranking with metadata boosting."""
        doc1 = VectorDocument(
            id="doc1", content="Doc 1", embedding=[0.1], 
            metadata={"priority": "high"}, created_at=datetime.utcnow(), 
            collection_name="test", tenant_id="tenant1"
        )
        doc2 = VectorDocument(
            id="doc2", content="Doc 2", embedding=[0.2], 
            metadata={"recent": True}, created_at=datetime.utcnow(), 
            collection_name="test", tenant_id="tenant1"
        )
        
        results = [
            SimilarityResult(document=doc1, similarity_score=0.7, distance=0.3, rank=1),
            SimilarityResult(document=doc2, similarity_score=0.8, distance=0.2, rank=2)
        ]
        
        request = VectorSearchRequest()
        
        reranked = await repository._rerank_results(results, request)
        
        assert len(reranked) == 2
        assert all(result.similarity_score > 0.7 for result in reranked)

    @pytest.mark.asyncio 
    async def test_kmeans_clustering_fallback(self, repository):
        """Test K-means clustering fallback when sklearn is not available."""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        result = await repository._kmeans_clustering(embeddings, 2)
        
        assert len(result) == 2
        assert all(0 <= label < 2 for label in result)

    @pytest.mark.asyncio
    async def test_dbscan_clustering_fallback(self, repository):
        """Test DBSCAN clustering fallback when sklearn is not available."""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.9, 1.0]])
        
        result = await repository._dbscan_clustering(embeddings)
        
        assert len(result) == 3
        assert all(label >= 0 for label in result)  # Fallback returns zeros

    def test_calculate_avg_query_time_empty(self, repository):
        """Test average query time calculation with no metrics."""
        avg_time = repository._calculate_avg_query_time("test_collection")
        assert avg_time == 0.0

    def test_calculate_avg_query_time_with_metrics(self, repository):
        """Test average query time calculation with metrics."""
        repository._query_metrics = [
            {"collection": "test_collection", "execution_time": 0.1},
            {"collection": "test_collection", "execution_time": 0.2},
            {"collection": "other_collection", "execution_time": 0.5}
        ]
        
        avg_time = repository._calculate_avg_query_time("test_collection")
        assert abs(avg_time - 150.0) < 0.001  # (0.1 + 0.2) / 2 * 1000 ms


class TestVectorDocument:
    """Test VectorDocument data class with multi-tenancy."""

    def test_vector_document_creation(self):
        """Test VectorDocument creation and to_dict method."""
        created_at = datetime.utcnow()
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            created_at=created_at,
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        assert doc.id == "test_id"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"type": "test"}
        assert doc.created_at == created_at
        assert doc.collection_name == "test_collection"
        assert doc.tenant_id == "tenant1"
        
        # Test to_dict
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "test_id"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["embedding"] == [0.1, 0.2, 0.3]
        assert doc_dict["created_at"] == created_at.isoformat()
        assert doc_dict["tenant_id"] == "tenant1"

    def test_to_qdrant_point(self):
        """Test conversion to Qdrant PointStruct."""
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test", "category": "sample"},
            created_at=datetime.utcnow(),
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        point = doc.to_qdrant_point()
        assert isinstance(point, PointStruct)
        assert point.id == "test_id"
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload["content"] == "Test content"
        assert point.payload["tenant_id"] == "tenant1"
        assert point.payload["type"] == "test"
        assert point.payload["category"] == "sample"


class TestVectorSearchRequest:
    """Test VectorSearchRequest data class with multi-tenancy."""

    def test_vector_search_request_defaults(self):
        """Test VectorSearchRequest with default values."""
        request = VectorSearchRequest()
        
        assert request.query_text is None
        assert request.query_embedding is None
        assert request.collection_name == ""
        assert request.n_results == 10
        assert request.similarity_threshold == 0.0
        assert request.algorithm == SimilarityAlgorithm.COSINE
        assert request.include_embeddings is False
        assert request.rerank is False
        assert request.tenant_id is None
        assert request.offset is None

    def test_vector_search_request_with_tenant(self):
        """Test VectorSearchRequest with tenant information."""
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3],
            collection_name="test_collection",
            n_results=5,
            similarity_threshold=0.8,
            algorithm=SimilarityAlgorithm.EUCLIDEAN,
            include_embeddings=True,
            rerank=True,
            tenant_id="tenant1",
            offset=10
        )
        
        assert request.query_embedding == [0.1, 0.2, 0.3]
        assert request.collection_name == "test_collection"
        assert request.n_results == 5
        assert request.similarity_threshold == 0.8
        assert request.algorithm == SimilarityAlgorithm.EUCLIDEAN
        assert request.include_embeddings is True
        assert request.rerank is True
        assert request.tenant_id == "tenant1"
        assert request.offset == 10


class TestBulkVectorOperation:
    """Test BulkVectorOperation data class with multi-tenancy."""

    def test_bulk_vector_operation_creation(self, sample_vector_documents):
        """Test BulkVectorOperation creation and to_dict method."""
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=500,
            tenant_id="tenant1"
        )
        
        assert operation.operation_type == "insert"
        assert operation.documents == sample_vector_documents
        assert operation.batch_size == 500
        assert operation.tenant_id == "tenant1"
        
        # Test to_dict
        op_dict = operation.to_dict()
        assert op_dict["operation_type"] == "insert"
        assert op_dict["document_count"] == 3
        assert op_dict["batch_size"] == 500
        assert op_dict["tenant_id"] == "tenant1"


class TestQdrantCollectionConfig:
    """Test QdrantCollectionConfig data class."""

    def test_qdrant_collection_config_defaults(self):
        """Test QdrantCollectionConfig with default values."""
        config = QdrantCollectionConfig(
            name="test_collection",
            vector_size=384,
            distance=QdrantDistance.COSINE
        )
        
        assert config.name == "test_collection"
        assert config.vector_size == 384
        assert config.distance == QdrantDistance.COSINE
        assert config.hnsw_config is None
        assert config.optimizers_config is None
        assert config.on_disk_payload is True

    def test_qdrant_collection_config_custom(self):
        """Test QdrantCollectionConfig with custom values."""
        config = QdrantCollectionConfig(
            name="test_collection",
            vector_size=512,
            distance=QdrantDistance.EUCLIDEAN,
            hnsw_config={"m": 32, "ef_construct": 200},
            optimizers_config={"deleted_threshold": 0.1},
            on_disk_payload=False
        )
        
        assert config.name == "test_collection"
        assert config.vector_size == 512
        assert config.distance == QdrantDistance.EUCLIDEAN
        assert config.hnsw_config == {"m": 32, "ef_construct": 200}
        assert config.optimizers_config == {"deleted_threshold": 0.1}
        assert config.on_disk_payload is False


class TestQdrantDistance:
    """Test QdrantDistance enum."""

    def test_qdrant_distance_values(self):
        """Test that all expected distance metrics are defined."""
        assert QdrantDistance.COSINE.value == Distance.COSINE
        assert QdrantDistance.EUCLIDEAN.value == Distance.EUCLID
        assert QdrantDistance.DOT_PRODUCT.value == Distance.DOT
        assert QdrantDistance.MANHATTAN.value == Distance.MANHATTAN
        
        # Test enum usage
        assert len(QdrantDistance) == 4


class TestBackwardCompatibility:
    """Test backward compatibility with ChromaDB interface."""

    @pytest.mark.asyncio
    async def test_legacy_create_collection(self, repository, mock_qdrant_client):
        """Test legacy create_collection method still works."""
        result = await repository.create_collection(
            "test_collection", 
            {"meta": "data"}, 
            vector_size=512, 
            distance="euclidean"
        )
        assert result is True
        mock_qdrant_client.create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_get_collection(self, repository, mock_qdrant_client):
        """Test legacy get_collection method."""
        result = await repository.get_collection("test_collection")
        assert result is True
        mock_qdrant_client.collection_exists.assert_called_once_with(collection_name="test_collection")

    @pytest.mark.asyncio
    async def test_legacy_get_collection_not_found(self, repository, mock_qdrant_client):
        """Test legacy get_collection when collection doesn't exist."""
        mock_qdrant_client.collection_exists.side_effect = Exception("Not found")
        
        result = await repository.get_collection("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_legacy_add_documents(self, repository, mock_qdrant_client):
        """Test legacy add_documents method with tenant support."""
        result = await repository.add_documents(
            collection_name="test_collection",
            documents=["Doc 1", "Doc 2"],
            ids=["id1", "id2"],
            metadatas=[{"type": "text"}, {"type": "text"}],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            tenant_id="tenant1"
        )
        
        assert result is True
        mock_qdrant_client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_query_documents(self, repository, mock_qdrant_client):
        """Test legacy query_documents method with tenant support."""
        results = await repository.query_documents(
            collection_name="test_collection",
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=3,
            where={"type": "text"},
            tenant_id="tenant1"
        )
        
        assert "ids" in results
        assert "documents" in results
        assert "distances" in results
        assert "metadatas" in results
        mock_qdrant_client.search.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_get_documents_by_ids(self, repository, mock_qdrant_client):
        """Test legacy get_documents method by IDs."""
        results = await repository.get_documents(
            collection_name="test_collection",
            ids=["id1", "id2"],
            tenant_id="tenant1"
        )
        
        assert "ids" in results
        assert "documents" in results
        assert "embeddings" in results
        assert "metadatas" in results
        mock_qdrant_client.retrieve.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_similarity_search(self, repository, mock_qdrant_client):
        """Test legacy similarity_search method with tenant support."""
        results = await repository.similarity_search(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=3,
            threshold=0.5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 2  # Based on mock data
        assert all("similarity" in result for result in results)
        assert all("document" in result for result in results)

    @pytest.mark.asyncio
    async def test_legacy_delete_documents_by_ids(self, repository, mock_qdrant_client):
        """Test legacy delete_documents method by IDs."""
        result = await repository.delete_documents(
            collection_name="test_collection",
            ids=["id1", "id2"]
        )
        
        assert result is True
        mock_qdrant_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_delete_documents_by_filter(self, repository, mock_qdrant_client):
        """Test legacy delete_documents method by filter."""
        result = await repository.delete_documents(
            collection_name="test_collection",
            where={"type": "text"},
            tenant_id="tenant1"
        )
        
        assert result is True
        mock_qdrant_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_count_documents(self, repository, mock_qdrant_client):
        """Test legacy count_documents method."""
        count = await repository.count_documents("test_collection")
        assert count == 100  # Based on mock data
        mock_qdrant_client.get_collection.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_delete_collection(self, repository, mock_qdrant_client):
        """Test legacy delete_collection method."""
        result = await repository.delete_collection("test_collection")
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")


class TestQdrantRepositoryAlias:
    """Test that QdrantRepository is an alias for EnhancedQdrantRepository."""

    def test_alias_compatibility(self, mock_qdrant_client):
        """Test that QdrantRepository is the same as EnhancedQdrantRepository."""
        assert QdrantRepository == EnhancedQdrantRepository
        
        # Test instantiation
        repo = QdrantRepository(client=mock_qdrant_client)
        assert isinstance(repo, EnhancedQdrantRepository)
        assert repo.client == mock_qdrant_client


class TestErrorHandling:
    """Test error handling in Qdrant repository."""

    @pytest.mark.asyncio
    async def test_create_collection_error(self, repository, mock_qdrant_client):
        """Test error handling in collection creation."""
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")
        
        config = QdrantCollectionConfig(name="test_collection", vector_size=384, distance=QdrantDistance.COSINE)
        
        with pytest.raises(RepositoryError, match="Failed to create optimized collection"):
            await repository.create_collection_optimized(
                collection_name=config.name,
                vector_size=config.vector_size,
                distance=config.distance.value
            )

    @pytest.mark.asyncio
    async def test_bulk_upsert_error(self, repository, mock_qdrant_client, sample_vector_documents):
        """Test error handling in bulk upsert."""
        mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=10
        )
        
        with pytest.raises(RepositoryError, match="Failed to bulk upsert documents"):
            await repository.bulk_upsert_documents("test_collection", operation)

    @pytest.mark.asyncio
    async def test_semantic_search_error(self, repository, mock_qdrant_client):
        """Test error handling in semantic search."""
        mock_qdrant_client.search.side_effect = Exception("Search failed")
        
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3],
            collection_name="test_collection"
        )
        
        with pytest.raises(RepositoryError, match="Failed to perform advanced semantic search"):
            await repository.semantic_search_advanced(request)

    @pytest.mark.asyncio
    async def test_clustering_error(self, repository, mock_qdrant_client):
        """Test error handling in document clustering."""
        mock_qdrant_client.scroll.side_effect = Exception("Scroll failed")
        
        with pytest.raises(RepositoryError, match="Failed to cluster documents"):
            await repository.cluster_documents("test_collection", n_clusters=2)

    @pytest.mark.asyncio
    async def test_unsupported_clustering_algorithm(self, repository, mock_qdrant_client):
        """Test error handling for unsupported clustering algorithm."""
        scroll_points = [
            Mock(id="doc1", vector=[0.1, 0.2], payload={"content": "Doc 1"}),
            Mock(id="doc2", vector=[0.3, 0.4], payload={"content": "Doc 2"})
        ]
        mock_qdrant_client.scroll.return_value = (scroll_points, None)
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            with pytest.raises(RepositoryError, match="Failed to cluster documents"):
                await repository.cluster_documents(
                    "test_collection",
                    algorithm="unsupported_algorithm"
                )


class TestMultiTenancy:
    """Test multi-tenancy features specific to Qdrant."""

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_search(self, repository, mock_qdrant_client):
        """Test that tenant isolation works in search operations."""
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3],
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            await repository.semantic_search_advanced(request)
        
        # Verify tenant filter was applied
        mock_qdrant_client.search.assert_called_once()
        call_kwargs = mock_qdrant_client.search.call_args.kwargs
        assert "filter" in call_kwargs

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_export(self, repository, mock_qdrant_client):
        """Test that tenant isolation works in export operations."""
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection_name="test_collection",
                tenant_id="tenant1"
            )
        
        assert result["tenant_id"] == "tenant1"
        mock_qdrant_client.scroll.assert_called()

    @pytest.mark.asyncio
    async def test_build_query_filter_with_tenant_and_metadata(self, repository):
        """Test building query filter with both tenant and metadata filters."""
        request = VectorSearchRequest(
            tenant_id="tenant1",
            metadata_filter={
                "type": "text",
                "score": {"gte": 0.5, "lte": 0.9}
            }
        )
        
        query_filter = repository._build_query_filter(request)
        
        assert query_filter is not None
        assert len(query_filter.must) == 3  # tenant_id, type, score range

    def test_build_query_filter_empty(self, repository):
        """Test building query filter with no filters."""
        request = VectorSearchRequest()
        
        query_filter = repository._build_query_filter(request)
        
        assert query_filter is None


class TestAsyncRepository:
    """Test async repository functionality."""

    @pytest.mark.asyncio
    async def test_async_semantic_search(self, async_repository):
        """Test async semantic search operations."""
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3],
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            results = await async_repository.semantic_search_advanced(request)
        
        assert isinstance(results, list)
        async_repository.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_collection_creation(self, async_repository):
        """Test async collection creation."""
        config = QdrantCollectionConfig(
            name="test_collection",
            vector_size=384,
            distance=QdrantDistance.COSINE
        )
        
        with patch('pcs.repositories.qdrant_http_repo.PerformanceMonitor'):
            result = await async_repository.create_collection_optimized(
                collection_name=config.name,
                vector_size=config.vector_size,
                distance=config.distance.value
            )
        
        assert result is True
<<<<<<< Current (Your changes)
        async_repository.client.create_collection.assert_called_once()
=======
        async_repository.client.create_collection_async.assert_called_once()
>>>>>>> Incoming (Background Agent changes)
