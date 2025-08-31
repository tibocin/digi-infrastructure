"""
Filepath: pcs/tests/unit/test_qdrant_core.py
Purpose: Core repository functionality tests for Qdrant
Related Components: Basic operations, CRUD, search
Tags: testing, core, qdrant, repository
"""

import pytest
from unittest.mock import patch, Mock
from datetime import datetime, UTC

from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
from pcs.repositories.qdrant_types import (
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorIndexType,
    VectorCollectionStats,
    BulkVectorOperation,
    VectorSearchRequest,
    QdrantCollectionConfig,
    QdrantDistance
)

# Create alias for backward compatibility in tests
QdrantRepository = EnhancedQdrantRepository
EnhancedQdrantHTTPRepository = EnhancedQdrantRepository


class TestEnhancedQdrantRepository:
    """Test suite for enhanced Qdrant repository functionality."""
    
    def test_initialization_with_client(self, mock_qdrant_client):
        """Test repository initialization with existing client."""
        repo = EnhancedQdrantHTTPRepository(client=mock_qdrant_client)
        assert repo.client == mock_qdrant_client
    
    def test_initialization_with_parameters(self, mock_qdrant_client):
        """Test repository initialization with connection parameters."""
        with patch('pcs.repositories.qdrant_repo.QdrantHTTPClient') as mock_client_class:
            repo = EnhancedQdrantHTTPRepository(
                host="localhost",
                port=6333,
                api_key="test_key"
            )
            assert repo.client is not None
    
    @pytest.mark.asyncio
    async def test_create_collection_optimized(self, repository, mock_qdrant_client):
        """Test creating optimized collection."""
        result = await repository.create_collection_optimized(
            collection_name="test_collection",
            vector_size=384
        )
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_upsert_documents(self, repository, mock_qdrant_client, sample_vector_documents):
        """Test bulk upsert operations with batch processing."""
        operation = BulkVectorOperation(
            operation_type="insert",
            collection_name="test_collection",
            documents=sample_vector_documents,
            batch_size=2,
            tenant_id="tenant1"
        )
        
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            result = await repository.bulk_upsert_documents("test_collection", operation)
        
        assert result["total_processed"] == 3
        assert result["batch_count"] == 2  # 2 docs in first batch, 1 in second
        assert result["execution_time_seconds"] > 0
        assert mock_qdrant_client.upsert_points.call_count == 2
    
    @pytest.mark.asyncio
    async def test_semantic_search_advanced(self, repository, mock_qdrant_client):
        """Test advanced semantic search with multi-tenancy and reranking."""
        # Configure mock to return search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        request = VectorSearchRequest(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        results = await repository.semantic_search_advanced(request)
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
    
    @pytest.mark.asyncio
    async def test_find_similar_documents_with_tenant(self, repository, mock_qdrant_client):
        """Test finding similar documents with tenant isolation."""
        # Configure mock to return search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        results = await repository.find_similar_documents(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics(self, repository, mock_qdrant_client):
        """Test getting comprehensive collection statistics."""
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            stats = await repository.get_collection_statistics("test_collection")
        
        assert isinstance(stats, VectorCollectionStats)
        assert stats.points_count == 100
        assert stats.config["dimension"] == 384
        assert stats.config["memory_usage_mb"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics_with_tenant(self, repository, mock_qdrant_client):
        """Test getting tenant-specific collection statistics."""
        # Mock scroll for tenant counting
        mock_qdrant_client.scroll.return_value = ([
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant2"})
        ], None)
        
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            stats = await repository.get_collection_statistics("test_collection", tenant_id="tenant1")
        
        assert isinstance(stats, VectorCollectionStats)
        assert stats.points_count == 2  # Tenant-specific count
    
    @pytest.mark.asyncio
    async def test_optimize_collection_performance(self, repository, mock_qdrant_client):
        """Test collection performance optimization."""
        with patch('pcs.repositories.qdrant_repo.PerformanceMonitor'):
            result = await repository.optimize_collection_performance("test_collection")
        
        assert "before_optimization" in result
        assert "optimizations_applied" in result
        assert "performance_improvements" in result
        assert result["before_optimization"]["memory_usage_mb"] == 100.0
        assert result["performance_improvements"]["memory_usage_mb"] == 80.0
    
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
    async def test_calculate_avg_query_time_empty(self, repository):
        """Test average query time calculation with empty metrics."""
        avg_time = repository._calculate_avg_query_time("test_collection")
        assert avg_time == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_avg_query_time_with_metrics(self, repository):
        """Test average query time calculation with metrics."""
        # Mock performance metrics
        repository.performance_metrics = {
            "test_collection": {
                "query_times": [0.1, 0.2]
            }
        }
        
        avg_time = repository._calculate_avg_query_time("test_collection")
        assert abs(avg_time - 150.0) < 0.001  # (0.1 + 0.2) / 2 * 1000 ms


class TestVectorDocument:
    """Test suite for VectorDocument dataclass."""
    
    def test_vector_document_creation(self):
        """Test VectorDocument creation."""
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        assert doc.id == "test_id"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata["type"] == "test"
        assert doc.tenant_id == "tenant1"
    
    def test_to_qdrant_point(self):
        """Test conversion to Qdrant point."""
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        point = doc.to_qdrant_point()
        assert point.id == "test_id"
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload["content"] == "Test content"
        assert point.payload["tenant_id"] == "tenant1"


class TestVectorSearchRequest:
    """Test suite for VectorSearchRequest dataclass."""
    
    def test_vector_search_request_defaults(self):
        """Test VectorSearchRequest with default values."""
        request = VectorSearchRequest(collection_name="test_collection")
        
        assert request.collection_name == "test_collection"
        assert request.n_results == 10
        assert request.similarity_threshold == 0.0
        assert request.algorithm == SimilarityAlgorithm.COSINE
    
    def test_vector_search_request_with_tenant(self):
        """Test VectorSearchRequest with tenant specification."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            tenant_id="tenant1",
            n_results=5
        )
        
        assert request.tenant_id == "tenant1"
        assert request.n_results == 5


class TestBulkVectorOperation:
    """Test suite for BulkVectorOperation dataclass."""
    
    def test_bulk_vector_operation_creation(self):
        """Test BulkVectorOperation creation."""
        operation = BulkVectorOperation(
            operation_type="insert",
            collection_name="test_collection",
            documents=[],
            tenant_id="tenant1",
            batch_size=50
        )
        
        assert operation.operation_type == "insert"
        assert operation.collection_name == "test_collection"
        assert operation.tenant_id == "tenant1"
        assert operation.batch_size == 50


class TestQdrantCollectionConfig:
    """Test suite for QdrantCollectionConfig dataclass."""
    
    def test_qdrant_collection_config_defaults(self):
        """Test QdrantCollectionConfig with default values."""
        config = QdrantCollectionConfig(name="test", vector_size=384)
        
        assert config.name == "test"
        assert config.vector_size == 384
        assert config.distance == QdrantDistance.COSINE
        assert config.on_disk_payload is True
    
    def test_qdrant_collection_config_custom(self):
        """Test QdrantCollectionConfig with custom values."""
        config = QdrantCollectionConfig(
            name="test",
            vector_size=768,
            distance=QdrantDistance.EUCLIDEAN,
            on_disk_payload=False
        )
        
        assert config.vector_size == 768
        assert config.distance == QdrantDistance.EUCLIDEAN
        assert config.on_disk_payload is False


class TestQdrantDistance:
    """Test suite for QdrantDistance enum."""
    
    def test_qdrant_distance_values(self):
        """Test QdrantDistance enum values."""
        assert QdrantDistance.COSINE.value == "Cosine"
        assert QdrantDistance.EUCLIDEAN.value == "Euclid"
        assert QdrantDistance.DOT_PRODUCT.value == "Dot"
        assert QdrantDistance.MANHATTAN.value == "Manhattan"
