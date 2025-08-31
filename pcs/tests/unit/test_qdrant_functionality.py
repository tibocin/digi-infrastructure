"""
Filepath: pcs/tests/unit/test_qdrant_functionality.py
Purpose: Comprehensive tests for Qdrant repository functionality
Related Components: EnhancedQdrantRepository, QdrantHTTPClient, vector operations
Tags: testing, qdrant, vector-database, functionality, mocking
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, UTC
from typing import List, Dict, Any

from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
from pcs.repositories.qdrant_types import (
    QdrantDistance,
    QdrantCollectionConfig,
    QdrantPoint,
    QdrantSearchResult,
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorSearchRequest,
    VectorCollectionStats,
    BulkVectorOperation
)
from pcs.repositories.qdrant_http_client import QdrantHTTPClient


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant HTTP client for testing."""
    client = Mock()
    
    # Mock health check
    client.health_check.return_value = {"version": "1.7.4", "status": "ok"}
    
    # Mock collections
    client.get_collections.return_value = [
        {"name": "test_collection", "vectors_count": 100}
    ]
    
    # Mock collection info
    client.get_collection.return_value = {
        "result": {
            "config": {
                "params": {
                    "vectors": {"size": 384, "distance": "Cosine"}
                }
            },
            "vectors_count": 100,
            "points_count": 100,
            "segments_count": 1,
            "status": "green"
        }
    }
    
    # Mock collection stats
    client.get_collection_stats.return_value = {
        "vectors_count": 100,
        "points_count": 100,
        "segments_count": 1,
        "status": "green"
    }
    
    # Mock collection creation
    client.create_collection.return_value = True
    
    # Mock collection deletion
    client.delete_collection.return_value = True
    
    # Mock points upsert
    client.upsert_points.return_value = {"result": {"status": "ok"}}
    
    # Mock points search
    client.search_points.return_value = [
        QdrantSearchResult(
            id=1,
            score=0.95,
            payload={
                "content": "Test document",
                "metadata": {"type": "text"},
                "created_at": datetime.now(UTC).isoformat(),
                "tenant_id": "tenant1"
            },
            vector=[0.1, 0.2, 0.3]
        )
    ]
    
    # Mock points deletion
    client.delete_points.return_value = {"result": {"status": "ok"}}
    
    # Mock scroll for export functionality
    client.scroll.return_value = ([
        Mock(
            vector=[0.1, 0.2, 0.3],
            payload={"content": "Doc 1", "tenant_id": "tenant1"}
        ),
        Mock(
            vector=[0.4, 0.5, 0.6],
            payload={"content": "Doc 2", "tenant_id": "tenant1"}
        )
    ], None)
    
    return client


@pytest.fixture
def repository(mock_qdrant_client):
    """Create an enhanced Qdrant repository for testing."""
    return EnhancedQdrantRepository(client=mock_qdrant_client)


@pytest.fixture
def sample_documents():
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
            metadata={"type": "text", "category": "example"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        )
    ]


class TestEnhancedQdrantRepository:
    """Test the enhanced Qdrant repository functionality."""

    def test_initialization_with_client(self, mock_qdrant_client):
        """Test repository initialization with existing client."""
        repo = EnhancedQdrantRepository(client=mock_qdrant_client)
        assert repo.client == mock_qdrant_client
        assert repo._is_async is False

    def test_initialization_with_parameters(self):
        """Test repository initialization with connection parameters."""
        repo = EnhancedQdrantRepository(
            host="test-host",
            port=6334,
            api_key="test-key",
            timeout=60.0,
            max_retries=5
        )
        assert repo.client.base_url == "http://test-host:6334"
        assert repo.client.api_key == "test-key"
        assert repo.client.timeout == 60.0
        assert repo.client.max_retries == 5

    def test_health_check(self, repository, mock_qdrant_client):
        """Test repository health check."""
        result = repository.health_check()
        assert result["version"] == "1.7.4"
        assert result["status"] == "ok"
        mock_qdrant_client.health_check.assert_called_once()

    def test_get_collections(self, repository, mock_qdrant_client):
        """Test getting collections."""
        collections = repository.get_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "test_collection"
        mock_qdrant_client.get_collections.assert_called_once()

    def test_get_collection_info(self, repository, mock_qdrant_client):
        """Test getting collection information."""
        info = repository.get_collection_info("test_collection")
        assert info["result"]["config"]["params"]["vectors"]["size"] == 384
        mock_qdrant_client.get_collection.assert_called_once_with("test_collection")

    def test_create_collection(self, repository, mock_qdrant_client):
        """Test collection creation."""
        result = repository.create_collection(
            collection_name="new_collection",
            vector_size=512,
            distance="euclidean"
        )
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

    def test_delete_collection(self, repository, mock_qdrant_client):
        """Test collection deletion."""
        result = repository.delete_collection("test_collection")
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")

    def test_upsert_documents(self, repository, mock_qdrant_client, sample_documents):
        """Test document upsert."""
        result = repository.upsert_documents("test_collection", sample_documents)
        assert result["result"]["status"] == "ok"
        mock_qdrant_client.upsert_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar(self, repository, mock_qdrant_client, sample_documents):
        """Test similarity search."""
        # First upsert documents
        repository.upsert_documents("test_collection", sample_documents)
        
        # Then search
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.15, 0.25, 0.35, 0.45],
            limit=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert isinstance(results[0], SimilarityResult)
        assert results[0].similarity_score == 0.95
        mock_qdrant_client.search_points.assert_called_once()

    def test_delete_documents(self, repository, mock_qdrant_client):
        """Test document deletion."""
        result = repository.delete_documents("test_collection", ["doc1", "doc2"])
        assert result["result"]["status"] == "ok"
        mock_qdrant_client.delete_points.assert_called_once_with("test_collection", ["doc1", "doc2"], True)

    def test_get_collection_stats(self, repository, mock_qdrant_client):
        """Test getting collection statistics."""
        stats = repository.get_collection_stats("test_collection")
        assert isinstance(stats, VectorCollectionStats)
        assert stats.vectors_count == 100
        assert stats.points_count == 100
        assert stats.status == "green"
        mock_qdrant_client.get_collection_stats.assert_called_once_with("test_collection")

    def test_calculate_similarity_cosine(self, repository):
        """Test cosine similarity calculation."""
        similarity = repository._calculate_similarity(0.8, SimilarityAlgorithm.COSINE)
        assert similarity == 0.8

    def test_calculate_similarity_euclidean(self, repository):
        """Test Euclidean similarity calculation."""
        similarity = repository._calculate_similarity(2.0, SimilarityAlgorithm.EUCLIDEAN)
        expected = 1.0 / (1.0 + 2.0)
        assert abs(similarity - expected) < 1e-6

    def test_calculate_similarity_manhattan(self, repository):
        """Test Manhattan similarity calculation."""
        similarity = repository._calculate_similarity(3.0, SimilarityAlgorithm.MANHATTAN)
        expected = 1.0 / (1.0 + 3.0)
        assert abs(similarity - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_semantic_search_advanced(self, repository, mock_qdrant_client):
        """Test advanced semantic search."""
        request = VectorSearchRequest(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            collection_name="test_collection",
            n_results=5,
            similarity_threshold=0.8,
            algorithm=SimilarityAlgorithm.COSINE,
            tenant_id="tenant1"
        )
        
        results = await repository.semantic_search_advanced(request)
        assert len(results) == 1
        assert isinstance(results[0], SimilarityResult)
        mock_qdrant_client.search_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_embeddings(self, repository, mock_qdrant_client):
        """Test embedding export."""
        # Mock scroll results for export
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.4, 0.5, 0.6],
                payload={"content": "Doc 2", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        assert result["collection_name"] == "test_collection"
        assert result["tenant_id"] == "tenant1"
        assert result["format"] == "numpy"
        assert len(result["embeddings"]) == 2

    @pytest.mark.asyncio
    async def test_create_collection_optimized(self, repository, mock_qdrant_client):
        """Test optimized collection creation."""
        result = await repository.create_collection_optimized(
            collection_name="optimized_collection",
            vector_size=768,
            distance="cosine"
        )
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()

    def test_build_query_filter(self, repository):
        """Test query filter building."""
        filter_conditions = repository._build_query_filter(
            tenant_id="tenant1",
            metadata_filters={"type": "text"}
        )
        
        # Should return filter conditions
        assert filter_conditions is not None
        assert "must" in filter_conditions
        assert len(filter_conditions["must"]) == 2  # tenant_id + metadata filter

    def test_context_manager(self, repository):
        """Test context manager functionality."""
        with repository as repo:
            assert repo == repository
        # Should not raise any errors


class TestQdrantHTTPClient:
    """Test the Qdrant HTTP client functionality."""

    def test_client_initialization(self):
        """Test HTTP client initialization."""
        client = QdrantHTTPClient(
            host="test-host",
            port=6334,
            api_key="test-key",
            timeout=60.0,
            max_retries=5
        )
        
        assert client.base_url == "http://test-host:6334"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0
        assert client.max_retries == 5
        assert "api-key" in client.headers

    def test_client_initialization_no_api_key(self):
        """Test HTTP client initialization without API key."""
        client = QdrantHTTPClient(host="test-host", port=6334)
        
        assert client.base_url == "http://test-host:6334"
        assert client.api_key is None
        assert "api-key" not in client.headers

    @patch('pcs.repositories.qdrant_http_client.Client')
    def test_make_request_success(self, mock_client_class):
        """Test successful HTTP request."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "success"}'
        mock_response.json.return_value = {"result": "success"}
        
        mock_client.get.return_value = mock_response
        
        client = QdrantHTTPClient(host="test-host", port=6334)
        result = client._make_request("GET", "/test")
        
        assert result == {"result": "success"}
        mock_client.get.assert_called_once()

    @patch('pcs.repositories.qdrant_http_client.Client')
    def test_make_request_error(self, mock_client_class):
        """Test HTTP request with error response."""
        mock_client = Mock()
        mock_client_class.return_value.__enter__.return_value = mock_client
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = '{"error": "bad request"}'
        
        mock_client.get.return_value = mock_response
        
        client = QdrantHTTPClient(host="test-host", port=6334)
        
        with pytest.raises(Exception) as exc_info:
            client._make_request("GET", "/test")
        
        assert "HTTP 400" in str(exc_info.value)


class TestDataStructures:
    """Test the data structure classes."""

    def test_qdrant_collection_config(self):
        """Test QdrantCollectionConfig."""
        config = QdrantCollectionConfig(
            name="test_collection",
            vector_size=384,
            distance=QdrantDistance.COSINE,
            on_disk_payload=True,
            hnsw_config={"m": 16},
            optimizers_config={"deleted_threshold": 0.1}
        )
        
        assert config.name == "test_collection"
        assert config.vector_size == 384
        assert config.distance == QdrantDistance.COSINE
        assert config.on_disk_payload is True
        assert config.hnsw_config == {"m": 16}
        assert config.optimizers_config == {"deleted_threshold": 0.1}

    def test_qdrant_point(self):
        """Test QdrantPoint."""
        point = QdrantPoint(
            id="test_id",
            vector=[0.1, 0.2, 0.3],
            payload={"content": "test"}
        )
        
        assert point.id == "test_id"
        assert point.vector == [0.1, 0.2, 0.3]
        assert point.payload["content"] == "test"

    def test_qdrant_search_result(self):
        """Test QdrantSearchResult."""
        result = QdrantSearchResult(
            id="test_id",
            score=0.95,
            payload={"content": "test"},
            vector=[0.1, 0.2, 0.3],
            version=1
        )
        
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.payload["content"] == "test"
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.version == 1

    def test_vector_collection_stats(self):
        """Test VectorCollectionStats."""
        stats = VectorCollectionStats(
            vectors_count=100,
            points_count=100,
            segments_count=1,
            status="green",
            config={"test": "config"},
            payload_schema={"test": "schema"}
        )
        
        assert stats.vectors_count == 100
        assert stats.points_count == 100
        assert stats.segments_count == 1
        assert stats.status == "green"
        assert stats.config == {"test": "config"}
        assert stats.payload_schema == {"test": "schema"}

    def test_bulk_vector_operation(self):
        """Test BulkVectorOperation."""
        operation = BulkVectorOperation(
            operation_type="upsert",
            collection_name="test_collection",
            documents=["doc1", "doc2"],
            tenant_id="tenant1",
            batch_size=500,
            metadata={"priority": "high"}
        )
        
        assert operation.operation_type == "upsert"
        assert operation.collection_name == "test_collection"
        assert operation.documents == ["doc1", "doc2"]
        assert operation.tenant_id == "tenant1"
        assert operation.batch_size == 500
        assert operation.metadata == {"priority": "high"}
        
        # Test to_dict
        op_dict = operation.to_dict()
        assert op_dict["operation_type"] == "upsert"
        assert op_dict["documents_count"] == 2
        assert op_dict["tenant_id"] == "tenant1"
        assert op_dict["batch_size"] == 500
        assert op_dict["metadata"] == {"priority": "high"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
