"""
Filepath: pcs/tests/unit/conftest.py
Purpose: Shared test fixtures and mocks for Qdrant tests
Related Components: Test fixtures, mocks, common test data
Tags: testing, fixtures, mocks, qdrant
"""

import pytest
from unittest.mock import Mock, AsyncMock
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
    client.upsert_points.return_value = Mock()
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
    mock_client = Mock()  # Use regular Mock instead of AsyncMock
    
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
                "created_at": datetime.now(UTC).isoformat(),
                "type": "text"
            },
            vector=[0.1, 0.2, 0.3]
        )
    ]
    
    # Setup mock methods - return actual values, not coroutines
    mock_client.get_collection.return_value = collection_info
    mock_client.collection_exists.return_value = True
    mock_client.create_collection.return_value = True
    mock_client.create_collection_async.return_value = True
    mock_client.search_points.return_value = scored_points
    mock_client.scroll.return_value = ([], None)
    mock_client.retrieve.return_value = []
    mock_client.upsert_points.return_value = {"status": "ok"}
    mock_client.search_points_async.return_value = scored_points
    mock_client.delete.return_value = {"status": "ok"}
    mock_client.delete_collection.return_value = True
    mock_client.get_collection_stats.return_value = {
        "points_count": 100,
        "vectors_count": 100,
        "segments_count": 1,
        "status": "green"
    }
    
    return EnhancedQdrantHTTPRepository(client=mock_client, use_async=True)


@pytest.fixture
def sample_vector_documents():
    """Create sample vector documents for testing."""
    return [
        VectorDocument(
            id="doc1",
            content="This is document 1",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"type": "text", "category": "general"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        ),
        VectorDocument(
            id="doc2",
            content="This is document 2",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"type": "text", "category": "specific"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant1"
        ),
        VectorDocument(
            id="doc3",
            content="This is document 3",
            embedding=[0.9, 1.0, 1.1, 1.2],
            metadata={"type": "text", "category": "general"},
            created_at=datetime.now(UTC),
            collection_name="test_collection",
            tenant_id="tenant2"
        )
    ]


@pytest.fixture
def sample_search_request():
    """Create a sample vector search request for testing."""
    return VectorSearchRequest(
        collection_name="test_collection",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        n_results=5,
        similarity_threshold=0.7,
        tenant_id="tenant1",
        algorithm=SimilarityAlgorithm.COSINE
    )
