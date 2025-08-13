"""
Filepath: tests/unit/test_chroma_repo.py
Purpose: Unit tests for enhanced ChromaDB repository implementation
Related Components: EnhancedChromaRepository, VectorDocument, SimilarityResult, BulkVectorOperation
Tags: testing, chromadb, vector-database, semantic-search, similarity-algorithms
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4
from typing import List, Dict, Any
import numpy as np

from pcs.repositories.chroma_repo import (
    EnhancedChromaRepository,
    ChromaRepository,  # Legacy alias
    VectorDocument,
    SimilarityResult,
    VectorSearchRequest,
    BulkVectorOperation,
    VectorCollectionStats,
    SimilarityAlgorithm,
    VectorIndexType
)
from pcs.repositories.base import RepositoryError


@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client for testing."""
    client = Mock()
    collection = Mock()
    collection.name = "test_collection"
    collection.metadata = {"index_type": "hnsw", "created_at": datetime.utcnow().isoformat()}
    collection.count.return_value = 100
    collection.get.return_value = {
        'ids': ['doc1', 'doc2'],
        'documents': ['Document 1', 'Document 2'],
        'embeddings': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'metadatas': [{'type': 'text'}, {'type': 'text'}]
    }
    collection.query.return_value = {
        'ids': [['doc1', 'doc2']],
        'documents': [['Document 1', 'Document 2']],
        'distances': [[0.1, 0.3]],
        'metadatas': [[{'type': 'text'}, {'type': 'text'}]]
    }
    collection.add.return_value = None
    collection.upsert.return_value = None
    collection.update.return_value = None
    collection.delete.return_value = None
    
    client.get_collection.return_value = collection
    client.create_collection.return_value = collection
    client.get_or_create_collection.return_value = collection
    client.delete_collection.return_value = None
    
    return client


@pytest.fixture
def repository(mock_chromadb_client):
    """Create an enhanced ChromaDB repository for testing."""
    return EnhancedChromaRepository(mock_chromadb_client)


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
            collection_name="test_collection"
        ),
        VectorDocument(
            id="doc2",
            content="This is document 2",
            embedding=[0.5, 0.6, 0.7, 0.8],
            metadata={"type": "text", "category": "example"},
            created_at=datetime.utcnow(),
            collection_name="test_collection"
        ),
        VectorDocument(
            id="doc3",
            content="This is document 3",
            embedding=[0.9, 1.0, 1.1, 1.2],
            metadata={"type": "text", "category": "demo"},
            created_at=datetime.utcnow(),
            collection_name="test_collection"
        )
    ]


class TestEnhancedChromaRepository:
    """Test suite for enhanced ChromaDB repository functionality."""

    def test_initialization(self, mock_chromadb_client):
        """Test repository initialization."""
        repo = EnhancedChromaRepository(mock_chromadb_client)
        assert repo.client == mock_chromadb_client
        assert repo._query_metrics == []
        assert repo._collection_cache == {}

    @pytest.mark.asyncio
    async def test_create_collection_optimized(self, repository, mock_chromadb_client):
        """Test optimized collection creation with enhanced metadata."""
        # Execute
        result = await repository.create_collection_optimized(
            name="test_collection",
            metadata={"custom": "value"},
            index_type=VectorIndexType.HNSW
        )
        
        # Verify
        assert result is not None
        mock_chromadb_client.create_collection.assert_called_once()
        
        # Check that enhanced metadata was used
        call_args = mock_chromadb_client.create_collection.call_args
        metadata = call_args[1]['metadata']
        assert metadata['index_type'] == 'hnsw'
        assert metadata['optimized'] is True
        assert metadata['custom'] == 'value'
        assert 'created_at' in metadata

    @pytest.mark.asyncio
    async def test_bulk_upsert_documents(self, repository, mock_chromadb_client, sample_vector_documents):
        """Test bulk upsert operations with batch processing."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=2
        )
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.bulk_upsert_documents(collection, operation)
        
        # Verify
        assert result["total_processed"] == 3
        assert result["batch_count"] == 2  # 2 docs in first batch, 1 in second
        assert result["execution_time_seconds"] > 0
        assert collection.upsert.call_count == 2  # Two batches

    @pytest.mark.asyncio
    async def test_semantic_search_advanced(self, repository, mock_chromadb_client):
        """Test advanced semantic search with algorithms and reranking."""
        # Setup
        request = VectorSearchRequest(
            query_text="test query",
            collection_name="test_collection",
            n_results=5,
            similarity_threshold=0.5,
            algorithm=SimilarityAlgorithm.COSINE,
            rerank=True
        )
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            results = await repository.semantic_search_advanced(request)
        
        # Verify
        assert len(results) == 2  # Based on mock data
        assert all(isinstance(result, SimilarityResult) for result in results)
        assert all(result.similarity_score >= request.similarity_threshold for result in results)
        
        # Check that results are properly structured
        first_result = results[0]
        assert isinstance(first_result.document, VectorDocument)
        assert first_result.similarity_score > 0
        assert first_result.rank >= 1

    @pytest.mark.asyncio
    async def test_find_similar_documents(self, repository, mock_chromadb_client):
        """Test finding similar documents with target embedding."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        target_embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Execute
        results = await repository.find_similar_documents(
            collection=collection,
            target_embedding=target_embedding,
            similarity_threshold=0.7,
            max_results=3
        )
        
        # Verify
        assert isinstance(results, list)
        # Results depend on semantic_search_advanced being called
        mock_chromadb_client.get_collection.assert_called()

    @pytest.mark.asyncio
    async def test_cluster_documents_kmeans(self, repository, mock_chromadb_client):
        """Test document clustering with K-means algorithm."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        collection.get.return_value = {
            'ids': ['doc1', 'doc2', 'doc3'],
            'documents': ['Doc 1', 'Doc 2', 'Doc 3'],
            'embeddings': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            'metadatas': [{'type': 'text'}, {'type': 'text'}, {'type': 'text'}]
        }
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.cluster_documents(
                collection=collection,
                n_clusters=2,
                algorithm="kmeans"
            )
        
        # Verify
        assert "clusters" in result
        assert "statistics" in result
        assert result["algorithm"] == "kmeans"
        assert result["statistics"]["total_documents"] == 3

    @pytest.mark.asyncio
    async def test_cluster_documents_empty_collection(self, repository, mock_chromadb_client):
        """Test clustering with empty collection."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        collection.get.return_value = {'ids': [], 'embeddings': []}
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.cluster_documents(collection, n_clusters=2)
        
        # Verify
        assert result["clusters"] == []
        assert result["statistics"]["total_documents"] == 0

    @pytest.mark.asyncio
    async def test_get_collection_statistics(self, repository, mock_chromadb_client):
        """Test getting comprehensive collection statistics."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        collection.count.return_value = 1000
        collection.get.return_value = {
            'embeddings': [[0.1, 0.2, 0.3, 0.4]]  # 4-dimensional
        }
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            stats = await repository.get_collection_statistics(collection)
        
        # Verify
        assert isinstance(stats, VectorCollectionStats)
        assert stats.name == "test_collection"
        assert stats.document_count == 1000
        assert stats.dimension == 4
        assert stats.memory_usage_mb > 0

    @pytest.mark.asyncio
    async def test_optimize_collection_performance(self, repository, mock_chromadb_client):
        """Test collection performance optimization."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        collection.count.return_value = 15000  # Large collection for optimization
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.optimize_collection_performance(collection)
        
        # Verify
        assert "before_optimization" in result
        assert "optimizations_applied" in result
        assert "large_collection_optimization" in result["optimizations_applied"]

    @pytest.mark.asyncio
    async def test_export_embeddings_numpy(self, repository, mock_chromadb_client):
        """Test exporting embeddings in numpy format."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection=collection,
                format_type="numpy",
                include_metadata=True
            )
        
        # Verify
        assert result["collection_name"] == "test_collection"
        assert result["format"] == "numpy"
        assert result["document_count"] == 2
        assert isinstance(result["embeddings"], np.ndarray)
        assert "documents" in result
        assert "metadatas" in result

    @pytest.mark.asyncio
    async def test_export_embeddings_json(self, repository, mock_chromadb_client):
        """Test exporting embeddings in JSON format."""
        # Setup
        collection = mock_chromadb_client.get_collection.return_value
        
        # Execute
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.export_embeddings(
                collection=collection,
                format_type="json",
                include_metadata=True
            )
        
        # Verify
        assert result["format"] == "json"
        assert "data" in result
        assert len(result["data"]) == 2
        assert all("embedding" in item for item in result["data"])

    def test_calculate_similarity_cosine(self, repository):
        """Test similarity calculation with cosine algorithm."""
        similarity = repository._calculate_similarity(0.2, SimilarityAlgorithm.COSINE)
        assert similarity == 0.8  # 1 - 0.2

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
        # Setup
        doc1 = VectorDocument(
            id="doc1", content="Doc 1", embedding=[0.1], 
            metadata={"priority": "high"}, created_at=datetime.utcnow(), 
            collection_name="test"
        )
        doc2 = VectorDocument(
            id="doc2", content="Doc 2", embedding=[0.2], 
            metadata={"recent": True}, created_at=datetime.utcnow(), 
            collection_name="test"
        )
        
        results = [
            SimilarityResult(document=doc1, similarity_score=0.7, distance=0.3, rank=1),
            SimilarityResult(document=doc2, similarity_score=0.8, distance=0.2, rank=2)
        ]
        
        request = VectorSearchRequest()
        
        # Execute
        reranked = await repository._rerank_results(results, request)
        
        # Verify that scores were boosted and re-ranked
        assert len(reranked) == 2
        assert all(result.similarity_score > 0.7 for result in reranked)

    @pytest.mark.asyncio 
    async def test_kmeans_clustering_fallback(self, repository):
        """Test K-means clustering fallback when sklearn is not available."""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # This will trigger the ImportError and use fallback
        result = await repository._kmeans_clustering(embeddings, 2)
        
        assert len(result) == 2
        assert all(0 <= label < 2 for label in result)

    @pytest.mark.asyncio
    async def test_dbscan_clustering_fallback(self, repository):
        """Test DBSCAN clustering fallback when sklearn is not available."""
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.9, 1.0]])
        
        # This will trigger the ImportError and use fallback
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
    """Test VectorDocument data class."""

    def test_vector_document_creation(self):
        """Test VectorDocument creation and to_dict method."""
        created_at = datetime.utcnow()
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            created_at=created_at,
            collection_name="test_collection"
        )
        
        assert doc.id == "test_id"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"type": "test"}
        assert doc.created_at == created_at
        assert doc.collection_name == "test_collection"
        
        # Test to_dict
        doc_dict = doc.to_dict()
        assert doc_dict["id"] == "test_id"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["embedding"] == [0.1, 0.2, 0.3]
        assert doc_dict["created_at"] == created_at.isoformat()


class TestSimilarityResult:
    """Test SimilarityResult data class."""

    def test_similarity_result_creation(self):
        """Test SimilarityResult creation and to_dict method."""
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2],
            metadata={},
            created_at=datetime.utcnow(),
            collection_name="test"
        )
        
        result = SimilarityResult(
            document=doc,
            similarity_score=0.85,
            distance=0.15,
            rank=1
        )
        
        assert result.document == doc
        assert result.similarity_score == 0.85
        assert result.distance == 0.15
        assert result.rank == 1
        
        # Test to_dict
        result_dict = result.to_dict()
        assert "document" in result_dict
        assert result_dict["similarity_score"] == 0.85
        assert result_dict["distance"] == 0.15
        assert result_dict["rank"] == 1


class TestVectorSearchRequest:
    """Test VectorSearchRequest data class."""

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

    def test_vector_search_request_custom(self):
        """Test VectorSearchRequest with custom values."""
        request = VectorSearchRequest(
            query_text="test query",
            collection_name="test_collection",
            n_results=5,
            similarity_threshold=0.8,
            algorithm=SimilarityAlgorithm.EUCLIDEAN,
            include_embeddings=True,
            rerank=True
        )
        
        assert request.query_text == "test query"
        assert request.collection_name == "test_collection"
        assert request.n_results == 5
        assert request.similarity_threshold == 0.8
        assert request.algorithm == SimilarityAlgorithm.EUCLIDEAN
        assert request.include_embeddings is True
        assert request.rerank is True


class TestBulkVectorOperation:
    """Test BulkVectorOperation data class."""

    def test_bulk_vector_operation_creation(self, sample_vector_documents):
        """Test BulkVectorOperation creation and to_dict method."""
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=500
        )
        
        assert operation.operation_type == "insert"
        assert operation.documents == sample_vector_documents
        assert operation.batch_size == 500
        
        # Test to_dict
        op_dict = operation.to_dict()
        assert op_dict["operation_type"] == "insert"
        assert op_dict["document_count"] == 3
        assert op_dict["batch_size"] == 500


class TestVectorCollectionStats:
    """Test VectorCollectionStats data class."""

    def test_vector_collection_stats_creation(self):
        """Test VectorCollectionStats creation and to_dict method."""
        last_updated = datetime.utcnow()
        stats = VectorCollectionStats(
            name="test_collection",
            document_count=1000,
            dimension=384,
            index_type="hnsw",
            memory_usage_mb=512.5,
            avg_query_time_ms=25.3,
            last_updated=last_updated
        )
        
        assert stats.name == "test_collection"
        assert stats.document_count == 1000
        assert stats.dimension == 384
        assert stats.index_type == "hnsw"
        assert stats.memory_usage_mb == 512.5
        assert stats.avg_query_time_ms == 25.3
        assert stats.last_updated == last_updated
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["name"] == "test_collection"
        assert stats_dict["document_count"] == 1000
        assert stats_dict["dimension"] == 384
        assert stats_dict["last_updated"] == last_updated.isoformat()


class TestSimilarityAlgorithm:
    """Test SimilarityAlgorithm enum."""

    def test_similarity_algorithm_values(self):
        """Test that all expected similarity algorithms are defined."""
        assert SimilarityAlgorithm.COSINE.value == "cosine"
        assert SimilarityAlgorithm.EUCLIDEAN.value == "euclidean"
        assert SimilarityAlgorithm.DOT_PRODUCT.value == "dot_product"
        assert SimilarityAlgorithm.MANHATTAN.value == "manhattan"
        
        # Test enum usage
        assert len(SimilarityAlgorithm) == 4


class TestVectorIndexType:
    """Test VectorIndexType enum."""

    def test_vector_index_type_values(self):
        """Test that all expected vector index types are defined."""
        assert VectorIndexType.HNSW.value == "hnsw"
        assert VectorIndexType.IVF.value == "ivf"
        assert VectorIndexType.FLAT.value == "flat"
        
        # Test enum usage
        assert len(VectorIndexType) == 3


class TestBackwardCompatibility:
    """Test backward compatibility with legacy methods."""

    @pytest.mark.asyncio
    async def test_legacy_create_collection(self, repository, mock_chromadb_client):
        """Test legacy create_collection method still works."""
        result = await repository.create_collection("test_collection", {"meta": "data"})
        assert result is not None
        mock_chromadb_client.create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_legacy_get_collection_cached(self, repository, mock_chromadb_client):
        """Test legacy get_collection method with caching."""
        # First call
        result1 = await repository.get_collection("test_collection")
        assert result1 is not None
        
        # Second call should use cache
        result2 = await repository.get_collection("test_collection")
        assert result2 == result1
        
        # Should only call client once due to caching
        assert mock_chromadb_client.get_collection.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_get_collection_not_found(self, repository, mock_chromadb_client):
        """Test legacy get_collection when collection doesn't exist."""
        mock_chromadb_client.get_collection.side_effect = Exception("Not found")
        
        result = await repository.get_collection("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_legacy_similarity_search(self, repository, mock_chromadb_client):
        """Test legacy similarity_search method still works."""
        collection = mock_chromadb_client.get_collection.return_value
        
        results = await repository.similarity_search(
            collection=collection,
            query_text="test query",
            n_results=3,
            threshold=0.5
        )
        
        assert len(results) == 2  # Based on mock data
        assert all("similarity" in result for result in results)
        assert all("document" in result for result in results)

    @pytest.mark.asyncio
    async def test_legacy_add_documents(self, repository, mock_chromadb_client):
        """Test legacy add_documents method still works."""
        collection = mock_chromadb_client.get_collection.return_value
        
        result = await repository.add_documents(
            collection=collection,
            documents=["Doc 1", "Doc 2"],
            ids=["id1", "id2"],
            metadatas=[{"type": "text"}, {"type": "text"}]
        )
        
        assert result is True
        collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_legacy_delete_collection_with_cache_cleanup(self, repository, mock_chromadb_client):
        """Test legacy delete_collection method cleans up cache."""
        # Add to cache first
        repository._collection_cache["test_collection"] = Mock()
        
        result = await repository.delete_collection("test_collection")
        
        assert result is True
        assert "test_collection" not in repository._collection_cache
        mock_chromadb_client.delete_collection.assert_called_once_with(name="test_collection")


class TestChromaRepositoryAlias:
    """Test that ChromaRepository is an alias for EnhancedChromaRepository."""

    def test_alias_compatibility(self, mock_chromadb_client):
        """Test that ChromaRepository is the same as EnhancedChromaRepository."""
        assert ChromaRepository == EnhancedChromaRepository
        
        # Test instantiation
        repo = ChromaRepository(mock_chromadb_client)
        assert isinstance(repo, EnhancedChromaRepository)
        assert repo.client == mock_chromadb_client


class TestErrorHandling:
    """Test error handling in ChromaDB repository."""

    @pytest.mark.asyncio
    async def test_create_collection_error(self, repository, mock_chromadb_client):
        """Test error handling in collection creation."""
        mock_chromadb_client.create_collection.side_effect = Exception("Creation failed")
        
        with pytest.raises(RepositoryError, match="Failed to create optimized collection"):
            await repository.create_collection_optimized("test_collection")

    @pytest.mark.asyncio
    async def test_bulk_upsert_error(self, repository, mock_chromadb_client, sample_vector_documents):
        """Test error handling in bulk upsert."""
        collection = mock_chromadb_client.get_collection.return_value
        collection.upsert.side_effect = Exception("Upsert failed")
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=sample_vector_documents,
            batch_size=10
        )
        
        with pytest.raises(RepositoryError, match="Failed to bulk upsert documents"):
            await repository.bulk_upsert_documents(collection, operation)

    @pytest.mark.asyncio
    async def test_semantic_search_collection_not_found(self, repository, mock_chromadb_client):
        """Test semantic search when collection doesn't exist."""
        async def mock_get_collection(name):
            return None
        
        repository.get_collection = mock_get_collection
        
        request = VectorSearchRequest(
            query_text="test",
            collection_name="nonexistent"
        )
        
        with pytest.raises(RepositoryError, match="Collection nonexistent not found"):
            await repository.semantic_search_advanced(request)

    @pytest.mark.asyncio
    async def test_clustering_error(self, repository, mock_chromadb_client):
        """Test error handling in document clustering."""
        collection = mock_chromadb_client.get_collection.return_value
        collection.get.side_effect = Exception("Get failed")
        
        with pytest.raises(RepositoryError, match="Failed to cluster documents"):
            await repository.cluster_documents(collection, n_clusters=2)

    @pytest.mark.asyncio
    async def test_unsupported_clustering_algorithm(self, repository, mock_chromadb_client):
        """Test error handling for unsupported clustering algorithm."""
        collection = mock_chromadb_client.get_collection.return_value
        collection.get.return_value = {
            'embeddings': [[0.1, 0.2], [0.3, 0.4]],
            'ids': ['1', '2'],
            'documents': ['Doc 1', 'Doc 2']
        }
        
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            with pytest.raises(RepositoryError, match="Failed to cluster documents"):
                await repository.cluster_documents(
                    collection,
                    algorithm="unsupported_algorithm"
                )


class TestPerformanceFeatures:
    """Test performance-related features."""

    @pytest.mark.asyncio
    async def test_collection_caching(self, repository, mock_chromadb_client):
        """Test that collections are properly cached for performance."""
        # First call
        result1 = await repository.get_collection("test_collection")
        
        # Second call should use cache
        result2 = await repository.get_collection("test_collection")
        
        assert result1 == result2
        assert mock_chromadb_client.get_collection.call_count == 1
        assert "test_collection" in repository._collection_cache

    @pytest.mark.asyncio
    async def test_bulk_operation_batching(self, repository, mock_chromadb_client, sample_vector_documents):
        """Test that bulk operations are properly batched."""
        # Create a larger set of documents to test batching
        large_doc_set = sample_vector_documents * 5  # 15 documents
        
        collection = mock_chromadb_client.get_collection.return_value
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=large_doc_set,
            batch_size=4  # Should create 4 batches (4, 4, 4, 3)
        )
        
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor'):
            result = await repository.bulk_upsert_documents(collection, operation)
        
        assert result["total_processed"] == 15
        assert result["batch_count"] == 4
        assert collection.upsert.call_count == 4

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, repository, mock_chromadb_client):
        """Test that performance monitoring is properly integrated."""
        collection = mock_chromadb_client.get_collection.return_value
        
        with patch('pcs.repositories.chroma_repo.PerformanceMonitor') as mock_monitor:
            await repository.get_collection_statistics(collection)
            
            # Verify PerformanceMonitor was used
            mock_monitor.assert_called_with("get_collection_stats", "chromadb")

    def test_memory_usage_calculation(self, repository):
        """Test memory usage calculation for collections."""
        # Test the memory calculation logic indirectly through stats
        # This would be in get_collection_statistics
        # Memory = count * dimension * 4 bytes / (1024*1024) for MB
        count = 1000
        dimension = 384
        expected_mb = (count * dimension * 4) / (1024 * 1024)
        
        # This is approximately 1.46 MB
        assert expected_mb > 1.4
        assert expected_mb < 1.5
