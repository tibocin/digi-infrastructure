"""
Filepath: pcs/tests/unit/test_qdrant_legacy.py
Purpose: Backward compatibility method tests for Qdrant
Related Components: Legacy API methods, compatibility layer
Tags: testing, legacy, compatibility, qdrant
"""

import pytest
from unittest.mock import Mock, patch
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


class TestLegacyCompatibility:
    """Test suite for backward compatibility methods."""
    
    @pytest.mark.asyncio
    async def test_get_collection_exists(self, repository, mock_qdrant_client):
        """Test legacy get_collection method."""
        result = await repository.get_collection("test_collection")
        assert result is True
        mock_qdrant_client.get_collection_info.assert_called_once_with("test_collection")
    
    @pytest.mark.asyncio
    async def test_get_collection_not_exists(self, repository, mock_qdrant_client):
        """Test legacy get_collection method when collection doesn't exist."""
        mock_qdrant_client.get_collection_info.side_effect = Exception("Not found")
        result = await repository.get_collection("nonexistent_collection")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_add_documents_legacy(self, repository, mock_qdrant_client, sample_vector_documents):
        """Test legacy add_documents method."""
        result = await repository.add_documents("test_collection", sample_vector_documents)
        
        # Should call upsert_documents internally
        assert result is not None
        # Verify that upsert_documents was called (this is the internal implementation)
    
    @pytest.mark.asyncio
    async def test_query_documents_legacy(self, repository, mock_qdrant_client):
        """Test legacy query_documents method."""
        # Mock search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        results = await repository.query_documents(
            "test_collection",
            [0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_get_documents_by_ids_legacy(self, repository):
        """Test legacy get_documents method (placeholder implementation)."""
        result = await repository.get_documents("test_collection", ["doc1", "doc2"])
        
        # Currently returns empty list (placeholder)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_similarity_search_legacy(self, repository, mock_qdrant_client):
        """Test legacy similarity_search method."""
        # Mock search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        results = await repository.similarity_search(
            "test_collection",
            [0.1, 0.2, 0.3],
            n_results=5,
            tenant_id="tenant1"
        )
        
        assert len(results) == 1
        assert results[0].similarity_score == 0.95
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_count_documents_legacy(self, repository, mock_qdrant_client):
        """Test legacy count_documents method."""
        result = await repository.count_documents("test_collection")
        assert result == 100  # From mock fixture
    
    @pytest.mark.asyncio
    async def test_count_documents_legacy_error(self, repository, mock_qdrant_client):
        """Test legacy count_documents method with error."""
        mock_qdrant_client.get_collection_stats.side_effect = Exception("Connection error")
        result = await repository.count_documents("test_collection")
        assert result == 0  # Should return 0 on error
    
    def test_delete_documents_legacy(self, repository, mock_qdrant_client):
        """Test legacy delete_documents method."""
        # Test with document_ids parameter
        result = repository.delete_documents(
            "test_collection",
            document_ids=["doc1", "doc2"]
        )
        mock_qdrant_client.delete_points.assert_called_once_with("test_collection", ["doc1", "doc2"])
        
        # Test with ids parameter (legacy)
        result = repository.delete_documents(
            "test_collection",
            ids=["doc3", "doc4"]
        )
        mock_qdrant_client.delete_points.assert_called_with("test_collection", ["doc3", "doc4"])
    
    def test_delete_documents_legacy_where_clause(self, repository, mock_qdrant_client):
        """Test legacy delete_documents method with where clause (warning)."""
        with patch('pcs.repositories.qdrant_repo.logger') as mock_logger:
            result = repository.delete_documents(
                "test_collection",
                where={"tenant_id": "tenant1"}
            )
            
            # Should log warning about where clause not implemented
            mock_logger.warning.assert_called_with(
                "Filter-based deletion not yet implemented, using document IDs"
            )
    
    def test_delete_documents_legacy_no_ids(self, repository, mock_qdrant_client):
        """Test legacy delete_documents method with no IDs."""
        result = repository.delete_documents("test_collection")
        mock_qdrant_client.delete_points.assert_called_once_with("test_collection", [])


class TestLegacyMethodSignatures:
    """Test suite for legacy method signature compatibility."""
    
    def test_legacy_method_availability(self, repository):
        """Test that all legacy methods are available."""
        assert hasattr(repository, 'get_collection')
        assert hasattr(repository, 'add_documents')
        assert hasattr(repository, 'query_documents')
        assert hasattr(repository, 'get_documents')
        assert hasattr(repository, 'similarity_search')
        assert hasattr(repository, 'count_documents')
        assert hasattr(repository, 'delete_documents')
    
    def test_legacy_method_callable(self, repository):
        """Test that all legacy methods are callable."""
        assert callable(repository.get_collection)
        assert callable(repository.add_documents)
        assert callable(repository.query_documents)
        assert callable(repository.get_documents)
        assert callable(repository.similarity_search)
        assert callable(repository.count_documents)
        assert callable(repository.delete_documents)


class TestLegacyParameterHandling:
    """Test suite for legacy parameter handling."""
    
    def test_delete_documents_parameter_precedence(self, repository, mock_qdrant_client):
        """Test parameter precedence in delete_documents."""
        # ids parameter should take precedence over document_ids
        result = repository.delete_documents(
            "test_collection",
            document_ids=["doc1"],
            ids=["doc2"]
        )
        
        # Should use ids parameter
        mock_qdrant_client.delete_points.assert_called_once_with("test_collection", ["doc2"])
    
    def test_delete_documents_empty_parameters(self, repository, mock_qdrant_client):
        """Test delete_documents with empty parameters."""
        result = repository.delete_documents("test_collection")
        mock_qdrant_client.delete_points.assert_called_once_with("test_collection", [])
        
        result = repository.delete_documents("test_collection", document_ids=[])
        mock_qdrant_client.delete_points.assert_called_with("test_collection", [])
        
        result = repository.delete_documents("test_collection", ids=[])
        mock_qdrant_client.delete_points.assert_called_with("test_collection", [])


class TestLegacyErrorHandling:
    """Test suite for legacy method error handling."""
    
    @pytest.mark.asyncio
    async def test_get_collection_error_handling(self, repository, mock_qdrant_client):
        """Test error handling in get_collection."""
        mock_qdrant_client.get_collection_info.side_effect = Exception("Test error")
        result = await repository.get_collection("test_collection")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count_documents_error_handling(self, repository, mock_qdrant_client):
        """Test error handling in count_documents."""
        mock_qdrant_client.get_collection_stats.side_effect = Exception("Test error")
        result = await repository.count_documents("test_collection")
        assert result == 0


class TestLegacyMethodDelegation:
    """Test suite for legacy method delegation to new methods."""
    
    @pytest.mark.asyncio
    async def test_add_documents_delegates_to_upsert(self, repository):
        """Test that add_documents delegates to upsert_documents."""
        with patch.object(repository, 'upsert_documents') as mock_upsert:
            mock_upsert.return_value = {"status": "success"}
            
            result = await repository.add_documents("test_collection", [])
            
            mock_upsert.assert_called_once_with("test_collection", [])
            assert result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_query_documents_delegates_to_search_similar(self, repository):
        """Test that query_documents delegates to search_similar."""
        with patch.object(repository, 'search_similar') as mock_search:
            mock_search.return_value = []
            
            result = await repository.query_documents("test_collection", [0.1, 0.2, 0.3])
            
            mock_search.assert_called_once()
            assert result == []
    
    @pytest.mark.asyncio
    async def test_similarity_search_delegates_to_search_similar(self, repository):
        """Test that similarity_search delegates to search_similar."""
        with patch.object(repository, 'search_similar') as mock_search:
            mock_search.return_value = []
            
            result = await repository.similarity_search("test_collection", [0.1, 0.2, 0.3])
            
            mock_search.assert_called_once()
            assert result == []
