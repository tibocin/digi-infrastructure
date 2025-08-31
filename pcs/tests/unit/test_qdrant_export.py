"""
Filepath: pcs/tests/unit/test_qdrant_export.py
Purpose: Export and data retrieval functionality tests for Qdrant
Related Components: Embedding export, data retrieval, tenant filtering
Tags: testing, export, retrieval, qdrant
"""

import pytest
import numpy as np
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


class TestExportFunctionality:
    """Test suite for export and data retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_export_embeddings_default_format(self, repository, mock_qdrant_client):
        """Test exporting embeddings in default format."""
        # Mock scroll results
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
        
        result = await repository.export_embeddings("test_collection")
        
        assert result["collection_name"] == "test_collection"
        assert len(result["embeddings"]) == 2
        assert np.array_equal(result["embeddings"][0], [0.1, 0.2, 0.3])
        assert np.array_equal(result["embeddings"][1], [0.4, 0.5, 0.6])
        assert result["format"] == "numpy"
        assert result["document_count"] == 2
        assert result["tenant_id"] is None
    
    @pytest.mark.asyncio
    async def test_export_embeddings_numpy_format(self, repository, mock_qdrant_client):
        """Test exporting embeddings in numpy format."""
        # Mock scroll results
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            format_type="numpy"
        )
        
        assert result["format"] == "numpy"
        assert isinstance(result["embeddings"], np.ndarray)
        assert result["embeddings"].shape == (1, 3)
    
    @pytest.mark.asyncio
    async def test_export_embeddings_json_format(self, repository, mock_qdrant_client):
        """Test exporting embeddings in JSON format."""
        # Mock scroll results
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            format_type="json"
        )
        
        assert result["format"] == "json"
        assert "data" in result
        assert len(result["data"]) == 1
        assert "embedding" in result["data"][0]
        assert "content" in result["data"][0]
        assert "tenant_id" in result["data"][0]
    
    @pytest.mark.asyncio
    async def test_export_embeddings_with_tenant_filter(self, repository, mock_qdrant_client):
        """Test exporting embeddings with tenant filtering."""
        # Mock scroll results with mixed tenants
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.4, 0.5, 0.6],
                payload={"content": "Doc 2", "tenant_id": "tenant2"}
            ),
            Mock(
                vector=[0.7, 0.8, 0.9],
                payload={"content": "Doc 3", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            tenant_id="tenant1"
        )
        
        assert result["tenant_id"] == "tenant1"
        assert len(result["embeddings"]) == 2  # Only tenant1 documents
        assert np.array_equal(result["embeddings"][0], [0.1, 0.2, 0.3])
        assert np.array_equal(result["embeddings"][1], [0.7, 0.8, 0.9])
    
    @pytest.mark.asyncio
    async def test_export_embeddings_empty_collection(self, repository, mock_qdrant_client):
        """Test exporting embeddings from empty collection."""
        mock_qdrant_client.scroll.return_value = ([], None)
        
        result = await repository.export_embeddings("empty_collection")
        
        assert result["embeddings"] == []
        assert result["documents"] == []
        assert result["metadatas"] == []
        assert result["document_count"] == 0
    
    @pytest.mark.asyncio
    async def test_export_embeddings_without_metadata(self, repository, mock_qdrant_client):
        """Test exporting embeddings without metadata."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            include_metadata=False
        )
        
        assert result["documents"] == []
        assert result["metadatas"] == []
        assert len(result["embeddings"]) == 1
    
    @pytest.mark.asyncio
    async def test_export_embeddings_numpy_import_error(self, repository, mock_qdrant_client):
        """Test exporting embeddings when numpy is not available."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            )
        ], None)
        
        with patch.dict('sys.modules', {'numpy': None}):
            result = await repository.export_embeddings(
                "test_collection",
                format_type="numpy"
            )
        
        # Should fall back to list format
        assert isinstance(result["embeddings"], list)
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]


class TestDataRetrieval:
    """Test suite for data retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics_basic(self, repository, mock_qdrant_client):
        """Test getting basic collection statistics."""
        result = await repository.get_collection_statistics("test_collection")
        
        assert isinstance(result, VectorCollectionStats)
        assert result.points_count == 100
        assert result.vectors_count == 100
        assert result.segments_count == 1
        assert result.status == "green"
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics_with_config(self, repository, mock_qdrant_client):
        """Test getting collection statistics with configuration."""
        result = await repository.get_collection_statistics("test_collection")
        
        assert result.config is not None
        assert result.config["name"] == "test_collection"
        assert result.config["dimension"] == 384
        assert result.config["document_count"] == 100
        assert result.config["memory_usage_mb"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics_tenant_specific(self, repository, mock_qdrant_client):
        """Test getting tenant-specific collection statistics."""
        # Mock scroll for tenant counting
        mock_qdrant_client.scroll.return_value = ([
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant2"})
        ], None)
        
        result = await repository.get_collection_statistics(
            "test_collection",
            tenant_id="tenant1"
        )
        
        assert result.points_count == 2  # Only tenant1 documents
        assert result.config["document_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_collection_statistics_error_handling(self, repository, mock_qdrant_client):
        """Test error handling in collection statistics."""
        mock_qdrant_client.get_collection_stats.side_effect = Exception("Connection error")
        
        with pytest.raises(Exception, match="Failed to get stats for test_collection: Connection error"):
            await repository.get_collection_statistics("test_collection")


class TestTenantFiltering:
    """Test suite for tenant filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_tenant_filtering_in_export(self, repository, mock_qdrant_client):
        """Test tenant filtering in export operations."""
        # Mock scroll results with mixed tenants
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.4, 0.5, 0.6],
                payload={"content": "Doc 2", "tenant_id": "tenant2"}
            )
        ], None)
        
        # Export for tenant1
        result_tenant1 = await repository.export_embeddings(
            "test_collection",
            tenant_id="tenant1"
        )
        assert len(result_tenant1["embeddings"]) == 1
        # Use numpy array comparison for proper array equality
        assert np.array_equal(result_tenant1["embeddings"][0], [0.1, 0.2, 0.3])
        
        # Export for tenant2
        result_tenant2 = await repository.export_embeddings(
            "test_collection",
            tenant_id="tenant2"
        )
        assert len(result_tenant2["embeddings"]) == 1
        # Use numpy array comparison for proper array equality
        assert np.array_equal(result_tenant2["embeddings"][0], [0.4, 0.5, 0.6])
    
    @pytest.mark.asyncio
    async def test_tenant_filtering_with_none_tenant(self, repository, mock_qdrant_client):
        """Test export behavior when tenant_id is None."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.4, 0.5, 0.6],
                payload={"content": "Doc 2", "tenant_id": "tenant2"}
            )
        ], None)
        
        result = await repository.export_embeddings("test_collection", tenant_id=None)
        
        # Should return all documents when tenant_id is None
        assert len(result["embeddings"]) == 2
        assert result["tenant_id"] is None


class TestFormatHandling:
    """Test suite for different output format handling."""
    
    @pytest.mark.asyncio
    async def test_list_format_default(self, repository, mock_qdrant_client):
        """Test default list format output."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            format_type="list"
        )
        
        assert isinstance(result["embeddings"], list)
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]
    
    @pytest.mark.asyncio
    async def test_json_format_structure(self, repository, mock_qdrant_client):
        """Test JSON format structure and content."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "category": "test"}
            )
        ], None)
        
        result = await repository.export_embeddings(
            "test_collection",
            format_type="json"
        )
        
        assert "data" in result
        assert len(result["data"]) == 1
        
        item = result["data"][0]
        assert "embedding" in item
        assert "content" in item
        assert "category" in item
        assert item["embedding"] == [0.1, 0.2, 0.3]
        assert item["content"] == "Doc 1"
        assert item["category"] == "test"
    
    @pytest.mark.asyncio
    async def test_invalid_format_handling(self, repository, mock_qdrant_client):
        """Test handling of invalid format types."""
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1"}
            )
        ], None)
        
        # Should default to list format for invalid types
        result = await repository.export_embeddings(
            "test_collection",
            format_type="invalid_format"
        )
        
        assert isinstance(result["embeddings"], list)
        assert result["format"] == "invalid_format"
