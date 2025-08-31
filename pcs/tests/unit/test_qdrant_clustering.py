"""
Filepath: pcs/tests/unit/test_qdrant_clustering.py
Purpose: Clustering functionality tests for Qdrant
Related Components: K-means, DBSCAN, document clustering
Tags: testing, clustering, algorithms, qdrant
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


class TestClusteringFunctionality:
    """Test suite for clustering functionality."""
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering_basic(self, repository):
        """Test basic K-means clustering."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7]
        ]
        
        result = await repository._kmeans_clustering(embeddings, n_clusters=3)
        
        assert result["algorithm"] == "kmeans"
        assert result["clusters"] == 3
        assert len(result["centroids"]) == 3
        assert len(result["labels"]) == 5
        assert all(label < 3 for label in result["labels"])
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering_empty_input(self, repository):
        """Test K-means clustering with empty input."""
        result = await repository._kmeans_clustering([], n_clusters=3)
        
        assert result["algorithm"] == "kmeans"
        assert result["clusters"] == 3
        assert result["centroids"] == []
        assert result["labels"] == []
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering_fewer_points_than_clusters(self, repository):
        """Test K-means clustering when points < clusters."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        result = await repository._kmeans_clustering(embeddings, n_clusters=5)
        
        assert result["algorithm"] == "kmeans"
        assert result["clusters"] == 5
        assert len(result["centroids"]) == 2  # Limited by available points
        assert len(result["labels"]) == 2
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering_basic(self, repository):
        """Test basic DBSCAN clustering."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        result = await repository._dbscan_clustering(embeddings)
        
        assert result["algorithm"] == "dbscan"
        assert result["clusters"] == 1
        assert len(result["centroids"]) == 1
        assert len(result["labels"]) == 3
        assert all(label == 0 for label in result["labels"])
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering_empty_input(self, repository):
        """Test DBSCAN clustering with empty input."""
        result = await repository._dbscan_clustering([])
        
        assert result["algorithm"] == "dbscan"
        assert result["clusters"] == 1
        assert result["centroids"] == []
        assert result["labels"] == []
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering_single_point(self, repository):
        """Test DBSCAN clustering with single point."""
        embeddings = [[0.1, 0.2, 0.3]]
        
        result = await repository._dbscan_clustering(embeddings)
        
        assert result["algorithm"] == "dbscan"
        assert result["clusters"] == 1
        assert len(result["centroids"]) == 1
        assert result["labels"] == [0]


class TestClusteringIntegration:
    """Test suite for clustering integration with repository."""
    
    @pytest.mark.asyncio
    async def test_clustering_with_exported_embeddings(self, repository, mock_qdrant_client):
        """Test clustering with embeddings exported from collection."""
        # Mock scroll results
        mock_qdrant_client.scroll.return_value = ([
            Mock(
                vector=[0.1, 0.2, 0.3],
                payload={"content": "Doc 1", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.4, 0.5, 0.6],
                payload={"content": "Doc 2", "tenant_id": "tenant1"}
            ),
            Mock(
                vector=[0.7, 0.8, 0.9],
                payload={"content": "Doc 3", "tenant_id": "tenant1"}
            )
        ], None)
        
        # Export embeddings
        export_result = await repository.export_embeddings("test_collection")
        embeddings = export_result["embeddings"]
        
        # Apply clustering
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=2)
        dbscan_result = await repository._dbscan_clustering(embeddings)
        
        assert kmeans_result["clusters"] == 2
        assert dbscan_result["clusters"] == 1
        assert len(kmeans_result["labels"]) == 3
        assert len(dbscan_result["labels"]) == 3
    
    @pytest.mark.asyncio
    async def test_clustering_with_tenant_filtered_embeddings(self, repository, mock_qdrant_client):
        """Test clustering with tenant-filtered embeddings."""
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
        
        # Export embeddings for tenant1 only
        export_result = await repository.export_embeddings(
            "test_collection",
            tenant_id="tenant1"
        )
        embeddings = export_result["embeddings"]
        
        # Should only have tenant1 embeddings
        assert len(embeddings) == 2
        
        # Apply clustering
        result = await repository._kmeans_clustering(embeddings, n_clusters=2)
        
        assert result["clusters"] == 2
        assert len(result["labels"]) == 2


class TestClusteringAlgorithms:
    """Test suite for different clustering algorithms."""
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering_consistency(self, repository):
        """Test that K-means clustering produces consistent results."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        # Run clustering multiple times
        result1 = await repository._kmeans_clustering(embeddings, n_clusters=2)
        result2 = await repository._kmeans_clustering(embeddings, n_clusters=2)
        
        # Results should be consistent (same structure)
        assert result1["algorithm"] == result2["algorithm"]
        assert result1["clusters"] == result2["clusters"]
        assert len(result1["centroids"]) == len(result2["centroids"])
        assert len(result1["labels"]) == len(result2["labels"])
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering_consistency(self, repository):
        """Test that DBSCAN clustering produces consistent results."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        # Run clustering multiple times
        result1 = await repository._dbscan_clustering(embeddings)
        result2 = await repository._dbscan_clustering(embeddings)
        
        # Results should be consistent (same structure)
        assert result1["algorithm"] == result2["algorithm"]
        assert result1["clusters"] == result2["clusters"]
        assert len(result1["centroids"]) == len(result2["centroids"])
        assert len(result1["labels"]) == len(result2["labels"])
    
    @pytest.mark.asyncio
    async def test_clustering_algorithm_comparison(self, repository):
        """Test comparison between different clustering algorithms."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7]
        ]
        
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=3)
        dbscan_result = await repository._dbscan_clustering(embeddings)
        
        # K-means should produce more clusters than DBSCAN for this data
        assert kmeans_result["clusters"] >= dbscan_result["clusters"]
        
        # Both should produce the same number of labels
        assert len(kmeans_result["labels"]) == len(dbscan_result["labels"])
        assert len(kmeans_result["labels"]) == len(embeddings)


class TestClusteringEdgeCases:
    """Test suite for clustering edge cases."""
    
    @pytest.mark.asyncio
    async def test_clustering_with_duplicate_embeddings(self, repository):
        """Test clustering with duplicate embeddings."""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],  # Duplicate
            [0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6]   # Duplicate
        ]
        
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=2)
        dbscan_result = await repository._dbscan_clustering(embeddings)
        
        assert len(kmeans_result["labels"]) == 4
        assert len(dbscan_result["labels"]) == 4
    
    @pytest.mark.asyncio
    async def test_clustering_with_single_dimension(self, repository):
        """Test clustering with single-dimensional embeddings."""
        embeddings = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=2)
        dbscan_result = await repository._dbscan_clustering(embeddings)
        
        assert len(kmeans_result["labels"]) == 5
        assert len(dbscan_result["labels"]) == 5
    
    @pytest.mark.asyncio
    async def test_clustering_with_high_dimensional_embeddings(self, repository):
        """Test clustering with high-dimensional embeddings."""
        # Create 1536-dimensional embeddings (like OpenAI ada-002)
        embeddings = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536
        ]
        
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=2)
        dbscan_result = await repository._dbscan_clustering(embeddings)
        
        assert len(kmeans_result["labels"]) == 3
        assert len(dbscan_result["labels"]) == 3
        assert len(kmeans_result["centroids"][0]) == 1536
        assert len(dbscan_result["centroids"][0]) == 1536


class TestClusteringPerformance:
    """Test suite for clustering performance."""
    
    @pytest.mark.asyncio
    async def test_clustering_performance_with_large_dataset(self, repository):
        """Test clustering performance with larger dataset."""
        import time
        
        # Create larger dataset
        embeddings = [[i * 0.1, i * 0.2, i * 0.3] for i in range(100)]
        
        # Measure K-means performance
        start_time = time.time()
        kmeans_result = await repository._kmeans_clustering(embeddings, n_clusters=10)
        kmeans_time = time.time() - start_time
        
        # Measure DBSCAN performance
        start_time = time.time()
        dbscan_result = await repository._dbscan_clustering(embeddings)
        dbscan_time = time.time() - start_time
        
        # Both should complete in reasonable time
        assert kmeans_time < 1.0  # Should complete within 1 second
        assert dbscan_time < 1.0  # Should complete within 1 second
        
        # Verify results
        assert len(kmeans_result["labels"]) == 100
        assert len(dbscan_result["labels"]) == 100
        assert kmeans_result["clusters"] == 10
        assert dbscan_result["clusters"] == 1
    
    @pytest.mark.asyncio
    async def test_clustering_memory_usage(self, repository):
        """Test clustering memory usage (basic check)."""
        import sys
        
        # Create embeddings
        embeddings = [[i * 0.1, i * 0.2, i * 0.3] for i in range(50)]
        
        # Get initial memory usage
        initial_memory = sys.getsizeof(embeddings)
        
        # Run clustering
        result = await repository._kmeans_clustering(embeddings, n_clusters=5)
        
        # Get final memory usage
        final_memory = sys.getsizeof(embeddings) + sys.getsizeof(result)
        
        # Memory usage should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 10000  # Less than 10KB increase


class TestClusteringErrorHandling:
    """Test suite for clustering error handling."""
    
    @pytest.mark.asyncio
    async def test_clustering_with_invalid_parameters(self, repository):
        """Test clustering with invalid parameters."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Test with invalid n_clusters
        with pytest.raises(ValueError):
            await repository._kmeans_clustering(embeddings, n_clusters=0)
        
        with pytest.raises(ValueError):
            await repository._kmeans_clustering(embeddings, n_clusters=-1)
    
    @pytest.mark.asyncio
    async def test_clustering_with_malformed_embeddings(self, repository):
        """Test clustering with malformed embeddings."""
        # Test with empty embeddings
        result = await repository._kmeans_clustering([], n_clusters=3)
        assert result["clusters"] == 3
        assert result["centroids"] == []
        assert result["labels"] == []
        
        # Test with None embeddings
        with pytest.raises(TypeError):
            await repository._kmeans_clustering(None, n_clusters=3)
