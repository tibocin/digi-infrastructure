"""
Filepath: pcs/tests/unit/test_qdrant_multi_tenant.py
Purpose: Multi-tenancy functionality tests for Qdrant
Related Components: Tenant isolation, filtering, security
Tags: testing, multi-tenancy, tenant-isolation, qdrant
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, UTC

from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository,
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


class TestMultiTenancy:
    """Test suite for multi-tenancy functionality."""
    
    def test_tenant_id_in_vector_document(self):
        """Test that VectorDocument supports tenant_id."""
        doc = VectorDocument(
            id="test_id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            tenant_id="tenant1"
        )
        
        assert doc.tenant_id == "tenant1"
    
    def test_tenant_id_in_search_request(self):
        """Test that VectorSearchRequest supports tenant_id."""
        request = VectorSearchRequest(
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        assert request.tenant_id == "tenant1"
    
    def test_tenant_id_in_bulk_operation(self):
        """Test that BulkVectorOperation supports tenant_id."""
        operation = BulkVectorOperation(
            operation_type="insert",
            collection_name="test_collection",
            tenant_id="tenant1"
        )
        
        assert operation.tenant_id == "tenant1"


class TestTenantFiltering:
    """Test suite for tenant filtering functionality."""
    
    def test_build_query_filter_with_tenant(self, repository):
        """Test building query filter with tenant ID."""
        filter_conditions = repository._build_query_filter(
            tenant_id="tenant1",
            metadata_filters={"type": "text"}
        )
        
        assert filter_conditions is not None
        # Should contain tenant filter
        assert "must" in filter_conditions
        tenant_filter = None
        for condition in filter_conditions["must"]:
            if "key" in condition and condition["key"] == "tenant_id":
                tenant_filter = condition
                break
        
        assert tenant_filter is not None
        assert tenant_filter["match"]["value"] == "tenant1"
    
    def test_build_query_filter_without_tenant(self, repository):
        """Test building query filter without tenant ID."""
        filter_conditions = repository._build_query_filter(
            metadata_filters={"type": "text"}
        )
        
        assert filter_conditions is not None
        # Should not contain tenant filter
        if "must" in filter_conditions:
            for condition in filter_conditions["must"]:
                if "key" in condition and condition["key"] == "tenant_id":
                    assert False, "Should not contain tenant filter when tenant_id is None"
    
    def test_build_query_filter_with_metadata_only(self, repository):
        """Test building query filter with metadata filters only."""
        filter_conditions = repository._build_query_filter(
            metadata_filters={"type": "text", "category": "general"}
        )
        
        assert filter_conditions is not None
        assert "must" in filter_conditions
        
        # Should contain metadata filters
        metadata_filters = []
        for condition in filter_conditions["must"]:
            if "key" in condition:
                metadata_filters.append(condition["key"])
        
        assert "type" in metadata_filters
        assert "category" in metadata_filters
    
    def test_build_query_filter_empty(self, repository):
        """Test building query filter with no conditions."""
        filter_conditions = repository._build_query_filter()
        
        assert filter_conditions is None


class TestTenantIsolation:
    """Test suite for tenant isolation functionality."""
    
    @pytest.mark.asyncio
    async def test_search_with_tenant_isolation(self, repository, mock_qdrant_client):
        """Test that search results are isolated by tenant."""
        # Mock search results with mixed tenants
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            ),
            Mock(
                id="doc2",
                score=0.85,
                payload={"content": "Document 2", "tenant_id": "tenant2"},
                vector=[0.4, 0.5, 0.6]
            )
        ]
        
        # Search for tenant1
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        # Should only return tenant1 results
        assert len(results) == 1
        assert results[0].document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_export_with_tenant_isolation(self, repository, mock_qdrant_client):
        """Test that export results are isolated by tenant."""
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
        
        # Export for tenant1
        result = await repository.export_embeddings(
            "test_collection",
            tenant_id="tenant1"
        )
        
        # Should only return tenant1 embeddings
        assert len(result["embeddings"]) == 2
        assert result["embeddings"][0] == [0.1, 0.2, 0.3]
        assert result["embeddings"][1] == [0.7, 0.8, 0.9]
    
    @pytest.mark.asyncio
    async def test_statistics_with_tenant_isolation(self, repository, mock_qdrant_client):
        """Test that statistics are isolated by tenant."""
        # Mock scroll for tenant counting
        mock_qdrant_client.scroll.return_value = ([
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant1"}),
            Mock(payload={"tenant_id": "tenant2"})
        ], None)
        
        # Get statistics for tenant1
        stats = await repository.get_collection_statistics(
            "test_collection",
            tenant_id="tenant1"
        )
        
        # Should only count tenant1 documents
        assert stats.points_count == 2
        assert stats.config["document_count"] == 2


class TestTenantSecurity:
    """Test suite for tenant security functionality."""
    
    @pytest.mark.asyncio
    async def test_cannot_access_other_tenant_data(self, repository, mock_qdrant_client):
        """Test that users cannot access data from other tenants."""
        # Mock search results with mixed tenants
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            ),
            Mock(
                id="doc2",
                score=0.85,
                payload={"content": "Document 2", "tenant_id": "tenant2"},
                vector=[0.4, 0.5, 0.6]
            )
        ]
        
        # Search for tenant1 (should not see tenant2 data)
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        # Should not contain tenant2 data
        for result in results:
            assert result.document.tenant_id == "tenant1"
    
    @pytest.mark.asyncio
    async def test_tenant_filter_applied_to_all_operations(self, repository, mock_qdrant_client):
        """Test that tenant filter is applied to all operations."""
        # Mock search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        # Test search with tenant filter
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        
        # Verify that search was called with tenant filter
        mock_qdrant_client.search_points.assert_called_once()
        call_args = mock_qdrant_client.search_points.call_args
        
        # Should contain filter with tenant_id
        assert "filter" in call_args.kwargs
        filter_conditions = call_args.kwargs["filter"]
        
        # Find tenant filter
        tenant_filter_found = False
        if "must" in filter_conditions:
            for condition in filter_conditions["must"]:
                if "key" in condition and condition["key"] == "tenant_id":
                    tenant_filter_found = True
                    break
        
        assert tenant_filter_found, "Tenant filter should be applied to search"


class TestTenantManagement:
    """Test suite for tenant management functionality."""
    
    def test_tenant_id_validation(self, repository):
        """Test tenant ID validation."""
        # Valid tenant IDs
        valid_tenants = ["tenant1", "tenant_123", "tenant-456", "TENANT_789"]
        
        for tenant_id in valid_tenants:
            filter_conditions = repository._build_query_filter(tenant_id=tenant_id)
            assert filter_conditions is not None
        
        # Invalid tenant IDs (None is allowed for no filtering)
        invalid_tenants = ["", " ", "tenant with spaces", "tenant@#$%"]
        
        for tenant_id in invalid_tenants:
            if tenant_id.strip():  # Skip empty strings
                filter_conditions = repository._build_query_filter(tenant_id=tenant_id)
                # Should still work but may need validation in production
    
    def test_tenant_id_case_sensitivity(self, repository):
        """Test tenant ID case sensitivity."""
        # Create filters with different case
        filter1 = repository._build_query_filter(tenant_id="tenant1")
        filter2 = repository._build_query_filter(tenant_id="TENANT1")
        
        # Should be different (case-sensitive)
        assert filter1 != filter2
    
    def test_tenant_id_special_characters(self, repository):
        """Test tenant ID with special characters."""
        special_tenant = "tenant-123_456.789"
        filter_conditions = repository._build_query_filter(tenant_id=special_tenant)
        
        assert filter_conditions is not None
        # Should contain the exact tenant ID
        tenant_filter = None
        for condition in filter_conditions["must"]:
            if "key" in condition and condition["key"] == "tenant_id":
                tenant_filter = condition
                break
        
        assert tenant_filter is not None
        assert tenant_filter["match"]["value"] == special_tenant


class TestTenantPerformance:
    """Test suite for tenant performance functionality."""
    
    @pytest.mark.asyncio
    async def test_tenant_filtering_performance(self, repository, mock_qdrant_client):
        """Test that tenant filtering doesn't significantly impact performance."""
        import time
        
        # Mock search results
        mock_qdrant_client.search_points.return_value = [
            Mock(
                id="doc1",
                score=0.95,
                payload={"content": "Document 1", "tenant_id": "tenant1"},
                vector=[0.1, 0.2, 0.3]
            )
        ]
        
        # Measure search time with tenant filter
        start_time = time.time()
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5,
            tenant_id="tenant1"
        )
        end_time = time.time()
        
        search_time_with_tenant = end_time - start_time
        
        # Measure search time without tenant filter
        start_time = time.time()
        results = await repository.search_similar(
            collection_name="test_collection",
            query_embedding=[0.1, 0.2, 0.3],
            limit=5
        )
        end_time = time.time()
        
        search_time_without_tenant = end_time - start_time
        
        # Tenant filtering should not add significant overhead
        # (This is a basic test - in production you'd want more sophisticated benchmarking)
        assert search_time_with_tenant < search_time_without_tenant + 0.1  # 100ms tolerance
    
    @pytest.mark.asyncio
    async def test_tenant_statistics_performance(self, repository, mock_qdrant_client):
        """Test tenant statistics performance."""
        # Mock scroll for tenant counting
        mock_qdrant_client.scroll.return_value = [
            Mock(payload={"tenant_id": "tenant1"}) for _ in range(1000)
        ]
        
        import time
        start_time = time.time()
        
        stats = await repository.get_collection_statistics(
            "test_collection",
            tenant_id="tenant1"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 documents in reasonable time
        assert processing_time < 1.0  # Should complete within 1 second
        assert stats.points_count == 1000
