"""
Filepath: pcs/tests/integration/test_qdrant_simple_integration.py
Purpose: Simple integration tests for Qdrant repository with real or mocked backend
Related Components: EnhancedQdrantRepository, QdrantHTTPClient, vector operations
Tags: testing, integration, qdrant, vector-database, simple
"""

import pytest
import asyncio
from datetime import datetime, UTC
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository,
    QdrantHTTPClient,
    QdrantDistance,
    QdrantCollectionConfig,
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm,
    VectorSearchRequest
)


class TestQdrantSimpleIntegration:
    """Simple integration tests for Qdrant repository."""
    
    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"test_integration_{timestamp}"
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample vector documents for testing."""
        return [
            VectorDocument(
                id="doc1",
                content="This is a test document about artificial intelligence",
                embedding=[0.1] * 384,  # 384-dimensional vector
                metadata={"category": "AI", "language": "en", "source": "test"},
                created_at=datetime.now(UTC),
                collection_name="test_collection",
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc2",
                content="Another test document about machine learning",
                embedding=[0.2] * 384,  # 384-dimensional vector
                metadata={"category": "ML", "language": "en", "source": "test"},
                created_at=datetime.now(UTC),
                collection_name="test_collection",
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc3",
                content="A third document about data science",
                embedding=[0.3] * 384,  # 384-dimensional vector
                metadata={"category": "DS", "language": "en", "source": "test"},
                created_at=datetime.now(UTC),
                collection_name="test_collection",
                tenant_id="tenant2"
            )
        ]
    
    def test_repository_initialization(self):
        """Test repository initialization with default parameters."""
        print("\n📋 Test: Repository Initialization")
        
        try:
            repo = EnhancedQdrantRepository()
            print("✅ Repository initialized successfully")
            assert repo is not None
            assert hasattr(repo, 'client')
            assert hasattr(repo, 'health_check')
        except Exception as e:
            print(f"❌ Repository initialization failed: {e}")
            raise
    
    def test_health_check_mock(self):
        """Test health check with mocked client."""
        print("\n📋 Test: Health Check (Mocked)")
        
        try:
            # Create mock client
            mock_client = Mock()
            mock_client.health_check.return_value = {"version": "1.7.4", "status": "ok"}
            
            repo = EnhancedQdrantRepository(client=mock_client)
            health = repo.health_check()
            
            print(f"✅ Health check successful: {health}")
            assert health["version"] == "1.7.4"
            assert health["status"] == "ok"
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            raise
    
    def test_collection_operations_mock(self, test_collection_name):
        """Test collection operations with mocked client."""
        print(f"\n📋 Test: Collection Operations (Mocked) - '{test_collection_name}'")
        
        try:
            # Create mock client
            mock_client = Mock()
            mock_client.create_collection.return_value = True
            mock_client.get_collections.return_value = [
                {"name": test_collection_name, "vectors_count": 0}
            ]
            mock_client.delete_collection.return_value = True
            
            repo = EnhancedQdrantRepository(client=mock_client)
            
            # Test collection creation
            success = asyncio.run(repo.create_collection(
                collection_name=test_collection_name,
                vector_size=384,
                distance="cosine"
            ))
            print(f"✅ Collection creation: {success}")
            assert success is True
            
            # Test getting collections
            collections = repo.get_collections()
            print(f"✅ Collections retrieved: {len(collections)}")
            assert len(collections) == 1
            assert collections[0]["name"] == test_collection_name
            
            # Test collection deletion
            delete_success = repo.delete_collection(test_collection_name)
            print(f"✅ Collection deletion: {delete_success}")
            assert delete_success is True
            
        except Exception as e:
            print(f"❌ Collection operations failed: {e}")
            raise
    
    def test_document_operations_mock(self, test_collection_name, sample_documents):
        """Test document operations with mocked client."""
        print(f"\n📋 Test: Document Operations (Mocked) - '{test_collection_name}'")
        
        try:
            # Create mock client
            mock_client = Mock()
            mock_client.create_collection.return_value = True
            mock_client.upsert_points.return_value = {"result": {"status": "ok"}}
            mock_client.search_points.return_value = [
                Mock(
                    id=1,
                    score=0.95,
                    payload={
                        "content": "This is a test document about artificial intelligence",
                        "metadata": {"category": "AI", "language": "en", "source": "test"},
                        "created_at": datetime.now(UTC).isoformat(),
                        "tenant_id": "tenant1"
                    },
                    vector=[0.1] * 384
                )
            ]
            mock_client.delete_points.return_value = {"result": {"status": "ok"}}
            
            repo = EnhancedQdrantRepository(client=mock_client)
            
            # Create collection
            repo.create_collection(
                collection_name=test_collection_name,
                vector_size=384,
                distance="cosine"
            )
            print("✅ Collection created for testing")
            
            # Upsert documents
            result = repo.upsert_documents(test_collection_name, sample_documents)
            print(f"✅ Documents upserted: {result}")
            assert result["result"]["status"] == "ok"
            
            # Search for similar documents
            query_vector = [0.15] * 384  # Similar to doc1 and doc2
            results = repo.search_similar(
                collection_name=test_collection_name,
                query_embedding=query_vector,
                limit=5,
                tenant_id="tenant1"
            )
            print(f"✅ Search results: {len(results)} documents found")
            assert len(results) >= 1
            
            # Delete documents
            delete_result = repo.delete_documents(
                test_collection_name,
                [doc.id for doc in sample_documents]
            )
            print(f"✅ Documents deleted: {delete_result}")
            assert delete_result["result"]["status"] == "ok"
            
        except Exception as e:
            print(f"❌ Document operations failed: {e}")
            raise
    
    def test_semantic_search_mock(self, test_collection_name):
        """Test semantic search with mocked client."""
        print(f"\n📋 Test: Semantic Search (Mocked) - '{test_collection_name}'")
        
        try:
            # Create mock client
            mock_client = Mock()
            mock_client.search_points.return_value = [
                Mock(
                    id=1,
                    score=0.92,
                    payload={
                        "content": "Test document content",
                        "metadata": {"category": "AI"},
                        "created_at": datetime.now(UTC).isoformat(),
                        "tenant_id": "tenant1"
                    },
                    vector=[0.1] * 384
                )
            ]
            
            repo = EnhancedQdrantRepository(client=mock_client)
            
            # Test advanced semantic search
            request = VectorSearchRequest(
                query_embedding=[0.1] * 384,
                collection_name=test_collection_name,
                n_results=5,
                similarity_threshold=0.8,
                algorithm=SimilarityAlgorithm.COSINE,
                tenant_id="tenant1"
            )
            
            results = asyncio.run(repo.semantic_search_advanced(request))
            print(f"✅ Semantic search returned {len(results)} results")
            assert len(results) >= 1
            assert isinstance(results[0], SimilarityResult)
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            raise
    
    def test_multi_tenancy_mock(self, test_collection_name):
        """Test multi-tenancy features with mocked client."""
        print(f"\n📋 Test: Multi-Tenancy (Mocked) - '{test_collection_name}'")
        
        try:
            # Create mock client
            mock_client = Mock()
            mock_client.search_points.return_value = [
                Mock(
                    id=1,
                    score=0.95,
                    payload={
                        "content": "Tenant-specific document",
                        "metadata": {"category": "test"},
                        "created_at": datetime.now(UTC).isoformat(),
                        "tenant_id": "tenant1"
                    },
                    vector=[0.1] * 384
                )
            ]
            
            repo = EnhancedQdrantRepository(client=mock_client)
            
            # Test tenant isolation in search
            request = VectorSearchRequest(
                query_embedding=[0.1] * 384,
                collection_name=test_collection_name,
                tenant_id="tenant1"
            )
            
            results = asyncio.run(repo.semantic_search_advanced(request))
            print(f"✅ Tenant-specific search returned {len(results)} results")
            assert len(results) >= 1
            
            # Test export with tenant
            export_result = asyncio.run(repo.export_embeddings(
                collection_name=test_collection_name,
                tenant_id="tenant1"
            ))
            print(f"✅ Export with tenant: {export_result}")
            assert export_result["tenant_id"] == "tenant1"
            
        except Exception as e:
            print(f"❌ Multi-tenancy test failed: {e}")
            raise
    
    def test_error_handling_mock(self):
        """Test error handling with mocked client."""
        print("\n📋 Test: Error Handling (Mocked)")
        
        try:
            # Create mock client that raises errors
            mock_client = Mock()
            mock_client.health_check.side_effect = Exception("Connection failed")
            
            repo = EnhancedQdrantRepository(client=mock_client)
            
            # Test error handling
            with pytest.raises(Exception) as exc_info:
                repo.health_check()
            
            print(f"✅ Error properly caught: {exc_info.value}")
            assert "Connection failed" in str(exc_info.value)
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            raise
    
    def test_similarity_calculations(self):
        """Test similarity calculation methods."""
        print("\n📋 Test: Similarity Calculations")
        
        try:
            repo = EnhancedQdrantRepository()
            
            # Test cosine similarity
            cosine_sim = repo._calculate_similarity(0.8, SimilarityAlgorithm.COSINE)
            print(f"✅ Cosine similarity: {cosine_sim}")
            assert cosine_sim == 0.8
            
            # Test Euclidean similarity
            euclidean_sim = repo._calculate_similarity(2.0, SimilarityAlgorithm.EUCLIDEAN)
            expected_euclidean = 1.0 / (1.0 + 2.0)
            print(f"✅ Euclidean similarity: {euclidean_sim} (expected: {expected_euclidean})")
            assert abs(euclidean_sim - expected_euclidean) < 1e-6
            
            # Test Manhattan similarity
            manhattan_sim = repo._calculate_similarity(3.0, SimilarityAlgorithm.MANHATTAN)
            expected_manhattan = 1.0 / (1.0 + 3.0)
            print(f"✅ Manhattan similarity: {manhattan_sim} (expected: {expected_manhattan})")
            assert abs(manhattan_sim - expected_manhattan) < 1e-6
            
        except Exception as e:
            print(f"❌ Similarity calculations failed: {e}")
            raise


if __name__ == "__main__":
    print("🚀 Starting Qdrant Simple Integration Tests")
    print("=" * 60)
    
    # Run tests manually
    test_instance = TestQdrantSimpleIntegration()
    
    try:
        test_instance.test_repository_initialization()
        test_instance.test_health_check_mock()
        test_instance.test_collection_operations_mock(f"test_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        test_instance.test_document_operations_mock(
            f"test_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_instance.sample_documents()
        )
        test_instance.test_semantic_search_mock(f"test_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        test_instance.test_multi_tenancy_mock(f"test_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        test_instance.test_error_handling_mock()
        test_instance.test_similarity_calculations()
        
        print("\n" + "=" * 60)
        print("🎉 All manual tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
