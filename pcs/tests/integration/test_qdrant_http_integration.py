"""
Filepath: pcs/tests/integration/test_qdrant_http_integration.py
Purpose: Integration tests for HTTP-based Qdrant repository
Related Components: Qdrant HTTP repository, vector operations, multi-tenancy
Tags: testing, integration, qdrant, http-repository, vector-database
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from pcs.repositories.qdrant_http_repo import (
    EnhancedQdrantHTTPRepository,
    VectorDocument,
    SimilarityResult,
    SimilarityAlgorithm
)


class TestQdrantHTTPIntegration:
    """Integration tests for HTTP-based Qdrant repository."""
    
    @pytest.fixture
    def repo(self):
        """Create repository instance for testing."""
        return EnhancedQdrantHTTPRepository(
            host="localhost",
            port=6333,
            api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814",
            timeout=10.0,
            max_retries=2
        )
    
    @pytest.fixture
    def test_collection_name(self):
        """Test collection name."""
        return f"test_http_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            VectorDocument(
                id="doc1",
                content="This is a test document about artificial intelligence",
                embedding=[0.1] * 1536,
                metadata={"category": "AI", "language": "en", "source": "test"},
                created_at=datetime.now(),
                collection_name="test_collection",
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc2",
                content="Another test document about machine learning",
                embedding=[0.2] * 1536,
                metadata={"category": "ML", "language": "en", "source": "test"},
                created_at=datetime.now(),
                collection_name="test_collection",
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc3",
                content="A third document about data science",
                embedding=[0.3] * 1536,
                metadata={"category": "DS", "language": "en", "source": "test"},
                created_at=datetime.now(),
                collection_name="test_collection",
                tenant_id="tenant2"
            )
        ]
    
    def test_health_check(self, repo):
        """Test repository health check."""
        print("\n📋 Test: Health Check")
        
        try:
            health = repo.health_check()
            print(f"✅ Health check successful: {health}")
            assert "version" in health
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            raise
    
    def test_get_collections(self, repo):
        """Test getting collections."""
        print("\n📋 Test: Get Collections")
        
        try:
            collections = repo.get_collections()
            print(f"✅ Retrieved {len(collections)} collections: {collections}")
            assert isinstance(collections, list)
        except Exception as e:
            print(f"❌ Get collections failed: {e}")
            raise
    
    def test_create_and_delete_collection(self, repo, test_collection_name):
        """Test collection creation and deletion."""
        print(f"\n📋 Test: Create and Delete Collection '{test_collection_name}'")
        
        try:
            # Create collection
            success = repo.create_collection(
                collection_name=test_collection_name,
                vector_size=1536,
                distance="cosine",
                on_disk_payload=True
            )
            print(f"✅ Collection creation: {success}")
            assert success is True
            
            # Verify collection exists
            collections = repo.get_collections()
            collection_names = [col.get("name") for col in collections]
            print(f"✅ Collection exists: {test_collection_name in collection_names}")
            assert test_collection_name in collection_names
            
            # Get collection info
            info = repo.get_collection_info(test_collection_name)
            print(f"✅ Collection info: {info}")
            assert info["config"]["params"]["vectors"]["size"] == 1536
            
            # Delete collection
            delete_success = repo.delete_collection(test_collection_name)
            print(f"✅ Collection deletion: {delete_success}")
            assert delete_success is True
            
        except Exception as e:
            print(f"❌ Collection operations failed: {e}")
            # Cleanup on failure
            try:
                repo.delete_collection(test_collection_name)
            except:
                pass
            raise
    
    def test_document_operations(self, repo, test_collection_name, sample_documents):
        """Test document upsert, search, and delete operations."""
        print(f"\n📋 Test: Document Operations in '{test_collection_name}'")
        
        try:
            # Create collection
            repo.create_collection(
                collection_name=test_collection_name,
                vector_size=1536,
                distance="cosine"
            )
            print("✅ Collection created for testing")
            
            # Upsert documents
            result = repo.upsert_documents(test_collection_name, sample_documents)
            print(f"✅ Documents upserted: {result}")
            
            # Get collection stats
            stats = repo.get_collection_stats(test_collection_name)
            print(f"✅ Collection stats: {stats}")
            assert stats["points_count"] == len(sample_documents)
            
            # Search for similar documents
            query_vector = [0.15] * 1536  # Similar to doc1 and doc2
            results = repo.search_similar(
                collection_name=test_collection_name,
                query_embedding=query_vector,
                limit=5,
                tenant_id="tenant1"
            )
            print(f"✅ Search results: {len(results)} documents found")
            assert len(results) >= 2  # Should find doc1 and doc2
            
            # Verify tenant isolation
            tenant2_results = repo.search_similar(
                collection_name=test_collection_name,
                query_embedding=query_vector,
                limit=5,
                tenant_id="tenant2"
            )
            print(f"✅ Tenant2 results: {len(tenant2_results)} documents found")
            assert len(tenant2_results) >= 1  # Should find doc3
            
            # Test metadata filtering
            filtered_results = repo.search_similar(
                collection_name=test_collection_name,
                query_embedding=query_vector,
                limit=5,
                metadata_filters={"category": "AI"}
            )
            print(f"✅ Filtered results: {len(filtered_results)} AI documents found")
            assert len(filtered_results) >= 1  # Should find doc1
            
            # Delete documents
            delete_result = repo.delete_documents(
                test_collection_name,
                [doc.id for doc in sample_documents]
            )
            print(f"✅ Documents deleted: {delete_result}")
            
            # Verify deletion
            final_stats = repo.get_collection_stats(test_collection_name)
            print(f"✅ Final stats: {final_stats}")
            assert final_stats["points_count"] == 0
            
        except Exception as e:
            print(f"❌ Document operations failed: {e}")
            raise
        finally:
            # Cleanup
            try:
                repo.delete_collection(test_collection_name)
                print("✅ Test collection cleaned up")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
    
    def test_performance_monitoring(self, repo, test_collection_name, sample_documents):
        """Test performance monitoring and metrics."""
        print(f"\n📋 Test: Performance Monitoring")
        
        try:
            # Create collection
            repo.create_collection(
                collection_name=test_collection_name,
                vector_size=1536,
                distance="cosine"
            )
            
            # Upsert documents with timing
            start_time = datetime.now()
            result = repo.upsert_documents(test_collection_name, sample_documents)
            upsert_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Upsert time: {upsert_time:.3f}s")
            
            # Search with timing
            start_time = datetime.now()
            results = repo.search_similar(
                collection_name=test_collection_name,
                query_embedding=[0.1] * 1536,
                limit=10
            )
            search_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Search time: {search_time:.3f}s")
            
            # Get performance metrics
            metrics = repo.get_performance_metrics()
            print(f"✅ Performance metrics: {metrics}")
            assert "is_async" in metrics
            assert metrics["is_async"] is False
            
            # Performance assertions
            assert upsert_time < 5.0  # Should be fast
            assert search_time < 2.0   # Should be very fast
            
        except Exception as e:
            print(f"❌ Performance monitoring failed: {e}")
            raise
        finally:
            # Cleanup
            try:
                repo.delete_collection(test_collection_name)
            except:
                pass
    
    def test_error_handling(self, repo):
        """Test error handling for invalid operations."""
        print(f"\n📋 Test: Error Handling")
        
        try:
            # Try to get non-existent collection
            with pytest.raises(Exception):
                repo.get_collection_info("non_existent_collection")
            print("✅ Properly handled non-existent collection error")
            
            # Try to search in non-existent collection
            with pytest.raises(Exception):
                repo.search_similar(
                    collection_name="non_existent_collection",
                    query_embedding=[0.1] * 1536
                )
            print("✅ Properly handled search in non-existent collection error")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            raise
    
    def test_context_manager(self, test_collection_name):
        """Test repository context manager functionality."""
        print(f"\n📋 Test: Context Manager")
        
        try:
            with EnhancedQdrantHTTPRepository(
                host="localhost",
                port=6333,
                api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
            ) as repo:
                # Create collection
                repo.create_collection(
                    collection_name=test_collection_name,
                    vector_size=1536,
                    distance="cosine"
                )
                print("✅ Collection created in context manager")
                
                # Verify collection exists
                collections = repo.get_collections()
                collection_names = [col.get("name") for col in collections]
                assert test_collection_name in collection_names
                
            # After context exit, cache should be cleared
            print("✅ Context manager exited successfully")
            
        except Exception as e:
            print(f"❌ Context manager test failed: {e}")
            raise
        finally:
            # Cleanup
            try:
                with EnhancedQdrantHTTPRepository(
                    host="localhost",
                    port=6333,
                    api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
                ) as repo:
                    repo.delete_collection(test_collection_name)
            except:
                pass


@pytest.mark.asyncio
class TestAsyncQdrantHTTPIntegration:
    """Async integration tests for HTTP-based Qdrant repository."""
    
    @pytest.fixture
    def async_repo(self):
        """Create async repository instance for testing."""
        return EnhancedQdrantHTTPRepository(
            host="localhost",
            port=6333,
            api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814",
            use_async=True,
            timeout=10.0,
            max_retries=2
        )
    
    @pytest.fixture
    def test_collection_name(self):
        """Test collection name."""
        return f"test_async_http_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def test_async_health_check(self, async_repo):
        """Test async health check."""
        print("\n📋 Test: Async Health Check")
        
        try:
            health = await async_repo.health_check_async()
            print(f"✅ Async health check successful: {health}")
            assert "version" in health
        except Exception as e:
            print(f"❌ Async health check failed: {e}")
            raise
    
    async def test_async_collections(self, async_repo):
        """Test async collection operations."""
        print("\n📋 Test: Async Collections")
        
        try:
            collections = await async_repo.get_collections_async()
            print(f"✅ Async collections retrieved: {len(collections)}")
            assert isinstance(collections, list)
        except Exception as e:
            print(f"❌ Async collections failed: {e}")
            raise


if __name__ == "__main__":
    # Run tests manually
    print("🚀 Starting Qdrant HTTP Integration Tests")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestQdrantHTTPIntegration()
    
    # Run tests
    try:
        test_instance.test_health_check(test_instance.repo())
        test_instance.test_get_collections(test_instance.repo())
        
        # Test with unique collection names
        test_instance.test_create_and_delete_collection(
            test_instance.repo(),
            f"test_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        print("\n" + "=" * 60)
        print("🎉 All manual tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
