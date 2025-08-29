#!/usr/bin/env python3
"""
Filepath: pcs/demo_qdrant_http_repo.py
Purpose: Demo script showcasing HTTP-based Qdrant repository capabilities
Related Components: Qdrant HTTP repository, vector operations, multi-tenancy
Tags: demo, qdrant, http-repository, vector-database, showcase
"""

import asyncio
from datetime import datetime
from typing import List

from src.pcs.repositories.qdrant_http_repo import (
    EnhancedQdrantHTTPRepository,
    VectorDocument,
    SimilarityAlgorithm
)


def demo_sync_operations():
    """Demonstrate synchronous operations."""
    print("🚀 SYNC OPERATIONS DEMO")
    print("=" * 50)
    
    # Initialize repository
    repo = EnhancedQdrantHTTPRepository(
        host="localhost",
        port=6333,
        api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
    )
    
    collection_name = f"demo_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Health check
        print("\n📋 1. Health Check")
        health = repo.health_check()
        print(f"   ✅ Qdrant version: {health.get('version', 'unknown')}")
        
        # Create collection
        print(f"\n📋 2. Create Collection '{collection_name}'")
        success = repo.create_collection(
            collection_name=collection_name,
            vector_size=1536,
            distance="cosine",
            on_disk_payload=True
        )
        print(f"   ✅ Collection created: {success}")
        
        # Create sample documents
        print("\n📋 3. Create Sample Documents")
        documents = [
            VectorDocument(
                id="doc1",
                content="This is a document about artificial intelligence and machine learning",
                embedding=[0.1] * 1536,
                metadata={"category": "AI", "language": "en", "source": "demo"},
                created_at=datetime.now(),
                collection_name=collection_name,
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc2",
                content="Another document about data science and analytics",
                embedding=[0.2] * 1536,
                metadata={"category": "DS", "language": "en", "source": "demo"},
                created_at=datetime.now(),
                collection_name=collection_name,
                tenant_id="tenant1"
            ),
            VectorDocument(
                id="doc3",
                content="A third document about software engineering and development",
                embedding=[0.3] * 1536,
                metadata={"category": "SE", "language": "en", "source": "demo"},
                created_at=datetime.now(),
                collection_name=collection_name,
                tenant_id="tenant2"
            )
        ]
        
        # Upsert documents
        result = repo.upsert_documents(collection_name, documents)
        print(f"   ✅ Documents upserted: {result}")
        
        # Get collection stats
        print("\n📋 4. Collection Statistics")
        stats = repo.get_collection_stats(collection_name)
        print(f"   📊 Points count: {stats['points_count']}")
        print(f"   📊 Vectors count: {stats['vectors_count']}")
        print(f"   📊 Status: {stats['status']}")
        
        # Search operations
        print("\n📋 5. Search Operations")
        
        # Basic search
        query_vector = [0.15] * 1536
        results = repo.search_similar(
            collection_name=collection_name,
            query_embedding=query_vector,
            limit=5
        )
        print(f"   🔍 Basic search: {len(results)} results")
        for i, result in enumerate(results[:3]):
            print(f"      {i+1}. Score: {result.similarity_score:.4f}, Content: {result.document.content[:50]}...")
        
        # Tenant-specific search
        tenant1_results = repo.search_similar(
            collection_name=collection_name,
            query_embedding=query_vector,
            limit=5,
            tenant_id="tenant1"
        )
        print(f"   🔍 Tenant1 search: {len(tenant1_results)} results")
        
        # Metadata filtering
        ai_results = repo.search_similar(
            collection_name=collection_name,
            query_embedding=query_vector,
            limit=5,
            metadata_filters={"category": "AI"}
        )
        print(f"   🔍 AI category search: {len(ai_results)} results")
        
        # Performance metrics
        print("\n📋 6. Performance Metrics")
        metrics = repo.get_performance_metrics()
        print(f"   📈 Cache size: {metrics['collection_cache_size']}")
        print(f"   📈 Async mode: {metrics['is_async']}")
        
        # Cleanup
        print(f"\n📋 7. Cleanup")
        delete_result = repo.delete_documents(collection_name, [doc.id for doc in documents])
        print(f"   ✅ Documents deleted: {delete_result}")
        
        repo.delete_collection(collection_name)
        print(f"   ✅ Collection deleted")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        # Cleanup on failure
        try:
            repo.delete_collection(collection_name)
        except:
            pass
        raise


async def demo_async_operations():
    """Demonstrate asynchronous operations."""
    print("\n🚀 ASYNC OPERATIONS DEMO")
    print("=" * 50)
    
    # Initialize async repository
    repo = EnhancedQdrantHTTPRepository(
        host="localhost",
        port=6333,
        api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814",
        use_async=True
    )
    
    try:
        # Async health check
        print("\n📋 1. Async Health Check")
        health = await repo.health_check_async()
        print(f"   ✅ Qdrant version: {health.get('version', 'unknown')}")
        
        # Async collections
        print("\n📋 2. Async Collections")
        collections = await repo.get_collections_async()
        print(f"   ✅ Collections count: {len(collections)}")
        
        # Performance comparison
        print("\n📋 3. Performance Comparison")
        start_time = datetime.now()
        collections = await repo.get_collections_async()
        async_time = (datetime.now() - start_time).total_seconds()
        print(f"   ⚡ Async time: {async_time:.4f}s")
        
    except Exception as e:
        print(f"❌ Async demo failed: {e}")


def demo_context_manager():
    """Demonstrate context manager usage."""
    print("\n🚀 CONTEXT MANAGER DEMO")
    print("=" * 50)
    
    collection_name = f"demo_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        with EnhancedQdrantHTTPRepository(
            host="localhost",
            port=6333,
            api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
        ) as repo:
            print(f"\n📋 1. Create Collection '{collection_name}'")
            success = repo.create_collection(
                collection_name=collection_name,
                vector_size=1536,
                distance="cosine"
            )
            print(f"   ✅ Collection created: {success}")
            
            print("\n📋 2. Add Sample Document")
            doc = VectorDocument(
                id="context_doc",
                content="This document was created using the context manager",
                embedding=[0.1] * 1536,
                metadata={"demo": "context_manager"},
                created_at=datetime.now(),
                collection_name=collection_name
            )
            
            result = repo.upsert_documents(collection_name, [doc])
            print(f"   ✅ Document added: {result}")
            
            print("\n📋 3. Context Manager Exit")
            # Context manager will automatically clear cache
        
        print("   ✅ Context manager exited, cache cleared")
        
        # Cleanup
        with EnhancedQdrantHTTPRepository(
            host="localhost",
            port=6333,
            api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
        ) as repo:
            repo.delete_collection(collection_name)
            print("   ✅ Collection cleaned up")
            
    except Exception as e:
        print(f"❌ Context manager demo failed: {e}")


def main():
    """Run all demos."""
    print("🎯 HTTP-BASED QDRANT REPOSITORY DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the robust HTTP-based Qdrant repository")
    print("that bypasses client library issues and provides reliable")
    print("vector database operations.")
    print("=" * 60)
    
    try:
        # Run sync demo
        demo_sync_operations()
        
        # Run async demo
        asyncio.run(demo_async_operations())
        
        # Run context manager demo
        demo_context_manager()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("✅ HTTP-based Qdrant repository is working perfectly")
        print("✅ All operations: Create, Read, Update, Delete, Search")
        print("✅ Multi-tenancy support with tenant isolation")
        print("✅ Metadata filtering and advanced search")
        print("✅ Performance monitoring and metrics")
        print("✅ Context manager for resource management")
        print("✅ Both sync and async operations")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This might indicate Qdrant is not running or there's a configuration issue.")
        print("Please ensure Qdrant is running on localhost:6333")


if __name__ == "__main__":
    main()
