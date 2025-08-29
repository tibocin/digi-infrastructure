#!/usr/bin/env python3
"""
Filepath: test_pcs_qdrant_integration.py
Purpose: Test PCS integration with Qdrant repository
Related Components: PCS, Qdrant repository, vector operations
Tags: testing, integration, pcs, qdrant, vectors
"""

import sys
import os
import asyncio
from pathlib import Path

# Add PCS source to path
pcs_src = Path(__file__).parent / "pcs" / "src"
sys.path.insert(0, str(pcs_src))

async def test_qdrant_repository_import():
    """Test that PCS can import and initialize Qdrant repository."""
    print("🔍 Testing Qdrant repository import...")
    
    try:
        from pcs.repositories import QdrantRepository
        print("✅ QdrantRepository imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import QdrantRepository: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing QdrantRepository: {e}")
        return False

async def test_qdrant_repository_initialization():
    """Test that Qdrant repository can be initialized with proper config."""
    print("\n🔍 Testing Qdrant repository initialization...")
    
    try:
        from pcs.repositories import QdrantRepository
        
        # Initialize with test configuration
        repo = QdrantRepository(
            host="localhost",
            port=6333,
            grpc_port=6334,
            api_key="qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814",
            prefer_grpc=False,  # Use HTTP for testing
            use_async=True
        )
        
        print("✅ QdrantRepository initialized successfully")
        return True, repo
        
    except Exception as e:
        print(f"❌ Failed to initialize QdrantRepository: {e}")
        return False, None

async def test_qdrant_collection_operations():
    """Test basic collection operations through Qdrant repository."""
    print("\n🔍 Testing Qdrant collection operations...")
    
    try:
        success, repo = await test_qdrant_repository_initialization()
        if not success:
            return False
            
        # Test collection creation
        collection_name = "test_pcs_integration"
        
        # Create collection using repository
        from pcs.repositories.qdrant_repo import QdrantCollectionConfig
        
        config = QdrantCollectionConfig(
            name=collection_name,
            vector_size=1536,
            distance="cosine"
        )
        
        result = await repo.create_collection_optimized(config)
        if result:
            print("✅ Collection created successfully through repository")
        else:
            print("❌ Collection creation failed through repository")
            return False
        
        # Verify collection exists
        collections = await repo.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_name in collection_names:
            print("✅ Collection verified through repository")
        else:
            print("❌ Collection not found through repository")
            return False
        
        # Clean up
        await repo.delete_collection(collection_name)
        print("✅ Collection cleaned up through repository")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection operations failed: {e}")
        return False

async def test_vector_operations():
    """Test vector storage and similarity search operations."""
    print("\n🔍 Testing vector operations...")
    
    try:
        success, repo = await test_qdrant_repository_initialization()
        if not success:
            return False
            
        collection_name = "test_vectors"
        
        # Create collection
        from pcs.repositories.qdrant_repo import QdrantCollectionConfig
        
        config = QdrantCollectionConfig(
            name=collection_name,
            vector_size=1536,
            distance="cosine"
        )
        
        await repo.create_collection_optimized(config)
        
        # Create test vectors
        from pcs.repositories.qdrant_repo import VectorDocument
        from datetime import datetime
        
        # Sample vector (1536 dimensions)
        sample_vector = [0.1] * 1536
        
        doc1 = VectorDocument(
            id="doc1",
            content="This is a test document about AI and machine learning",
            embedding=sample_vector,
            metadata={"category": "AI", "language": "en"},
            created_at=datetime.now(),
            collection_name=collection_name
        )
        
        doc2 = VectorDocument(
            id="doc2", 
            content="Another document about artificial intelligence and neural networks",
            embedding=sample_vector,
            metadata={"category": "AI", "language": "en"},
            created_at=datetime.now(),
            collection_name=collection_name
        )
        
        # Add vectors to collection
        await repo.add_vectors([doc1, doc2], collection_name)
        print("✅ Vectors added successfully")
        
        # Test similarity search
        search_results = await repo.similarity_search(
            query_vector=sample_vector,
            collection_name=collection_name,
            limit=5,
            score_threshold=0.0
        )
        
        if len(search_results) >= 2:
            print(f"✅ Similarity search returned {len(search_results)} results")
        else:
            print(f"❌ Similarity search returned only {len(search_results)} results")
            return False
        
        # Clean up
        await repo.delete_collection(collection_name)
        print("✅ Test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector operations failed: {e}")
        return False

async def main():
    """Run all PCS-Qdrant integration tests."""
    print("🚀 Starting PCS-Qdrant Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Repository Import", test_qdrant_repository_import),
        ("Repository Initialization", test_qdrant_repository_initialization),
        ("Collection Operations", test_qdrant_collection_operations),
        ("Vector Operations", test_vector_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if await test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PCS-Qdrant integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
