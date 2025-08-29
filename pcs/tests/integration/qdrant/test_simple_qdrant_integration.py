#!/usr/bin/env python3
"""
Filepath: test_simple_qdrant_integration.py
Purpose: Simple test script to verify basic Qdrant functionality
Related Components: Qdrant, basic integration testing
Tags: testing, integration, qdrant, simple
"""

import httpx
import json

def test_basic_qdrant_operations():
    """Test basic Qdrant operations using direct HTTP requests."""
    print("🚀 Starting Simple Qdrant Integration Test")
    print("=" * 50)
    
    base_url = "http://localhost:6333"
    api_key = "qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"
    headers = {"api-key": api_key}
    
    # Test 1: Check Qdrant health
    print("\n📋 Test 1: Qdrant Health Check")
    try:
        with httpx.Client() as client:
            response = client.get(f"{base_url}/", headers=headers)
            if response.status_code == 200:
                print("✅ Qdrant is healthy")
                print(f"   Version: {response.json().get('version')}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: List collections
    print("\n📋 Test 2: List Collections")
    try:
        with httpx.Client() as client:
            response = client.get(f"{base_url}/collections", headers=headers)
            if response.status_code == 200:
                collections = response.json()
                print("✅ Collections retrieved successfully")
                print(f"   Collections: {collections}")
            else:
                print(f"❌ List collections failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ List collections error: {e}")
        return False
    
    # Test 3: Create a test collection
    print("\n📋 Test 3: Create Test Collection")
    collection_name = "simple_test_collection"
    create_data = {
        "vectors": {
            "size": 1536,
            "distance": "Cosine"
        }
    }
    
    try:
        with httpx.Client() as client:
            response = client.put(
                f"{base_url}/collections/{collection_name}",
                headers={**headers, "Content-Type": "application/json"},
                json=create_data
            )
            if response.status_code == 200:
                print("✅ Test collection created successfully")
            else:
                print(f"❌ Collection creation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Collection creation error: {e}")
        return False
    
    # Test 4: Verify collection exists
    print("\n📋 Test 4: Verify Collection")
    try:
        with httpx.Client() as client:
            response = client.get(f"{base_url}/collections/{collection_name}", headers=headers)
            if response.status_code == 200:
                collection_info = response.json()
                print("✅ Collection verified successfully")
                print(f"   Collection info: {collection_info}")
            else:
                print(f"❌ Collection verification failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Collection verification error: {e}")
        return False
    
    # Test 5: Add a test vector
    print("\n📋 Test 5: Add Test Vector")
    test_vector = [0.1] * 1536  # 1536-dimensional vector
    point_data = {
        "points": [
            {
                "id": 1,
                "vector": test_vector,
                "payload": {
                    "text": "This is a test document",
                    "category": "test"
                }
            }
        ]
    }
    
    try:
        with httpx.Client() as client:
            response = client.put(
                f"{base_url}/collections/{collection_name}/points",
                headers={**headers, "Content-Type": "application/json"},
                json=point_data
            )
            if response.status_code == 200:
                print("✅ Test vector added successfully")
            else:
                print(f"❌ Vector addition failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Vector addition error: {e}")
        return False
    
    # Test 6: Search for similar vectors
    print("\n📋 Test 6: Vector Search")
    search_data = {
        "vector": test_vector,
        "limit": 5
    }
    
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{base_url}/collections/{collection_name}/points/search",
                headers={**headers, "Content-Type": "application/json"},
                json=search_data
            )
            if response.status_code == 200:
                search_results = response.json()
                print("✅ Vector search successful")
                print(f"   Results: {search_results}")
            else:
                print(f"❌ Vector search failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        return False
    
    # Test 7: Clean up test collection
    print("\n📋 Test 7: Cleanup")
    try:
        with httpx.Client() as client:
            response = client.delete(f"{base_url}/collections/{collection_name}", headers=headers)
            if response.status_code == 200:
                print("✅ Test collection cleaned up successfully")
            else:
                print(f"⚠️  Cleanup failed: {response.status_code}")
                print(f"   Response: {response.text}")
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All basic Qdrant tests passed!")
    print("✅ Qdrant is working correctly with HTTP API")
    return True

if __name__ == "__main__":
    success = test_basic_qdrant_operations()
    exit(0 if success else 1)
