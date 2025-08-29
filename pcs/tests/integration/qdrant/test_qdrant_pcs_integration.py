#!/usr/bin/env python3
"""
Filepath: test_qdrant_pcs_integration.py
Purpose: Test script to verify PCS integration with Qdrant
Related Components: PCS, Qdrant, integration testing
Tags: testing, integration, qdrant, pcs
"""

import httpx
import json
import time
import asyncio

def test_qdrant_direct():
    """Test direct Qdrant API access."""
    print("🔍 Testing direct Qdrant API access...")
    
    # Test Qdrant health
    try:
        response = httpx.get("http://localhost:6333/collections", 
                              headers={"api-key": "qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"})
        if response.status_code == 200:
            print("✅ Qdrant API accessible")
            collections = response.json()
            print(f"   Collections: {collections}")
        else:
            print(f"❌ Qdrant API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant API connection failed: {e}")
        return False
    
    return True

def test_pcs_health():
    """Test PCS health endpoint."""
    print("\n🔍 Testing PCS health...")
    
    try:
        response = httpx.get("http://localhost:8000/api/v1/health/")
        if response.status_code == 200:
            print("✅ PCS health check passed")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Version: {health_data.get('version')}")
        else:
            print(f"❌ PCS health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ PCS connection failed: {e}")
        return False
    
    return True

def test_qdrant_collection_creation():
    """Test creating a test collection in Qdrant."""
    print("\n🔍 Testing Qdrant collection creation...")
    
    collection_name = "test_pcs_integration"
    
    # Create collection
    try:
        create_data = {
            "vectors": {
                "size": 1536,
                "distance": "Cosine"
            }
        }
        
        response = httpx.put(
            f"http://localhost:6333/collections/{collection_name}",
            headers={
                "api-key": "qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814",
                "Content-Type": "application/json"
            },
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
    
    # Verify collection exists
    try:
        response = httpx.get(
            f"http://localhost:6333/collections/{collection_name}",
            headers={"api-key": "qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"}
        )
        
        if response.status_code == 200:
            print("✅ Test collection verified")
            collection_info = response.json()
            print(f"   Collection info: {collection_info.get('result', {}).get('name')}")
        else:
            print(f"❌ Collection verification failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Collection verification error: {e}")
        return False
    
    # Clean up test collection
    try:
        response = httpx.delete(
            f"http://localhost:6333/collections/{collection_name}",
            headers={"api-key": "qd_0197b3371bcf3c99bfacb50d71c40b868a8a81bf6b9731a7965276f4c3f79814"}
        )
        
        if response.status_code == 200:
            print("✅ Test collection cleaned up")
        else:
            print(f"⚠️  Collection cleanup failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Collection cleanup error: {e}")
    
    return True

def main():
    """Run all integration tests."""
    print("🚀 Starting Qdrant-PCS Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Direct Qdrant API", test_qdrant_direct),
        ("PCS Health", test_pcs_health),
        ("Qdrant Collection Operations", test_qdrant_collection_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Qdrant-PCS integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
