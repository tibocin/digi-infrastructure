#!/usr/bin/env python3
"""
Filepath: scripts/quick_migration_test.py
Purpose: Quick migration test script for ChromaDB to Qdrant with sample data
Related Components: ChromaDB, Qdrant, migration testing, sample data
Tags: migration, testing, chromadb, qdrant, sample-data
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any
import uuid

import chromadb
from chromadb import Collection as ChromaCollection
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickMigrationTest:
    """Quick migration test to validate the migration process with sample data."""
    
    def __init__(self):
        self.chroma_client = None
        self.qdrant_client = None
        self.test_collection_name = "test_migration"
        self.qdrant_collection_name = "test_migrated"
        
    async def setup_clients(self):
        """Setup ChromaDB and Qdrant clients."""
        try:
            # ChromaDB client
            self.chroma_client = chromadb.HttpClient(host="localhost", port=8001)
            logger.info("Connected to ChromaDB")
            
            # Qdrant client
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
            logger.info("Connected to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to connect to clients: {e}")
            raise
    
    def create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample data for testing."""
        sample_docs = [
            {
                "id": "doc_1",
                "content": "This is the first test document about artificial intelligence and machine learning.",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4],  # 384 dimensions
                "metadata": {"type": "AI", "category": "tech", "priority": "high"}
            },
            {
                "id": "doc_2", 
                "content": "Second document discussing natural language processing and deep learning networks.",
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6] * 76 + [0.2, 0.3, 0.4, 0.5],  # 384 dimensions
                "metadata": {"type": "NLP", "category": "tech", "priority": "medium"}
            },
            {
                "id": "doc_3",
                "content": "Third document about computer vision and image recognition technologies.",
                "embedding": [0.3, 0.4, 0.5, 0.6, 0.7] * 76 + [0.3, 0.4, 0.5, 0.6],  # 384 dimensions  
                "metadata": {"type": "CV", "category": "tech", "priority": "low"}
            },
            {
                "id": "doc_4",
                "content": "Fourth document covering robotics and autonomous systems development.",
                "embedding": [0.4, 0.5, 0.6, 0.7, 0.8] * 76 + [0.4, 0.5, 0.6, 0.7],  # 384 dimensions
                "metadata": {"type": "Robotics", "category": "hardware", "priority": "high"}
            },
            {
                "id": "doc_5",
                "content": "Fifth document about quantum computing and quantum machine learning algorithms.",
                "embedding": [0.5, 0.6, 0.7, 0.8, 0.9] * 76 + [0.5, 0.6, 0.7, 0.8],  # 384 dimensions
                "metadata": {"type": "Quantum", "category": "research", "priority": "medium"}
            }
        ]
        return sample_docs
    
    async def populate_chromadb(self, sample_docs: List[Dict[str, Any]]):
        """Populate ChromaDB with sample data."""
        try:
            # Create or get collection
            try:
                collection = self.chroma_client.get_collection(self.test_collection_name)
                logger.info(f"Found existing collection: {self.test_collection_name}")
                # Clear existing data
                collection.delete()
            except:
                pass
                
            collection = self.chroma_client.create_collection(
                name=self.test_collection_name,
                metadata={"description": "Test collection for migration"}
            )
            
            # Add documents
            ids = [doc["id"] for doc in sample_docs]
            documents = [doc["content"] for doc in sample_docs] 
            embeddings = [doc["embedding"] for doc in sample_docs]
            metadatas = [doc["metadata"] for doc in sample_docs]
            
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(sample_docs)} documents to ChromaDB collection")
            
            # Verify data
            count = collection.count()
            logger.info(f"ChromaDB collection now contains {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to populate ChromaDB: {e}")
            raise
    
    async def create_qdrant_collection(self):
        """Create target collection in Qdrant."""
        try:
            # Delete collection if it exists
            try:
                self.qdrant_client.delete_collection(self.qdrant_collection_name)
                logger.info(f"Deleted existing Qdrant collection: {self.qdrant_collection_name}")
            except:
                pass
            
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {self.qdrant_collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            raise
    
    async def migrate_data(self):
        """Migrate data from ChromaDB to Qdrant."""
        try:
            # Get data from ChromaDB
            chroma_collection = self.chroma_client.get_collection(self.test_collection_name)
            
            results = chroma_collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            logger.info(f"Retrieved {len(results['ids'])} documents from ChromaDB")
            
            # Convert to Qdrant format
            points = []
            for i, doc_id in enumerate(results["ids"]):
                # Add tenant information for multi-tenancy testing
                metadata = results["metadatas"][i].copy()
                metadata.update({
                    "tenant_id": "test_tenant",
                    "source_collection": self.test_collection_name,
                    "migrated_at": datetime.now().isoformat(),
                    "content": results["documents"][i]
                })
                
                point = PointStruct(
                    id=doc_id,
                    vector=results["embeddings"][i],
                    payload=metadata
                )
                points.append(point)
            
            # Insert into Qdrant
            operation_info = self.qdrant_client.upsert(
                collection_name=self.qdrant_collection_name,
                points=points
            )
            
            logger.info(f"Inserted {len(points)} points into Qdrant")
            logger.info(f"Operation info: {operation_info}")
            
        except Exception as e:
            logger.error(f"Failed to migrate data: {e}")
            raise
    
    async def verify_migration(self) -> bool:
        """Verify the migration was successful."""
        try:
            # Get counts
            chroma_collection = self.chroma_client.get_collection(self.test_collection_name)
            chroma_count = chroma_collection.count()
            
            qdrant_info = self.qdrant_client.get_collection(self.qdrant_collection_name)
            qdrant_count = qdrant_info.points_count
            
            logger.info(f"ChromaDB count: {chroma_count}")
            logger.info(f"Qdrant count: {qdrant_count}")
            
            if chroma_count != qdrant_count:
                logger.error("Document counts don't match!")
                return False
            
            # Test search functionality
            query_vector = [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            logger.info(f"Qdrant search returned {len(search_results)} results")
            
            for i, result in enumerate(search_results):
                logger.info(f"Result {i+1}: ID={result.id}, Score={result.score:.4f}")
                logger.info(f"  Metadata: {result.payload}")
            
            # Test tenant filtering
            tenant_results = self.qdrant_client.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_vector,
                query_filter={
                    "must": [
                        {
                            "key": "tenant_id",
                            "match": {"value": "test_tenant"}
                        }
                    ]
                },
                limit=3
            )
            
            logger.info(f"Tenant-filtered search returned {len(tenant_results)} results")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    async def performance_test(self):
        """Run basic performance comparison."""
        try:
            query_vector = [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.1, 0.2, 0.3, 0.4]
            
            # Test ChromaDB performance
            chroma_collection = self.chroma_client.get_collection(self.test_collection_name)
            
            chroma_times = []
            for _ in range(5):
                start_time = time.time()
                chroma_results = chroma_collection.query(
                    query_embeddings=[query_vector],
                    n_results=3
                )
                chroma_times.append(time.time() - start_time)
            
            avg_chroma_time = sum(chroma_times) / len(chroma_times)
            
            # Test Qdrant performance
            qdrant_times = []
            for _ in range(5):
                start_time = time.time()
                qdrant_results = self.qdrant_client.search(
                    collection_name=self.qdrant_collection_name,
                    query_vector=query_vector,
                    limit=3
                )
                qdrant_times.append(time.time() - start_time)
            
            avg_qdrant_time = sum(qdrant_times) / len(qdrant_times)
            
            logger.info("=" * 50)
            logger.info("PERFORMANCE COMPARISON")
            logger.info("=" * 50)
            logger.info(f"ChromaDB average query time: {avg_chroma_time*1000:.2f}ms")
            logger.info(f"Qdrant average query time: {avg_qdrant_time*1000:.2f}ms")
            logger.info(f"Performance improvement: {avg_chroma_time/avg_qdrant_time:.2f}x faster")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
    
    async def cleanup(self):
        """Clean up test data."""
        try:
            # Delete ChromaDB collection
            try:
                self.chroma_client.delete_collection(self.test_collection_name)
                logger.info(f"Deleted ChromaDB collection: {self.test_collection_name}")
            except:
                pass
            
            # Delete Qdrant collection
            try:
                self.qdrant_client.delete_collection(self.qdrant_collection_name)
                logger.info(f"Deleted Qdrant collection: {self.qdrant_collection_name}")
            except:
                pass
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def run_test(self, cleanup_after: bool = True):
        """Run the complete migration test."""
        try:
            logger.info("Starting Quick Migration Test")
            logger.info("=" * 50)
            
            # Setup
            await self.setup_clients()
            
            # Create sample data
            sample_docs = self.create_sample_data()
            logger.info(f"Created {len(sample_docs)} sample documents")
            
            # Populate ChromaDB
            await self.populate_chromadb(sample_docs)
            
            # Create Qdrant collection
            await self.create_qdrant_collection()
            
            # Migrate data
            await self.migrate_data()
            
            # Verify migration
            verification_passed = await self.verify_migration()
            
            if verification_passed:
                logger.info("✅ Migration verification PASSED")
                
                # Run performance test
                await self.performance_test()
                
            else:
                logger.error("❌ Migration verification FAILED")
                return False
            
            # Cleanup
            if cleanup_after:
                await self.cleanup()
                logger.info("🧹 Cleanup completed")
            
            logger.info("=" * 50)
            logger.info("✅ Quick Migration Test COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick Migration Test FAILED: {e}")
            return False


async def main():
    """Main function to run the quick migration test."""
    test = QuickMigrationTest()
    success = await test.run_test(cleanup_after=True)
    
    if success:
        print("✅ Quick migration test passed! Ready for full migration.")
        exit(0)
    else:
        print("❌ Quick migration test failed. Check logs for details.")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
