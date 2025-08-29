#!/usr/bin/env python3
"""
Filepath: scripts/migrate_chromadb_to_qdrant.py
Purpose: Comprehensive migration script from ChromaDB to Qdrant with multi-tenancy support
Related Components: ChromaDB, Qdrant, data migration, multi-tenancy, vector database
Tags: migration, chromadb, qdrant, data-transfer, multi-tenant, vector-database
"""

import asyncio
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid

import chromadb
from chromadb import Collection as ChromaCollection
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, OptimizersConfig, HnswConfig
)

from pcs.repositories.chroma_repo import EnhancedChromaRepository
from pcs.repositories.qdrant_repo import (
    EnhancedQdrantRepository, 
    QdrantCollectionConfig, 
    QdrantDistance,
    VectorDocument
)


@dataclass
class MigrationConfig:
    """Configuration for the migration process."""
    # Source ChromaDB configuration
    chromadb_host: str = "localhost"
    chromadb_port: int = 8001
    chromadb_collections: Optional[List[str]] = None
    
    # Target Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_use_https: bool = False
    
    # Migration settings
    batch_size: int = 1000
    tenant_mapping: Optional[Dict[str, str]] = None  # chromadb_collection -> tenant_id
    target_collection_name: str = "migrated_knowledge"
    preserve_ids: bool = True
    verify_migration: bool = True
    
    # Performance settings
    max_concurrent_batches: int = 5
    retry_count: int = 3
    retry_delay: float = 1.0
    
    # Qdrant collection configuration
    vector_size: int = 384
    distance_metric: str = "cosine"
    enable_quantization: bool = True
    quantization_type: str = "scalar"


@dataclass
class MigrationStats:
    """Statistics for the migration process."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_documents: int = 0
    migrated_documents: int = 0
    failed_documents: int = 0
    collections_processed: int = 0
    total_batches: int = 0
    processed_batches: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_documents": self.total_documents,
            "migrated_documents": self.migrated_documents,
            "failed_documents": self.failed_documents,
            "collections_processed": self.collections_processed,
            "total_batches": self.total_batches,
            "processed_batches": self.processed_batches,
            "success_rate": self.migrated_documents / self.total_documents if self.total_documents > 0 else 0,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "errors": self.errors
        }


class ChromaDBToQdrantMigrator:
    """
    Comprehensive migration tool from ChromaDB to Qdrant with multi-tenancy support.
    
    Features:
    - Batch processing with configurable sizes
    - Multi-tenant data isolation
    - Progress tracking and reporting
    - Error handling and retry logic
    - Data verification and validation
    - Performance optimization
    """
    
    def __init__(self, config: MigrationConfig):
        """Initialize the migrator with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.stats = MigrationStats(start_time=datetime.now())
        
        # Initialize clients
        self.chroma_client = None
        self.qdrant_client = None
        self.chroma_repo = None
        self.qdrant_repo = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def connect_clients(self) -> None:
        """Initialize and connect to ChromaDB and Qdrant clients."""
        try:
            # Initialize ChromaDB client
            self.logger.info(f"Connecting to ChromaDB at {self.config.chromadb_host}:{self.config.chromadb_port}")
            self.chroma_client = chromadb.HttpClient(
                host=self.config.chromadb_host,
                port=self.config.chromadb_port
            )
            self.chroma_repo = EnhancedChromaRepository(self.chroma_client)
            
            # Test ChromaDB connection
            collections = self.chroma_client.list_collections()
            self.logger.info(f"Connected to ChromaDB. Found {len(collections)} collections.")
            
            # Initialize Qdrant client
            self.logger.info(f"Connecting to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
            self.qdrant_client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key,
                https=self.config.qdrant_use_https
            )
            self.qdrant_repo = EnhancedQdrantRepository(client=self.qdrant_client)
            
            # Test Qdrant connection
            qdrant_collections = self.qdrant_client.get_collections()
            self.logger.info(f"Connected to Qdrant. Found {len(qdrant_collections.collections)} collections.")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to databases: {str(e)}")
            raise
    
    async def create_target_collection(self) -> None:
        """Create the target collection in Qdrant with optimized configuration."""
        try:
            # Check if collection already exists
            collection_exists = self.qdrant_client.collection_exists(
                collection_name=self.config.target_collection_name
            )
            
            if collection_exists:
                self.logger.info(f"Target collection '{self.config.target_collection_name}' already exists")
                return
            
            # Map distance metrics
            distance_map = {
                "cosine": QdrantDistance.COSINE,
                "euclidean": QdrantDistance.EUCLIDEAN,
                "dot_product": QdrantDistance.DOT_PRODUCT,
                "manhattan": QdrantDistance.MANHATTAN
            }
            
            # Configure quantization
            quantization_config = None
            if self.config.enable_quantization:
                quantization_config = {
                    "type": self.config.quantization_type,
                    "quantile": 0.99,
                    "always_ram": False
                }
            
            # Create collection configuration
            collection_config = QdrantCollectionConfig(
                name=self.config.target_collection_name,
                vector_size=self.config.vector_size,
                distance=distance_map.get(self.config.distance_metric, QdrantDistance.COSINE),
                hnsw_config={
                    "m": 16,
                    "ef_construct": 100,
                    "full_scan_threshold": 10000,
                    "max_indexing_threads": 0,
                    "on_disk": False
                },
                optimizers_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 0,
                    "indexing_threshold": 20000,
                    "flush_interval_sec": 5,
                    "max_optimization_threads": 1
                },
                quantization_config=quantization_config,
                on_disk_payload=True
            )
            
            # Create collection
            await self.qdrant_repo.create_collection_optimized(
                config=collection_config,
                metadata={
                    "migration_source": "chromadb",
                    "migration_date": datetime.now().isoformat(),
                    "migration_tool": "chromadb_to_qdrant_migrator"
                }
            )
            
            self.logger.info(f"Created target collection '{self.config.target_collection_name}' with optimizations")
            
        except Exception as e:
            self.logger.error(f"Failed to create target collection: {str(e)}")
            raise
    
    async def get_chromadb_collections(self) -> List[str]:
        """Get list of ChromaDB collections to migrate."""
        try:
            all_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if self.config.chromadb_collections:
                # Filter to specified collections
                collections = [col for col in self.config.chromadb_collections if col in all_collections]
                missing = set(self.config.chromadb_collections) - set(collections)
                if missing:
                    self.logger.warning(f"Collections not found: {missing}")
            else:
                # Use all collections
                collections = all_collections
            
            self.logger.info(f"Collections to migrate: {collections}")
            return collections
            
        except Exception as e:
            self.logger.error(f"Failed to get ChromaDB collections: {str(e)}")
            raise
    
    async def migrate_collection(self, collection_name: str) -> Tuple[int, int]:
        """
        Migrate a single ChromaDB collection to Qdrant.
        
        Returns:
            Tuple of (successful_documents, failed_documents)
        """
        try:
            self.logger.info(f"Starting migration of collection: {collection_name}")
            
            # Get ChromaDB collection
            chroma_collection = self.chroma_client.get_collection(collection_name)
            
            # Get collection count
            collection_count = chroma_collection.count()
            self.logger.info(f"Collection '{collection_name}' contains {collection_count} documents")
            
            if collection_count == 0:
                self.logger.info(f"Collection '{collection_name}' is empty, skipping")
                return 0, 0
            
            # Determine tenant ID for this collection
            tenant_id = self._get_tenant_id(collection_name)
            
            # Migrate documents in batches
            successful_docs = 0
            failed_docs = 0
            offset = 0
            
            while offset < collection_count:
                # Get batch of documents
                batch_size = min(self.config.batch_size, collection_count - offset)
                
                try:
                    # Get documents from ChromaDB
                    results = chroma_collection.get(
                        limit=batch_size,
                        offset=offset,
                        include=["documents", "metadatas", "embeddings"]
                    )
                    
                    if not results["ids"]:
                        break
                    
                    # Convert to VectorDocument format
                    vector_documents = []
                    for i, doc_id in enumerate(results["ids"]):
                        try:
                            vector_doc = VectorDocument(
                                id=doc_id if self.config.preserve_ids else str(uuid.uuid4()),
                                content=results["documents"][i] if results.get("documents") else "",
                                embedding=results["embeddings"][i] if results.get("embeddings") else [],
                                metadata={
                                    **(results["metadatas"][i] if results.get("metadatas") and i < len(results["metadatas"]) else {}),
                                    "source_collection": collection_name,
                                    "migrated_at": datetime.now().isoformat()
                                },
                                created_at=datetime.now(),
                                collection_name=self.config.target_collection_name,
                                tenant_id=tenant_id
                            )
                            vector_documents.append(vector_doc)
                        except Exception as e:
                            self.logger.error(f"Failed to convert document {doc_id}: {str(e)}")
                            failed_docs += 1
                    
                    # Insert batch into Qdrant
                    if vector_documents:
                        try:
                            points = [doc.to_qdrant_point() for doc in vector_documents]
                            
                            self.qdrant_client.upsert(
                                collection_name=self.config.target_collection_name,
                                points=points
                            )
                            
                            successful_docs += len(vector_documents)
                            self.stats.processed_batches += 1
                            
                            self.logger.info(
                                f"Migrated batch {self.stats.processed_batches}/{self.stats.total_batches}: "
                                f"{len(vector_documents)} documents from {collection_name}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Failed to insert batch into Qdrant: {str(e)}")
                            failed_docs += len(vector_documents)
                            self.stats.errors.append(f"Batch insert failed for {collection_name}: {str(e)}")
                    
                    offset += batch_size
                    
                    # Add small delay to prevent overwhelming the databases
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process batch at offset {offset}: {str(e)}")
                    failed_docs += batch_size
                    offset += batch_size
                    self.stats.errors.append(f"Batch processing failed for {collection_name} at offset {offset}: {str(e)}")
            
            self.logger.info(
                f"Completed migration of collection '{collection_name}': "
                f"{successful_docs} successful, {failed_docs} failed"
            )
            
            return successful_docs, failed_docs
            
        except Exception as e:
            self.logger.error(f"Failed to migrate collection {collection_name}: {str(e)}")
            raise
    
    def _get_tenant_id(self, collection_name: str) -> str:
        """Get tenant ID for a collection based on mapping or collection name."""
        if self.config.tenant_mapping and collection_name in self.config.tenant_mapping:
            return self.config.tenant_mapping[collection_name]
        else:
            # Use collection name as tenant ID by default
            return collection_name
    
    async def verify_migration(self) -> bool:
        """Verify the migration by comparing document counts and sampling data."""
        try:
            self.logger.info("Starting migration verification...")
            
            # Get total count from Qdrant
            qdrant_collection = self.qdrant_client.get_collection(self.config.target_collection_name)
            qdrant_count = qdrant_collection.points_count
            
            self.logger.info(f"Qdrant collection contains {qdrant_count} documents")
            
            # Compare with expected count
            if qdrant_count != self.stats.migrated_documents:
                self.logger.warning(
                    f"Document count mismatch: expected {self.stats.migrated_documents}, "
                    f"found {qdrant_count} in Qdrant"
                )
                return False
            
            # Sample verification - check a few random documents
            if qdrant_count > 0:
                sample_size = min(10, qdrant_count)
                
                # Get sample points from Qdrant
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.config.target_collection_name,
                    limit=sample_size,
                    with_payload=True,
                    with_vector=True
                )
                
                sample_points = scroll_result[0]
                
                for point in sample_points:
                    # Verify point structure
                    if not point.payload or not point.vector:
                        self.logger.warning(f"Point {point.id} missing payload or vector")
                        return False
                    
                    # Verify tenant_id is present (for multi-tenancy)
                    if "tenant_id" not in point.payload:
                        self.logger.warning(f"Point {point.id} missing tenant_id")
                        return False
                
                self.logger.info(f"Sample verification passed for {len(sample_points)} documents")
            
            self.logger.info("Migration verification completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration verification failed: {str(e)}")
            return False
    
    async def run_migration(self) -> MigrationStats:
        """Run the complete migration process."""
        try:
            self.logger.info("Starting ChromaDB to Qdrant migration...")
            
            # Connect to databases
            await self.connect_clients()
            
            # Create target collection
            await self.create_target_collection()
            
            # Get collections to migrate
            collections = await self.get_chromadb_collections()
            
            if not collections:
                self.logger.warning("No collections found to migrate")
                return self.stats
            
            # Calculate total batches for progress tracking
            total_docs = 0
            for collection_name in collections:
                chroma_collection = self.chroma_client.get_collection(collection_name)
                collection_count = chroma_collection.count()
                total_docs += collection_count
                self.stats.total_batches += (collection_count + self.config.batch_size - 1) // self.config.batch_size
            
            self.stats.total_documents = total_docs
            self.logger.info(f"Total documents to migrate: {total_docs} across {len(collections)} collections")
            
            # Migrate each collection
            for collection_name in collections:
                try:
                    successful, failed = await self.migrate_collection(collection_name)
                    self.stats.migrated_documents += successful
                    self.stats.failed_documents += failed
                    self.stats.collections_processed += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to migrate collection {collection_name}: {str(e)}")
                    self.stats.errors.append(f"Collection {collection_name} migration failed: {str(e)}")
            
            # Verify migration if requested
            if self.config.verify_migration:
                verification_passed = await self.verify_migration()
                if not verification_passed:
                    self.stats.errors.append("Migration verification failed")
            
            # Complete migration
            self.stats.end_time = datetime.now()
            
            # Log final statistics
            self._log_final_stats()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"Migration failed: {str(e)}")
            self.stats.end_time = datetime.now()
            self.stats.errors.append(f"Migration failed: {str(e)}")
            raise
    
    def _log_final_stats(self) -> None:
        """Log final migration statistics."""
        duration = (self.stats.end_time - self.stats.start_time).total_seconds()
        success_rate = (self.stats.migrated_documents / self.stats.total_documents * 100) if self.stats.total_documents > 0 else 0
        
        self.logger.info("=" * 60)
        self.logger.info("MIGRATION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Collections processed: {self.stats.collections_processed}")
        self.logger.info(f"Total documents: {self.stats.total_documents}")
        self.logger.info(f"Migrated documents: {self.stats.migrated_documents}")
        self.logger.info(f"Failed documents: {self.stats.failed_documents}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Throughput: {self.stats.migrated_documents / duration:.2f} docs/sec")
        
        if self.stats.errors:
            self.logger.warning(f"Errors encountered: {len(self.stats.errors)}")
            for error in self.stats.errors:
                self.logger.warning(f"  - {error}")
        
        self.logger.info("=" * 60)
    
    def save_migration_report(self, filename: Optional[str] = None) -> str:
        """Save migration report to JSON file."""
        if not filename:
            filename = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
        
        self.logger.info(f"Migration report saved to: {filename}")
        return filename


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate data from ChromaDB to Qdrant")
    
    # ChromaDB configuration
    parser.add_argument("--chromadb-host", default="localhost", help="ChromaDB host")
    parser.add_argument("--chromadb-port", type=int, default=8001, help="ChromaDB port")
    parser.add_argument("--chromadb-collections", nargs="+", help="Specific collections to migrate")
    
    # Qdrant configuration
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--qdrant-api-key", help="Qdrant API key")
    parser.add_argument("--qdrant-https", action="store_true", help="Use HTTPS for Qdrant")
    
    # Migration settings
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for migration")
    parser.add_argument("--target-collection", default="migrated_knowledge", help="Target collection name")
    parser.add_argument("--tenant-mapping", help="JSON file with collection->tenant mapping")
    parser.add_argument("--preserve-ids", action="store_true", default=True, help="Preserve document IDs")
    parser.add_argument("--no-verify", action="store_true", help="Skip migration verification")
    
    # Performance settings
    parser.add_argument("--vector-size", type=int, default=384, help="Vector dimension size")
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "euclidean", "dot_product", "manhattan"])
    parser.add_argument("--enable-quantization", action="store_true", default=True, help="Enable vector quantization")
    parser.add_argument("--quantization-type", default="scalar", choices=["scalar", "product", "binary"])
    
    # Output
    parser.add_argument("--report-file", help="Migration report filename")
    
    return parser.parse_args()


async def main():
    """Main migration function."""
    args = parse_arguments()
    
    # Load tenant mapping if provided
    tenant_mapping = None
    if args.tenant_mapping:
        with open(args.tenant_mapping, 'r') as f:
            tenant_mapping = json.load(f)
    
    # Create migration configuration
    config = MigrationConfig(
        chromadb_host=args.chromadb_host,
        chromadb_port=args.chromadb_port,
        chromadb_collections=args.chromadb_collections,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_use_https=args.qdrant_https,
        batch_size=args.batch_size,
        tenant_mapping=tenant_mapping,
        target_collection_name=args.target_collection,
        preserve_ids=args.preserve_ids,
        verify_migration=not args.no_verify,
        vector_size=args.vector_size,
        distance_metric=args.distance_metric,
        enable_quantization=args.enable_quantization,
        quantization_type=args.quantization_type
    )
    
    # Create and run migrator
    migrator = ChromaDBToQdrantMigrator(config)
    
    try:
        stats = await migrator.run_migration()
        
        # Save migration report
        report_file = migrator.save_migration_report(args.report_file)
        
        # Exit with appropriate code
        if stats.failed_documents > 0:
            print(f"Migration completed with {stats.failed_documents} failures. Check {report_file} for details.")
            exit(1)
        else:
            print(f"Migration completed successfully! Report saved to {report_file}")
            exit(0)
            
    except Exception as e:
        print(f"Migration failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())