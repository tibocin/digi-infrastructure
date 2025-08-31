#!/usr/bin/env python3
"""
Filepath: scripts/validate_qdrant_integration.py
Purpose: Comprehensive validation script for Qdrant integration with tenant applications
Related Components: Qdrant, tenant applications, integration testing, performance validation
Tags: validation, integration, qdrant, performance, multi-tenant
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

from pcs.repositories.qdrant_repo import EnhancedQdrantRepository
from pcs.repositories.qdrant_types import (
    VectorSearchRequest,
    VectorDocument,
    BulkVectorOperation,
    QdrantCollectionConfig,
    QdrantDistance
)


@dataclass
class ValidationResults:
    """Container for validation test results."""
    test_name: str
    status: str  # "passed", "failed", "skipped"
    duration_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
            "error_message": self.error_message
        }


@dataclass
class PerformanceMetrics:
    """Container for performance test results."""
    operation: str
    tenant_id: str
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    total_operations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "tenant_id": self.tenant_id,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "total_operations": self.total_operations
        }


class QdrantIntegrationValidator:
    """
    Comprehensive validation suite for Qdrant integration.
    
    Tests:
    - Basic connectivity and health
    - Multi-tenant data isolation
    - CRUD operations
    - Search functionality
    - Performance benchmarks
    - Error handling
    - Data consistency
    """
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        
        self.logger = self._setup_logging()
        self.vector_repo = None
        self.test_collection = "validation_test_collection"
        self.test_tenants = ["validation_tenant_1", "validation_tenant_2", "validation_tenant_3"]
        
        self.results: List[ValidationResults] = []
        self.performance_metrics: List[PerformanceMetrics] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation tests."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'qdrant_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def setup(self) -> None:
        """Initialize Qdrant repository and test environment."""
        try:
            self.logger.info("Setting up Qdrant integration validator...")
            
            # Initialize repository
            self.vector_repo = EnhancedQdrantRepository(
                host=self.qdrant_host,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key,
                prefer_grpc=False,  # Use HTTP instead of gRPC for testing
                use_async=True,     # Use async client
                https=False
            )
            
            # Clean up any existing test collection
            try:
                await self.vector_repo.delete_collection(self.test_collection)
                self.logger.info(f"Cleaned up existing test collection: {self.test_collection}")
            except:
                pass
            
            # Create test collection with optimized configuration
            config = QdrantCollectionConfig(
                name=self.test_collection,
                vector_size=384,
                distance=QdrantDistance.COSINE,
                hnsw_config={
                    "m": 16,
                    "ef_construct": 100
                },
                quantization_config={
                    "type": "scalar",
                    "quantile": 0.99
                }
            )
            
            await self.vector_repo.create_collection_optimized(config)
            self.logger.info(f"Created test collection: {self.test_collection}")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up test environment."""
        try:
            if self.vector_repo:
                await self.vector_repo.delete_collection(self.test_collection)
                self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {str(e)}")
    
    async def run_test(self, test_name: str, test_func) -> ValidationResults:
        """Run a single validation test with error handling and timing."""
        self.logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            details = await test_func()
            duration = time.time() - start_time
            
            result = ValidationResults(
                test_name=test_name,
                status="passed",
                duration_seconds=duration,
                details=details
            )
            
            self.logger.info(f"✅ {test_name} PASSED ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = ValidationResults(
                test_name=test_name,
                status="failed",
                duration_seconds=duration,
                details={},
                error_message=str(e)
            )
            
            self.logger.error(f"❌ {test_name} FAILED: {str(e)} ({duration:.2f}s)")
        
        self.results.append(result)
        return result
    
    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic Qdrant connectivity and health."""
        # Test collection exists
        exists = await self.vector_repo.get_collection(self.test_collection)
        if not exists:
            raise Exception("Test collection not found")
        
        # Test basic operations
        stats = await self.vector_repo.get_collection_statistics(self.test_collection)
        
        return {
            "collection_exists": True,
            "collection_name": stats.name,
            "dimension": stats.dimension,
            "index_type": stats.index_type
        }
    
    async def test_multi_tenant_isolation(self) -> Dict[str, Any]:
        """Test multi-tenant data isolation."""
        # Add documents for different tenants
        tenant_docs = {}
        
        for i, tenant_id in enumerate(self.test_tenants):
            # Create unique documents for each tenant
            docs = []
            for j in range(5):
                doc_id = f"{tenant_id}_doc_{j}"
                content = f"Document {j} for tenant {tenant_id} with unique content."
                embedding = [float(i + j * 0.1)] * 384  # Unique embeddings per tenant
                
                vector_doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    embedding=embedding,
                    metadata={
                        "tenant_specific_data": f"data_for_{tenant_id}",
                        "doc_number": j,
                        "tenant_category": f"category_{i}"
                    },
                    created_at=datetime.utcnow(),
                    collection_name=self.test_collection,
                    tenant_id=tenant_id
                )
                docs.append(vector_doc)
            
            tenant_docs[tenant_id] = docs
            
            # Add documents for this tenant
            operation = BulkVectorOperation(
                operation_type="insert",
                documents=docs,
                tenant_id=tenant_id
            )
            
            result = await self.vector_repo.bulk_upsert_documents(
                self.test_collection,
                operation
            )
            
            if result["total_processed"] != len(docs):
                raise Exception(f"Failed to add all documents for tenant {tenant_id}")
        
        # Test tenant isolation
        isolation_results = {}
        
        for tenant_id in self.test_tenants:
            # Search within tenant
            tenant_query = tenant_docs[tenant_id][0].embedding
            
            search_results = await self.vector_repo.similarity_search(
                collection_name=self.test_collection,
                query_embedding=tenant_query,
                n_results=20,  # Request more than any single tenant has
                tenant_id=tenant_id
            )
            
            # Verify all results belong to this tenant
            tenant_result_ids = [result["id"] for result in search_results]
            expected_ids = [doc.id for doc in tenant_docs[tenant_id]]
            
            # Check that we only get this tenant's documents
            unexpected_docs = [doc_id for doc_id in tenant_result_ids if not doc_id.startswith(tenant_id)]
            missing_docs = [doc_id for doc_id in expected_ids if doc_id not in tenant_result_ids]
            
            isolation_results[tenant_id] = {
                "returned_count": len(search_results),
                "expected_count": len(tenant_docs[tenant_id]),
                "unexpected_docs": unexpected_docs,
                "missing_docs": missing_docs,
                "isolation_perfect": len(unexpected_docs) == 0
            }
            
            if unexpected_docs:
                raise Exception(f"Tenant isolation violated for {tenant_id}: found docs {unexpected_docs}")
        
        return {
            "tenants_tested": len(self.test_tenants),
            "documents_per_tenant": len(tenant_docs[self.test_tenants[0]]),
            "isolation_results": isolation_results,
            "all_tenants_isolated": all(r["isolation_perfect"] for r in isolation_results.values())
        }
    
    async def test_crud_operations(self) -> Dict[str, Any]:
        """Test Create, Read, Update, Delete operations."""
        tenant_id = "crud_test_tenant"
        
        # CREATE
        doc_id = f"crud_test_doc_{uuid.uuid4()}"
        original_content = "Original content for CRUD testing."
        original_embedding = [0.5] * 384
        original_metadata = {"type": "crud_test", "version": 1}
        
        create_success = await self.vector_repo.add_documents(
            collection_name=self.test_collection,
            documents=[original_content],
            ids=[doc_id],
            embeddings=[original_embedding],
            metadatas=[original_metadata],
            tenant_id=tenant_id
        )
        
        if not create_success:
            raise Exception("Failed to create document")
        
        # READ
        read_results = await self.vector_repo.get_documents(
            collection_name=self.test_collection,
            ids=[doc_id],
            tenant_id=tenant_id
        )
        
        if not read_results["ids"] or read_results["ids"][0] != doc_id:
            raise Exception("Failed to read created document")
        
        if read_results["documents"][0] != original_content:
            raise Exception("Document content mismatch on read")
        
        # UPDATE
        updated_content = "Updated content for CRUD testing."
        updated_metadata = {"type": "crud_test", "version": 2, "updated": True}
        
        update_success = await self.vector_repo.update_documents(
            collection_name=self.test_collection,
            ids=[doc_id],
            documents=[updated_content],
            metadatas=[updated_metadata],
            tenant_id=tenant_id
        )
        
        if not update_success:
            raise Exception("Failed to update document")
        
        # Verify update
        updated_results = await self.vector_repo.get_documents(
            collection_name=self.test_collection,
            ids=[doc_id],
            tenant_id=tenant_id
        )
        
        if updated_results["documents"][0] != updated_content:
            raise Exception("Document content not updated")
        
        if updated_results["metadatas"][0]["version"] != 2:
            raise Exception("Document metadata not updated")
        
        # DELETE
        delete_success = await self.vector_repo.delete_documents(
            collection_name=self.test_collection,
            ids=[doc_id],
            tenant_id=tenant_id
        )
        
        if not delete_success:
            raise Exception("Failed to delete document")
        
        # Verify deletion
        deleted_results = await self.vector_repo.get_documents(
            collection_name=self.test_collection,
            ids=[doc_id],
            tenant_id=tenant_id
        )
        
        if deleted_results["ids"]:
            raise Exception("Document still exists after deletion")
        
        return {
            "create_success": True,
            "read_success": True,
            "update_success": True,
            "delete_success": True,
            "content_verified": True,
            "metadata_verified": True
        }
    
    async def test_search_functionality(self) -> Dict[str, Any]:
        """Test various search functionality."""
        tenant_id = "search_test_tenant"
        
        # Create test documents with varied content
        test_docs = [
            {
                "id": "search_doc_1",
                "content": "Machine learning and artificial intelligence research paper.",
                "embedding": [0.1] * 384,
                "metadata": {"category": "AI", "priority": "high", "year": 2024}
            },
            {
                "id": "search_doc_2", 
                "content": "Natural language processing and deep learning tutorial.",
                "embedding": [0.2] * 384,
                "metadata": {"category": "NLP", "priority": "medium", "year": 2023}
            },
            {
                "id": "search_doc_3",
                "content": "Computer vision and image recognition algorithms.",
                "embedding": [0.3] * 384,
                "metadata": {"category": "CV", "priority": "low", "year": 2024}
            },
            {
                "id": "search_doc_4",
                "content": "Robotics and autonomous systems development guide.",
                "embedding": [0.4] * 384,
                "metadata": {"category": "Robotics", "priority": "high", "year": 2022}
            }
        ]
        
        # Add test documents
        vector_docs = []
        for doc in test_docs:
            vector_doc = VectorDocument(
                id=doc["id"],
                content=doc["content"],
                embedding=doc["embedding"],
                metadata=doc["metadata"],
                created_at=datetime.utcnow(),
                collection_name=self.test_collection,
                tenant_id=tenant_id
            )
            vector_docs.append(vector_doc)
        
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=vector_docs,
            tenant_id=tenant_id
        )
        
        await self.vector_repo.bulk_upsert_documents(self.test_collection, operation)
        
        # Test basic similarity search
        query_embedding = [0.15] * 384  # Should be closest to doc_1
        basic_results = await self.vector_repo.similarity_search(
            collection_name=self.test_collection,
            query_embedding=query_embedding,
            n_results=4,
            tenant_id=tenant_id
        )
        
        if len(basic_results) != 4:
            raise Exception(f"Expected 4 search results, got {len(basic_results)}")
        
        # Test filtered search
        filtered_results = await self.vector_repo.similarity_search(
            collection_name=self.test_collection,
            query_embedding=query_embedding,
            n_results=10,
            metadata_filter={"year": 2024},
            tenant_id=tenant_id
        )
        
        # Should only return docs from 2024
        for result in filtered_results:
            if result["metadata"]["year"] != 2024:
                raise Exception("Filtered search returned incorrect results")
        
        # Test advanced search
        advanced_request = VectorSearchRequest(
            query_embedding=query_embedding,
            collection_name=self.test_collection,
            n_results=4,
            similarity_threshold=0.0,
            metadata_filter={"priority": "high"},
            tenant_id=tenant_id,
            rerank=True
        )
        
        advanced_results = await self.vector_repo.semantic_search_advanced(advanced_request)
        
        # Should only return high priority docs
        high_priority_results = [r for r in advanced_results if r.document.metadata["priority"] == "high"]
        if len(high_priority_results) != len(advanced_results):
            raise Exception("Advanced search filtering failed")
        
        # Test threshold filtering
        threshold_results = await self.vector_repo.similarity_search(
            collection_name=self.test_collection,
            query_embedding=query_embedding,
            n_results=10,
            threshold=0.8,  # High threshold
            tenant_id=tenant_id
        )
        
        # Verify all results meet threshold
        for result in threshold_results:
            if result["similarity"] < 0.8:
                raise Exception(f"Result below threshold: {result['similarity']}")
        
        return {
            "basic_search_count": len(basic_results),
            "filtered_search_count": len(filtered_results),
            "advanced_search_count": len(advanced_results),
            "threshold_search_count": len(threshold_results),
            "all_searches_successful": True
        }
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks for key operations."""
        tenant_id = "perf_test_tenant"
        
        # Benchmark data preparation
        benchmark_docs = []
        for i in range(1000):
            doc_id = f"perf_doc_{i}"
            content = f"Performance test document {i} with varied content for benchmarking."
            embedding = [float(i % 100) / 100.0] * 384  # Varied embeddings
            metadata = {"doc_index": i, "batch": i // 100, "type": "performance_test"}
            
            vector_doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.utcnow(),
                collection_name=self.test_collection,
                tenant_id=tenant_id
            )
            benchmark_docs.append(vector_doc)
        
        # Benchmark bulk insert
        insert_times = []
        batch_size = 100
        
        for i in range(0, len(benchmark_docs), batch_size):
            batch = benchmark_docs[i:i + batch_size]
            
            start_time = time.time()
            operation = BulkVectorOperation(
                operation_type="insert",
                documents=batch,
                tenant_id=tenant_id
            )
            
            await self.vector_repo.bulk_upsert_documents(self.test_collection, operation)
            insert_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Benchmark search operations
        search_times = []
        query_embeddings = [[float(i) / 10.0] * 384 for i in range(50)]
        
        for query_embedding in query_embeddings:
            start_time = time.time()
            await self.vector_repo.similarity_search(
                collection_name=self.test_collection,
                query_embedding=query_embedding,
                n_results=10,
                tenant_id=tenant_id
            )
            search_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate performance metrics
        insert_metrics = PerformanceMetrics(
            operation="bulk_insert",
            tenant_id=tenant_id,
            avg_latency_ms=statistics.mean(insert_times),
            min_latency_ms=min(insert_times),
            max_latency_ms=max(insert_times),
            p95_latency_ms=statistics.quantiles(insert_times, n=20)[18],  # 95th percentile
            p99_latency_ms=statistics.quantiles(insert_times, n=100)[98],  # 99th percentile
            throughput_ops_per_sec=(batch_size * len(insert_times)) / (sum(insert_times) / 1000),
            total_operations=len(insert_times)
        )
        
        search_metrics = PerformanceMetrics(
            operation="search",
            tenant_id=tenant_id,
            avg_latency_ms=statistics.mean(search_times),
            min_latency_ms=min(search_times),
            max_latency_ms=max(search_times),
            p95_latency_ms=statistics.quantiles(search_times, n=20)[18],
            p99_latency_ms=statistics.quantiles(search_times, n=100)[98],
            throughput_ops_per_sec=len(search_times) / (sum(search_times) / 1000),
            total_operations=len(search_times)
        )
        
        self.performance_metrics.extend([insert_metrics, search_metrics])
        
        # Performance thresholds (adjust based on requirements)
        performance_checks = {
            "insert_avg_latency_acceptable": insert_metrics.avg_latency_ms < 500,  # <500ms avg
            "search_avg_latency_acceptable": search_metrics.avg_latency_ms < 50,   # <50ms avg
            "search_p95_latency_acceptable": search_metrics.p95_latency_ms < 100,  # <100ms p95
            "insert_throughput_acceptable": insert_metrics.throughput_ops_per_sec > 100,  # >100 docs/sec
            "search_throughput_acceptable": search_metrics.throughput_ops_per_sec > 100   # >100 searches/sec
        }
        
        return {
            "documents_inserted": len(benchmark_docs),
            "search_queries_executed": len(query_embeddings),
            "insert_performance": insert_metrics.to_dict(),
            "search_performance": search_metrics.to_dict(),
            "performance_checks": performance_checks,
            "all_performance_checks_passed": all(performance_checks.values())
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and resilience."""
        tenant_id = "error_test_tenant"
        
        error_tests = {}
        
        # Test invalid collection name
        try:
            await self.vector_repo.similarity_search(
                collection_name="nonexistent_collection",
                query_embedding=[0.1] * 384,
                n_results=5,
                tenant_id=tenant_id
            )
            error_tests["invalid_collection"] = "failed_to_raise_error"
        except Exception:
            error_tests["invalid_collection"] = "properly_handled"
        
        # Test invalid embedding dimensions
        try:
            await self.vector_repo.similarity_search(
                collection_name=self.test_collection,
                query_embedding=[0.1] * 100,  # Wrong dimension
                n_results=5,
                tenant_id=tenant_id
            )
            error_tests["invalid_dimensions"] = "failed_to_raise_error"
        except Exception:
            error_tests["invalid_dimensions"] = "properly_handled"
        
        # Test empty search
        try:
            result = await self.vector_repo.similarity_search(
                collection_name=self.test_collection,
                query_embedding=[0.1] * 384,
                n_results=5,
                tenant_id="nonexistent_tenant"
            )
            # Should return empty results, not error
            error_tests["empty_tenant_search"] = "properly_handled" if len(result) == 0 else "unexpected_results"
        except Exception:
            error_tests["empty_tenant_search"] = "unexpected_error"
        
        # Test invalid document operations
        try:
            await self.vector_repo.get_documents(
                collection_name=self.test_collection,
                ids=["nonexistent_doc"],
                tenant_id=tenant_id
            )
            error_tests["nonexistent_document"] = "properly_handled"
        except Exception:
            error_tests["nonexistent_document"] = "unexpected_error"
        
        return {
            "error_handling_tests": error_tests,
            "all_errors_handled_properly": all(
                status == "properly_handled" for status in error_tests.values()
            )
        }
    
    async def test_data_consistency(self) -> Dict[str, Any]:
        """Test data consistency across operations."""
        tenant_id = "consistency_test_tenant"
        
        # Add test documents
        test_docs = []
        for i in range(50):
            doc_id = f"consistency_doc_{i}"
            content = f"Consistency test document {i}."
            embedding = [float(i) / 50.0] * 384
            metadata = {"index": i, "type": "consistency_test"}
            
            vector_doc = VectorDocument(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.utcnow(),
                collection_name=self.test_collection,
                tenant_id=tenant_id
            )
            test_docs.append(vector_doc)
        
        # Bulk insert
        operation = BulkVectorOperation(
            operation_type="insert",
            documents=test_docs,
            tenant_id=tenant_id
        )
        
        insert_result = await self.vector_repo.bulk_upsert_documents(
            self.test_collection,
            operation
        )
        
        # Verify all documents were inserted
        if insert_result["total_processed"] != len(test_docs):
            raise Exception("Not all documents were inserted")
        
        # Count documents
        doc_count = await self.vector_repo.count_documents(
            collection_name=self.test_collection,
            tenant_id=tenant_id
        )
        
        if doc_count != len(test_docs):
            raise Exception(f"Document count mismatch: expected {len(test_docs)}, got {doc_count}")
        
        # Get all documents
        all_docs = await self.vector_repo.get_documents(
            collection_name=self.test_collection,
            tenant_id=tenant_id,
            limit=len(test_docs)
        )
        
        if len(all_docs["ids"]) != len(test_docs):
            raise Exception("Retrieved document count doesn't match inserted count")
        
        # Verify document IDs match
        inserted_ids = {doc.id for doc in test_docs}
        retrieved_ids = set(all_docs["ids"])
        
        if inserted_ids != retrieved_ids:
            missing_ids = inserted_ids - retrieved_ids
            extra_ids = retrieved_ids - inserted_ids
            raise Exception(f"Document ID mismatch. Missing: {missing_ids}, Extra: {extra_ids}")
        
        # Test search consistency
        search_results = await self.vector_repo.similarity_search(
            collection_name=self.test_collection,
            query_embedding=[0.5] * 384,
            n_results=len(test_docs),
            tenant_id=tenant_id
        )
        
        search_ids = {result["id"] for result in search_results}
        if not search_ids.issubset(inserted_ids):
            raise Exception("Search returned unexpected document IDs")
        
        return {
            "documents_inserted": len(test_docs),
            "documents_counted": doc_count,
            "documents_retrieved": len(all_docs["ids"]),
            "documents_searchable": len(search_results),
            "id_consistency": True,
            "count_consistency": True,
            "search_consistency": True
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        self.logger.info("Starting comprehensive Qdrant integration validation...")
        
        try:
            await self.setup()
            
            # Run all test suites
            await self.run_test("Basic Connectivity", self.test_basic_connectivity)
            await self.run_test("Multi-Tenant Isolation", self.test_multi_tenant_isolation)
            await self.run_test("CRUD Operations", self.test_crud_operations)
            await self.run_test("Search Functionality", self.test_search_functionality)
            await self.run_test("Performance Benchmarks", self.test_performance_benchmarks)
            await self.run_test("Error Handling", self.test_error_handling)
            await self.run_test("Data Consistency", self.test_data_consistency)
            
        finally:
            await self.cleanup()
        
        # Compile final results
        passed_tests = [r for r in self.results if r.status == "passed"]
        failed_tests = [r for r in self.results if r.status == "failed"]
        
        total_duration = sum(r.duration_seconds for r in self.results)
        
        summary = {
            "validation_completed": True,
            "total_tests": len(self.results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(passed_tests) / len(self.results) * 100,
            "total_duration_seconds": total_duration,
            "test_results": [r.to_dict() for r in self.results],
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "overall_status": "PASS" if len(failed_tests) == 0 else "FAIL"
        }
        
        self._log_summary(summary)
        return summary
    
    def _log_summary(self, summary: Dict[str, Any]):
        """Log validation summary."""
        self.logger.info("=" * 70)
        self.logger.info("QDRANT INTEGRATION VALIDATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Tests: {summary['total_tests']}")
        self.logger.info(f"Passed: {summary['passed_tests']}")
        self.logger.info(f"Failed: {summary['failed_tests']}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"Duration: {summary['total_duration_seconds']:.2f} seconds")
        self.logger.info(f"Overall Status: {summary['overall_status']}")
        
        if summary['failed_tests'] > 0:
            self.logger.error("Failed Tests:")
            for result in self.results:
                if result.status == "failed":
                    self.logger.error(f"  - {result.test_name}: {result.error_message}")
        
        if self.performance_metrics:
            self.logger.info("\nPerformance Summary:")
            for metric in self.performance_metrics:
                self.logger.info(
                    f"  {metric.operation}: {metric.avg_latency_ms:.1f}ms avg, "
                    f"{metric.throughput_ops_per_sec:.1f} ops/sec"
                )
        
        self.logger.info("=" * 70)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save validation results to JSON file."""
        if not filename:
            filename = f"qdrant_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            "validation_timestamp": datetime.now().isoformat(),
            "qdrant_config": {
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "api_key_configured": bool(self.qdrant_api_key)
            },
            "test_results": [r.to_dict() for r in self.results],
            "performance_metrics": [m.to_dict() for m in self.performance_metrics]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Validation results saved to: {filename}")
        return filename


async def main():
    """Main function to run validation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Qdrant integration")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    validator = QdrantIntegrationValidator(
        qdrant_host=args.host,
        qdrant_port=args.port,
        qdrant_api_key=args.api_key
    )
    
    try:
        summary = await validator.run_all_tests()
        validator.save_results(args.output)
        
        if summary["overall_status"] == "PASS":
            print("✅ All Qdrant integration tests PASSED!")
            exit(0)
        else:
            print(f"❌ {summary['failed_tests']} tests FAILED!")
            exit(1)
            
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
