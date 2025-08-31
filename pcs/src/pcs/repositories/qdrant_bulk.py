"""
Filepath: pcs/src/pcs/repositories/qdrant_bulk.py
Purpose: Bulk operations for Qdrant including batch processing, multi-tenancy, and bulk document operations
Related Components: Batch processing, bulk upsert, multi-tenancy, performance optimization
Tags: qdrant, bulk-operations, batch-processing, multi-tenancy, performance
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime

from .qdrant_types import QdrantPoint

logger = logging.getLogger(__name__)


@dataclass
class BulkVectorOperation:
    """Container for bulk vector operations with multi-tenancy support."""
    operation_type: str  # "upsert", "delete", "update"
    collection_name: str
    documents: Optional[List[Any]] = None  # List of VectorDocument or similar
    tenant_id: Optional[str] = None
    batch_size: int = 100
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "collection_name": self.collection_name,
            "documents_count": len(self.documents) if self.documents else 0,
            "tenant_id": self.tenant_id,
            "batch_size": self.batch_size,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }


@dataclass
class BulkOperationResult:
    """Result of a bulk operation with detailed metrics."""
    operation_type: str
    collection_name: str
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[Dict[str, Any]]
    execution_time: float
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "collection_name": self.collection_name,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }


class QdrantBulkOperations:
    """
    Bulk operations for Qdrant vector database.
    
    This class handles batch processing, multi-tenancy, and bulk document
    operations with performance optimization and error handling.
    """
    
    def __init__(self, core_operations, max_concurrent_batches: int = 5):
        """
        Initialize bulk operations with core operations.
        
        Args:
            core_operations: QdrantCoreOperations instance for basic operations
            max_concurrent_batches: Maximum number of concurrent batch operations
        """
        self.core = core_operations
        self.max_concurrent_batches = max_concurrent_batches
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._operation_metrics = []
        self._batch_sizes = []
    
    async def bulk_upsert_documents(
        self,
        collection_name: str,
        documents: List[Any],
        batch_size: int = 100,
        tenant_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        retry_failed: bool = True,
        max_retries: int = 3
    ) -> BulkOperationResult:
        """
        Bulk upsert documents with batching and error handling.
        
        Args:
            collection_name: Name of the collection to upsert to
            documents: List of documents to upsert
            batch_size: Number of documents per batch
            tenant_id: Optional tenant identifier for multi-tenancy
            progress_callback: Optional callback for progress updates
            retry_failed: Whether to retry failed operations
            max_retries: Maximum number of retry attempts
            
        Returns:
            BulkOperationResult with operation metrics
        """
        start_time = datetime.utcnow()
        total_items = len(documents)
        successful_items = 0
        failed_items = 0
        errors = []
        
        try:
            self.logger.info(
                f"Starting bulk upsert for {total_items} documents in {collection_name} "
                f"(batch_size: {batch_size}, tenant: {tenant_id})"
            )
            
            # Split documents into batches
            batches = self._create_batches(documents, batch_size)
            total_batches = len(batches)
            
            if progress_callback:
                progress_callback(0, total_batches, "Starting bulk upsert...")
            
            # Process batches with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent_batches)
            
            async def process_batch(batch: List[Any], batch_num: int) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        # Convert documents to QdrantPoint format
                        points = self._convert_documents_to_points(batch, tenant_id)
                        
                        # Upsert batch
                        result = self.core.upsert_points(collection_name, points, wait=True)
                        
                        if result.get("status") == "ok":
                            batch_success = len(batch)
                            batch_failed = 0
                        else:
                            batch_success = 0
                            batch_failed = len(batch)
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(batch_num + 1, total_batches, f"Processed batch {batch_num + 1}")
                        
                        return {
                            "batch_num": batch_num,
                            "success": batch_success,
                            "failed": batch_failed,
                            "result": result
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Batch {batch_num} failed: {e}")
                        return {
                            "batch_num": batch_num,
                            "success": 0,
                            "failed": len(batch),
                            "error": str(e)
                        }
            
            # Process all batches concurrently
            batch_tasks = [
                process_batch(batch, i) for i, batch in enumerate(batches)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Aggregate results
            for batch_result in batch_results:
                if isinstance(batch_result, dict):
                    successful_items += batch_result.get("success", 0)
                    failed_items += batch_result.get("failed", 0)
                    
                    if "error" in batch_result:
                        errors.append({
                            "batch_num": batch_result.get("batch_num"),
                            "error": batch_result.get("error"),
                            "type": "batch_processing_error"
                        })
                else:
                    # Exception occurred
                    failed_items += batch_size
                    errors.append({
                        "error": str(batch_result),
                        "type": "batch_exception"
                    })
            
            # Retry failed items if requested
            if retry_failed and failed_items > 0:
                self.logger.info(f"Retrying {failed_items} failed items...")
                retry_result = await self._retry_failed_operations(
                    collection_name, documents, batch_size, tenant_id, max_retries
                )
                successful_items += retry_result.successful_items
                failed_items = retry_result.failed_items
                errors.extend(retry_result.errors)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = BulkOperationResult(
                operation_type="bulk_upsert",
                collection_name=collection_name,
                total_items=total_items,
                successful_items=successful_items,
                failed_items=failed_items,
                errors=errors,
                execution_time=execution_time,
                tenant_id=tenant_id,
                metadata={
                    "batch_size": batch_size,
                    "total_batches": total_batches,
                    "max_concurrent_batches": self.max_concurrent_batches
                }
            )
            
            # Track metrics
            self._operation_metrics.append(result.to_dict())
            self._batch_sizes.append(batch_size)
            
            self.logger.info(
                f"Bulk upsert completed: {successful_items}/{total_items} successful "
                f"({result.success_rate:.1f}%) in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Bulk upsert failed: {e}")
            
            return BulkOperationResult(
                operation_type="bulk_upsert",
                collection_name=collection_name,
                total_items=total_items,
                successful_items=successful_items,
                failed_items=total_items,
                errors=[{"error": str(e), "type": "operation_failure"}],
                execution_time=execution_time,
                tenant_id=tenant_id
            )
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches of specified size."""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def _convert_documents_to_points(
        self, 
        documents: List[Any], 
        tenant_id: Optional[str] = None
    ) -> List[QdrantPoint]:
        """Convert documents to QdrantPoint format with tenant support."""
        points = []
        
        for doc in documents:
            try:
                # Extract vector and metadata from document
                # This is a simplified implementation - adjust based on your document structure
                if hasattr(doc, 'vector') and hasattr(doc, 'id'):
                    payload = {}
                    
                    # Add tenant ID if provided
                    if tenant_id:
                        payload["tenant_id"] = tenant_id
                    
                    # Add document metadata
                    if hasattr(doc, 'metadata'):
                        payload.update(doc.metadata)
                    
                    # Add document content if available
                    if hasattr(doc, 'content'):
                        payload["content"] = doc.content
                    
                    point = QdrantPoint(
                        id=doc.id,
                        vector=doc.vector,
                        payload=payload
                    )
                    points.append(point)
                else:
                    self.logger.warning(f"Document missing required fields: {doc}")
                    
            except Exception as e:
                self.logger.error(f"Failed to convert document to point: {e}")
                continue
        
        return points
    
    async def _retry_failed_operations(
        self,
        collection_name: str,
        documents: List[Any],
        batch_size: int,
        tenant_id: Optional[str],
        max_retries: int
    ) -> BulkOperationResult:
        """Retry failed operations with exponential backoff."""
        # This is a simplified retry implementation
        # In production, you might want more sophisticated retry logic
        
        retry_successful = 0
        retry_failed = 0
        retry_errors = []
        
        for attempt in range(max_retries):
            try:
                # Wait before retry with exponential backoff
                if attempt > 0:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                
                self.logger.info(f"Retry attempt {attempt + 1} for failed operations...")
                
                # For now, just return a simple result
                # In production, implement actual retry logic
                retry_successful = 0
                retry_failed = len(documents)
                retry_errors.append({
                    "attempt": attempt + 1,
                    "error": "Retry not fully implemented in this version"
                })
                
            except Exception as e:
                retry_errors.append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        return BulkOperationResult(
            operation_type="retry",
            collection_name=collection_name,
            total_items=len(documents),
            successful_items=retry_successful,
            failed_items=retry_failed,
            errors=retry_errors,
            execution_time=0.0,
            tenant_id=tenant_id
        )
    
    async def bulk_delete_documents(
        self,
        collection_name: str,
        document_ids: List[Union[str, int]],
        batch_size: int = 100,
        tenant_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> BulkOperationResult:
        """
        Bulk delete documents with batching.
        
        Args:
            collection_name: Name of the collection to delete from
            document_ids: List of document IDs to delete
            batch_size: Number of IDs per batch
            tenant_id: Optional tenant identifier for multi-tenancy
            progress_callback: Optional callback for progress updates
            
        Returns:
            BulkOperationResult with operation metrics
        """
        start_time = datetime.utcnow()
        total_items = len(document_ids)
        successful_items = 0
        failed_items = 0
        errors = []
        
        try:
            self.logger.info(
                f"Starting bulk delete for {total_items} documents in {collection_name} "
                f"(batch_size: {batch_size}, tenant: {tenant_id})"
            )
            
            # Split IDs into batches
            batches = self._create_batches(document_ids, batch_size)
            total_batches = len(batches)
            
            if progress_callback:
                progress_callback(0, total_batches, "Starting bulk delete...")
            
            # Process batches
            for i, batch in enumerate(batches):
                try:
                    # Delete batch
                    result = self.core.delete_points(collection_name, batch, wait=True)
                    
                    if result.get("status") == "ok":
                        successful_items += len(batch)
                    else:
                        failed_items += len(batch)
                        errors.append({
                            "batch_num": i,
                            "error": f"Delete failed: {result}",
                            "type": "delete_failure"
                        })
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(i + 1, total_batches, f"Processed batch {i + 1}")
                    
                except Exception as e:
                    failed_items += len(batch)
                    errors.append({
                        "batch_num": i,
                        "error": str(e),
                        "type": "batch_exception"
                    })
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = BulkOperationResult(
                operation_type="bulk_delete",
                collection_name=collection_name,
                total_items=total_items,
                successful_items=successful_items,
                failed_items=failed_items,
                errors=errors,
                execution_time=execution_time,
                tenant_id=tenant_id,
                metadata={"batch_size": batch_size, "total_batches": total_batches}
            )
            
            self.logger.info(
                f"Bulk delete completed: {successful_items}/{total_items} successful "
                f"({result.success_rate:.1f}%) in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Bulk delete failed: {e}")
            
            return BulkOperationResult(
                operation_type="bulk_delete",
                collection_name=collection_name,
                total_items=total_items,
                successful_items=0,
                failed_items=total_items,
                errors=[{"error": str(e), "type": "operation_failure"}],
                execution_time=execution_time,
                tenant_id=tenant_id
            )
    
    def get_operation_metrics(self) -> List[Dict[str, Any]]:
        """Get historical operation metrics for analysis."""
        return self._operation_metrics.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from tracked metrics."""
        if not self._operation_metrics:
            return {}
        
        total_operations = len(self._operation_metrics)
        avg_execution_time = sum(m["execution_time"] for m in self._operation_metrics) / total_operations
        avg_success_rate = sum(m["success_rate"] for m in self._operation_metrics) / total_operations
        
        return {
            "total_operations": total_operations,
            "average_execution_time": avg_execution_time,
            "average_success_rate": avg_success_rate,
            "average_batch_size": sum(self._batch_sizes) / len(self._batch_sizes) if self._batch_sizes else 0
        }
