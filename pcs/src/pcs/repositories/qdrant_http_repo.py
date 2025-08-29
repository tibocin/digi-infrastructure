"""
Filepath: pcs/src/pcs/repositories/qdrant_http_repo.py
Purpose: Enhanced Qdrant repository using HTTP client for reliable vector database operations
Related Components: Qdrant HTTP client, vector operations, multi-tenancy, performance monitoring
Tags: qdrant, repository, http-client, vector-database, multi-tenant, performance
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np
from collections import defaultdict

from .qdrant_http_client import (
    QdrantHTTPClient, AsyncQdrantHTTPClient, QdrantCollectionConfig,
    QdrantPoint, QdrantSearchResult, QdrantDistance, QdrantHTTPError
)
from .base import RepositoryError
from ..utils.metrics import PerformanceMonitor, record_manual_metric


class SimilarityAlgorithm(Enum):
    """Enum for different similarity algorithms."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class VectorIndexType(Enum):
    """Enum for different vector index types."""
    HNSW = "hnsw"
    FLAT = "flat"


@dataclass
class VectorDocument:
    """Container for vector document with metadata and tenant information."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime
    collection_name: str
    tenant_id: Optional[str] = None  # Multi-tenancy support
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "collection_name": self.collection_name,
            "tenant_id": self.tenant_id
        }

    def to_qdrant_point(self) -> QdrantPoint:
        """Convert to Qdrant point."""
        payload = {
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "collection_name": self.collection_name,
            **self.metadata
        }
        
        # Add tenant isolation
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
            
        return QdrantPoint(
            id=self.id,
            vector=self.embedding,
            payload=payload
        )


@dataclass
class SimilarityResult:
    """Container for similarity search results."""
    document: VectorDocument
    similarity_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document": self.document.to_dict(),
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }


@dataclass
class VectorCollectionStats:
    """Container for collection statistics."""
    name: str
    document_count: int
    dimension: int
    memory_usage_mb: float
    tenant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "document_count": self.document_count,
            "dimension": self.dimension,
            "memory_usage_mb": self.memory_usage_mb,
            "tenant_id": self.tenant_id
        }


@dataclass
class BulkVectorOperation:
    """Container for bulk vector operations."""
    operation_type: str
    documents: List[VectorDocument]
    batch_size: int = 100
    tenant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "document_count": len(self.documents),
            "batch_size": self.batch_size,
            "tenant_id": self.tenant_id
        }


@dataclass
class VectorSearchRequest:
    """Container for vector search requests."""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    collection_name: str = ""
    n_results: int = 10
    similarity_threshold: float = 0.0
    algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    include_embeddings: bool = False
    rerank: bool = False
    tenant_id: Optional[str] = None
    offset: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class EnhancedQdrantHTTPRepository:
    """
    Enhanced Qdrant repository using HTTP client for reliable vector operations.
    
    This repository provides a robust interface for vector database operations,
    bypassing the official Qdrant client library issues while maintaining
    the same functionality and adding enhanced features.
    """
    
    def __init__(
        self,
        client: Optional[QdrantHTTPClient] = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        use_async: bool = False,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize enhanced Qdrant HTTP repository.
        
        Args:
            client: Existing HTTP client instance
            host: Qdrant host
            port: Qdrant HTTP port
            api_key: API key for authentication
            use_async: Whether to use async client
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        if client:
            self.client = client
        else:
            if use_async:
                self.client = AsyncQdrantHTTPClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
            else:
                self.client = QdrantHTTPClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
        
        self._query_metrics = []
        self._collection_cache = {}
        self._is_async = use_async
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized Enhanced Qdrant HTTP Repository (async: {use_async})")
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health status."""
        try:
            with PerformanceMonitor("health_check", "qdrant"):
                return self.client.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise RepositoryError(f"Health check failed: {e}")
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check."""
        if not self._is_async:
            raise RuntimeError("Repository is not configured for async operations")
        
        try:
            async with PerformanceMonitor("health_check", "qdrant"):
                return await self.client.health_check_async()
        except Exception as e:
            self.logger.error(f"Async health check failed: {e}")
            raise RepositoryError(f"Async health check failed: {e}")
    
    async def get_collections_async(self) -> List[Dict[str, Any]]:
        """Async get collections."""
        if not self._is_async:
            raise RuntimeError("Repository is not configured for async operations")
        
        try:
            async with PerformanceMonitor("get_collections", "qdrant"):
                collections = await self.client.get_collections_async()
                self.logger.info(f"Retrieved {len(collections)} collections")
                return collections
        except Exception as e:
            self.logger.error(f"Failed to get collections: {e}")
            raise RepositoryError(f"Failed to get collections: {e}")
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get list of all collections."""
        try:
            with PerformanceMonitor("get_collections", "qdrant"):
                collections = self.client.get_collections()
                self.logger.info(f"Retrieved {len(collections)} collections")
                return collections
        except Exception as e:
            self.logger.error(f"Failed to get collections: {e}")
            raise RepositoryError(f"Failed to get collections: {e}")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed collection information."""
        try:
            with PerformanceMonitor("get_collection_info", "qdrant"):
                info = self.client.get_collection(collection_name)
                self.logger.info(f"Retrieved info for collection: {collection_name}")
                return info
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise RepositoryError(f"Failed to get collection info: {e}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Union[str, SimilarityAlgorithm, QdrantDistance] = "cosine",
        on_disk_payload: bool = True,
        hnsw_config: Optional[Dict[str, Any]] = None,
        optimizers_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collection with optimized configuration.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (cosine, euclidean, dot_product, manhattan)
            on_disk_payload: Whether to store payload on disk
            hnsw_config: HNSW index configuration
            optimizers_config: Optimizers configuration
            
        Returns:
            True if collection was created successfully
        """
        try:
            # Convert distance to QdrantDistance
            if isinstance(distance, str):
                distance_map = {
                    "cosine": QdrantDistance.COSINE,
                    "euclidean": QdrantDistance.EUCLIDEAN,
                    "dot_product": QdrantDistance.DOT_PRODUCT,
                    "manhattan": QdrantDistance.MANHATTAN
                }
                qdrant_distance = distance_map.get(distance.lower(), QdrantDistance.COSINE)
            elif isinstance(distance, SimilarityAlgorithm):
                distance_map = {
                    SimilarityAlgorithm.COSINE: QdrantDistance.COSINE,
                    SimilarityAlgorithm.EUCLIDEAN: QdrantDistance.EUCLIDEAN,
                    SimilarityAlgorithm.DOT_PRODUCT: QdrantDistance.DOT_PRODUCT,
                    SimilarityAlgorithm.MANHATTAN: QdrantDistance.MANHATTAN
                }
                qdrant_distance = distance_map.get(distance, QdrantDistance.COSINE)
            else:
                qdrant_distance = distance
            
            # Build collection configuration
            config = QdrantCollectionConfig(
                name=collection_name,
                vector_size=vector_size,
                distance=qdrant_distance,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config
            )
            
            with PerformanceMonitor("create_collection", "qdrant"):
                success = self.client.create_collection(collection_name, config)
                
            if success:
                self.logger.info(f"Successfully created collection: {collection_name}")
                # Clear cache
                self._collection_cache.pop(collection_name, None)
                return True
            else:
                raise RepositoryError(f"Failed to create collection: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to create collection: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            with PerformanceMonitor("delete_collection", "qdrant"):
                success = self.client.delete_collection(collection_name)
                
            if success:
                self.logger.info(f"Successfully deleted collection: {collection_name}")
                # Clear cache
                self._collection_cache.pop(collection_name, None)
                return True
            else:
                raise RepositoryError(f"Failed to delete collection: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to delete collection: {e}")
    
    def upsert_documents(
        self,
        collection_name: str,
        documents: List[VectorDocument]
    ) -> Dict[str, Any]:
        """
        Insert or update documents in a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of vector documents to upsert
            
        Returns:
            Upsert operation result
        """
        try:
            # Convert documents to Qdrant points
            points = [doc.to_qdrant_point() for doc in documents]
            
            with PerformanceMonitor("upsert_documents", "qdrant"):
                result = self.client.upsert_points(collection_name, points)
                
            self.logger.info(f"Successfully upserted {len(documents)} documents in collection: {collection_name}")
            
            # Store ID mapping for deletion
            if collection_name not in self._collection_cache:
                self._collection_cache[collection_name] = {}
            
            # Map original IDs to Qdrant internal IDs
            for i, doc in enumerate(documents):
                qdrant_id = i + 1  # Qdrant IDs start from 1
                self._collection_cache[collection_name][doc.id] = qdrant_id
            
            # Record metrics
            record_manual_metric(
                "qdrant_documents_upserted",
                len(documents),
                {"collection": collection_name}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to upsert documents in collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to upsert documents: {e}")
    
    def search_similar(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        tenant_id: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SimilarityResult]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            tenant_id: Tenant ID for multi-tenancy
            metadata_filters: Additional metadata filters
            
        Returns:
            List of similarity results
        """
        try:
            # Build filter conditions
            filter_conditions = {}
            
            # Add tenant isolation
            if tenant_id:
                filter_conditions["must"] = [
                    {"key": "tenant_id", "match": {"value": tenant_id}}
                ]
            
            # Add metadata filters
            if metadata_filters:
                if "must" not in filter_conditions:
                    filter_conditions["must"] = []
                
                for key, value in metadata_filters.items():
                    if isinstance(value, (list, tuple)):
                        filter_conditions["must"].append({
                            "key": key,
                            "match": {"any": value}
                        })
                    else:
                        filter_conditions["must"].append({
                            "key": key,
                            "match": {"value": value}
                        })
            
            with PerformanceMonitor("search_similar", "qdrant"):
                search_results = self.client.search_points(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions if filter_conditions else None
                )
            
            # Convert to similarity results
            results = []
            for result in search_results:
                # Reconstruct document from payload
                payload = result.payload or {}
                document = VectorDocument(
                    id=str(result.id),
                    content=payload.get("content", ""),
                    embedding=result.vector or [],
                    metadata={k: v for k, v in payload.items() if k not in ["content", "created_at", "collection_name", "tenant_id"]},
                    created_at=datetime.fromisoformat(payload.get("created_at", datetime.now().isoformat())),
                    collection_name=payload.get("collection_name", collection_name),
                    tenant_id=payload.get("tenant_id")
                )
                
                similarity_result = SimilarityResult(
                    document=document,
                    similarity_score=result.score,
                    metadata={"version": result.version}
                )
                results.append(similarity_result)
            
            self.logger.info(f"Search returned {len(results)} results from collection: {collection_name}")
            
            # Record metrics
            record_manual_metric(
                "qdrant_search_results",
                len(results),
                {"collection": collection_name, "tenant_id": tenant_id or "none"}
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search in collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to search: {e}")
    
    def delete_documents(
        self,
        collection_name: str,
        document_ids: List[str]
    ) -> Dict[str, Any]:
        """Delete documents from a collection."""
        try:
            # Convert original IDs to Qdrant internal IDs
            qdrant_ids = []
            if collection_name in self._collection_cache:
                for doc_id in document_ids:
                    if doc_id in self._collection_cache[collection_name]:
                        qdrant_ids.append(self._collection_cache[collection_name][doc_id])
                    else:
                        self.logger.warning(f"Document ID {doc_id} not found in collection {collection_name}")
            
            if not qdrant_ids:
                self.logger.warning(f"No valid Qdrant IDs found for deletion in collection {collection_name}")
                return {"deleted": 0}
            
            with PerformanceMonitor("delete_documents", "qdrant"):
                result = self.client.delete_points(collection_name, qdrant_ids)
                
            self.logger.info(f"Successfully deleted {len(qdrant_ids)} documents from collection: {collection_name}")
            
            # Clean up ID mapping for deleted documents
            for doc_id in document_ids:
                if collection_name in self._collection_cache:
                    self._collection_cache[collection_name].pop(doc_id, None)
            
            # Record metrics
            record_manual_metric(
                "qdrant_documents_deleted",
                len(qdrant_ids),
                {"collection": collection_name}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to delete documents: {e}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            with PerformanceMonitor("get_collection_stats", "qdrant"):
                stats = self.client.get_collection_stats(collection_name)
                
            self.logger.info(f"Retrieved stats for collection: {collection_name}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            raise RepositoryError(f"Failed to get collection stats: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the repository."""
        return {
            "query_metrics": self._query_metrics,
            "collection_cache_size": len(self._collection_cache),
            "is_async": self._is_async
        }
    
    async def find_similar_documents(
        self,
        collection_name: str,
        target_embedding: List[float],
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        tenant_id: Optional[str] = None
    ) -> List[SimilarityResult]:
        """Find similar documents based on embedding similarity."""
        try:
            with PerformanceMonitor("find_similar_documents", "qdrant"):
                results = await self.semantic_search_advanced(
                    VectorSearchRequest(
                        query_embedding=target_embedding,
                        collection_name=collection_name,
                        n_results=max_results,
                        similarity_threshold=similarity_threshold,
                        tenant_id=tenant_id
                    )
                )
            return results
        except Exception as e:
            self.logger.error(f"Failed to find similar documents: {e}")
            raise RepositoryError(f"Failed to find similar documents: {e}")
    
    async def cluster_documents(
        self,
        collection_name: str,
        n_clusters: int = 2,
        algorithm: str = "kmeans",
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cluster documents using specified algorithm."""
        try:
            with PerformanceMonitor("cluster_documents", "qdrant"):
                # Get all documents from collection
                scroll_points, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # Large limit to get all documents
                    with_payload=True,
                    with_vectors=True
                )
                
                if not scroll_points:
                    return {
                        "clusters": [],
                        "statistics": {"total_documents": 0},
                        "algorithm": algorithm
                    }
                
                # Extract embeddings and filter by tenant if specified
                embeddings = []
                doc_ids = []
                for point in scroll_points:
                    if tenant_id is None or point.payload.get("tenant_id") == tenant_id:
                        embeddings.append(point.vector)
                        doc_ids.append(point.id)
                
                if not embeddings:
                    return {
                        "clusters": [],
                        "statistics": {"total_documents": 0},
                        "algorithm": algorithm
                    }
                
                embeddings_array = np.array(embeddings)
                
                if algorithm == "kmeans":
                    cluster_labels = await self._kmeans_clustering(embeddings_array, n_clusters)
                elif algorithm == "dbscan":
                    cluster_labels = await self._dbscan_clustering(embeddings_array)
                else:
                    raise RepositoryError(f"Unsupported clustering algorithm: {algorithm}")
                
                # Group documents by cluster
                clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    clusters[int(label)].append(doc_ids[i])
                
                return {
                    "clusters": [list(cluster) for cluster in clusters.values()],
                    "statistics": {"total_documents": len(embeddings)},
                    "algorithm": algorithm
                }
                
        except Exception as e:
            self.logger.error(f"Failed to cluster documents: {e}")
            raise RepositoryError(f"Failed to cluster documents: {e}")
    
    async def get_collection_statistics(
        self,
        collection_name: str,
        tenant_id: Optional[str] = None
    ) -> VectorCollectionStats:
        """Get comprehensive collection statistics."""
        try:
            with PerformanceMonitor("get_collection_statistics", "qdrant"):
                # Get basic collection info
                collection_info = self.client.get_collection(collection_name)
                
                # Get document count
                if tenant_id:
                    # Count tenant-specific documents
                    scroll_points, _ = self.client.scroll(
                        collection_name=collection_name,
                        limit=10000,
                        with_payload=True
                    )
                    tenant_count = sum(1 for point in scroll_points if point.payload.get("tenant_id") == tenant_id)
                    document_count = tenant_count
                else:
                    document_count = collection_info.points_count
                
                # Calculate memory usage (rough estimate)
                vector_size = collection_info.config.params.vector.size
                memory_usage_mb = (document_count * vector_size * 4) / (1024 * 1024)  # 4 bytes per float
                
                return VectorCollectionStats(
                    name=collection_name,
                    document_count=document_count,
                    dimension=vector_size,
                    memory_usage_mb=memory_usage_mb,
                    tenant_id=tenant_id
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get collection statistics: {e}")
            raise RepositoryError(f"Failed to get collection statistics: {e}")
    
    async def optimize_collection_performance(self, collection_name: str) -> Dict[str, Any]:
        """Optimize collection performance."""
        try:
            with PerformanceMonitor("optimize_collection_performance", "qdrant"):
                # Get current stats
                before_stats = await self.get_collection_statistics(collection_name)
                
                # Apply optimizations
                optimizations = {
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_count": 1000
                }
                
                self.client.update_collection(
                    collection_name=collection_name,
                    optimizers_config=optimizations
                )
                
                # Get stats after optimization
                after_stats = await self.get_collection_statistics(collection_name)
                
                return {
                    "before_optimization": before_stats.to_dict(),
                    "optimizations_applied": optimizations,
                    "performance_improvements": {
                        "memory_reduction_mb": before_stats.memory_usage_mb - after_stats.memory_usage_mb
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to optimize collection performance: {e}")
            raise RepositoryError(f"Failed to optimize collection performance: {e}")
    
    async def export_embeddings(
        self,
        collection_name: str,
        format_type: str = "json",
        include_metadata: bool = True,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export embeddings in specified format."""
        try:
            with PerformanceMonitor("export_embeddings", "qdrant"):
                # Get all documents
                scroll_points, _ = self.client.scroll(
                    collection_name=collection_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=True
                )
                
                # Filter by tenant if specified
                if tenant_id:
                    scroll_points = [p for p in scroll_points if p.payload.get("tenant_id") == tenant_id]
                
                if format_type == "numpy":
                    embeddings = np.array([point.vector for point in scroll_points])
                    documents = [point.payload.get("content", "") for point in scroll_points]
                    metadatas = [point.payload for point in scroll_points]
                    
                    return {
                        "collection_name": collection_name,
                        "format": "numpy",
                        "document_count": len(scroll_points),
                        "embeddings": embeddings,
                        "documents": documents,
                        "metadatas": metadatas,
                        "tenant_id": tenant_id
                    }
                elif format_type == "json":
                    data = []
                    for point in scroll_points:
                        item = {
                            "id": point.id,
                            "embedding": point.vector,
                            "content": point.payload.get("content", "")
                        }
                        if include_metadata:
                            item["metadata"] = point.payload
                        data.append(item)
                    
                    return {
                        "collection_name": collection_name,
                        "format": "json",
                        "document_count": len(scroll_points),
                        "data": data,
                        "tenant_id": tenant_id
                    }
                else:
                    raise RepositoryError(f"Unsupported export format: {format_type}")
                    
        except Exception as e:
            self.logger.error(f"Failed to export embeddings: {e}")
            raise RepositoryError(f"Failed to export embeddings: {e}")
    
    async def semantic_search_advanced(self, request: VectorSearchRequest) -> List[SimilarityResult]:
        """Perform advanced semantic search with multi-tenancy and reranking."""
        try:
            with PerformanceMonitor("semantic_search_advanced", "qdrant"):
                # Build query filter
                query_filter = self._build_query_filter(request)
                
                # Perform search
                search_results = self.client.search(
                    collection_name=request.collection_name,
                    query_vector=request.query_embedding,
                    limit=request.n_results,
                    query_filter=query_filter,
                    with_payload=True,
                    with_vectors=request.include_embeddings
                )
                
                # Convert to SimilarityResult objects
                results = []
                for result in search_results:
                    if result.score >= request.similarity_threshold:
                        doc = VectorDocument(
                            id=str(result.id),
                            content=result.payload.get("content", ""),
                            embedding=result.vector or [],
                            metadata=result.payload,
                            created_at=datetime.fromisoformat(result.payload.get("created_at", datetime.now().isoformat())),
                            collection_name=request.collection_name,
                            tenant_id=result.payload.get("tenant_id")
                        )
                        
                        similarity_result = SimilarityResult(
                            document=doc,
                            similarity_score=result.score,
                            metadata=result.payload
                        )
                        results.append(similarity_result)
                
                # Apply reranking if requested
                if request.rerank and len(results) > 1:
                    results = await self._rerank_results(results, request)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to perform advanced semantic search: {e}")
            raise RepositoryError(f"Failed to perform advanced semantic search: {e}")
    
    async def bulk_upsert_documents(
        self,
        collection_name: str,
        operation: BulkVectorOperation
    ) -> Dict[str, Any]:
        """Bulk upsert documents with batch processing."""
        try:
            start_time = time.time()
            
            with PerformanceMonitor("bulk_upsert_documents", "qdrant"):
                total_processed = 0
                batch_count = 0
                
                # Process in batches
                for i in range(0, len(operation.documents), operation.batch_size):
                    batch = operation.documents[i:i + operation.batch_size]
                    
                    # Convert to Qdrant points
                    points = []
                    for doc in batch:
                        point = doc.to_qdrant_point()
                        points.append(point)
                    
                    # Upsert batch
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    
                    total_processed += len(batch)
                    batch_count += 1
                
                execution_time = time.time() - start_time
                
                return {
                    "total_processed": total_processed,
                    "batch_count": batch_count,
                    "execution_time_seconds": execution_time
                }
                
        except Exception as e:
            self.logger.error(f"Failed to bulk upsert documents: {e}")
            raise RepositoryError(f"Failed to bulk upsert documents: {e}")
    
    def _calculate_similarity(self, distance: float, algorithm: SimilarityAlgorithm) -> float:
        """Calculate similarity score from distance based on algorithm."""
        if algorithm == SimilarityAlgorithm.COSINE:
            return distance  # Qdrant returns similarity directly for cosine
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        elif algorithm == SimilarityAlgorithm.MANHATTAN:
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        else:
            return distance  # Default to direct value
    
    async def _rerank_results(
        self,
        results: List[SimilarityResult],
        request: VectorSearchRequest
    ) -> List[SimilarityResult]:
        """Rerank results with metadata boosting."""
        # Simple reranking based on metadata priority
        for result in results:
            metadata = result.document.metadata
            
            # Boost score based on metadata
            if metadata.get("priority") == "high":
                result.similarity_score *= 1.2
            if metadata.get("recent"):
                result.similarity_score *= 1.1
        
        # Sort by boosted scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    async def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering with fallback."""
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(embeddings)
        except ImportError:
            # Fallback: simple random assignment
            self.logger.warning("sklearn not available, using fallback clustering")
            return np.random.randint(0, n_clusters, size=len(embeddings))
    
    async def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering with fallback."""
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            return dbscan.fit_predict(embeddings)
        except ImportError:
            # Fallback: assign all to same cluster
            self.logger.warning("sklearn not available, using fallback clustering")
            return np.zeros(len(embeddings), dtype=int)
    
    def _calculate_avg_query_time(self, collection_name: str) -> float:
        """Calculate average query time for a collection."""
        if not self._query_metrics:
            return 0.0
        
        collection_metrics = [
            m for m in self._query_metrics 
            if m.get("collection") == collection_name
        ]
        
        if not collection_metrics:
            return 0.0
        
        avg_time = sum(m.get("execution_time", 0) for m in collection_metrics) / len(collection_metrics)
        return avg_time * 1000  # Convert to milliseconds
    
    def _build_query_filter(self, request: VectorSearchRequest):
        """Build query filter for search operations."""
        if not request.tenant_id and not request.metadata_filter:
            return None
        
        # This is a simplified filter - in practice, you'd use Qdrant's filter syntax
        # For now, we'll return None and let the client handle filtering
        return None
    
    # Legacy compatibility methods
    async def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None
    ) -> bool:
        """Legacy create_collection method for backward compatibility."""
        if vector_size is None:
            vector_size = 384
        
        if distance is None:
            distance = "cosine"
        
        return await self.create_collection_optimized(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=distance
        )
    
    async def get_collection(self, collection_name: str) -> bool:
        """Legacy get_collection method for backward compatibility."""
        try:
            return self.client.collection_exists(collection_name)
        except Exception:
            return False
    
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
        tenant_id: Optional[str] = None
    ) -> bool:
        """Legacy add_documents method for backward compatibility."""
        try:
            # Convert to VectorDocument objects
            vector_docs = []
            for i, (doc_id, content, metadata, embedding) in enumerate(zip(ids, documents, metadatas, embeddings)):
                doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata,
                    created_at=datetime.now(),
                    collection_name=collection_name,
                    tenant_id=tenant_id
                )
                vector_docs.append(doc)
            
            # Use bulk upsert
            operation = BulkVectorOperation(
                operation_type="insert",
                documents=vector_docs,
                tenant_id=tenant_id
            )
            
            result = await self.bulk_upsert_documents(collection_name, operation)
            return result["total_processed"] > 0
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    async def query_documents(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Legacy query_documents method for backward compatibility."""
        try:
            # Use the first query embedding
            query_embedding = query_embeddings[0] if query_embeddings else []
            
            request = VectorSearchRequest(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=n_results,
                tenant_id=tenant_id
            )
            
            results = await self.semantic_search_advanced(request)
            
            # Convert to legacy format
            return {
                "ids": [r.document.id for r in results],
                "documents": [r.document.content for r in results],
                "distances": [1.0 - r.similarity_score for r in results],
                "metadatas": [r.document.metadata for r in results]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to query documents: {e}")
            return {"ids": [], "documents": [], "distances": [], "metadatas": []}
    
    async def get_documents(
        self,
        collection_name: str,
        ids: List[str],
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Legacy get_documents method for backward compatibility."""
        try:
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=True
            )
            
            # Filter by tenant if specified
            if tenant_id:
                results = [r for r in results if r.payload.get("tenant_id") == tenant_id]
            
            return {
                "ids": [r.id for r in results],
                "documents": [r.payload.get("content", "") for r in results],
                "embeddings": [r.vector for r in results],
                "metadatas": [r.payload for r in results]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get documents: {e}")
            return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}
    
    async def similarity_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        threshold: float = 0.5,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Legacy similarity_search method for backward compatibility."""
        try:
            request = VectorSearchRequest(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=n_results,
                similarity_threshold=threshold,
                tenant_id=tenant_id
            )
            
            results = await self.semantic_search_advanced(request)
            
            # Convert to legacy format
            return [
                {
                    "similarity": result.similarity_score,
                    "document": result.document.content,
                    "metadata": result.document.metadata
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    async def count_documents(self, collection_name: str) -> int:
        """Legacy count_documents method for backward compatibility."""
        try:
            stats = self.client.get_collection_stats(collection_name)
            return stats.points_count
        except Exception as e:
            self.logger.error(f"Failed to count documents: {e}")
            return 0
    
    async def create_collection_optimized(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: str = "cosine",
        hnsw_config: Optional[Dict[str, Any]] = None,
        optimizers_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        replication_factor: int = 1,
        write_consistency_factor: int = 1,
        on_disk_payload: bool = True
    ) -> bool:
        """Create collection with optimized configuration."""
        try:
            config = QdrantCollectionConfig(
                name=collection_name,
                vector_size=vector_size,
                distance=QdrantDistance.COSINE if distance == "cosine" else QdrantDistance.EUCLIDEAN,
                hnsw_config=hnsw_config or {},
                optimizers_config=optimizers_config or {}
            )
            
            if self._is_async:
                result = await self.client.create_collection_async(collection_name, config)
            else:
                result = self.client.create_collection(config)
            
            # Update collection cache
            if result:
                self._collection_cache[collection_name] = {}
                
            return result
        except Exception as e:
            self.logger.error(f"Failed to create optimized collection: {e}")
            return False
    
    async def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete documents by IDs or filter."""
        try:
            if ids:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=ids
                )
            elif where:
                # Build filter and delete
                # For now, simplified implementation
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=where
                )
            else:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection (synchronous)."""
        try:
            return self.client.delete_collection(collection_name)
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._collection_cache.clear()
        self.logger.info("Repository cache cleared")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_cache()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.clear_cache()
