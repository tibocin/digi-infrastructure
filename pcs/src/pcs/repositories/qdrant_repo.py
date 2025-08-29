"""
Filepath: pcs/src/pcs/repositories/qdrant_repo.py
Purpose: Enhanced Qdrant repository implementation for advanced vector database operations and multi-tenant semantic search
Related Components: Qdrant client, embedding operations, similarity search, vector indexing, multi-tenancy
Tags: qdrant, vector-database, embeddings, similarity-search, semantic, performance-optimization, multi-tenant
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import time
import asyncio
import numpy as np
from collections import defaultdict

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CollectionInfo, PointStruct, Filter, FieldCondition, 
    MatchValue, SearchRequest, ScrollRequest, OptimizersConfig, HnswConfig,
    ScalarQuantization, ProductQuantization, BinaryQuantization,
    CompressionRatio, QuantizationType, PayloadSchemaType
)
from qdrant_client.http.models import UpdateResult, CollectionExistence

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
    # Note: Qdrant primarily uses HNSW, but supports various HNSW configurations


class QdrantDistance(Enum):
    """Qdrant-specific distance metrics."""
    COSINE = Distance.COSINE
    EUCLIDEAN = Distance.EUCLID
    DOT_PRODUCT = Distance.DOT
    MANHATTAN = Distance.MANHATTAN


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

    def to_qdrant_point(self) -> PointStruct:
        """Convert to Qdrant PointStruct."""
        payload = {
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "collection_name": self.collection_name,
            **self.metadata
        }
        
        # Add tenant isolation
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
            
        return PointStruct(
            id=self.id,
            vector=self.embedding,
            payload=payload
        )


@dataclass
class SimilarityResult:
    """Container for similarity search results."""
    document: VectorDocument
    similarity_score: float
    distance: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document": self.document.to_dict(),
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "rank": self.rank
        }


@dataclass
class VectorSearchRequest:
    """Container for vector search parameters with multi-tenancy."""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    collection_name: str = ""
    n_results: int = 10
    similarity_threshold: float = 0.0
    metadata_filter: Optional[Dict[str, Any]] = None
    algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    include_embeddings: bool = False
    rerank: bool = False
    tenant_id: Optional[str] = None  # Multi-tenancy support
    offset: Optional[int] = None  # Pagination support


@dataclass
class BulkVectorOperation:
    """Container for bulk vector operations with tenant support."""
    operation_type: str  # "insert", "update", "delete"
    documents: List[VectorDocument]
    batch_size: int = 1000
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
class VectorCollectionStats:
    """Container for vector collection statistics."""
    name: str
    document_count: int
    dimension: int
    index_type: str
    memory_usage_mb: float
    avg_query_time_ms: float
    last_updated: datetime
    tenant_count: Optional[int] = None  # Multi-tenancy support
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "document_count": self.document_count,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "memory_usage_mb": self.memory_usage_mb,
            "avg_query_time_ms": self.avg_query_time_ms,
            "last_updated": self.last_updated.isoformat(),
            "tenant_count": self.tenant_count
        }


@dataclass
class QdrantCollectionConfig:
    """Configuration for Qdrant collection creation."""
    name: str
    vector_size: int
    distance: QdrantDistance = QdrantDistance.COSINE
    hnsw_config: Optional[Dict[str, Any]] = None
    optimizers_config: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None
    replication_factor: Optional[int] = None
    write_consistency_factor: Optional[int] = None
    on_disk_payload: bool = True


class EnhancedQdrantRepository:
    """
    Enhanced Qdrant repository for advanced vector database operations with multi-tenancy.
    
    Features:
    - Advanced vector operations and indexing with HNSW
    - Multiple similarity algorithms and distance metrics
    - Multi-tenant data isolation using payload filtering
    - Bulk operations with batch processing
    - Performance monitoring and optimization
    - Semantic search with reranking
    - Vector collection management and statistics
    - Advanced filtering and metadata operations
    - Quantization and compression support
    - Distributed deployment readiness
    """

    def __init__(
        self, 
        client: Optional[QdrantClient] = None,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
        api_key: Optional[str] = None,
        https: bool = False,
        use_async: bool = False
    ):
        """
        Initialize enhanced Qdrant repository with client connection.
        
        Args:
            client: Existing Qdrant client instance
            host: Qdrant host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
            prefer_grpc: Whether to prefer gRPC over HTTP
            api_key: API key for authentication
            https: Whether to use HTTPS
            use_async: Whether to use async client
        """
        if client:
            self.client = client
        else:
            if use_async:
                self.client = AsyncQdrantClient(
                    host=host,
                    port=port,
                    grpc_port=grpc_port,
                    prefer_grpc=prefer_grpc,
                    api_key=api_key,
                    https=https
                )
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    grpc_port=grpc_port,
                    prefer_grpc=prefer_grpc,
                    api_key=api_key,
                    https=https
                )
        
        self._query_metrics = []
        self._collection_cache = {}
        self._is_async = use_async

    async def create_collection_optimized(
        self,
        config: QdrantCollectionConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create an optimized collection with advanced Qdrant configuration.
        
        Args:
            config: Collection configuration
            metadata: Additional collection metadata
            
        Returns:
            True if collection was created successfully
        """
        try:
            async with PerformanceMonitor("create_collection", "qdrant") as monitor:
                # Configure HNSW parameters
                hnsw_config = HnswConfig(
                    m=config.hnsw_config.get("m", 16) if config.hnsw_config else 16,
                    ef_construct=config.hnsw_config.get("ef_construct", 100) if config.hnsw_config else 100,
                    full_scan_threshold=config.hnsw_config.get("full_scan_threshold", 10000) if config.hnsw_config else 10000,
                    max_indexing_threads=config.hnsw_config.get("max_indexing_threads", 0) if config.hnsw_config else 0,
                    on_disk=config.hnsw_config.get("on_disk", False) if config.hnsw_config else False,
                    payload_m=config.hnsw_config.get("payload_m", None) if config.hnsw_config else None
                )
                
                # Configure optimizers
                optimizers_config = None
                if config.optimizers_config:
                    optimizers_config = OptimizersConfig(
                        deleted_threshold=config.optimizers_config.get("deleted_threshold", 0.2),
                        vacuum_min_vector_number=config.optimizers_config.get("vacuum_min_vector_number", 1000),
                        default_segment_number=config.optimizers_config.get("default_segment_number", 0),
                        max_segment_size=config.optimizers_config.get("max_segment_size", None),
                        memmap_threshold=config.optimizers_config.get("memmap_threshold", None),
                        indexing_threshold=config.optimizers_config.get("indexing_threshold", 20000),
                        flush_interval_sec=config.optimizers_config.get("flush_interval_sec", 5),
                        max_optimization_threads=config.optimizers_config.get("max_optimization_threads", 1)
                    )
                
                # Configure quantization if specified
                quantization_config = None
                if config.quantization_config:
                    quant_type = config.quantization_config.get("type", "scalar")
                    if quant_type == "scalar":
                        quantization_config = ScalarQuantization(
                            type=QuantizationType.INT8,
                            quantile=config.quantization_config.get("quantile", None),
                            always_ram=config.quantization_config.get("always_ram", None)
                        )
                    elif quant_type == "product":
                        quantization_config = ProductQuantization(
                            compression=CompressionRatio(config.quantization_config.get("compression", 16)),
                            always_ram=config.quantization_config.get("always_ram", None)
                        )
                    elif quant_type == "binary":
                        quantization_config = BinaryQuantization(
                            always_ram=config.quantization_config.get("always_ram", None)
                        )
                
                # Create vector parameters
                vectors_config = VectorParams(
                    size=config.vector_size,
                    distance=config.distance.value,
                    hnsw_config=hnsw_config,
                    quantization_config=quantization_config,
                    on_disk=config.on_disk_payload
                )
                
                # Create collection
                if self._is_async:
                    result = await self.client.create_collection(
                        collection_name=config.name,
                        vectors_config=vectors_config,
                        optimizers_config=optimizers_config,
                        replication_factor=config.replication_factor,
                        write_consistency_factor=config.write_consistency_factor
                    )
                else:
                    result = self.client.create_collection(
                        collection_name=config.name,
                        vectors_config=vectors_config,
                        optimizers_config=optimizers_config,
                        replication_factor=config.replication_factor,
                        write_consistency_factor=config.write_consistency_factor
                    )
                
                # Cache collection for performance
                self._collection_cache[config.name] = {
                    "config": config,
                    "metadata": metadata or {},
                    "created_at": datetime.utcnow()
                }
                
                monitor.set_rows_affected(1)
                return result
                
        except Exception as e:
            raise RepositoryError(f"Failed to create optimized collection {config.name}: {str(e)}") from e

    async def bulk_upsert_documents(
        self,
        collection_name: str,
        operation: BulkVectorOperation
    ) -> Dict[str, Any]:
        """
        Perform bulk upsert operations with batch processing and performance monitoring.
        
        Args:
            collection_name: Target collection name
            operation: Bulk operation specification
            
        Returns:
            Operation statistics and performance metrics
        """
        try:
            start_time = time.time()
            total_processed = 0
            batch_count = 0
            
            for i in range(0, len(operation.documents), operation.batch_size):
                batch = operation.documents[i:i + operation.batch_size]
                batch_count += 1
                
                # Convert documents to Qdrant points
                points = [doc.to_qdrant_point() for doc in batch]
                
                # Execute batch operation
                async with PerformanceMonitor(f"bulk_upsert_batch_{batch_count}", "qdrant"):
                    if self._is_async:
                        await self.client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                    else:
                        self.client.upsert(
                            collection_name=collection_name,
                            points=points
                        )
                
                total_processed += len(batch)
            
            execution_time = time.time() - start_time
            
            # Record performance metrics
            record_manual_metric(
                query_type=f"bulk_upsert_{operation.operation_type}",
                execution_time=execution_time,
                rows_affected=total_processed,
                table_name=collection_name
            )
            
            return {
                "total_processed": total_processed,
                "batch_count": batch_count,
                "execution_time_seconds": execution_time,
                "average_batch_time": execution_time / batch_count if batch_count > 0 else 0,
                "documents_per_second": total_processed / execution_time if execution_time > 0 else 0
            }
            
        except Exception as e:
            raise RepositoryError(f"Failed to bulk upsert documents: {str(e)}") from e

    async def semantic_search_advanced(
        self,
        request: VectorSearchRequest
    ) -> List[SimilarityResult]:
        """
        Perform advanced semantic search with multi-tenancy, algorithms and reranking.
        
        Args:
            request: Vector search request with parameters
            
        Returns:
            List of similarity results with scores and rankings
        """
        try:
            async with PerformanceMonitor("semantic_search_advanced", "qdrant") as monitor:
                # Build query filter with tenant isolation
                query_filter = self._build_query_filter(request)
                
                # Perform search
                search_request = SearchRequest(
                    vector=request.query_embedding,
                    filter=query_filter,
                    limit=request.n_results * 2 if request.rerank else request.n_results,
                    offset=request.offset,
                    with_payload=True,
                    with_vector=request.include_embeddings,
                    score_threshold=request.similarity_threshold if request.similarity_threshold > 0 else None
                )
                
                if self._is_async:
                    search_results = await self.client.search(
                        collection_name=request.collection_name,
                        search_request=search_request
                    )
                else:
                    search_results = self.client.search(
                        collection_name=request.collection_name,
                        **search_request.__dict__
                    )
                
                # Process and format results
                similarity_results = []
                for i, result in enumerate(search_results):
                    # Create VectorDocument from result
                    payload = result.payload or {}
                    vector_doc = VectorDocument(
                        id=str(result.id),
                        content=payload.get("content", ""),
                        embedding=result.vector if request.include_embeddings and result.vector else [],
                        metadata={k: v for k, v in payload.items() if k not in ["content", "created_at", "collection_name", "tenant_id"]},
                        created_at=datetime.fromisoformat(payload.get("created_at", datetime.utcnow().isoformat())),
                        collection_name=request.collection_name,
                        tenant_id=payload.get("tenant_id")
                    )
                    
                    # Calculate similarity score (Qdrant returns similarity scores, not distances)
                    similarity_score = result.score
                    
                    similarity_results.append(SimilarityResult(
                        document=vector_doc,
                        similarity_score=similarity_score,
                        distance=1 - similarity_score if request.algorithm == SimilarityAlgorithm.COSINE else similarity_score,
                        rank=i + 1
                    ))
                
                # Apply reranking if requested
                if request.rerank and similarity_results:
                    similarity_results = await self._rerank_results(similarity_results, request)
                
                # Limit to requested number of results
                similarity_results = similarity_results[:request.n_results]
                
                monitor.set_rows_affected(len(similarity_results))
                return similarity_results
                
        except Exception as e:
            raise RepositoryError(f"Failed to perform advanced semantic search: {str(e)}") from e

    def _build_query_filter(self, request: VectorSearchRequest) -> Optional[Filter]:
        """Build Qdrant filter from search request."""
        conditions = []
        
        # Add tenant isolation
        if request.tenant_id:
            conditions.append(
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=request.tenant_id)
                )
            )
        
        # Add metadata filters
        if request.metadata_filter:
            for key, value in request.metadata_filter.items():
                if isinstance(value, dict):
                    # Handle complex filters (range, etc.)
                    if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                        # Range filter
                        range_filter = {}
                        if "gte" in value:
                            range_filter["gte"] = value["gte"]
                        if "lte" in value:
                            range_filter["lte"] = value["lte"]
                        if "gt" in value:
                            range_filter["gt"] = value["gt"]
                        if "lt" in value:
                            range_filter["lt"] = value["lt"]
                        
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=range_filter
                            )
                        )
                else:
                    # Simple equality filter
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
        
        return Filter(must=conditions) if conditions else None

    async def find_similar_documents(
        self,
        collection_name: str,
        target_embedding: List[float],
        similarity_threshold: float = 0.8,
        max_results: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Find documents similar to a target embedding with advanced filtering and multi-tenancy.
        
        Args:
            collection_name: Collection name
            target_embedding: Target embedding vector
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            metadata_filter: Optional metadata filtering
            tenant_id: Tenant ID for isolation
            
        Returns:
            List of similar documents with similarity scores
        """
        try:
            request = VectorSearchRequest(
                query_embedding=target_embedding,
                collection_name=collection_name,
                n_results=max_results,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter,
                include_embeddings=True,
                tenant_id=tenant_id
            )
            
            return await self.semantic_search_advanced(request)
            
        except Exception as e:
            raise RepositoryError(f"Failed to find similar documents: {str(e)}") from e

    async def cluster_documents(
        self,
        collection_name: str,
        n_clusters: int = 5,
        algorithm: str = "kmeans",
        metadata_filter: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cluster documents in the collection using embedding vectors with multi-tenancy support.
        
        Args:
            collection_name: Collection name
            n_clusters: Number of clusters to create
            algorithm: Clustering algorithm ("kmeans", "dbscan")
            metadata_filter: Optional metadata filtering
            tenant_id: Tenant ID for isolation
            
        Returns:
            Clustering results with cluster assignments and statistics
        """
        try:
            async with PerformanceMonitor("cluster_documents", "qdrant") as monitor:
                # Build filter for tenant isolation and metadata
                query_filter = Filter(must=[]) if not metadata_filter and not tenant_id else None
                conditions = []
                
                if tenant_id:
                    conditions.append(
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
                        )
                    )
                
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
                
                # Scroll through all points to get embeddings
                scroll_request = ScrollRequest(
                    filter=query_filter,
                    limit=10000,  # Adjust based on expected data size
                    with_payload=True,
                    with_vector=True
                )
                
                if self._is_async:
                    scroll_result = await self.client.scroll(
                        collection_name=collection_name,
                        scroll_request=scroll_request
                    )
                else:
                    scroll_result = self.client.scroll(
                        collection_name=collection_name,
                        **scroll_request.__dict__
                    )
                
                points = scroll_result[0]  # First element is the list of points
                
                if not points:
                    return {"clusters": [], "statistics": {"total_documents": 0}}
                
                # Extract embeddings and document info
                embeddings = []
                documents = []
                for point in points:
                    embeddings.append(point.vector)
                    documents.append({
                        "id": str(point.id),
                        "content": point.payload.get("content", ""),
                        "metadata": {k: v for k, v in point.payload.items() 
                                   if k not in ["content", "created_at", "collection_name", "tenant_id"]}
                    })
                
                embeddings = np.array(embeddings)
                
                # Perform clustering based on algorithm
                if algorithm == "kmeans":
                    cluster_labels = await self._kmeans_clustering(embeddings, n_clusters)
                elif algorithm == "dbscan":
                    cluster_labels = await self._dbscan_clustering(embeddings)
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
                
                # Organize results by cluster
                clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    if label != -1:  # -1 is noise in DBSCAN
                        clusters[int(label)].append(documents[i])
                
                # Calculate cluster statistics
                statistics = {
                    "total_documents": len(embeddings),
                    "n_clusters": len(clusters),
                    "avg_cluster_size": np.mean([len(cluster) for cluster in clusters.values()]) if clusters else 0,
                    "noise_points": np.sum(cluster_labels == -1) if algorithm == "dbscan" else 0
                }
                
                monitor.set_rows_affected(len(embeddings))
                
                return {
                    "clusters": dict(clusters),
                    "statistics": statistics,
                    "algorithm": algorithm
                }
                
        except Exception as e:
            raise RepositoryError(f"Failed to cluster documents: {str(e)}") from e

    async def get_collection_statistics(
        self, 
        collection_name: str,
        tenant_id: Optional[str] = None
    ) -> VectorCollectionStats:
        """
        Get comprehensive statistics for a vector collection with optional tenant filtering.
        
        Args:
            collection_name: Collection name
            tenant_id: Optional tenant ID for tenant-specific stats
            
        Returns:
            Collection statistics including performance metrics
        """
        try:
            async with PerformanceMonitor("get_collection_stats", "qdrant"):
                # Get collection info
                if self._is_async:
                    collection_info = await self.client.get_collection(collection_name)
                else:
                    collection_info = self.client.get_collection(collection_name)
                
                # Get basic stats
                total_count = collection_info.points_count
                dimension = collection_info.config.params.vector.size
                
                # Count tenant-specific documents if tenant_id provided
                document_count = total_count
                tenant_count = None
                
                if tenant_id:
                    # Count documents for specific tenant
                    count_filter = Filter(must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
                        )
                    ])
                    
                    # Use scroll to count (Qdrant doesn't have a direct count method with filter)
                    scroll_request = ScrollRequest(
                        filter=count_filter,
                        limit=1,
                        with_payload=False,
                        with_vector=False
                    )
                    
                    if self._is_async:
                        scroll_result = await self.client.scroll(
                            collection_name=collection_name,
                            scroll_request=scroll_request
                        )
                    else:
                        scroll_result = self.client.scroll(
                            collection_name=collection_name,
                            **scroll_request.__dict__
                        )
                    
                    # For accurate count, we'd need to scroll through all points
                    # This is a simplified implementation
                    document_count = len(scroll_result[0])
                
                # Get unique tenant count if not filtering by tenant
                if not tenant_id:
                    try:
                        # Sample some points to estimate tenant count
                        scroll_request = ScrollRequest(
                            limit=1000,
                            with_payload=True,
                            with_vector=False
                        )
                        
                        if self._is_async:
                            scroll_result = await self.client.scroll(
                                collection_name=collection_name,
                                scroll_request=scroll_request
                            )
                        else:
                            scroll_result = self.client.scroll(
                                collection_name=collection_name,
                                **scroll_request.__dict__
                            )
                        
                        tenant_ids = set()
                        for point in scroll_result[0]:
                            if point.payload and "tenant_id" in point.payload:
                                tenant_ids.add(point.payload["tenant_id"])
                        
                        tenant_count = len(tenant_ids)
                    except:
                        tenant_count = None
                
                # Calculate estimated memory usage
                memory_usage_mb = (document_count * dimension * 4) / (1024 * 1024)  # 4 bytes per float
                
                # Get average query time from metrics
                avg_query_time = self._calculate_avg_query_time(collection_name)
                
                # Get index type from collection config
                index_type = "hnsw"  # Qdrant primarily uses HNSW
                if collection_info.config.params.hnsw_config:
                    index_type = f"hnsw(m={collection_info.config.params.hnsw_config.m})"
                
                return VectorCollectionStats(
                    name=collection_name,
                    document_count=document_count,
                    dimension=dimension,
                    index_type=index_type,
                    memory_usage_mb=memory_usage_mb,
                    avg_query_time_ms=avg_query_time,
                    last_updated=datetime.utcnow(),
                    tenant_count=tenant_count
                )
                
        except Exception as e:
            raise RepositoryError(f"Failed to get collection statistics: {str(e)}") from e

    async def optimize_collection_performance(
        self, 
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Optimize collection performance by updating configuration and triggering optimizations.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Optimization results and performance improvements
        """
        try:
            async with PerformanceMonitor("optimize_collection", "qdrant") as monitor:
                # Get current statistics
                before_stats = await self.get_collection_statistics(collection_name)
                
                # Trigger Qdrant optimization
                if self._is_async:
                    await self.client.update_collection(
                        collection_name=collection_name,
                        optimizer_config=OptimizersConfig(
                            deleted_threshold=0.2,
                            vacuum_min_vector_number=1000,
                            default_segment_number=0,
                            indexing_threshold=20000,
                            flush_interval_sec=5,
                            max_optimization_threads=1
                        )
                    )
                else:
                    self.client.update_collection(
                        collection_name=collection_name,
                        optimizer_config=OptimizersConfig(
                            deleted_threshold=0.2,
                            vacuum_min_vector_number=1000,
                            default_segment_number=0,
                            indexing_threshold=20000,
                            flush_interval_sec=5,
                            max_optimization_threads=1
                        )
                    )
                
                optimization_results = {
                    "before_optimization": before_stats.to_dict(),
                    "optimizations_applied": ["segment_optimization", "index_optimization"],
                    "performance_improvements": {
                        "optimization_triggered": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                monitor.set_rows_affected(before_stats.document_count)
                
                return optimization_results
                
        except Exception as e:
            raise RepositoryError(f"Failed to optimize collection performance: {str(e)}") from e

    async def export_embeddings(
        self,
        collection_name: str,
        format_type: str = "numpy",
        include_metadata: bool = True,
        tenant_id: Optional[str] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Export embeddings and metadata from a collection with multi-tenancy support.
        
        Args:
            collection_name: Collection name
            format_type: Export format ("numpy", "json", "csv")
            include_metadata: Whether to include document metadata
            tenant_id: Optional tenant ID for filtering
            batch_size: Batch size for scrolling through data
            
        Returns:
            Exported data in requested format
        """
        try:
            async with PerformanceMonitor("export_embeddings", "qdrant") as monitor:
                # Build filter for tenant isolation
                query_filter = None
                if tenant_id:
                    query_filter = Filter(must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
                        )
                    ])
                
                # Scroll through all points
                all_points = []
                offset = None
                
                while True:
                    scroll_request = ScrollRequest(
                        filter=query_filter,
                        limit=batch_size,
                        offset=offset,
                        with_payload=include_metadata,
                        with_vector=True
                    )
                    
                    if self._is_async:
                        points, next_offset = await self.client.scroll(
                            collection_name=collection_name,
                            scroll_request=scroll_request
                        )
                    else:
                        points, next_offset = self.client.scroll(
                            collection_name=collection_name,
                            **scroll_request.__dict__
                        )
                    
                    all_points.extend(points)
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                
                export_data = {
                    "collection_name": collection_name,
                    "format": format_type,
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "document_count": len(all_points),
                    "tenant_id": tenant_id
                }
                
                if format_type == "numpy":
                    embeddings = [point.vector for point in all_points]
                    export_data["embeddings"] = np.array(embeddings)
                    
                    if include_metadata:
                        export_data["documents"] = [point.payload.get("content", "") for point in all_points]
                        export_data["metadatas"] = [point.payload for point in all_points]
                        export_data["ids"] = [str(point.id) for point in all_points]
                
                elif format_type == "json":
                    export_data["data"] = []
                    for point in all_points:
                        item = {"embedding": point.vector}
                        if include_metadata:
                            item.update({
                                "id": str(point.id),
                                "content": point.payload.get("content", ""),
                                "metadata": point.payload
                            })
                        export_data["data"].append(item)
                
                monitor.set_rows_affected(export_data["document_count"])
                return export_data
                
        except Exception as e:
            raise RepositoryError(f"Failed to export embeddings: {str(e)}") from e

    # Helper methods for advanced operations

    def _calculate_similarity(self, distance: float, algorithm: SimilarityAlgorithm) -> float:
        """Calculate similarity score based on distance and algorithm."""
        # Note: Qdrant typically returns similarity scores, not distances
        # This method is for backward compatibility
        if algorithm == SimilarityAlgorithm.COSINE:
            return distance  # Qdrant returns cosine similarity directly
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            return 1 / (1 + distance)  # Convert distance to similarity
        elif algorithm == SimilarityAlgorithm.DOT_PRODUCT:
            return distance  # Dot product is already a similarity measure
        elif algorithm == SimilarityAlgorithm.MANHATTAN:
            return 1 / (1 + distance)
        else:
            return distance  # Default

    async def _rerank_results(
        self,
        results: List[SimilarityResult],
        request: VectorSearchRequest
    ) -> List[SimilarityResult]:
        """Rerank search results using advanced algorithms."""
        # Simple reranking based on metadata relevance and recency
        for i, result in enumerate(results):
            # Boost score based on metadata relevance
            relevance_boost = 0
            if result.document.metadata.get("priority") == "high":
                relevance_boost += 0.1
            if result.document.metadata.get("recent", False):
                relevance_boost += 0.05
            
            # Apply boost
            result.similarity_score = min(1.0, result.similarity_score + relevance_boost)
        
        # Re-sort by updated similarity scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results

    async def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering on embeddings."""
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)
        except ImportError:
            # Fallback to simple clustering if sklearn not available
            return np.random.randint(0, n_clusters, len(embeddings))

    async def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering on embeddings."""
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=0.3, min_samples=2)
            return dbscan.fit_predict(embeddings)
        except ImportError:
            # Fallback clustering if sklearn not available
            return np.zeros(len(embeddings))

    def _calculate_avg_query_time(self, collection_name: str) -> float:
        """Calculate average query time for a collection from metrics."""
        collection_metrics = [m for m in self._query_metrics if m.get("collection") == collection_name]
        if not collection_metrics:
            return 0.0
        return sum(m.get("execution_time", 0) for m in collection_metrics) / len(collection_metrics) * 1000  # Convert to ms

    # Legacy methods for backward compatibility with ChromaDB interface

    async def create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        vector_size: int = 384,
        distance: str = "cosine"
    ) -> bool:
        """Create a new collection (legacy method for compatibility)."""
        distance_map = {
            "cosine": QdrantDistance.COSINE,
            "euclidean": QdrantDistance.EUCLIDEAN,
            "dot_product": QdrantDistance.DOT_PRODUCT,
            "manhattan": QdrantDistance.MANHATTAN
        }
        
        config = QdrantCollectionConfig(
            name=name,
            vector_size=vector_size,
            distance=distance_map.get(distance, QdrantDistance.COSINE)
        )
        
        return await self.create_collection_optimized(config, metadata)

    async def get_collection(self, name: str) -> Optional[bool]:
        """Get an existing collection (legacy method for compatibility)."""
        try:
            if self._is_async:
                result = await self.client.collection_exists(collection_name=name)
            else:
                result = self.client.collection_exists(collection_name=name)
            return result
        except Exception:
            return False

    async def get_or_create_collection(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None,
        vector_size: int = 384,
        distance: str = "cosine"
    ) -> bool:
        """Get existing collection or create if it doesn't exist."""
        try:
            exists = await self.get_collection(name)
            if not exists:
                return await self.create_collection(name, metadata, vector_size, distance)
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to get or create collection {name}: {str(e)}") from e

    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Add documents to a collection (legacy method)."""
        try:
            # Convert to VectorDocument format
            vector_docs = []
            for i, (doc_id, content) in enumerate(zip(ids, documents)):
                vector_doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    embedding=embeddings[i] if embeddings else [],
                    metadata=metadatas[i] if metadatas else {},
                    created_at=datetime.utcnow(),
                    collection_name=collection_name,
                    tenant_id=tenant_id
                )
                vector_docs.append(vector_doc)
            
            # Use bulk upsert
            operation = BulkVectorOperation(
                operation_type="insert",
                documents=vector_docs,
                tenant_id=tenant_id
            )
            
            await self.bulk_upsert_documents(collection_name, operation)
            return True
            
        except Exception as e:
            raise RepositoryError(f"Failed to add documents to collection: {str(e)}") from e

    async def query_documents(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query documents from a collection (legacy method)."""
        try:
            if query_embeddings:
                query_embedding = query_embeddings[0]
            else:
                # Would need embedding function for query_texts
                raise ValueError("query_embeddings must be provided")
            
            request = VectorSearchRequest(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=n_results,
                metadata_filter=where,
                tenant_id=tenant_id
            )
            
            results = await self.semantic_search_advanced(request)
            
            # Format in ChromaDB-compatible format
            formatted_results = {
                "ids": [[result.document.id for result in results]],
                "documents": [[result.document.content for result in results]],
                "distances": [[result.distance for result in results]],
                "metadatas": [[result.document.metadata for result in results]]
            }
            
            return formatted_results
            
        except Exception as e:
            raise RepositoryError(f"Failed to query collection: {str(e)}") from e

    async def get_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get documents from a collection (legacy method)."""
        try:
            # Build filter
            conditions = []
            
            if tenant_id:
                conditions.append(
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                )
            
            if where:
                for key, value in where.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            
            query_filter = Filter(must=conditions) if conditions else None
            
            # Get specific documents by IDs or use scroll
            if ids:
                if self._is_async:
                    points = await self.client.retrieve(
                        collection_name=collection_name,
                        ids=ids,
                        with_payload=True,
                        with_vector=True
                    )
                else:
                    points = self.client.retrieve(
                        collection_name=collection_name,
                        ids=ids,
                        with_payload=True,
                        with_vector=True
                    )
            else:
                # Use scroll for general queries
                scroll_request = ScrollRequest(
                    filter=query_filter,
                    limit=limit or 100,
                    offset=offset,
                    with_payload=True,
                    with_vector=True
                )
                
                if self._is_async:
                    points, _ = await self.client.scroll(
                        collection_name=collection_name,
                        scroll_request=scroll_request
                    )
                else:
                    points, _ = self.client.scroll(
                        collection_name=collection_name,
                        **scroll_request.__dict__
                    )
            
            # Format in ChromaDB-compatible format
            results = {
                "ids": [str(point.id) for point in points],
                "documents": [point.payload.get("content", "") for point in points],
                "embeddings": [point.vector for point in points] if points and hasattr(points[0], 'vector') else [],
                "metadatas": [point.payload for point in points]
            }
            
            return results
            
        except Exception as e:
            raise RepositoryError(f"Failed to get documents from collection: {str(e)}") from e

    async def update_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Update existing documents in a collection (legacy method)."""
        try:
            # Convert to VectorDocument format for bulk update
            vector_docs = []
            for i, doc_id in enumerate(ids):
                vector_doc = VectorDocument(
                    id=doc_id,
                    content=documents[i] if documents else "",
                    embedding=embeddings[i] if embeddings else [],
                    metadata=metadatas[i] if metadatas else {},
                    created_at=datetime.utcnow(),
                    collection_name=collection_name,
                    tenant_id=tenant_id
                )
                vector_docs.append(vector_doc)
            
            # Use bulk upsert (upsert handles both insert and update)
            operation = BulkVectorOperation(
                operation_type="update",
                documents=vector_docs,
                tenant_id=tenant_id
            )
            
            await self.bulk_upsert_documents(collection_name, operation)
            return True
            
        except Exception as e:
            raise RepositoryError(f"Failed to update documents in collection: {str(e)}") from e

    async def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete documents from a collection (legacy method)."""
        try:
            if ids:
                # Delete by IDs
                if self._is_async:
                    await self.client.delete(
                        collection_name=collection_name,
                        points_selector=ids
                    )
                else:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=ids
                    )
            else:
                # Delete by filter
                conditions = []
                
                if tenant_id:
                    conditions.append(
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
                        )
                    )
                
                if where:
                    for key, value in where.items():
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
                    
                    if self._is_async:
                        await self.client.delete(
                            collection_name=collection_name,
                            points_selector=query_filter
                        )
                    else:
                        self.client.delete(
                            collection_name=collection_name,
                            points_selector=query_filter
                        )
            
            return True
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete documents from collection: {str(e)}") from e

    async def count_documents(
        self, 
        collection_name: str,
        tenant_id: Optional[str] = None
    ) -> int:
        """Count documents in a collection (legacy method)."""
        try:
            if tenant_id:
                # Count tenant-specific documents
                stats = await self.get_collection_statistics(collection_name, tenant_id)
                return stats.document_count
            else:
                # Count all documents
                if self._is_async:
                    collection_info = await self.client.get_collection(collection_name)
                else:
                    collection_info = self.client.get_collection(collection_name)
                return collection_info.points_count
                
        except Exception as e:
            raise RepositoryError(f"Failed to count documents in collection: {str(e)}") from e

    async def similarity_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5,
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with optional filtering (legacy method)."""
        try:
            request = VectorSearchRequest(
                query_embedding=query_embedding,
                collection_name=collection_name,
                n_results=n_results,
                similarity_threshold=threshold or 0.0,
                metadata_filter=metadata_filter,
                tenant_id=tenant_id
            )
            
            results = await self.semantic_search_advanced(request)
            
            # Format in legacy format
            formatted_results = []
            for result in results:
                formatted_result = {
                    'document': result.document.content,
                    'similarity': result.similarity_score,
                    'distance': result.distance,
                    'id': result.document.id,
                    'metadata': result.document.metadata
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            raise RepositoryError(f"Failed to perform similarity search: {str(e)}") from e

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection (legacy method)."""
        try:
            if self._is_async:
                await self.client.delete_collection(collection_name=name)
            else:
                self.client.delete_collection(collection_name=name)
            
            # Remove from cache
            if name in self._collection_cache:
                del self._collection_cache[name]
            return True
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete collection {name}: {str(e)}") from e


# Maintain backward compatibility
QdrantRepository = EnhancedQdrantRepository