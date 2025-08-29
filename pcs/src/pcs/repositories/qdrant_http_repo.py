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
