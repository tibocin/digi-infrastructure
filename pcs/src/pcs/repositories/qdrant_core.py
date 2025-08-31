"""
Filepath: pcs/src/pcs/repositories/qdrant_core.py
Purpose: Core CRUD operations for Qdrant vector database including collections, points, and basic search
Related Components: Collection management, vector operations, basic search, health checks
Tags: qdrant, core-operations, collections, vectors, search
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .qdrant_types import (
    QdrantDistance, 
    QdrantCollectionConfig, 
    QdrantPoint, 
    QdrantSearchResult,
    VectorCollectionStats
)
from .qdrant_http_client import QdrantHTTPClient

logger = logging.getLogger(__name__)


class QdrantCoreOperations:
    """
    Core CRUD operations for Qdrant vector database.
    
    This class handles the fundamental operations for managing collections,
    vectors, and performing basic searches. It delegates HTTP operations
    to the QdrantHTTPClient for reliability.
    """
    
    def __init__(self, client: QdrantHTTPClient):
        """
        Initialize core operations with HTTP client.
        
        Args:
            client: Configured QdrantHTTPClient instance
        """
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health status."""
        try:
            return self.client.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise Exception(f"Health check failed: {e}")
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get list of all collections."""
        try:
            return self.client.get_collections()
        except Exception as e:
            self.logger.error(f"Failed to get collections: {e}")
            raise Exception(f"Failed to get collections: {e}")
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific collection."""
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            raise Exception(f"Failed to get collection info for {collection_name}: {e}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Union[str, QdrantDistance] = "cosine",
        on_disk_payload: bool = True,
        hnsw_config: Optional[Dict[str, Any]] = None,
        optimizers_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new collection with optimized configuration."""
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
            
            success = self.client.create_collection(collection_name, config)
            
            if success:
                self.logger.info(f"Successfully created collection: {collection_name}")
            else:
                self.logger.warning(f"Collection creation may have failed: {collection_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise Exception(f"Failed to create collection {collection_name}: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data."""
        try:
            success = self.client.delete_collection(collection_name)
            if success:
                self.logger.info(f"Successfully deleted collection: {collection_name}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise Exception(f"Failed to delete collection {collection_name}: {e}")
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[QdrantPoint],
        wait: bool = True
    ) -> Dict[str, Any]:
        """Upsert points to a collection."""
        try:
            # Convert QdrantPoint objects to dictionary format
            qdrant_points = []
            for point in points:
                point_dict = {
                    "id": point.id,
                    "vector": point.vector
                }
                if point.payload:
                    point_dict["payload"] = point.payload
                qdrant_points.append(point_dict)
            
            result = self.client.upsert_points(collection_name, qdrant_points, wait)
            
            if result.get("status") == "ok":
                self.logger.info(f"Successfully upserted {len(points)} points to {collection_name}")
            else:
                self.logger.warning(f"Upsert operation may have issues: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to upsert points to {collection_name}: {e}")
            raise Exception(f"Failed to upsert points to {collection_name}: {e}")
    
    def search_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[QdrantSearchResult]:
        """Search for similar points in a collection."""
        try:
            # Build search parameters
            search_params = {
                "vector": query_vector,
                "limit": limit
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            if filter_conditions:
                search_params["filter"] = filter_conditions
            
            # Perform search
            search_result = self.client.search_points(collection_name, search_params)
            
            # Convert to QdrantSearchResult objects
            results = []
            
            # Handle both dictionary with "result" key and direct list
            if isinstance(search_result, dict):
                points = search_result.get("result", [])
            else:
                points = search_result
            
            for point in points:
                # Handle both dictionary and Mock objects
                if hasattr(point, 'id'):
                    # Mock object or object with attributes
                    result = QdrantSearchResult(
                        id=getattr(point, 'id'),
                        score=getattr(point, 'score', 0.0),
                        payload=getattr(point, 'payload'),
                        vector=getattr(point, 'vector'),
                        version=getattr(point, 'version')
                    )
                else:
                    # Dictionary
                    result = QdrantSearchResult(
                        id=point.get("id"),
                        score=point.get("score", 0.0),
                        payload=point.get("payload"),
                        vector=point.get("vector"),
                        version=point.get("version")
                    )
                results.append(result)
            
            self.logger.info(f"Search completed for {collection_name}, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed for {collection_name}: {e}")
            raise Exception(f"Search failed for {collection_name}: {e}")
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: List[Union[str, int]],
        wait: bool = True
    ) -> Dict[str, Any]:
        """Delete points from a collection."""
        try:
            result = self.client.delete_points(collection_name, point_ids, wait)
            
            if result.get("status") == "ok":
                self.logger.info(f"Successfully deleted {len(point_ids)} points from {collection_name}")
            else:
                self.logger.warning(f"Delete operation may have issues: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to delete points from {collection_name}: {e}")
            raise Exception(f"Failed to delete points from {collection_name}: {e}")
    
    def get_collection_stats(self, collection_name: str) -> VectorCollectionStats:
        """Get collection statistics and metadata."""
        try:
            stats_data = self.client.get_collection_stats(collection_name)
            
            # Extract relevant statistics
            stats = VectorCollectionStats(
                vectors_count=stats_data.get("vectors_count", 0),
                points_count=stats_data.get("points_count", 0),
                segments_count=stats_data.get("segments_count", 0),
                status=stats_data.get("status", "unknown"),
                config=stats_data.get("config"),
                payload_schema=stats_data.get("payload_schema")
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get stats for {collection_name}: {e}")
            raise Exception(f"Failed to get stats for {collection_name}: {e}")
    
    def count_points(self, collection_name: str) -> int:
        """Get the total number of points in a collection."""
        try:
            stats = self.get_collection_stats(collection_name)
            return stats.points_count
        except Exception as e:
            self.logger.error(f"Failed to count points in {collection_name}: {e}")
            raise Exception(f"Failed to count points in {collection_name}: {e}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.get_collections()
            collection_names = [col.get("name") for col in collections]
            return collection_name in collection_names
        except Exception as e:
            self.logger.error(f"Failed to check if collection {collection_name} exists: {e}")
            return False
