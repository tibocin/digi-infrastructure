"""
Filepath: pcs/src/pcs/repositories/qdrant_http_client.py
Purpose: HTTP-based Qdrant client wrapper for reliable vector database operations
Related Components: Qdrant HTTP API, vector operations, authentication, error handling
Tags: qdrant, http-client, vector-database, reliability, authentication
"""

import json
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import httpx
from httpx import Client, AsyncClient, Response, HTTPStatusError

logger = logging.getLogger(__name__)


class QdrantDistance(Enum):
    """Qdrant distance metrics."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclid"
    DOT_PRODUCT = "Dot"
    MANHATTAN = "Manhattan"


@dataclass
class QdrantCollectionConfig:
    """Configuration for Qdrant collections."""
    name: str
    vector_size: int
    distance: QdrantDistance
    on_disk_payload: bool = True
    hnsw_config: Optional[Dict[str, Any]] = None
    optimizers_config: Optional[Dict[str, Any]] = None


@dataclass
class QdrantPoint:
    """Qdrant point structure."""
    id: Union[str, int]
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None


@dataclass
class QdrantSearchResult:
    """Search result from Qdrant."""
    id: Union[str, int]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    version: Optional[int] = None


class QdrantHTTPError(Exception):
    """Custom exception for Qdrant HTTP operations."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)


class QdrantHTTPClient:
    """
    HTTP-based Qdrant client for reliable vector database operations.
    
    This client bypasses the official Qdrant client library issues and provides
    a robust, production-ready interface for vector operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize HTTP-based Qdrant client.
        
        Args:
            host: Qdrant host
            port: Qdrant HTTP port
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = f"http://{host}:{port}"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key
            
        logger.info(f"Initialized Qdrant HTTP client for {self.base_url}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Qdrant with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request payload
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            QdrantHTTPError: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                with Client(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = client.get(url, headers=self.headers, params=params)
                    elif method.upper() == "POST":
                        response = client.post(url, headers=self.headers, json=data, params=params)
                    elif method.upper() == "PUT":
                        response = client.put(url, headers=self.headers, json=data, params=params)
                    elif method.upper() == "DELETE":
                        response = client.delete(url, headers=self.headers, params=params)
                    else:
                        raise QdrantHTTPError(f"Unsupported HTTP method: {method}")
                    
                    # Check response status
                    if response.status_code >= 400:
                        raise QdrantHTTPError(
                            f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_text=response.text
                        )
                    
                    # Parse response
                    if response.text:
                        return response.json()
                    return {}
                    
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Request timeout after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Request failed after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                time.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Unexpected error after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                time.sleep(self.retry_delay * (2 ** attempt))
    
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health status."""
        try:
            return self._make_request("GET", "/")
        except QdrantHTTPError as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get list of all collections."""
        try:
            response = self._make_request("GET", "/collections")
            return response.get("result", {}).get("collections", [])
        except QdrantHTTPError as e:
            logger.error(f"Failed to get collections: {e}")
            raise
    
    def get_collection(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        try:
            response = self._make_request("GET", f"/collections/{collection_name}")
            return response.get("result", {})
        except QdrantHTTPError as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise
    
    def create_collection(
        self,
        collection_name: str,
        config: QdrantCollectionConfig
    ) -> bool:
        """Create a new collection."""
        try:
            # Build collection configuration
            collection_config = {
                "vectors": {
                    "size": config.vector_size,
                    "distance": config.distance.value
                },
                "on_disk_payload": config.on_disk_payload
            }
            
            # Add HNSW configuration if provided
            if config.hnsw_config:
                collection_config["hnsw_config"] = config.hnsw_config
            
            # Add optimizers configuration if provided
            if config.optimizers_config:
                collection_config["optimizers_config"] = config.optimizers_config
            
            self._make_request("PUT", f"/collections/{collection_name}", data=collection_config)
            logger.info(f"Successfully created collection: {collection_name}")
            return True
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self._make_request("DELETE", f"/collections/{collection_name}")
            logger.info(f"Successfully deleted collection: {collection_name}")
            return True
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[QdrantPoint]
    ) -> Dict[str, Any]:
        """Insert or update points in a collection."""
        try:
            # Convert points to Qdrant format - Qdrant 1.7.4 requires integer IDs
            qdrant_points = []
            for i, point in enumerate(points):
                # Qdrant 1.7.4 only accepts integer IDs, so we generate sequential IDs
                # and store the original ID in the payload for reference
                point_id = i + 1  # Start from 1
                
                # Add original ID to payload for reference
                payload = point.payload or {}
                payload["original_id"] = str(point.id)
                
                qdrant_point = {
                    "id": point_id,
                    "vector": point.vector,
                    "payload": payload
                }
                qdrant_points.append(qdrant_point)
            
            # Use the exact format that works in our simple test
            data = {"points": qdrant_points}
            response = self._make_request("PUT", f"/collections/{collection_name}/points", data=data)
            
            logger.info(f"Successfully upserted {len(points)} points in collection: {collection_name}")
            return response.get("result", {})
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to upsert points in collection {collection_name}: {e}")
            raise
    
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
            # Build search request
            search_data = {
                "vector": query_vector,
                "limit": limit
            }
            
            if score_threshold is not None:
                search_data["score_threshold"] = score_threshold
            
            if filter_conditions:
                search_data["filter"] = filter_conditions
            
            response = self._make_request("POST", f"/collections/{collection_name}/points/search", data=search_data)
            
            # Convert response to search results
            results = []
            for item in response.get("result", []):
                # Extract original ID from payload if available
                payload = item.get("payload") or {}
                original_id = payload.get("original_id", str(item["id"]))
                
                result = QdrantSearchResult(
                    id=original_id,  # Use original ID for consistency
                    score=item["score"],
                    payload=payload,
                    vector=item.get("vector"),
                    version=item.get("version")
                )
                results.append(result)
            
            logger.info(f"Search returned {len(results)} results from collection: {collection_name}")
            return results
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to search points in collection {collection_name}: {e}")
            raise
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: List[Union[str, int]]
    ) -> Dict[str, Any]:
        """Delete points from a collection."""
        try:
            data = {"points": point_ids}
            response = self._make_request("POST", f"/collections/{collection_name}/points/delete", data=data)
            
            logger.info(f"Successfully deleted {len(point_ids)} points from collection: {collection_name}")
            return response.get("result", {})
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to delete points from collection {collection_name}: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            response = self._make_request("GET", f"/collections/{collection_name}")
            result = response.get("result", {})
            
            stats = {
                "vectors_count": result.get("vectors_count", 0),
                "points_count": result.get("points_count", 0),
                "segments_count": result.get("segments_count", 0),
                "status": result.get("status", "unknown")
            }
            
            return stats
            
        except QdrantHTTPError as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            raise


class AsyncQdrantHTTPClient(QdrantHTTPClient):
    """Async version of the HTTP-based Qdrant client."""
    
    async def _make_request_async(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make async HTTP request to Qdrant with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with AsyncClient(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=self.headers, params=params)
                    elif method.upper() == "POST":
                        response = await client.post(url, headers=self.headers, json=data, params=params)
                    elif method.upper() == "PUT":
                        response = await client.put(url, headers=self.headers, json=data, params=params)
                    elif method.upper() == "DELETE":
                        response = await client.delete(url, headers=self.headers, params=params)
                    else:
                        raise QdrantHTTPError(f"Unsupported HTTP method: {method}")
                    
                    # Check response status
                    if response.status_code >= 400:
                        raise QdrantHTTPError(
                            f"HTTP {response.status_code}: {response.text}",
                            status_code=response.status_code,
                            response_text=response.text
                        )
                    
                    # Parse response
                    if response.text:
                        return response.json()
                    return {}
                    
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Request timeout after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Request failed after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Request error (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                
            except Exception as e:
                if attempt == self.max_retries:
                    raise QdrantHTTPError(f"Unexpected error after {self.max_retries + 1} attempts: {e}")
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{self.max_retries + 1}), retrying...")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def health_check_async(self) -> Dict[str, Any]:
        """Async health check."""
        return await self._make_request_async("GET", "/")
    
    async def get_collections_async(self) -> List[Dict[str, Any]]:
        """Async get collections."""
        response = await self._make_request_async("GET", "/collections")
        return response.get("result", {}).get("collections", [])
    
    async def create_collection_async(
        self,
        collection_name: str,
        config: 'QdrantCollectionConfig'
    ) -> bool:
        """Async create a new collection."""
        # Build collection configuration
        collection_config = {
            "vectors": {
                "size": config.vector_size,
                "distance": config.distance.value
            },
            "hnsw_config": config.hnsw_config or {},
            "optimizers_config": config.optimizers_config or {},
            "on_disk_payload": config.on_disk_payload
        }
        
        try:
            response = await self._make_request_async(
                "PUT", 
                f"/collections/{collection_name}",
                data=collection_config
            )
            return response.get("result", False)
        except HTTPStatusError as e:
            if e.response.status_code == 409:  # Collection already exists
                self.logger.warning(f"Collection {collection_name} already exists")
                return True
            raise QdrantHTTPError(f"Failed to create collection: {e}", e.response.status_code)
    # Add other async methods as needed...
