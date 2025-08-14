"""
Filepath: pcs/src/pcs/repositories/redis_repo.py
Purpose: Enhanced Redis repository implementation for advanced caching and context management operations
Related Components: Redis client, context caching, session management, cache invalidation, performance optimization
Tags: redis, caching, context, async, ttl, cache-invalidation, performance-optimization, multi-level-cache
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Union, Tuple, Callable
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import asyncio

# Optional imports for testing environments
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    # Create mock Redis classes and exceptions for testing
    class MockRedisError(Exception):
        pass
    
    class MockRedis:
        RedisError = MockRedisError
        
        def __init__(self, *args, **kwargs):
            pass
        
        async def close(self):
            pass
    
    class Redis:
        def __init__(self, *args, **kwargs):
            pass
        
        async def close(self):
            pass
    
    # Create a mock redis module-like object
    redis = MockRedis()

from ..utils.logger import get_logger

from .base import RepositoryError
from ..utils.metrics import PerformanceMonitor, record_manual_metric


class CacheStrategy(Enum):
    """Enum for different caching strategies."""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    CACHE_ASIDE = "cache_aside"


class CacheLevel(Enum):
    """Enum for cache hierarchy levels."""
    L1 = "l1"  # Memory cache (fastest, smallest)
    L2 = "l2"  # Redis cache (fast, medium)
    L3 = "l3"  # Persistent cache (slower, largest)


class InvalidationType(Enum):
    """Enum for cache invalidation types."""
    TTL = "ttl"
    TAG_BASED = "tag_based"
    PATTERN_BASED = "pattern_based"
    DEPENDENCY_BASED = "dependency_based"
    EVENT_DRIVEN = "event_driven"


@dataclass
class CacheEntry:
    """Container for cache entry with metadata."""
    key: str
    value: Any
    ttl: Optional[int] = None
    tags: List[str] = None
    dependencies: List[str] = None
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_accessed is None:
            self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes
        }


@dataclass
class CacheStats:
    """Container for cache statistics and analytics."""
    total_operations: int = 0
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    avg_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    total_keys: int = 0
    expired_keys: int = 0
    evicted_keys: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
        if self.total_operations > 0:
            self.hit_ratio = self.hits / self.total_operations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_operations": self.total_operations,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "avg_response_time_ms": self.avg_response_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "total_keys": self.total_keys,
            "expired_keys": self.expired_keys,
            "evicted_keys": self.evicted_keys,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class InvalidationRequest:
    """Container for cache invalidation requests."""
    invalidation_type: InvalidationType
    target: str  # Key, pattern, or tag
    cascade: bool = False
    immediate: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "invalidation_type": self.invalidation_type.value,
            "target": self.target,
            "cascade": self.cascade,
            "immediate": self.immediate,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class BatchOperation:
    """Container for batch cache operations."""
    operation_type: str  # "set", "get", "delete"
    operations: List[Dict[str, Any]]
    batch_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "operation_count": len(self.operations),
            "batch_size": self.batch_size
        }


class EnhancedRedisRepository:
    """
    Enhanced Redis repository for advanced caching and data storage.
    
    Features:
    - Multi-level caching strategies
    - Intelligent cache invalidation
    - Batch operations with pipelining
    - Performance monitoring and optimization
    - Advanced Redis data structures
    - Cache analytics and statistics
    - Memory optimization and compression
    - Event-driven cache management
    """
    
    def __init__(self, redis_client: "redis.Redis"):
        """
        Initialize enhanced Redis repository.
        
        Args:
            redis_client: Redis async client instance
        """
        self.redis = redis_client
        self.logger = get_logger(__name__)
        self._cache_stats = CacheStats()
        self._operation_metrics = []
        self._tag_cache = {}  # In-memory tag-to-keys mapping
        self._dependency_graph = defaultdict(set)  # Dependency tracking
        self._l1_cache = {}  # In-memory L1 cache for hot data
        self._l1_max_size = 1000  # Maximum L1 cache entries
    
    def _safe_deserialize(self, value_str: str) -> Any:
        """
        Safely deserialize a string value.
        
        Only deserializes proper JSON objects and arrays, keeps simple 
        strings and numbers as strings to preserve type expectations.
        
        Args:
            value_str: String value to potentially deserialize
            
        Returns:
            Deserialized object for JSON objects/arrays, original string otherwise
        """
        # Only attempt JSON deserialization for values that look like JSON objects or arrays
        stripped = value_str.strip()
        if stripped.startswith(('{', '[')):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                return value_str
        return value_str
    
    def _make_hashable(self, obj: Any) -> Any:
        """
        Convert an object to a hashable form for use in sets.
        
        Args:
            obj: Object to make hashable
            
        Returns:
            Hashable representation of the object
        """
        if isinstance(obj, dict):
            # Convert dict to a hashable tuple of sorted items
            return tuple(sorted(obj.items()))
        elif isinstance(obj, list):
            # Convert list to tuple
            return tuple(obj)
        return obj

    def _calculate_size_bytes(self, value: Any) -> int:
        """Calculate approximate size in bytes for a value."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Approximate size
            elif isinstance(value, bytes):
                return len(value)
            else:
                # For complex objects, estimate based on JSON serialization
                return len(json.dumps(value, default=str).encode('utf-8'))
        except Exception:
            return 0

    def _update_cache_stats(self, operation: str, hit: bool = False, response_time_ms: float = 0.0):
        """Update cache statistics."""
        self._cache_stats.total_operations += 1
        if hit:
            self._cache_stats.hits += 1
        else:
            self._cache_stats.misses += 1
        
        # Update hit ratio
        self._cache_stats.hit_ratio = self._cache_stats.hits / self._cache_stats.total_operations
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        self._cache_stats.avg_response_time_ms = (
            alpha * response_time_ms + 
            (1 - alpha) * self._cache_stats.avg_response_time_ms
        )
        
        self._cache_stats.last_updated = datetime.utcnow()

    def _manage_l1_cache(self, key: str, value: Any = None, remove: bool = False):
        """Manage L1 (in-memory) cache for hot data."""
        if remove:
            self._l1_cache.pop(key, None)
            return
        
        if value is not None:
            # Add to L1 cache
            if len(self._l1_cache) >= self._l1_max_size:
                # Remove oldest entry (simple LRU approximation)
                oldest_key = next(iter(self._l1_cache))
                del self._l1_cache[oldest_key]
            
            self._l1_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }

    async def set_advanced(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE,
        level: CacheLevel = CacheLevel.L2,
        compress: bool = False
    ) -> bool:
        """
        Advanced set operation with caching strategies and metadata.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live (seconds or timedelta)
            tags: Tags for tag-based invalidation
            dependencies: Dependencies for dependency-based invalidation
            strategy: Caching strategy to use
            level: Cache level for multi-level caching
            compress: Whether to compress large values
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            async with PerformanceMonitor("cache_set_advanced", "redis") as monitor:
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl.total_seconds() if isinstance(ttl, timedelta) else ttl,
                    tags=tags or [],
                    dependencies=dependencies or []
                )
                entry.size_bytes = self._calculate_size_bytes(value)
                
                # Handle multi-level caching
                if level == CacheLevel.L1:
                    self._manage_l1_cache(key, value)
                    monitor.set_rows_affected(1)
                    return True
                
                # Serialize value if needed
                if not isinstance(value, (str, bytes, int, float)):
                    value = json.dumps(value, default=str)
                
                # Store metadata separately
                metadata_key = f"meta:{key}"
                metadata = {
                    "tags": entry.tags,
                    "dependencies": entry.dependencies,
                    "created_at": entry.created_at.isoformat(),
                    "size_bytes": entry.size_bytes
                }
                
                # Use pipeline for atomic operations
                pipe = self.redis.pipeline()
                
                # Set main value
                if ttl:
                    ttl_seconds = ttl.total_seconds() if isinstance(ttl, timedelta) else ttl
                    pipe.setex(key, int(ttl_seconds), value)
                    pipe.setex(metadata_key, int(ttl_seconds), json.dumps(metadata))
                else:
                    pipe.set(key, value)
                    pipe.set(metadata_key, json.dumps(metadata))
                
                # Update tag mappings
                for tag in entry.tags:
                    tag_key = f"tag:{tag}"
                    pipe.sadd(tag_key, key)
                    if ttl:
                        pipe.expire(tag_key, int(ttl_seconds))
                
                # Update dependency graph
                for dep in entry.dependencies:
                    dep_key = f"dep:{dep}"
                    pipe.sadd(dep_key, key)
                    if ttl:
                        pipe.expire(dep_key, int(ttl_seconds))
                
                await pipe.execute()
                
                # Update in-memory structures
                if entry.tags:
                    for tag in entry.tags:
                        if tag not in self._tag_cache:
                            self._tag_cache[tag] = set()
                        self._tag_cache[tag].add(key)
                
                if entry.dependencies:
                    for dep in entry.dependencies:
                        self._dependency_graph[dep].add(key)
                
                # Cache in L1 if it's frequently accessed data
                if strategy in [CacheStrategy.WRITE_THROUGH, CacheStrategy.CACHE_ASIDE]:
                    self._manage_l1_cache(key, value)
                
                response_time = (time.time() - start_time) * 1000
                self._update_cache_stats("set", response_time_ms=response_time)
                monitor.set_rows_affected(1)
                
                return True
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_cache_stats("set", response_time_ms=response_time)
            raise RepositoryError(f"Failed to set advanced Redis key {key}: {str(e)}") from e

    async def get_advanced(
        self,
        key: str,
        update_access: bool = True,
        from_level: Optional[CacheLevel] = None
    ) -> Optional[Any]:
        """
        Advanced get operation with multi-level caching and access tracking.
        
        Args:
            key: Redis key
            update_access: Whether to update access statistics
            from_level: Specific cache level to read from
            
        Returns:
            Value if found, None otherwise
        """
        start_time = time.time()
        hit = False
        
        try:
            async with PerformanceMonitor("cache_get_advanced", "redis") as monitor:
                # Try L1 cache first (if not specified level)
                if from_level is None or from_level == CacheLevel.L1:
                    if key in self._l1_cache:
                        hit = True
                        value = self._l1_cache[key]['value']
                        
                        if update_access:
                            self._l1_cache[key]['timestamp'] = time.time()
                        
                        response_time = (time.time() - start_time) * 1000
                        self._update_cache_stats("get", hit=True, response_time_ms=response_time)
                        monitor.set_rows_affected(1)
                        return value
                
                # Try L2 cache (Redis)
                if from_level is None or from_level == CacheLevel.L2:
                    value = await self.redis.get(key)
                    if value is not None:
                        hit = True
                        
                        # Deserialize if needed
                        if isinstance(value, (str, bytes)):
                            value_str = value.decode() if isinstance(value, bytes) else value
                            try:
                                deserialized_value = json.loads(value_str)
                            except json.JSONDecodeError:
                                deserialized_value = value_str
                        else:
                            deserialized_value = value
                        
                        # Update access metadata if requested
                        if update_access:
                            metadata_key = f"meta:{key}"
                            await self.redis.hset(metadata_key, "last_accessed", datetime.utcnow().isoformat())
                            await self.redis.hincrby(metadata_key, "access_count", 1)
                        
                        # Promote to L1 cache for frequently accessed data
                        self._manage_l1_cache(key, deserialized_value)
                        
                        response_time = (time.time() - start_time) * 1000
                        self._update_cache_stats("get", hit=True, response_time_ms=response_time)
                        monitor.set_rows_affected(1)
                        return deserialized_value
                
                # Cache miss
                response_time = (time.time() - start_time) * 1000
                self._update_cache_stats("get", hit=False, response_time_ms=response_time)
                monitor.set_rows_affected(0)
                return None
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_cache_stats("get", hit=hit, response_time_ms=response_time)
            raise RepositoryError(f"Failed to get advanced Redis key {key}: {str(e)}") from e

    async def batch_operations(self, batch: BatchOperation) -> Dict[str, Any]:
        """
        Execute batch operations with pipelining for performance.
        
        Args:
            batch: Batch operation specification
            
        Returns:
            Results dictionary with operation statistics
        """
        start_time = time.time()
        
        try:
            async with PerformanceMonitor("cache_batch_operations", "redis") as monitor:
                results = {
                    "total_operations": len(batch.operations),
                    "successful": 0,
                    "failed": 0,
                    "results": []
                }
                
                # Process in batches for memory efficiency
                for i in range(0, len(batch.operations), batch.batch_size):
                    batch_ops = batch.operations[i:i + batch.batch_size]
                    
                    pipe = self.redis.pipeline()
                    
                    # Add operations to pipeline
                    for op in batch_ops:
                        if batch.operation_type == "set":
                            key = op["key"]
                            value = op["value"]
                            ttl = op.get("ttl")
                            
                            if not isinstance(value, (str, bytes, int, float)):
                                value = json.dumps(value, default=str)
                            
                            if ttl:
                                pipe.setex(key, ttl, value)
                            else:
                                pipe.set(key, value)
                                
                        elif batch.operation_type == "get":
                            pipe.get(op["key"])
                            
                        elif batch.operation_type == "delete":
                            pipe.delete(op["key"])
                    
                    # Execute pipeline
                    batch_results = await pipe.execute()
                    
                    # Process results
                    for j, result in enumerate(batch_results):
                        if result is not None and result is not False:
                            results["successful"] += 1
                        else:
                            results["failed"] += 1
                        
                        results["results"].append({
                            "operation_index": i + j,
                            "result": result,
                            "success": result is not None and result is not False
                        })
                
                execution_time = time.time() - start_time
                results["execution_time_seconds"] = execution_time
                results["operations_per_second"] = len(batch.operations) / execution_time if execution_time > 0 else 0
                
                # Record performance metrics
                record_manual_metric(
                    query_type=f"batch_{batch.operation_type}",
                    execution_time=execution_time,
                    rows_affected=results["successful"],
                    table_name="redis_batch"
                )
                
                monitor.set_rows_affected(results["successful"])
                return results
                
        except Exception as e:
            raise RepositoryError(f"Failed to execute batch operations: {str(e)}") from e

    async def invalidate_cache(self, request: InvalidationRequest) -> Dict[str, Any]:
        """
        Intelligent cache invalidation with various strategies.
        
        Args:
            request: Invalidation request specification
            
        Returns:
            Invalidation results and statistics
        """
        start_time = time.time()
        
        try:
            async with PerformanceMonitor("cache_invalidation", "redis") as monitor:
                invalidated_keys = set()
                
                if request.invalidation_type == InvalidationType.TAG_BASED:
                    # Invalidate all keys with specific tag
                    tag_key = f"tag:{request.target}"
                    tagged_keys = await self.redis.smembers(tag_key)
                    
                    if tagged_keys:
                        # Convert bytes to strings
                        key_list = [k.decode() if isinstance(k, bytes) else k for k in tagged_keys]
                        invalidated_keys.update(key_list)
                        
                        # Delete keys and metadata
                        pipe = self.redis.pipeline()
                        for key in key_list:
                            pipe.delete(key)
                            pipe.delete(f"meta:{key}")
                            # Remove from L1 cache
                            self._manage_l1_cache(key, remove=True)
                        
                        pipe.delete(tag_key)
                        await pipe.execute()
                        
                        # Update in-memory tag cache
                        if request.target in self._tag_cache:
                            del self._tag_cache[request.target]
                
                elif request.invalidation_type == InvalidationType.PATTERN_BASED:
                    # Invalidate keys matching pattern
                    matching_keys = await self.redis.keys(request.target)
                    
                    if matching_keys:
                        key_list = [k.decode() if isinstance(k, bytes) else k for k in matching_keys]
                        invalidated_keys.update(key_list)
                        
                        pipe = self.redis.pipeline()
                        for key in key_list:
                            pipe.delete(key)
                            pipe.delete(f"meta:{key}")
                            self._manage_l1_cache(key, remove=True)
                        
                        await pipe.execute()
                
                elif request.invalidation_type == InvalidationType.DEPENDENCY_BASED:
                    # Invalidate all keys dependent on target
                    dep_key = f"dep:{request.target}"
                    dependent_keys = await self.redis.smembers(dep_key)
                    
                    if dependent_keys:
                        key_list = [k.decode() if isinstance(k, bytes) else k for k in dependent_keys]
                        invalidated_keys.update(key_list)
                        
                        pipe = self.redis.pipeline()
                        for key in key_list:
                            pipe.delete(key)
                            pipe.delete(f"meta:{key}")
                            self._manage_l1_cache(key, remove=True)
                        
                        pipe.delete(dep_key)
                        await pipe.execute()
                        
                        # Update dependency graph
                        if request.target in self._dependency_graph:
                            del self._dependency_graph[request.target]
                
                elif request.invalidation_type == InvalidationType.TTL:
                    # Set TTL for immediate expiration
                    await self.redis.expire(request.target, 1)
                    invalidated_keys.add(request.target)
                    self._manage_l1_cache(request.target, remove=True)
                
                # Cascade invalidation if requested
                if request.cascade and invalidated_keys:
                    cascade_keys = set()
                    for key in invalidated_keys:
                        # Find keys that depend on this key
                        if key in self._dependency_graph:
                            cascade_keys.update(self._dependency_graph[key])
                    
                    if cascade_keys:
                        pipe = self.redis.pipeline()
                        for key in cascade_keys:
                            pipe.delete(key)
                            pipe.delete(f"meta:{key}")
                            self._manage_l1_cache(key, remove=True)
                        
                        await pipe.execute()
                        invalidated_keys.update(cascade_keys)
                
                execution_time = time.time() - start_time
                
                results = {
                    "invalidation_type": request.invalidation_type.value,
                    "target": request.target,
                    "invalidated_count": len(invalidated_keys),
                    "invalidated_keys": list(invalidated_keys),
                    "cascade": request.cascade,
                    "execution_time_seconds": execution_time
                }
                
                monitor.set_rows_affected(len(invalidated_keys))
                return results
                
        except Exception as e:
            raise RepositoryError(f"Failed to invalidate cache: {str(e)}") from e

    async def warm_cache(
        self,
        data_source: Callable,
        keys: List[str],
        strategy: str = "preload",
        ttl: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Warm cache by preloading data from a source.
        
        Args:
            data_source: Async function to fetch data for a key
            keys: Keys to warm
            strategy: Warming strategy ("preload", "lazy", "scheduled")
            ttl: TTL for warmed entries
            
        Returns:
            Warming results and statistics
        """
        start_time = time.time()
        
        try:
            async with PerformanceMonitor("cache_warming", "redis") as monitor:
                results = {
                    "strategy": strategy,
                    "total_keys": len(keys),
                    "warmed_successfully": 0,
                    "failed": 0,
                    "errors": []
                }
                
                # Warm cache entries
                for key in keys:
                    try:
                        # Check if already cached
                        if await self.redis.exists(key):
                            continue
                        
                        # Fetch data from source
                        data = await data_source(key)
                        if data is not None:
                            # Store in cache
                            await self.set_advanced(
                                key=key,
                                value=data,
                                ttl=ttl,
                                strategy=CacheStrategy.WRITE_THROUGH
                            )
                            results["warmed_successfully"] += 1
                    
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append({
                            "key": key,
                            "error": str(e)
                        })
                
                execution_time = time.time() - start_time
                results["execution_time_seconds"] = execution_time
                results["keys_per_second"] = len(keys) / execution_time if execution_time > 0 else 0
                
                monitor.set_rows_affected(results["warmed_successfully"])
                return results
                
        except Exception as e:
            raise RepositoryError(f"Failed to warm cache: {str(e)}") from e

    async def get_cache_statistics(self) -> CacheStats:
        """
        Get comprehensive cache statistics and analytics.
        
        Returns:
            Cache statistics object
        """
        try:
            async with PerformanceMonitor("cache_statistics", "redis") as monitor:
                # Get Redis memory info
                info = await self.redis.info("memory")
                memory_usage_mb = info.get("used_memory", 0) / (1024 * 1024)
                
                # Get key statistics
                dbsize = await self.redis.dbsize()
                
                # Update statistics
                self._cache_stats.memory_usage_mb = memory_usage_mb
                self._cache_stats.total_keys = dbsize
                self._cache_stats.last_updated = datetime.utcnow()
                
                monitor.set_rows_affected(1)
                return self._cache_stats
                
        except Exception as e:
            raise RepositoryError(f"Failed to get cache statistics: {str(e)}") from e

    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Optimize cache performance by analyzing and reorganizing data.
        
        Returns:
            Optimization results and recommendations
        """
        try:
            async with PerformanceMonitor("cache_optimization", "redis") as monitor:
                results = {
                    "optimizations_applied": [],
                    "recommendations": [],
                    "before_stats": {},
                    "after_stats": {}
                }
                
                # Get current statistics
                before_stats = await self.get_cache_statistics()
                results["before_stats"] = before_stats.to_dict()
                
                # Clean up expired keys
                expired_count = 0
                all_keys = await self.redis.keys("*")
                
                pipe = self.redis.pipeline()
                for key_bytes in all_keys:
                    key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                    ttl = await self.redis.ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        pipe.delete(key)
                        pipe.delete(f"meta:{key}")
                        expired_count += 1
                
                if expired_count > 0:
                    await pipe.execute()
                    results["optimizations_applied"].append(f"Cleaned up {expired_count} expired keys")
                
                # Optimize L1 cache size based on hit ratio
                if self._cache_stats.hit_ratio < 0.8 and len(self._l1_cache) < 2000:
                    self._l1_max_size = min(2000, self._l1_max_size * 2)
                    results["optimizations_applied"].append(f"Increased L1 cache size to {self._l1_max_size}")
                
                # Generate recommendations
                if self._cache_stats.hit_ratio < 0.7:
                    results["recommendations"].append("Consider increasing cache TTL for frequently accessed data")
                
                if self._cache_stats.memory_usage_mb > 100:
                    results["recommendations"].append("Consider implementing data compression for large values")
                
                # Get updated statistics
                after_stats = await self.get_cache_statistics()
                results["after_stats"] = after_stats.to_dict()
                
                monitor.set_rows_affected(1)
                return results
                
        except Exception as e:
            raise RepositoryError(f"Failed to optimize cache performance: {str(e)}") from e

    # Advanced Redis data structure operations

    async def sorted_set_add_with_score(
        self,
        key: str,
        member: str,
        score: float,
        ttl: Optional[int] = None
    ) -> bool:
        """Add member to sorted set with score."""
        try:
            result = await self.redis.zadd(key, {member: score})
            if ttl:
                await self.redis.expire(key, ttl)
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to add to sorted set {key}: {str(e)}") from e

    async def sorted_set_get_top(
        self,
        key: str,
        count: int = 10,
        with_scores: bool = True
    ) -> List[Union[str, Tuple[str, float]]]:
        """Get top N members from sorted set."""
        try:
            if with_scores:
                return await self.redis.zrevrange(key, 0, count - 1, withscores=True)
            else:
                return await self.redis.zrevrange(key, 0, count - 1)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get top from sorted set {key}: {str(e)}") from e

    async def hyperloglog_add(self, key: str, *elements: str) -> bool:
        """Add elements to HyperLogLog for cardinality estimation."""
        try:
            result = await self.redis.pfadd(key, *elements)
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to add to HyperLogLog {key}: {str(e)}") from e

    async def hyperloglog_count(self, key: str) -> int:
        """Get cardinality estimate from HyperLogLog."""
        try:
            return await self.redis.pfcount(key)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to count HyperLogLog {key}: {str(e)}") from e

    # Legacy methods for backward compatibility
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[Union[int, timedelta]] = None,
        serialize: bool = True
    ) -> bool:
        """Legacy set method for backward compatibility."""
        return await self.set_advanced(key, value, ttl)

    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """Legacy get method for backward compatibility."""
        return await self.get_advanced(key)

    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if not keys:
                return 0
            
            # Remove from L1 cache
            for key in keys:
                self._manage_l1_cache(key, remove=True)
            
            return await self.redis.delete(*keys)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to delete Redis keys: {str(e)}") from e

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Redis key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            # Check L1 cache first
            if key in self._l1_cache:
                return True
            
            return bool(await self.redis.exists(key))
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to check existence of Redis key {key}: {str(e)}") from e

    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """
        Set TTL on an existing key.
        
        Args:
            key: Redis key
            ttl: Time to live (seconds or timedelta)
            
        Returns:
            True if TTL was set, False if key doesn't exist
        """
        try:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            return bool(await self.redis.expire(key, ttl))
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to set TTL on Redis key {key}: {str(e)}") from e

    async def ttl(self, key: str) -> int:
        """
        Get TTL of a key.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            return await self.redis.ttl(key)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get TTL of Redis key {key}: {str(e)}") from e

    # Hash operations
    async def hset(self, key: str, mapping: Dict[str, Any], serialize: bool = True) -> int:
        """
        Set hash field values.
        
        Args:
            key: Redis hash key
            mapping: Dictionary of field-value pairs
            serialize: Whether to JSON serialize values
            
        Returns:
            Number of fields added
        """
        try:
            if serialize:
                serialized_mapping = {}
                for field, value in mapping.items():
                    if isinstance(value, (str, int, float)):
                        serialized_mapping[field] = value
                    else:
                        serialized_mapping[field] = json.dumps(value, default=str)
                mapping = serialized_mapping
            
            return await self.redis.hset(key, mapping=mapping)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to set Redis hash {key}: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise RepositoryError(f"Failed to serialize hash values for {key}: {str(e)}") from e

    async def hget(self, key: str, field: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get hash field value.
        
        Args:
            key: Redis hash key
            field: Hash field name
            deserialize: Whether to JSON deserialize the value
            
        Returns:
            Field value if found, None otherwise
        """
        try:
            value = await self.redis.hget(key, field)
            if value is None:
                return None
            
            if deserialize and isinstance(value, (str, bytes)):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode() if isinstance(value, bytes) else value
            
            return value.decode() if isinstance(value, bytes) else value
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis hash field {key}.{field}: {str(e)}") from e

    async def hgetall(self, key: str, deserialize: bool = True) -> Dict[str, Any]:
        """
        Get all hash fields and values.
        
        Args:
            key: Redis hash key
            deserialize: Whether to JSON deserialize values
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            hash_data = await self.redis.hgetall(key)
            
            if not deserialize:
                return {k.decode() if isinstance(k, bytes) else k: 
                       v.decode() if isinstance(v, bytes) else v 
                       for k, v in hash_data.items()}
            
            result = {}
            for field, value in hash_data.items():
                field_str = field.decode() if isinstance(field, bytes) else field
                
                if isinstance(value, (str, bytes)):
                    value_str = value.decode() if isinstance(value, bytes) else value
                    result[field_str] = self._safe_deserialize(value_str)
                else:
                    result[field_str] = value
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get all Redis hash fields for {key}: {str(e)}") from e

    async def hdel(self, key: str, *fields: str) -> int:
        """
        Delete hash fields.
        
        Args:
            key: Redis hash key
            *fields: Field names to delete
            
        Returns:
            Number of fields deleted
        """
        try:
            if not fields:
                return 0
            return await self.redis.hdel(key, *fields)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to delete Redis hash fields from {key}: {str(e)}") from e

    # Set operations
    async def sadd(self, key: str, *members: Any, serialize: bool = True) -> int:
        """
        Add members to a set.
        
        Args:
            key: Redis set key
            *members: Members to add
            serialize: Whether to JSON serialize members
            
        Returns:
            Number of members added
        """
        try:
            if serialize:
                serialized_members = []
                for member in members:
                    if isinstance(member, (str, int, float)):
                        serialized_members.append(member)
                    else:
                        serialized_members.append(json.dumps(member, default=str))
                members = serialized_members
            
            return await self.redis.sadd(key, *members)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to add to Redis set {key}: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise RepositoryError(f"Failed to serialize set members for {key}: {str(e)}") from e

    async def smembers(self, key: str, deserialize: bool = True) -> set:
        """
        Get all set members.
        
        Args:
            key: Redis set key
            deserialize: Whether to JSON deserialize members
            
        Returns:
            Set of members
        """
        try:
            members = await self.redis.smembers(key)
            
            if not deserialize:
                return {m.decode() if isinstance(m, bytes) else m for m in members}
            
            result = set()
            for member in members:
                if isinstance(member, (str, bytes)):
                    member_str = member.decode() if isinstance(member, bytes) else member
                    deserialized = self._safe_deserialize(member_str)
                    # Make hashable for sets
                    hashable_value = self._make_hashable(deserialized)
                    result.add(hashable_value)
                else:
                    result.add(member)
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis set members for {key}: {str(e)}") from e

    async def srem(self, key: str, *members: Any, serialize: bool = True) -> int:
        """
        Remove members from a set.
        
        Args:
            key: Redis set key
            *members: Members to remove
            serialize: Whether to JSON serialize members
            
        Returns:
            Number of members removed
        """
        try:
            if serialize:
                serialized_members = []
                for member in members:
                    if isinstance(member, (str, int, float)):
                        serialized_members.append(member)
                    else:
                        serialized_members.append(json.dumps(member, default=str))
                members = serialized_members
            
            return await self.redis.srem(key, *members)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to remove from Redis set {key}: {str(e)}") from e

    # List operations
    async def lpush(self, key: str, *values: Any, serialize: bool = True) -> int:
        """
        Push values to the left (beginning) of a list.
        
        Args:
            key: Redis list key
            *values: Values to push
            serialize: Whether to JSON serialize values
            
        Returns:
            New list length
        """
        try:
            if serialize:
                serialized_values = []
                for value in values:
                    if isinstance(value, (str, int, float)):
                        serialized_values.append(value)
                    else:
                        serialized_values.append(json.dumps(value, default=str))
                values = serialized_values
            
            return await self.redis.lpush(key, *values)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to lpush to Redis list {key}: {str(e)}") from e

    async def rpush(self, key: str, *values: Any, serialize: bool = True) -> int:
        """
        Push values to the right (end) of a list.
        
        Args:
            key: Redis list key
            *values: Values to push
            serialize: Whether to JSON serialize values
            
        Returns:
            New list length
        """
        try:
            if serialize:
                serialized_values = []
                for value in values:
                    if isinstance(value, (str, int, float)):
                        serialized_values.append(value)
                    else:
                        serialized_values.append(json.dumps(value, default=str))
                values = serialized_values
            
            return await self.redis.rpush(key, *values)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to rpush to Redis list {key}: {str(e)}") from e

    async def lrange(self, key: str, start: int = 0, end: int = -1, deserialize: bool = True) -> List[Any]:
        """
        Get list elements by range.
        
        Args:
            key: Redis list key
            start: Start index
            end: End index (-1 for all)
            deserialize: Whether to JSON deserialize values
            
        Returns:
            List of values
        """
        try:
            values = await self.redis.lrange(key, start, end)
            
            if not deserialize:
                return [v.decode() if isinstance(v, bytes) else v for v in values]
            
            result = []
            for value in values:
                if isinstance(value, (str, bytes)):
                    value_str = value.decode() if isinstance(value, bytes) else value
                    result.append(self._safe_deserialize(value_str))
                else:
                    result.append(value)
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis list range for {key}: {str(e)}") from e

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """
        Trim list to specified range.
        
        Args:
            key: Redis list key
            start: Start index
            end: End index
            
        Returns:
            True if successful
        """
        try:
            result = await self.redis.ltrim(key, start, end)
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to trim Redis list {key}: {str(e)}") from e

    # Utility methods
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        try:
            keys = await self.redis.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis keys with pattern {pattern}: {str(e)}") from e

    async def flushdb(self) -> bool:
        """
        Clear all keys in the current database.
        
        Returns:
            True if successful
        """
        try:
            # Clear L1 cache
            self._l1_cache.clear()
            self._tag_cache.clear()
            self._dependency_graph.clear()
            
            result = await self.redis.flushdb()
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to flush Redis database: {str(e)}") from e

    async def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connection is alive
        """
        try:
            result = await self.redis.ping()
            return bool(result)
        except redis.RedisError:
            return False

# Maintain backward compatibility
RedisRepository = EnhancedRedisRepository
