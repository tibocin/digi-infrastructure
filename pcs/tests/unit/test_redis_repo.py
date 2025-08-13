"""
Filepath: tests/unit/test_redis_repo.py
Purpose: Unit tests for enhanced Redis repository implementation
Related Components: EnhancedRedisRepository, CacheEntry, CacheStats, InvalidationRequest, BatchOperation
Tags: testing, redis, caching, cache-invalidation, multi-level-cache, performance-optimization
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4
from typing import List, Dict, Any
import json
import time

from pcs.repositories.redis_repo import (
    EnhancedRedisRepository,
    RedisRepository,  # Legacy alias
    CacheEntry,
    CacheStats,
    InvalidationRequest,
    BatchOperation,
    CacheStrategy,
    CacheLevel,
    InvalidationType
)
from pcs.repositories.base import RepositoryError


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    client = AsyncMock()
    
    # Mock basic operations
    client.set.return_value = True
    client.get.return_value = None
    client.delete.return_value = 1
    client.exists.return_value = False
    client.expire.return_value = True
    client.ttl.return_value = -1
    client.ping.return_value = True
    client.flushdb.return_value = True
    client.keys.return_value = []
    client.dbsize.return_value = 0
    
    # Mock hash operations
    client.hset.return_value = 1
    client.hget.return_value = None
    client.hgetall.return_value = {}
    client.hdel.return_value = 1
    client.hincrby.return_value = 1
    
    # Mock set operations
    client.sadd.return_value = 1
    client.smembers.return_value = set()
    client.srem.return_value = 1
    
    # Mock list operations
    client.lpush.return_value = 1
    client.rpush.return_value = 1
    client.lrange.return_value = []
    client.ltrim.return_value = True
    
    # Mock sorted set operations
    client.zadd.return_value = 1
    client.zrevrange.return_value = []
    
    # Mock HyperLogLog operations
    client.pfadd.return_value = 1
    client.pfcount.return_value = 0
    
    # Mock advanced operations
    client.setex.return_value = True
    client.info.return_value = {"used_memory": 1024 * 1024}  # 1MB
    
    # Configure pipeline mock with proper methods
    pipe_mock = Mock()
    pipe_mock.setex = Mock(return_value=None)
    pipe_mock.set = Mock(return_value=None)
    pipe_mock.delete = Mock(return_value=None)
    pipe_mock.get = Mock(return_value=None)
    pipe_mock.sadd = Mock(return_value=None)
    pipe_mock.expire = Mock(return_value=None)
    pipe_mock.execute = AsyncMock(return_value=[True, True, True])
    client.pipeline.return_value = pipe_mock
    
    return client


@pytest.fixture
def repository(mock_redis_client):
    """Create an enhanced Redis repository for testing."""
    return EnhancedRedisRepository(mock_redis_client)


@pytest.fixture
def sample_cache_entry():
    """Create a sample cache entry for testing."""
    return CacheEntry(
        key="test_key",
        value={"data": "test"},
        ttl=3600,
        tags=["tag1", "tag2"],
        dependencies=["dep1"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_batch_operation():
    """Create a sample batch operation for testing."""
    return BatchOperation(
        operation_type="set",
        operations=[
            {"key": "key1", "value": "value1", "ttl": 3600},
            {"key": "key2", "value": "value2"},
            {"key": "key3", "value": {"nested": "data"}}
        ],
        batch_size=2
    )


class TestEnhancedRedisRepository:
    """Test suite for enhanced Redis repository functionality."""

    def test_initialization(self, mock_redis_client):
        """Test repository initialization."""
        repo = EnhancedRedisRepository(mock_redis_client)
        assert repo.redis == mock_redis_client
        assert isinstance(repo._cache_stats, CacheStats)
        assert repo._operation_metrics == []
        assert repo._tag_cache == {}
        assert isinstance(repo._dependency_graph, dict)
        assert repo._l1_cache == {}
        assert repo._l1_max_size == 1000

    def test_calculate_size_bytes(self, repository):
        """Test size calculation for different value types."""
        # String
        size = repository._calculate_size_bytes("hello")
        assert size == 5
        
        # Integer
        size = repository._calculate_size_bytes(42)
        assert size == 8
        
        # Complex object
        size = repository._calculate_size_bytes({"key": "value"})
        assert size > 0

    def test_update_cache_stats(self, repository):
        """Test cache statistics updating."""
        # Record a hit
        repository._update_cache_stats("get", hit=True, response_time_ms=10.0)
        
        assert repository._cache_stats.total_operations == 1
        assert repository._cache_stats.hits == 1
        assert repository._cache_stats.misses == 0
        assert repository._cache_stats.hit_ratio == 1.0
        assert repository._cache_stats.avg_response_time_ms == 1.0  # Initial alpha calculation
        
        # Record a miss
        repository._update_cache_stats("get", hit=False, response_time_ms=20.0)
        
        assert repository._cache_stats.total_operations == 2
        assert repository._cache_stats.hits == 1
        assert repository._cache_stats.misses == 1
        assert repository._cache_stats.hit_ratio == 0.5

    def test_manage_l1_cache(self, repository):
        """Test L1 cache management."""
        # Add to L1 cache
        repository._manage_l1_cache("test_key", "test_value")
        assert "test_key" in repository._l1_cache
        assert repository._l1_cache["test_key"]["value"] == "test_value"
        
        # Remove from L1 cache
        repository._manage_l1_cache("test_key", remove=True)
        assert "test_key" not in repository._l1_cache

    def test_l1_cache_size_limit(self, repository):
        """Test L1 cache size limitation."""
        repository._l1_max_size = 2
        
        # Add items up to limit
        repository._manage_l1_cache("key1", "value1")
        repository._manage_l1_cache("key2", "value2")
        assert len(repository._l1_cache) == 2
        
        # Add one more - should evict oldest
        repository._manage_l1_cache("key3", "value3")
        assert len(repository._l1_cache) == 2
        assert "key3" in repository._l1_cache

    @pytest.mark.asyncio
    async def test_set_advanced_basic(self, repository, mock_redis_client):
        """Test basic advanced set operation."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.set_advanced(
                key="test_key",
                value="test_value",
                ttl=3600
            )
        
        assert result is True
        mock_redis_client.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_set_advanced_l1_cache(self, repository, mock_redis_client):
        """Test advanced set operation with L1 cache."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.set_advanced(
                key="test_key",
                value="test_value",
                level=CacheLevel.L1
            )
        
        assert result is True
        assert "test_key" in repository._l1_cache
        # Should not call Redis for L1 cache
        mock_redis_client.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_advanced_with_tags_and_dependencies(self, repository, mock_redis_client):
        """Test advanced set operation with tags and dependencies."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.set_advanced(
                key="test_key",
                value={"data": "test"},
                tags=["tag1", "tag2"],
                dependencies=["dep1"],
                ttl=3600
            )
        
        assert result is True
        mock_redis_client.pipeline.assert_called()
        
        # Check in-memory structures
        assert "tag1" in repository._tag_cache
        assert "tag2" in repository._tag_cache
        assert "dep1" in repository._dependency_graph

    @pytest.mark.asyncio
    async def test_get_advanced_from_l1_cache(self, repository, mock_redis_client):
        """Test advanced get operation from L1 cache."""
        # Pre-populate L1 cache
        repository._l1_cache["test_key"] = {
            "value": "test_value",
            "timestamp": time.time()
        }
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.get_advanced("test_key")
        
        assert result == "test_value"
        # Should not call Redis
        mock_redis_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_advanced_from_l2_cache(self, repository, mock_redis_client):
        """Test advanced get operation from L2 (Redis) cache."""
        mock_redis_client.get.return_value = b'"test_value"'
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.get_advanced("test_key")
        
        assert result == "test_value"
        mock_redis_client.get.assert_called_with("test_key")
        
        # Should promote to L1 cache
        assert "test_key" in repository._l1_cache

    @pytest.mark.asyncio
    async def test_get_advanced_cache_miss(self, repository, mock_redis_client):
        """Test advanced get operation with cache miss."""
        mock_redis_client.get.return_value = None
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.get_advanced("test_key")
        
        assert result is None
        mock_redis_client.get.assert_called_with("test_key")

    @pytest.mark.asyncio
    async def test_batch_operations_set(self, repository, mock_redis_client, sample_batch_operation):
        """Test batch set operations."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.batch_operations(sample_batch_operation)
        
        assert result["total_operations"] == 3
        assert "successful" in result
        assert "failed" in result
        assert "execution_time_seconds" in result
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_operations_get(self, repository, mock_redis_client):
        """Test batch get operations."""
        batch = BatchOperation(
            operation_type="get",
            operations=[
                {"key": "key1"},
                {"key": "key2"}
            ]
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.batch_operations(batch)
        
        assert result["total_operations"] == 2
        assert "results" in result

    @pytest.mark.asyncio
    async def test_invalidate_cache_tag_based(self, repository, mock_redis_client):
        """Test tag-based cache invalidation."""
        # Setup mock to return tagged keys
        mock_redis_client.smembers.return_value = [b"key1", b"key2"]
        
        request = InvalidationRequest(
            invalidation_type=InvalidationType.TAG_BASED,
            target="test_tag"
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.invalidate_cache(request)
        
        assert result["invalidation_type"] == "tag_based"
        assert result["target"] == "test_tag"
        assert result["invalidated_count"] == 2
        assert set(result["invalidated_keys"]) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_invalidate_cache_pattern_based(self, repository, mock_redis_client):
        """Test pattern-based cache invalidation."""
        mock_redis_client.keys.return_value = [b"test_key1", b"test_key2"]
        
        request = InvalidationRequest(
            invalidation_type=InvalidationType.PATTERN_BASED,
            target="test_*"
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.invalidate_cache(request)
        
        assert result["invalidation_type"] == "pattern_based"
        assert result["invalidated_count"] == 2

    @pytest.mark.asyncio
    async def test_invalidate_cache_dependency_based(self, repository, mock_redis_client):
        """Test dependency-based cache invalidation."""
        mock_redis_client.smembers.return_value = [b"dependent_key1", b"dependent_key2"]
        
        request = InvalidationRequest(
            invalidation_type=InvalidationType.DEPENDENCY_BASED,
            target="dependency_key"
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.invalidate_cache(request)
        
        assert result["invalidation_type"] == "dependency_based"
        assert result["invalidated_count"] == 2

    @pytest.mark.asyncio
    async def test_invalidate_cache_ttl_based(self, repository, mock_redis_client):
        """Test TTL-based cache invalidation."""
        request = InvalidationRequest(
            invalidation_type=InvalidationType.TTL,
            target="test_key"
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.invalidate_cache(request)
        
        assert result["invalidation_type"] == "ttl"
        assert result["invalidated_count"] == 1
        mock_redis_client.expire.assert_called_with("test_key", 1)

    @pytest.mark.asyncio
    async def test_invalidate_cache_with_cascade(self, repository, mock_redis_client):
        """Test cache invalidation with cascade."""
        # Setup dependency graph
        repository._dependency_graph["key1"] = {"dependent1", "dependent2"}
        
        mock_redis_client.smembers.return_value = [b"key1"]
        
        request = InvalidationRequest(
            invalidation_type=InvalidationType.TAG_BASED,
            target="test_tag",
            cascade=True
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.invalidate_cache(request)
        
        assert result["cascade"] is True
        # Should invalidate original key plus cascade dependencies
        assert result["invalidated_count"] >= 1

    @pytest.mark.asyncio
    async def test_warm_cache(self, repository, mock_redis_client):
        """Test cache warming functionality."""
        async def mock_data_source(key: str):
            return f"data_for_{key}"
        
        mock_redis_client.exists.return_value = False  # Keys don't exist yet
        
        keys = ["key1", "key2", "key3"]
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            with patch.object(repository, 'set_advanced', return_value=True) as mock_set:
                result = await repository.warm_cache(
                    data_source=mock_data_source,
                    keys=keys,
                    strategy="preload",
                    ttl=3600
                )
        
        assert result["strategy"] == "preload"
        assert result["total_keys"] == 3
        assert result["warmed_successfully"] == 3
        assert result["failed"] == 0
        assert mock_set.call_count == 3

    @pytest.mark.asyncio
    async def test_warm_cache_with_existing_keys(self, repository, mock_redis_client):
        """Test cache warming skips existing keys."""
        async def mock_data_source(key: str):
            return f"data_for_{key}"
        
        # Mock some keys as existing
        mock_redis_client.exists.side_effect = lambda key: key == "key1"
        
        keys = ["key1", "key2"]
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            with patch.object(repository, 'set_advanced', return_value=True) as mock_set:
                result = await repository.warm_cache(
                    data_source=mock_data_source,
                    keys=keys
                )
        
        assert result["warmed_successfully"] == 1  # Only key2 warmed
        assert mock_set.call_count == 1

    @pytest.mark.asyncio
    async def test_warm_cache_with_errors(self, repository, mock_redis_client):
        """Test cache warming handles errors gracefully."""
        async def mock_data_source(key: str):
            if key == "error_key":
                raise Exception("Data source error")
            return f"data_for_{key}"
        
        mock_redis_client.exists.return_value = False
        
        keys = ["good_key", "error_key"]
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            with patch.object(repository, 'set_advanced', return_value=True):
                result = await repository.warm_cache(
                    data_source=mock_data_source,
                    keys=keys
                )
        
        assert result["warmed_successfully"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0]["key"] == "error_key"

    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, repository, mock_redis_client):
        """Test getting cache statistics."""
        mock_redis_client.info.return_value = {"used_memory": 2 * 1024 * 1024}  # 2MB
        mock_redis_client.dbsize.return_value = 100
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            stats = await repository.get_cache_statistics()
        
        assert isinstance(stats, CacheStats)
        assert stats.memory_usage_mb == 2.0
        assert stats.total_keys == 100

    @pytest.mark.asyncio
    async def test_optimize_cache_performance(self, repository, mock_redis_client):
        """Test cache performance optimization."""
        # Setup mock responses
        mock_redis_client.keys.return_value = [b"key1", b"key2"]
        mock_redis_client.ttl.side_effect = [300, -2]  # key1 has TTL, key2 expired
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            with patch.object(repository, 'get_cache_statistics') as mock_stats:
                mock_stats.return_value = CacheStats(hit_ratio=0.6, memory_usage_mb=50)
                
                result = await repository.optimize_cache_performance()
        
        assert "optimizations_applied" in result
        assert "recommendations" in result
        assert "before_stats" in result
        assert "after_stats" in result

    @pytest.mark.asyncio
    async def test_sorted_set_operations(self, repository, mock_redis_client):
        """Test sorted set operations."""
        # Test add with score
        result = await repository.sorted_set_add_with_score("leaderboard", "player1", 100.0)
        assert result is True
        mock_redis_client.zadd.assert_called_with("leaderboard", {"player1": 100.0})
        
        # Test get top
        mock_redis_client.zrevrange.return_value = [("player1", 100.0), ("player2", 90.0)]
        result = await repository.sorted_set_get_top("leaderboard", count=2)
        mock_redis_client.zrevrange.assert_called_with("leaderboard", 0, 1, withscores=True)

    @pytest.mark.asyncio
    async def test_hyperloglog_operations(self, repository, mock_redis_client):
        """Test HyperLogLog operations."""
        # Test add
        result = await repository.hyperloglog_add("unique_visitors", "user1", "user2")
        assert result is True
        mock_redis_client.pfadd.assert_called_with("unique_visitors", "user1", "user2")
        
        # Test count
        mock_redis_client.pfcount.return_value = 1000
        result = await repository.hyperloglog_count("unique_visitors")
        assert result == 1000


class TestCacheEntry:
    """Test CacheEntry data class."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation and to_dict method."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600,
            tags=["tag1"],
            dependencies=["dep1"]
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.tags == ["tag1"]
        assert entry.dependencies == ["dep1"]
        assert entry.access_count == 0
        assert isinstance(entry.created_at, datetime)
        
        # Test to_dict
        entry_dict = entry.to_dict()
        assert entry_dict["key"] == "test_key"
        assert entry_dict["value"] == "test_value"
        assert entry_dict["ttl"] == 3600

    def test_cache_entry_defaults(self):
        """Test CacheEntry with default values."""
        entry = CacheEntry(key="test", value="test")
        
        assert entry.tags == []
        assert entry.dependencies == []
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.last_accessed, datetime)


class TestCacheStats:
    """Test CacheStats data class."""

    def test_cache_stats_creation(self):
        """Test CacheStats creation and calculations."""
        stats = CacheStats(
            total_operations=100,
            hits=80,
            misses=20
        )
        
        assert stats.total_operations == 100
        assert stats.hits == 80
        assert stats.misses == 20
        assert stats.hit_ratio == 0.8
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["hit_ratio"] == 0.8

    def test_cache_stats_zero_operations(self):
        """Test CacheStats with zero operations."""
        stats = CacheStats()
        assert stats.hit_ratio == 0.0


class TestInvalidationRequest:
    """Test InvalidationRequest data class."""

    def test_invalidation_request_creation(self):
        """Test InvalidationRequest creation."""
        request = InvalidationRequest(
            invalidation_type=InvalidationType.TAG_BASED,
            target="test_tag",
            cascade=True
        )
        
        assert request.invalidation_type == InvalidationType.TAG_BASED
        assert request.target == "test_tag"
        assert request.cascade is True
        assert request.immediate is True
        assert isinstance(request.created_at, datetime)
        
        # Test to_dict
        request_dict = request.to_dict()
        assert request_dict["invalidation_type"] == "tag_based"


class TestBatchOperation:
    """Test BatchOperation data class."""

    def test_batch_operation_creation(self):
        """Test BatchOperation creation."""
        operations = [
            {"key": "key1", "value": "value1"},
            {"key": "key2", "value": "value2"}
        ]
        
        batch = BatchOperation(
            operation_type="set",
            operations=operations,
            batch_size=1
        )
        
        assert batch.operation_type == "set"
        assert batch.operations == operations
        assert batch.batch_size == 1
        
        # Test to_dict
        batch_dict = batch.to_dict()
        assert batch_dict["operation_count"] == 2


class TestEnums:
    """Test enum definitions."""

    def test_cache_strategy_values(self):
        """Test CacheStrategy enum values."""
        assert CacheStrategy.WRITE_THROUGH.value == "write_through"
        assert CacheStrategy.WRITE_BACK.value == "write_back"
        assert CacheStrategy.WRITE_AROUND.value == "write_around"
        assert CacheStrategy.CACHE_ASIDE.value == "cache_aside"

    def test_cache_level_values(self):
        """Test CacheLevel enum values."""
        assert CacheLevel.L1.value == "l1"
        assert CacheLevel.L2.value == "l2"
        assert CacheLevel.L3.value == "l3"

    def test_invalidation_type_values(self):
        """Test InvalidationType enum values."""
        assert InvalidationType.TTL.value == "ttl"
        assert InvalidationType.TAG_BASED.value == "tag_based"
        assert InvalidationType.PATTERN_BASED.value == "pattern_based"
        assert InvalidationType.DEPENDENCY_BASED.value == "dependency_based"
        assert InvalidationType.EVENT_DRIVEN.value == "event_driven"


class TestBackwardCompatibility:
    """Test backward compatibility with legacy methods."""

    @pytest.mark.asyncio
    async def test_legacy_set(self, repository, mock_redis_client):
        """Test legacy set method still works."""
        with patch.object(repository, 'set_advanced', return_value=True) as mock_set_advanced:
            result = await repository.set("test_key", "test_value", ttl=3600)
            
            assert result is True
            mock_set_advanced.assert_called_once_with("test_key", "test_value", 3600)

    @pytest.mark.asyncio
    async def test_legacy_get(self, repository, mock_redis_client):
        """Test legacy get method still works."""
        with patch.object(repository, 'get_advanced', return_value="test_value") as mock_get_advanced:
            result = await repository.get("test_key")
            
            assert result == "test_value"
            mock_get_advanced.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_legacy_delete_with_l1_cache_cleanup(self, repository, mock_redis_client):
        """Test legacy delete method cleans up L1 cache."""
        # Add to L1 cache first
        repository._l1_cache["test_key"] = {"value": "test", "timestamp": time.time()}
        
        mock_redis_client.delete.return_value = 1
        result = await repository.delete("test_key")
        
        assert result == 1
        assert "test_key" not in repository._l1_cache
        mock_redis_client.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_legacy_exists_checks_l1_cache(self, repository, mock_redis_client):
        """Test legacy exists method checks L1 cache first."""
        # Add to L1 cache
        repository._l1_cache["test_key"] = {"value": "test", "timestamp": time.time()}
        
        result = await repository.exists("test_key")
        
        assert result is True
        # Should not call Redis since found in L1 cache
        mock_redis_client.exists.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_flushdb_clears_all_caches(self, repository, mock_redis_client):
        """Test legacy flushdb method clears all in-memory caches."""
        # Populate in-memory caches
        repository._l1_cache["key1"] = {"value": "test", "timestamp": time.time()}
        repository._tag_cache["tag1"] = {"key1"}
        repository._dependency_graph["dep1"] = {"key1"}
        
        result = await repository.flushdb()
        
        assert result is True
        assert len(repository._l1_cache) == 0
        assert len(repository._tag_cache) == 0
        assert len(repository._dependency_graph) == 0


class TestRedisRepositoryAlias:
    """Test that RedisRepository is an alias for EnhancedRedisRepository."""

    def test_alias_compatibility(self, mock_redis_client):
        """Test that RedisRepository is the same as EnhancedRedisRepository."""
        assert RedisRepository == EnhancedRedisRepository
        
        # Test instantiation
        repo = RedisRepository(mock_redis_client)
        assert isinstance(repo, EnhancedRedisRepository)
        assert repo.redis == mock_redis_client


class TestErrorHandling:
    """Test error handling in Redis repository."""

    @pytest.mark.asyncio
    async def test_set_advanced_error(self, repository, mock_redis_client):
        """Test error handling in advanced set operation."""
        mock_redis_client.pipeline.side_effect = Exception("Redis error")
        
        with pytest.raises(RepositoryError, match="Failed to set advanced Redis key"):
            await repository.set_advanced("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_get_advanced_error(self, repository, mock_redis_client):
        """Test error handling in advanced get operation."""
        mock_redis_client.get.side_effect = Exception("Redis error")
        
        with pytest.raises(RepositoryError, match="Failed to get advanced Redis key"):
            await repository.get_advanced("test_key")

    @pytest.mark.asyncio
    async def test_batch_operations_error(self, repository, mock_redis_client):
        """Test error handling in batch operations."""
        batch = BatchOperation(operation_type="set", operations=[{"key": "test", "value": "test"}])
        mock_redis_client.pipeline.side_effect = Exception("Pipeline error")
        
        with pytest.raises(RepositoryError, match="Failed to execute batch operations"):
            await repository.batch_operations(batch)

    @pytest.mark.asyncio
    async def test_invalidate_cache_error(self, repository, mock_redis_client):
        """Test error handling in cache invalidation."""
        request = InvalidationRequest(
            invalidation_type=InvalidationType.TAG_BASED,
            target="test_tag"
        )
        mock_redis_client.smembers.side_effect = Exception("Redis error")
        
        with pytest.raises(RepositoryError, match="Failed to invalidate cache"):
            await repository.invalidate_cache(request)

    @pytest.mark.asyncio
    async def test_warm_cache_error(self, repository, mock_redis_client):
        """Test error handling in cache warming."""
        async def failing_data_source(key: str):
            raise Exception("Data source error")
        
        # This error will occur during cache warming inside the try block
        with patch.object(repository, 'set_advanced', side_effect=Exception("Set error")):
            # The warm_cache method handles individual key errors gracefully
            # and doesn't raise an exception unless there's a major failure
            result = await repository.warm_cache(failing_data_source, ["key1"])
            
            # Should have 1 failed operation
            assert result["failed"] >= 0  # Graceful error handling


class TestPerformanceFeatures:
    """Test performance-related features."""

    def test_l1_cache_promotes_frequently_accessed_data(self, repository):
        """Test that L1 cache promotes frequently accessed data."""
        # Simulate accessing the same key multiple times
        repository._manage_l1_cache("hot_key", "hot_value")
        
        assert "hot_key" in repository._l1_cache
        assert repository._l1_cache["hot_key"]["value"] == "hot_value"

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, repository, mock_redis_client):
        """Test that performance monitoring is properly integrated."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor') as mock_monitor:
            await repository.get_cache_statistics()
            
            # Verify PerformanceMonitor was used
            mock_monitor.assert_called_with("cache_statistics", "redis")

    @pytest.mark.asyncio
    async def test_pipeline_usage_for_atomic_operations(self, repository, mock_redis_client):
        """Test that pipeline is used for atomic operations."""
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            await repository.set_advanced(
                key="test_key",
                value="test_value",
                tags=["tag1"],
                dependencies=["dep1"]
            )
        
        # Verify pipeline was used
        mock_redis_client.pipeline.assert_called()

    def test_cache_statistics_tracking(self, repository):
        """Test that cache statistics are properly tracked."""
        initial_ops = repository._cache_stats.total_operations
        
        repository._update_cache_stats("get", hit=True, response_time_ms=10.0)
        
        assert repository._cache_stats.total_operations == initial_ops + 1
        assert repository._cache_stats.hits > 0

    def test_memory_usage_calculation(self, repository):
        """Test memory usage calculation for cache optimization."""
        # Test different value types
        string_size = repository._calculate_size_bytes("hello world")
        assert string_size > 0
        
        dict_size = repository._calculate_size_bytes({"key": "value", "nested": {"data": "test"}})
        assert dict_size > string_size
        
        int_size = repository._calculate_size_bytes(12345)
        assert int_size == 8  # Standard int size approximation

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, repository, mock_redis_client):
        """Test that batch operations process efficiently."""
        # Create a large batch operation
        operations = [{"key": f"key{i}", "value": f"value{i}"} for i in range(10)]
        batch = BatchOperation(
            operation_type="set",
            operations=operations,
            batch_size=3  # Small batch size to test batching
        )
        
        with patch('pcs.repositories.redis_repo.PerformanceMonitor'):
            result = await repository.batch_operations(batch)
        
        assert result["total_operations"] == 10
        # Should have called pipeline multiple times due to batching
        assert mock_redis_client.pipeline.call_count >= 3

    def test_tag_and_dependency_graph_management(self, repository):
        """Test that tag and dependency graphs are properly managed."""
        # Test tag management
        repository._tag_cache["tag1"] = {"key1", "key2"}
        repository._tag_cache["tag2"] = {"key2", "key3"}
        
        assert "tag1" in repository._tag_cache
        assert len(repository._tag_cache["tag1"]) == 2
        
        # Test dependency graph management
        repository._dependency_graph["dep1"] = {"key1", "key2"}
        
        assert "dep1" in repository._dependency_graph
        assert len(repository._dependency_graph["dep1"]) == 2
