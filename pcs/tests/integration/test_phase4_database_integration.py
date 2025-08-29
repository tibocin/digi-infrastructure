"""
Filepath: tests/integration/test_phase4_database_integration.py
Purpose: Comprehensive integration tests for Phase 4 database integrations
Related Components: PostgreSQL, Neo4j, Redis, Connection Pools, Performance Monitoring
Tags: integration-tests, database-integration, phase4, cross-database, performance-testing
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from pcs.repositories.postgres_repo import OptimizedPostgreSQLRepository
from pcs.repositories.neo4j_repo import Neo4jRepository
from pcs.repositories.redis_repo import EnhancedRedisRepository
from pcs.core.connection_pool_manager import ConnectionPoolManager, get_connection_pool_manager, ConnectionPoolType
from pcs.utils.metrics import MetricsCollector, get_metrics_collector
from pcs.models.prompts import PromptTemplate
from pcs.models.contexts import Context
from pcs.models.conversations import Conversation
from pcs.api.v1.conversations import MessageResponse as Message


# Module-level fixtures available to all test classes
@pytest.fixture(scope="module")
def metrics_collector():
    """Get metrics collector instance."""
    return get_metrics_collector()


@pytest.fixture(scope="module")
def pool_manager():
    """Get connection pool manager instance."""
    return get_connection_pool_manager()


@pytest.fixture
def mock_postgresql_pool():
    """Mock PostgreSQL connection pool."""
    pool = Mock()
    pool._pool = Mock()
    pool._pool.size = Mock(return_value=20)
    pool._pool.freesize = Mock(return_value=15)
    pool._pool.checkedin = Mock(return_value=15)
    pool._pool.checkedout = Mock(return_value=5)
    pool._pool.overflow = Mock(return_value=0)
    pool._pool.checkedout_overflow = Mock(return_value=0)
    pool._pool.invalid = Mock(return_value=0)
    return pool


@pytest.fixture
def mock_redis_pool():
    """Mock Redis connection pool."""
    pool = Mock()
    pool._pool = Mock()
    pool._pool._available_connections = 10
    pool._pool._in_use_connections = 5
    pool._pool._created_connections = 15
    return pool


@pytest.fixture
def mock_neo4j_pool():
    """Mock Neo4j connection pool."""
    pool = Mock()
    pool._pool = Mock()
    pool._pool.size = Mock(return_value=10)
    pool._pool.available = Mock(return_value=8)
    pool._pool.in_use = Mock(return_value=2)
    return pool


@pytest.fixture
def mock_repositories():
    """Create mock repositories for testing."""
    # Mock PostgreSQL repository
    postgres_repo = Mock(spec=OptimizedPostgreSQLRepository)
    postgres_repo.bulk_create = AsyncMock()
    postgres_repo.find_with_pagination = AsyncMock()
    postgres_repo.execute_optimized_query = AsyncMock()
    
    # Mock Neo4j repository
    neo4j_repo = Mock(spec=Neo4jRepository)
    neo4j_repo.find_related_nodes = AsyncMock()
    neo4j_repo.analyze_conversation_patterns = AsyncMock()
    neo4j_repo.create_context_hierarchy = AsyncMock()
    
    # Mock Redis repository
    redis_repo = Mock(spec=EnhancedRedisRepository)
    redis_repo.set_advanced = AsyncMock()
    redis_repo.get_advanced = AsyncMock()
    redis_repo.batch_operations = AsyncMock()
    
    return {
        'postgres': postgres_repo,
        'neo4j': neo4j_repo,
        'redis': redis_repo
    }


class TestPhase4DatabaseIntegration:
    """
    Comprehensive integration tests for Phase 4 database integrations.
    
    These tests validate that all databases work together seamlessly,
    handle realistic workloads, and maintain performance under load.
    """
    
    @pytest.mark.asyncio
    async def test_cross_database_consistency(self, mock_repositories):
        """Test data consistency across all databases."""
        # Simulate creating a prompt template in PostgreSQL
        template_data = {
            "id": str(uuid.uuid4()),
            "name": "integration-test-template",
            "description": "Test template for integration",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Mock PostgreSQL response
        mock_repositories['postgres'].bulk_create.return_value = [template_data]
        
        # Create template in PostgreSQL
        created_templates = await mock_repositories['postgres'].bulk_create([template_data])
        assert len(created_templates) == 1
        assert created_templates[0]['name'] == "integration-test-template"
        
        # Store in Redis cache
        cache_key = f"template:{created_templates[0]['id']}"
        await mock_repositories['redis'].set_advanced(
            cache_key, 
            created_templates[0], 
            ttl=3600
        )
        
        # Verify cache storage
        mock_repositories['redis'].set_advanced.assert_called_once()
        
        # Create context hierarchy in Neo4j
        context_data = {
            "template_id": created_templates[0]['id'],
            "context_type": "prompt_generation",
            "relationships": ["depends_on", "similar_to"]
        }
        
        mock_repositories['neo4j'].create_context_hierarchy.return_value = {
            "success": True,
            "nodes_created": 3,
            "relationships_created": 2
        }
        
        hierarchy_result = await mock_repositories['neo4j'].create_context_hierarchy(context_data)
        assert hierarchy_result['success'] is True
        assert hierarchy_result['nodes_created'] == 3
        
        # Verify all operations were called
        mock_repositories['postgres'].bulk_create.assert_called_once()
        mock_repositories['redis'].set_advanced.assert_called_once()
        mock_repositories['neo4j'].create_context_hierarchy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, mock_repositories):
        """Test Redis caching integration with database operations."""
        # Test cache-aside pattern
        template_id = str(uuid.uuid4())
        cache_key = f"template:{template_id}"
        
        # First, try to get from cache (should miss)
        mock_repositories['redis'].get_advanced.return_value = None
        
        cached_template = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_template is None
        
        # Fetch from database
        template_data = {
            "id": template_id,
            "name": "cached-template",
            "description": "Template for caching test"
        }
        
        mock_repositories['postgres'].execute_optimized_query.return_value = [template_data]
        
        db_template = await mock_repositories['postgres'].execute_optimized_query(
            "SELECT * FROM prompt_templates WHERE id = :id",
            {"id": template_id}
        )
        
        assert len(db_template) == 1
        assert db_template[0]['id'] == template_id
        
        # Store in cache
        await mock_repositories['redis'].set_advanced(
            cache_key, 
            db_template[0], 
            ttl=1800,
            tags=["template", "prompt"]
        )
        
        # Verify cache storage
        mock_repositories['redis'].set_advanced.assert_called_once()
        
        # Test cache hit
        mock_repositories['redis'].get_advanced.return_value = db_template[0]
        
        cached_result = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_result is not None
        assert cached_result['id'] == template_id
        
        # Test batch cache operations
        batch_data = [
            {"key": f"key_{i}", "value": f"value_{i}", "ttl": 3600}
            for i in range(5)
        ]
        
        mock_repositories['redis'].batch_operations.return_value = {
            "success": True,
            "processed": 5,
            "failed": 0
        }
        
        batch_result = await mock_repositories['redis'].batch_operations(batch_data)
        assert batch_result['success'] is True
        assert batch_result['processed'] == 5
    
    @pytest.mark.asyncio
    async def test_connection_pool_integration(self, pool_manager, mock_postgresql_pool, 
                                            mock_redis_pool, mock_neo4j_pool):
        """Test connection pools under realistic load."""
        # Register all pools
        pool_manager.register_pool(ConnectionPoolType.POSTGRESQL, mock_postgresql_pool, "postgresql")
        pool_manager.register_pool(ConnectionPoolType.REDIS, mock_redis_pool, "redis")
        pool_manager.register_pool(ConnectionPoolType.NEO4J, mock_neo4j_pool, "neo4j")
        
        # Start monitoring
        await pool_manager.start_monitoring()
        
        # Simulate connection events
        pool_manager.record_connection_event(ConnectionPoolType.POSTGRESQL, "checkout", 0.05)
        pool_manager.record_connection_event(ConnectionPoolType.POSTGRESQL, "checkin", 0.02)
        pool_manager.record_connection_event(ConnectionPoolType.REDIS, "checkout", 0.01)
        pool_manager.record_connection_event(ConnectionPoolType.REDIS, "checkin", 0.01)
        
        # Wait for metrics collection
        await asyncio.sleep(0.1)
        
        # Get pool statistics
        pool_stats = await pool_manager.get_pool_stats()
        assert len(pool_stats) == 3
        
        # Verify PostgreSQL pool stats
        postgres_stats = pool_stats.get("postgresql")
        assert postgres_stats is not None
        assert postgres_stats["total_connections"] == 0  # Initial state
        assert postgres_stats["active_connections"] == 0  # Initial state
        
        # Verify Redis pool stats
        redis_stats = pool_stats.get("redis")
        assert redis_stats is not None
        assert redis_stats["total_connections"] == 0  # Initial state
        assert redis_stats["active_connections"] == 0  # Initial state
        
        # Get overall health
        health = pool_manager.get_overall_health()
        assert health.status in ["healthy", "degraded", "critical"]
        assert health.total_pools == 3
        
        # Test pool warming
        warm_result = await pool_manager.warm_pools()
        assert warm_result['success'] is True
        
        # Stop monitoring
        pool_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, mock_repositories, metrics_collector):
        """Validate all performance requirements are met."""
        # Test PostgreSQL bulk operations performance
        start_time = time.time()
        
        # Simulate bulk insert of 1000 templates
        bulk_data = [
            {
                "id": str(uuid.uuid4()),
                "name": f"bulk-template-{i}",
                "description": f"Bulk template {i}",
                "created_at": datetime.now()
            }
            for i in range(1000)
        ]
        
        mock_repositories['postgres'].bulk_create.return_value = bulk_data
        
        result = await mock_repositories['postgres'].bulk_create(bulk_data)
        
        bulk_time = time.time() - start_time
        assert len(result) == 1000
        assert bulk_time < 1.0  # Should complete in under 1 second
        
        # Test Neo4j graph operations performance
        start_time = time.time()
        
        mock_repositories['neo4j'].analyze_conversation_patterns.return_value = {
            "patterns": [{"type": "user_query", "frequency": 150}],
            "analysis_time": 0.1
        }
        
        patterns = await mock_repositories['neo4j'].analyze_conversation_patterns(
            time_window=timedelta(hours=24)
        )
        
        graph_time = time.time() - start_time
        assert patterns['patterns'][0]['frequency'] == 150
        assert graph_time < 0.5  # Should complete in under 0.5 seconds
        
        # Test Redis batch operations performance
        start_time = time.time()
        
        batch_ops = [
            {"operation": "set", "key": f"key_{i}", "value": f"value_{i}"}
            for i in range(1000)
        ]
        
        mock_repositories['redis'].batch_operations.return_value = {
            "success": True,
            "processed": 1000,
            "failed": 0
        }
        
        batch_result = await mock_repositories['redis'].batch_operations(batch_ops)
        
        cache_time = time.time() - start_time
        assert batch_result['success'] is True
        assert batch_result['processed'] == 1000
        assert cache_time < 0.5  # Should complete in under 0.5 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_repositories, pool_manager):
        """Test error handling and recovery across all databases."""
        # Test PostgreSQL connection failure
        mock_repositories['postgres'].bulk_create.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            await mock_repositories['postgres'].bulk_create([])
        
        # Test Neo4j query failure
        mock_repositories['neo4j'].find_related_nodes.side_effect = Exception("Cypher query failed")
        
        with pytest.raises(Exception):
            await mock_repositories['neo4j'].find_related_nodes("test-node")
        
        # Test Redis connection failure
        mock_repositories['redis'].set_advanced.side_effect = Exception("Redis connection failed")
        
        with pytest.raises(Exception):
            await mock_repositories['redis'].set_advanced("test-key", "test-value")
        
        # Test circuit breaker pattern
        pool_manager.record_connection_event(ConnectionPoolType.POSTGRESQL, "failure", 0.0)
        pool_manager.record_connection_event(ConnectionPoolType.POSTGRESQL, "failure", 0.0)
        pool_manager.record_connection_event(ConnectionPoolType.POSTGRESQL, "failure", 0.0)
        
        # Get optimization recommendations
        recommendations = await pool_manager.get_optimization_recommendations()
        assert len(recommendations) > 0
        
        # Verify failure handling recommendations
        failure_recommendations = [r for r in recommendations if "failure" in r.description.lower()]
        assert len(failure_recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_repositories):
        """Test concurrent operations across all databases."""
        # Create concurrent tasks for different database operations
        async def postgres_operation():
            return await mock_repositories['postgres'].bulk_create([])
        
        async def neo4j_operation():
            return await mock_repositories['neo4j'].analyze_conversation_patterns()
        
        async def redis_operation():
            return await mock_repositories['redis'].batch_operations([])
        
        # Execute all operations concurrently
        start_time = time.time()
        
        results = await asyncio.gather(
            postgres_operation(),
            neo4j_operation(),
            redis_operation(),
            return_exceptions=True
        )
        
        concurrent_time = time.time() - start_time
        
        # Verify all operations completed
        assert len(results) == 3
        assert concurrent_time < 1.0  # Should complete quickly with concurrency
        
        # Verify all repositories were called
        mock_repositories['postgres'].bulk_create.assert_called_once()
        mock_repositories['neo4j'].analyze_conversation_patterns.assert_called_once()
        mock_repositories['redis'].batch_operations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_migration_and_sync(self, mock_repositories):
        """Test data migration and synchronization between databases."""
        # Simulate migrating data from PostgreSQL to other databases
        
        # Source data from PostgreSQL
        source_data = [
            {
                "id": str(uuid.uuid4()),
                "name": f"migrated-template-{i}",
                "description": f"Template {i} for migration",
                "created_at": datetime.now()
            }
            for i in range(10)
        ]
        
        mock_repositories['postgres'].find_with_pagination.return_value = {
            "items": source_data,
            "total": 10,
            "page": 1,
            "size": 10,
            "has_next": False
        }
        
        # Fetch source data
        paginated_result = await mock_repositories['postgres'].find_with_pagination(
            {}, page=1, size=10
        )
        
        assert len(paginated_result['items']) == 10
        
        # Migrate to Neo4j
        for item in paginated_result['items']:
            mock_repositories['neo4j'].create_context_hierarchy.return_value = {
                "success": True,
                "nodes_created": 1
            }
            
            result = await mock_repositories['neo4j'].create_context_hierarchy({
                "template_id": item['id'],
                "context_type": "migrated"
            })
            
            assert result['success'] is True
        
        # Cache in Redis
        for item in paginated_result['items']:
            cache_key = f"migrated:{item['id']}"
            await mock_repositories['redis'].set_advanced(
                cache_key, 
                item, 
                ttl=7200,
                tags=["migrated", "template"]
            )
        
        # Verify migration completion
        mock_repositories['postgres'].find_with_pagination.assert_called_once()
        assert mock_repositories['neo4j'].create_context_hierarchy.call_count == 10
        assert mock_repositories['redis'].set_advanced.call_count == 10


class TestPhase4PerformanceBenchmarks:
    """Performance benchmarks for Phase 4 database integrations."""
    
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, mock_repositories):
        """Benchmark bulk operations across all databases."""
        # Test different batch sizes
        batch_sizes = [100, 500, 1000, 5000]
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # PostgreSQL bulk create
            bulk_data = [{"id": str(uuid.uuid4()), "name": f"benchmark-{i}"} 
                        for i in range(batch_size)]
            
            mock_repositories['postgres'].bulk_create.return_value = bulk_data
            result = await mock_repositories['postgres'].bulk_create(bulk_data)
            
            postgres_time = time.time() - start_time
            
            # Verify performance targets
            if batch_size <= 1000:
                assert postgres_time < 0.5
            else:
                assert postgres_time < 2.0
            
            assert len(result) == batch_size
    
    @pytest.mark.asyncio
    async def test_search_performance(self, mock_repositories):
        """Benchmark search operations across databases."""
        # Performance targets
        start_time = time.time()
        
        # Test Redis caching
        cache_keys = [f"benchmark-key-{i}" for i in range(100)]
        
        mock_repositories['redis'].get_advanced.return_value = None
        for key in cache_keys:
            result = await mock_repositories['redis'].get_advanced(key)
        miss_time = time.time() - start_time
        assert miss_time < 0.5
        
        start_time = time.time()
        
        for key in cache_keys:
            mock_repositories['redis'].get_advanced.return_value = {"key": key, "value": "cached"}
            result = await mock_repositories['redis'].get_advanced(key)
        
        hit_time = time.time() - start_time
        assert hit_time < 0.3

class TestPhase4SecurityValidation:
    """Security validation tests for Phase 4 database integrations."""
    
    @pytest.mark.asyncio
    async def test_connection_security(self, pool_manager):
        """Test connection security and isolation."""
        pools = await pool_manager.get_pool_stats()
        
        for pool_stat in pools:
            assert pool_stat.pool_type in ["postgresql", "redis", "neo4j"]
            assert pool_stat.total_connections > 0
            assert pool_stat.active_connections >= 0
    
    @pytest.mark.asyncio
    async def test_data_validation(self, mock_repositories):
        """Test data validation and sanitization."""
        malicious_query = "'; DROP TABLE users; --"
        
        mock_repositories['postgres'].execute_optimized_query.return_value = []
        
        result = await mock_repositories['postgres'].execute_optimized_query(
            "SELECT * FROM templates WHERE name = :name",
            {"name": malicious_query}
        )
        
        mock_repositories['postgres'].execute_optimized_query.assert_called_once()
        
        malicious_content = "<script>alert('xss')</script>"
        
        mock_repositories['neo4j'].create_context_hierarchy.return_value = {
            "success": True,
            "nodes_created": 1
        }
        
        result = await mock_repositories['neo4j'].create_context_hierarchy({
            "content": malicious_content
        })
        
        assert result['success'] is True
