"""
Filepath: tests/unit/test_connection_pool_manager.py
Purpose: Unit tests for unified connection pool manager implementation
Related Components: ConnectionPoolManager, CircuitBreaker, PoolStats, OptimizationRecommendation
Tags: testing, connection-pool, optimization, circuit-breaker, monitoring
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from pcs.core.connection_pool_manager import (
    ConnectionPoolManager,
    ConnectionPoolType,
    PoolHealthStatus,
    ConnectionState,
    ConnectionStats,
    PoolStats,
    OptimizationRecommendation,
    CircuitBreaker,
    get_connection_pool_manager,
    initialize_connection_pool_monitoring
)


@pytest.fixture
def pool_manager():
    """Create a connection pool manager for testing."""
    return ConnectionPoolManager()


@pytest.fixture
def mock_postgresql_pool():
    """Create a mock PostgreSQL pool."""
    pool = Mock()
    pool.size.return_value = 10
    pool.checkedout.return_value = 3
    pool.checkedin.return_value = 7
    pool.invalid.return_value = 0
    return pool


@pytest.fixture
def mock_redis_pool():
    """Create a mock Redis pool."""
    pool = Mock()
    pool.connection_pool = Mock()
    pool.connection_pool.max_connections = 20
    pool.connection_pool._in_use_connections = [1, 2, 3, 4, 5]
    pool.connection_pool._available_connections = [6, 7, 8, 9, 10]
    return pool


@pytest.fixture
def mock_health_check():
    """Create a mock health check function."""
    async def health_check():
        await asyncio.sleep(0.01)  # Simulate some work
        return {"status": "healthy"}
    return health_check


class TestConnectionStats:
    """Test ConnectionStats data class."""

    def test_connection_stats_creation(self):
        """Test ConnectionStats creation and to_dict method."""
        stats = ConnectionStats(
            connection_id="conn_123",
            pool_type=ConnectionPoolType.POSTGRESQL,
            state=ConnectionState.ACTIVE,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow(),
            total_uses=10,
            total_errors=1,
            avg_response_time_ms=150.0,
            current_query="SELECT * FROM users"
        )
        
        assert stats.connection_id == "conn_123"
        assert stats.pool_type == ConnectionPoolType.POSTGRESQL
        assert stats.state == ConnectionState.ACTIVE
        assert stats.total_uses == 10
        assert stats.total_errors == 1
        assert stats.avg_response_time_ms == 150.0
        assert stats.current_query == "SELECT * FROM users"
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["connection_id"] == "conn_123"
        assert stats_dict["pool_type"] == "postgresql"
        assert stats_dict["state"] == "active"
        assert stats_dict["total_uses"] == 10


class TestPoolStats:
    """Test PoolStats data class."""

    def test_pool_stats_creation(self):
        """Test PoolStats creation and calculations."""
        stats = PoolStats(
            pool_type=ConnectionPoolType.REDIS,
            total_connections=20,
            active_connections=5,
            idle_connections=15,
            busy_connections=5,
            failed_connections=0,
            pool_utilization=0.25,
            avg_response_time_ms=100.0,
            error_rate=0.02,
            health_status=PoolHealthStatus.HEALTHY,
            last_updated=datetime.utcnow(),
            max_pool_size=30,
            min_pool_size=5
        )
        
        assert stats.pool_type == ConnectionPoolType.REDIS
        assert stats.total_connections == 20
        assert stats.active_connections == 5
        assert stats.pool_utilization == 0.25
        assert stats.health_status == PoolHealthStatus.HEALTHY
        
        # Test to_dict
        stats_dict = stats.to_dict()
        assert stats_dict["pool_type"] == "redis"
        assert stats_dict["total_connections"] == 20
        assert stats_dict["pool_utilization"] == 0.25


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation data class."""

    def test_optimization_recommendation_creation(self):
        """Test OptimizationRecommendation creation."""
        recommendation = OptimizationRecommendation(
            pool_type=ConnectionPoolType.NEO4J,
            recommendation_type="increase_pool_size",
            current_value=10,
            recommended_value=15,
            reason="High utilization detected",
            priority="high",
            estimated_impact="Improved performance"
        )
        
        assert recommendation.pool_type == ConnectionPoolType.NEO4J
        assert recommendation.recommendation_type == "increase_pool_size"
        assert recommendation.current_value == 10
        assert recommendation.recommended_value == 15
        assert recommendation.priority == "high"
        
        # Test to_dict
        rec_dict = recommendation.to_dict()
        assert rec_dict["pool_type"] == "neo4j"
        assert rec_dict["recommendation_type"] == "increase_pool_size"
        assert rec_dict["priority"] == "high"


class TestCircuitBreaker:
    """Test CircuitBreaker implementation."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.failure_count == 0
        assert cb.state == "closed"
        assert cb.can_execute() is True

    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=30)
        
        # Record first failure
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.state == "closed"
        assert cb.can_execute() is True
        
        # Record second failure - should open circuit
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        # Trigger circuit open
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(1.1)
        assert cb.can_execute() is True  # Should be half_open now
        assert cb.state == "half_open"
        
        # Record success - should close circuit
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_circuit_breaker_success_reset(self):
        """Test that success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "closed"


class TestConnectionPoolManager:
    """Test ConnectionPoolManager functionality."""

    def test_initialization(self, pool_manager):
        """Test pool manager initialization."""
        assert pool_manager._pools == {}
        assert pool_manager._pool_stats == {}
        assert pool_manager._connection_stats == {}
        assert pool_manager._circuit_breakers == {}
        assert pool_manager._monitoring_enabled is True

    @pytest.mark.asyncio
    async def test_register_pool(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test pool registration."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check,
            min_size=5,
            max_size=20
        )
        
        assert ConnectionPoolType.POSTGRESQL in pool_manager._pools
        assert ConnectionPoolType.POSTGRESQL in pool_manager._circuit_breakers
        assert ConnectionPoolType.POSTGRESQL in pool_manager._pool_stats
        
        pool_config = pool_manager._pools[ConnectionPoolType.POSTGRESQL]
        assert pool_config["instance"] == mock_postgresql_pool
        assert pool_config["health_check"] == mock_health_check
        assert pool_config["min_size"] == 5
        assert pool_config["max_size"] == 20

    @pytest.mark.asyncio
    async def test_collect_postgresql_metrics(self, pool_manager, mock_health_check):
        """Test PostgreSQL metrics collection."""
        # Create mock engine with pool
        mock_engine = Mock()
        mock_engine.pool = Mock()
        mock_engine.pool.size.return_value = 10
        mock_engine.pool.checkedout.return_value = 3
        mock_engine.pool.checkedin.return_value = 7
        mock_engine.pool.invalid.return_value = 0
        
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_engine,
            mock_health_check
        )
        
        await pool_manager._collect_postgresql_metrics(mock_engine)
        
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        assert stats.total_connections == 10
        assert stats.active_connections == 3
        assert stats.idle_connections == 7
        assert stats.pool_utilization == 0.3
        assert stats.health_status == PoolHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_collect_redis_metrics(self, pool_manager, mock_redis_pool, mock_health_check):
        """Test Redis metrics collection."""
        pool_manager.register_pool(
            ConnectionPoolType.REDIS,
            mock_redis_pool,
            mock_health_check
        )
        
        await pool_manager._collect_redis_metrics(mock_redis_pool)
        
        stats = pool_manager._pool_stats[ConnectionPoolType.REDIS]
        assert stats.total_connections == 20
        assert stats.active_connections == 5
        assert stats.idle_connections == 5
        assert stats.health_status == PoolHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_collect_neo4j_metrics(self, pool_manager, mock_health_check):
        """Test Neo4j metrics collection."""
        mock_neo4j = Mock()
        
        pool_manager.register_pool(
            ConnectionPoolType.NEO4J,
            mock_neo4j,
            mock_health_check
        )
        
        await pool_manager._collect_neo4j_metrics(mock_neo4j)
        
        stats = pool_manager._pool_stats[ConnectionPoolType.NEO4J]
        assert stats.total_connections == 10
        assert stats.active_connections == 3
        assert stats.health_status == PoolHealthStatus.HEALTHY


    @pytest.mark.asyncio
    async def test_update_circuit_breaker_states(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test circuit breaker state updates."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Simulate high error rate
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        stats.error_rate = 0.15  # 15% error rate
        
        await pool_manager._update_circuit_breaker_states()
        
        assert stats.health_status == PoolHealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_generate_optimization_recommendations(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test optimization recommendation generation."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Set high utilization to trigger recommendation
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        stats.pool_utilization = 0.85
        stats.total_connections = 10
        stats.max_pool_size = 20
        
        await pool_manager._generate_optimization_recommendations()
        
        recommendations = pool_manager._optimization_history
        assert len(recommendations) > 0
        
        high_util_rec = next((r for r in recommendations if r.recommendation_type == "increase_pool_size"), None)
        assert high_util_rec is not None
        assert high_util_rec.priority == "high"
        assert high_util_rec.recommended_value == 15  # 10 + 5

    @pytest.mark.asyncio
    async def test_get_overall_health(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test overall health status retrieval."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        with patch('pcs.core.connection_pool_manager.PerformanceMonitor'):
            health = await pool_manager.get_overall_health()
        
        assert "overall_status" in health
        assert "healthy_pools" in health
        assert "total_pools" in health
        assert "aggregate_metrics" in health
        assert "pool_details" in health
        
        assert health["total_pools"] == 1
        assert health["healthy_pools"] == 1
        assert health["overall_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_pool_stats(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test pool statistics retrieval."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Test getting specific pool stats
        pg_stats = await pool_manager.get_pool_stats(ConnectionPoolType.POSTGRESQL)
        assert pg_stats["pool_type"] == "postgresql"
        
        # Test getting all pool stats
        all_stats = await pool_manager.get_pool_stats()
        assert "postgresql" in all_stats

    @pytest.mark.asyncio
    async def test_get_optimization_recommendations(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test optimization recommendation retrieval."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Add a test recommendation
        rec = OptimizationRecommendation(
            pool_type=ConnectionPoolType.POSTGRESQL,
            recommendation_type="test",
            current_value=10,
            recommended_value=15,
            reason="test",
            priority="high",
            estimated_impact="test"
        )
        pool_manager._optimization_history.append(rec)
        
        # Test getting all recommendations
        all_recs = await pool_manager.get_optimization_recommendations()
        assert len(all_recs) == 1
        
        # Test filtering by pool type
        pg_recs = await pool_manager.get_optimization_recommendations(
            pool_type=ConnectionPoolType.POSTGRESQL
        )
        assert len(pg_recs) == 1
        
        # Test filtering by priority
        high_recs = await pool_manager.get_optimization_recommendations(priority="high")
        assert len(high_recs) == 1

    @pytest.mark.asyncio
    async def test_warm_pools(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test pool warming functionality."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check,
            min_size=3
        )
        
        results = await pool_manager.warm_pools()
        
        assert "postgresql" in results
        assert results["postgresql"]["status"] == "success"
        assert results["postgresql"]["target_connections"] == 3
        assert "warm_time_ms" in results["postgresql"]

    @pytest.mark.asyncio
    async def test_rebalance_pools(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test pool rebalancing functionality."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check,
            min_size=5,
            max_size=20
        )
        
        # Set high utilization
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        stats.total_connections = 10
        stats.pool_utilization = 0.85
        
        results = await pool_manager.rebalance_pools()
        
        assert "postgresql" in results
        assert results["postgresql"]["current_size"] == 10
        assert results["postgresql"]["optimal_size"] == 13  # 10 + 3
        assert results["postgresql"]["action"] == "increase"

    def test_record_connection_event(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test connection event recording."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Test success event
        pool_manager.record_connection_event(
            ConnectionPoolType.POSTGRESQL,
            "success",
            connection_id="conn_123",
            execution_time_ms=150.0
        )
        
        cb = pool_manager._circuit_breakers[ConnectionPoolType.POSTGRESQL]
        assert cb.failure_count == 0
        
        # Test failure event
        pool_manager.record_connection_event(
            ConnectionPoolType.POSTGRESQL,
            "failure",
            connection_id="conn_123"
        )
        
        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_close_all_pools(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test closing all pools."""
        # Mock close method - use close instead of dispose for the test
        mock_postgresql_pool.close = Mock()
        
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        await pool_manager.close_all_pools()
        
        mock_postgresql_pool.close.assert_called_once()
        assert pool_manager._monitoring_enabled is False

    @pytest.mark.asyncio
    async def test_error_handling_in_metrics_collection(self, pool_manager, mock_health_check):
        """Test error handling during metrics collection."""
        # Create a mock that raises an exception
        mock_faulty_pool = Mock()
        mock_faulty_pool.pool = Mock()
        mock_faulty_pool.pool.size.side_effect = Exception("Connection error")
        
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_faulty_pool,
            mock_health_check
        )
        
        await pool_manager._collect_pool_metrics()
        
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        assert stats.health_status == PoolHealthStatus.UNHEALTHY
        
        cb = pool_manager._circuit_breakers[ConnectionPoolType.POSTGRESQL]
        assert cb.failure_count > 0


class TestEnums:
    """Test enum definitions."""

    def test_connection_pool_type_values(self):
        """Test ConnectionPoolType enum values."""
        assert ConnectionPoolType.POSTGRESQL.value == "postgresql"
        assert ConnectionPoolType.REDIS.value == "redis"
        assert ConnectionPoolType.NEO4J.value == "neo4j"

    def test_pool_health_status_values(self):
        """Test PoolHealthStatus enum values."""
        assert PoolHealthStatus.HEALTHY.value == "healthy"
        assert PoolHealthStatus.DEGRADED.value == "degraded"
        assert PoolHealthStatus.UNHEALTHY.value == "unhealthy"
        assert PoolHealthStatus.CIRCUIT_OPEN.value == "circuit_open"

    def test_connection_state_values(self):
        """Test ConnectionState enum values."""
        assert ConnectionState.ACTIVE.value == "active"
        assert ConnectionState.IDLE.value == "idle"
        assert ConnectionState.BUSY.value == "busy"
        assert ConnectionState.FAILED.value == "failed"
        assert ConnectionState.WARMING.value == "warming"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_connection_pool_manager(self):
        """Test global pool manager instance."""
        manager1 = get_connection_pool_manager()
        manager2 = get_connection_pool_manager()
        
        # Should return the same instance
        assert manager1 is manager2
        assert isinstance(manager1, ConnectionPoolManager)

    @pytest.mark.asyncio
    async def test_initialize_connection_pool_monitoring(self):
        """Test connection pool monitoring initialization."""
        with patch('pcs.core.connection_pool_manager.asyncio.create_task') as mock_create_task:
            manager = await initialize_connection_pool_monitoring()
            
            assert isinstance(manager, ConnectionPoolManager)
            mock_create_task.assert_called_once()


class TestPerformanceFeatures:
    """Test performance-related features."""

    @pytest.mark.asyncio
    async def test_performance_history_tracking(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test that performance history is properly tracked."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        # Simulate multiple performance data points
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        stats.pool_utilization = 0.5
        stats.avg_response_time_ms = 100.0
        stats.error_rate = 0.01
        
        await pool_manager._analyze_performance_trends()
        
        history = pool_manager._performance_history[ConnectionPoolType.POSTGRESQL]
        assert len(history) == 1
        assert history[0]["utilization"] == 0.5
        assert history[0]["response_time"] == 100.0

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, pool_manager):
        """Test that monitoring loop handles errors gracefully."""
        # Mock the collect_pool_metrics to raise an exception
        with patch.object(pool_manager, '_collect_pool_metrics', side_effect=Exception("Test error")):
            # Start monitoring for a short time
            pool_manager._monitoring_enabled = True
            
            # Start monitoring task and let it run briefly
            monitoring_task = asyncio.create_task(pool_manager.start_monitoring(interval_seconds=0.1))
            
            # Let it run for a moment
            await asyncio.sleep(0.2)
            
            # Stop monitoring
            pool_manager.stop_monitoring()
            
            # Wait for task to complete
            try:
                await asyncio.wait_for(monitoring_task, timeout=1.0)
            except asyncio.TimeoutError:
                monitoring_task.cancel()
            
            # Test passed if no unhandled exception occurred

    def test_circuit_breaker_integration(self, pool_manager, mock_postgresql_pool, mock_health_check):
        """Test circuit breaker integration with pool manager."""
        pool_manager.register_pool(
            ConnectionPoolType.POSTGRESQL,
            mock_postgresql_pool,
            mock_health_check
        )
        
        cb = pool_manager._circuit_breakers[ConnectionPoolType.POSTGRESQL]
        
        # Trigger circuit breaker
        for _ in range(5):  # Default threshold is 5
            cb.record_failure()
        
        assert cb.state == "open"
        
        # Update circuit breaker states
        asyncio.run(pool_manager._update_circuit_breaker_states())
        
        stats = pool_manager._pool_stats[ConnectionPoolType.POSTGRESQL]
        assert stats.circuit_breaker_open is True
        assert stats.health_status == PoolHealthStatus.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_optimization_recommendation_filtering(self, pool_manager):
        """Test optimization recommendation filtering and storage."""
        # Add multiple recommendations
        recommendations = [
            OptimizationRecommendation(
                pool_type=ConnectionPoolType.POSTGRESQL,
                recommendation_type="increase_pool_size",
                current_value=10,
                recommended_value=15,
                reason="High utilization",
                priority="high",
                estimated_impact="Better performance"
            ),
            OptimizationRecommendation(
                pool_type=ConnectionPoolType.REDIS,
                recommendation_type="decrease_pool_size",
                current_value=20,
                recommended_value=15,
                reason="Low utilization",
                priority="low",
                estimated_impact="Resource savings"
            )
        ]
        
        pool_manager._optimization_history.extend(recommendations)
        
        # Test filtering by pool type
        pg_recs = await pool_manager.get_optimization_recommendations(
            pool_type=ConnectionPoolType.POSTGRESQL
        )
        assert len(pg_recs) == 1
        assert pg_recs[0]["pool_type"] == "postgresql"
        
        # Test filtering by priority
        high_recs = await pool_manager.get_optimization_recommendations(priority="high")
        assert len(high_recs) == 1
        assert high_recs[0]["priority"] == "high"

    def test_recommendation_history_size_limit(self, pool_manager):
        """Test that recommendation history respects size limits."""
        # Add more than 100 recommendations
        for i in range(105):
            rec = OptimizationRecommendation(
                pool_type=ConnectionPoolType.POSTGRESQL,
                recommendation_type="test",
                current_value=i,
                recommended_value=i+1,
                reason=f"Test {i}",
                priority="low",
                estimated_impact="Test"
            )
            pool_manager._optimization_history.append(rec)
        
        # Trigger cleanup
        asyncio.run(pool_manager._generate_optimization_recommendations())
        
        # Should keep only last 100
        assert len(pool_manager._optimization_history) == 100
        # First recommendation should be index 5 (100 kept from 105 total)
        assert pool_manager._optimization_history[0].current_value == 5
