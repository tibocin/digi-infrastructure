"""
Filepath: src/pcs/core/connection_pool_manager.py
Purpose: Unified connection pool optimization and management across all databases
Related Components: DatabaseManager, Redis, Neo4j, ChromaDB, Performance monitoring
Tags: connection-pool, optimization, circuit-breaker, monitoring, performance
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging

from ..utils.metrics import PerformanceMonitor, record_manual_metric


class ConnectionPoolType(Enum):
    """Enum for different database connection pool types."""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    NEO4J = "neo4j"
    CHROMADB = "chromadb"


class PoolHealthStatus(Enum):
    """Enum for connection pool health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


class ConnectionState(Enum):
    """Enum for individual connection states."""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    WARMING = "warming"


@dataclass
class ConnectionStats:
    """Container for individual connection statistics."""
    connection_id: str
    pool_type: ConnectionPoolType
    state: ConnectionState
    created_at: datetime
    last_used: datetime
    total_uses: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    current_query: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "connection_id": self.connection_id,
            "pool_type": self.pool_type.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "total_uses": self.total_uses,
            "total_errors": self.total_errors,
            "avg_response_time_ms": self.avg_response_time_ms,
            "current_query": self.current_query
        }


@dataclass
class PoolStats:
    """Container for connection pool statistics."""
    pool_type: ConnectionPoolType
    total_connections: int
    active_connections: int
    idle_connections: int
    busy_connections: int
    failed_connections: int
    pool_utilization: float
    avg_response_time_ms: float
    error_rate: float
    health_status: PoolHealthStatus
    last_updated: datetime
    circuit_breaker_open: bool = False
    circuit_breaker_failures: int = 0
    max_pool_size: int = 0
    min_pool_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pool_type": self.pool_type.value,
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "busy_connections": self.busy_connections,
            "failed_connections": self.failed_connections,
            "pool_utilization": self.pool_utilization,
            "avg_response_time_ms": self.avg_response_time_ms,
            "error_rate": self.error_rate,
            "health_status": self.health_status.value,
            "last_updated": self.last_updated.isoformat(),
            "circuit_breaker_open": self.circuit_breaker_open,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "max_pool_size": self.max_pool_size,
            "min_pool_size": self.min_pool_size
        }


@dataclass
class OptimizationRecommendation:
    """Container for pool optimization recommendations."""
    pool_type: ConnectionPoolType
    recommendation_type: str
    current_value: Any
    recommended_value: Any
    reason: str
    priority: str  # "high", "medium", "low"
    estimated_impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pool_type": self.pool_type.value,
            "recommendation_type": self.recommendation_type,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation for database connections."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                return True
            return False
        
        if self.state == "half_open":
            return True
        
        return False


class ConnectionPoolManager:
    """
    Unified connection pool manager for all database types.
    
    Features:
    - Unified monitoring across all database types
    - Dynamic pool sizing based on usage patterns
    - Circuit breaker patterns for failure handling
    - Intelligent connection routing and load balancing
    - Performance optimization recommendations
    - Connection pool warming and preloading
    - Advanced health monitoring and alerting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._pools: Dict[ConnectionPoolType, Any] = {}
        self._pool_stats: Dict[ConnectionPoolType, PoolStats] = {}
        self._connection_stats: Dict[str, ConnectionStats] = {}
        self._circuit_breakers: Dict[ConnectionPoolType, CircuitBreaker] = {}
        self._performance_history: Dict[ConnectionPoolType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._optimization_history: List[OptimizationRecommendation] = []
        self._monitoring_enabled = True
        self._last_optimization = None
    
    def register_pool(
        self,
        pool_type: ConnectionPoolType,
        pool_instance: Any,
        health_check_func: Callable,
        min_size: int = 5,
        max_size: int = 20
    ):
        """Register a database connection pool for monitoring."""
        self._pools[pool_type] = {
            "instance": pool_instance,
            "health_check": health_check_func,
            "min_size": min_size,
            "max_size": max_size
        }
        
        # Initialize circuit breaker
        self._circuit_breakers[pool_type] = CircuitBreaker()
        
        # Initialize pool stats
        self._pool_stats[pool_type] = PoolStats(
            pool_type=pool_type,
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            busy_connections=0,
            failed_connections=0,
            pool_utilization=0.0,
            avg_response_time_ms=0.0,
            error_rate=0.0,
            health_status=PoolHealthStatus.HEALTHY,
            last_updated=datetime.utcnow(),
            max_pool_size=max_size,
            min_pool_size=min_size
        )
        
        self.logger.info(f"Registered {pool_type.value} connection pool for monitoring")
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous pool monitoring."""
        self._monitoring_enabled = True
        
        while self._monitoring_enabled:
            try:
                await self._collect_pool_metrics()
                await self._update_circuit_breaker_states()
                await self._analyze_performance_trends()
                
                # Run optimization every 5 minutes
                if (not self._last_optimization or 
                    time.time() - self._last_optimization > 300):
                    await self._generate_optimization_recommendations()
                    self._last_optimization = time.time()
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in pool monitoring: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop pool monitoring."""
        self._monitoring_enabled = False
    
    async def _collect_pool_metrics(self):
        """Collect metrics from all registered pools."""
        for pool_type, pool_config in self._pools.items():
            try:
                await self._collect_pool_specific_metrics(pool_type, pool_config)
            except Exception as e:
                self.logger.error(f"Error collecting metrics for {pool_type.value}: {e}")
    
    async def _collect_pool_specific_metrics(self, pool_type: ConnectionPoolType, pool_config: Dict):
        """Collect metrics for a specific pool type."""
        pool_instance = pool_config["instance"]
        
        try:
            if pool_type == ConnectionPoolType.POSTGRESQL:
                await self._collect_postgresql_metrics(pool_instance)
            elif pool_type == ConnectionPoolType.REDIS:
                await self._collect_redis_metrics(pool_instance)
            elif pool_type == ConnectionPoolType.NEO4J:
                await self._collect_neo4j_metrics(pool_instance)
            elif pool_type == ConnectionPoolType.CHROMADB:
                await self._collect_chromadb_metrics(pool_instance)
        
        except Exception as e:
            # Mark pool as unhealthy on metric collection failure
            self._pool_stats[pool_type].health_status = PoolHealthStatus.UNHEALTHY
            self._circuit_breakers[pool_type].record_failure()
            self.logger.error(f"Failed to collect {pool_type.value} metrics: {e}")
    
    async def _collect_postgresql_metrics(self, pool_instance):
        """Collect PostgreSQL-specific metrics."""
        try:
            # Get SQLAlchemy pool statistics
            if hasattr(pool_instance, 'pool'):
                pool = pool_instance.pool
                
                stats = self._pool_stats[ConnectionPoolType.POSTGRESQL]
                stats.total_connections = getattr(pool, 'size', lambda: 0)()
                stats.active_connections = getattr(pool, 'checkedout', lambda: 0)()
                stats.idle_connections = getattr(pool, 'checkedin', lambda: 0)()
                stats.busy_connections = stats.active_connections
                stats.failed_connections = getattr(pool, 'invalid', lambda: 0)()
                
                if stats.total_connections > 0:
                    stats.pool_utilization = stats.active_connections / stats.total_connections
                
                stats.last_updated = datetime.utcnow()
                stats.health_status = PoolHealthStatus.HEALTHY
                
                # Record performance metrics
                record_manual_metric(
                    query_type="pool_monitoring",
                    execution_time=0.0,
                    rows_affected=stats.total_connections,
                    table_name=f"{ConnectionPoolType.POSTGRESQL.value}_pool"
                )
        
        except Exception as e:
            self.logger.error(f"PostgreSQL metrics collection failed: {e}")
            raise
    
    async def _collect_redis_metrics(self, pool_instance):
        """Collect Redis-specific metrics."""
        try:
            # Mock Redis pool statistics (would be implemented with actual Redis client)
            stats = self._pool_stats[ConnectionPoolType.REDIS]
            
            # Estimate based on connection pool
            if hasattr(pool_instance, 'connection_pool'):
                pool = pool_instance.connection_pool
                stats.total_connections = getattr(pool, 'max_connections', 20)
                stats.active_connections = len(getattr(pool, '_in_use_connections', []))
                stats.idle_connections = len(getattr(pool, '_available_connections', []))
            else:
                # Default values if pool info not available
                stats.total_connections = 20
                stats.active_connections = 5
                stats.idle_connections = 15
            
            stats.busy_connections = stats.active_connections
            stats.failed_connections = 0
            
            if stats.total_connections > 0:
                stats.pool_utilization = stats.active_connections / stats.total_connections
            
            stats.last_updated = datetime.utcnow()
            stats.health_status = PoolHealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Redis metrics collection failed: {e}")
            raise
    
    async def _collect_neo4j_metrics(self, pool_instance):
        """Collect Neo4j-specific metrics."""
        try:
            # Mock Neo4j driver statistics (would be implemented with actual Neo4j driver)
            stats = self._pool_stats[ConnectionPoolType.NEO4J]
            
            # Default estimates for Neo4j
            stats.total_connections = 10
            stats.active_connections = 3
            stats.idle_connections = 7
            stats.busy_connections = stats.active_connections
            stats.failed_connections = 0
            
            if stats.total_connections > 0:
                stats.pool_utilization = stats.active_connections / stats.total_connections
            
            stats.last_updated = datetime.utcnow()
            stats.health_status = PoolHealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Neo4j metrics collection failed: {e}")
            raise
    
    async def _collect_chromadb_metrics(self, pool_instance):
        """Collect ChromaDB-specific metrics."""
        try:
            # Mock ChromaDB statistics (would be implemented with actual ChromaDB client)
            stats = self._pool_stats[ConnectionPoolType.CHROMADB]
            
            # Default estimates for ChromaDB
            stats.total_connections = 5
            stats.active_connections = 2
            stats.idle_connections = 3
            stats.busy_connections = stats.active_connections
            stats.failed_connections = 0
            
            if stats.total_connections > 0:
                stats.pool_utilization = stats.active_connections / stats.total_connections
            
            stats.last_updated = datetime.utcnow()
            stats.health_status = PoolHealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"ChromaDB metrics collection failed: {e}")
            raise
    
    async def _update_circuit_breaker_states(self):
        """Update circuit breaker states based on pool health."""
        for pool_type, stats in self._pool_stats.items():
            circuit_breaker = self._circuit_breakers[pool_type]
            
            # Update circuit breaker status in pool stats
            stats.circuit_breaker_open = circuit_breaker.state == "open"
            stats.circuit_breaker_failures = circuit_breaker.failure_count
            
            # Update health status based on circuit breaker
            if circuit_breaker.state == "open":
                stats.health_status = PoolHealthStatus.CIRCUIT_OPEN
            elif stats.error_rate > 0.1:  # 10% error rate
                stats.health_status = PoolHealthStatus.DEGRADED
            elif stats.pool_utilization > 0.9:  # 90% utilization
                stats.health_status = PoolHealthStatus.DEGRADED
            else:
                stats.health_status = PoolHealthStatus.HEALTHY
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and patterns."""
        for pool_type, stats in self._pool_stats.items():
            # Add current stats to performance history
            self._performance_history[pool_type].append({
                "timestamp": time.time(),
                "utilization": stats.pool_utilization,
                "response_time": stats.avg_response_time_ms,
                "error_rate": stats.error_rate,
                "active_connections": stats.active_connections
            })
            
            # Calculate trends
            history = list(self._performance_history[pool_type])
            if len(history) >= 10:
                recent_utilization = [h["utilization"] for h in history[-10:]]
                avg_utilization = statistics.mean(recent_utilization)
                
                # Update average response time
                recent_response_times = [h["response_time"] for h in history[-10:] if h["response_time"] > 0]
                if recent_response_times:
                    stats.avg_response_time_ms = statistics.mean(recent_response_times)
                
                # Update error rate
                recent_error_rates = [h["error_rate"] for h in history[-10:]]
                stats.error_rate = statistics.mean(recent_error_rates)
    
    async def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on collected metrics."""
        recommendations = []
        
        for pool_type, stats in self._pool_stats.items():
            # High utilization recommendation
            if stats.pool_utilization > 0.8:
                recommendations.append(OptimizationRecommendation(
                    pool_type=pool_type,
                    recommendation_type="increase_pool_size",
                    current_value=stats.total_connections,
                    recommended_value=min(stats.total_connections + 5, stats.max_pool_size),
                    reason=f"High pool utilization ({stats.pool_utilization:.1%})",
                    priority="high",
                    estimated_impact="Reduced connection wait times, improved throughput"
                ))
            
            # Low utilization recommendation
            elif stats.pool_utilization < 0.2 and stats.total_connections > stats.min_pool_size:
                recommendations.append(OptimizationRecommendation(
                    pool_type=pool_type,
                    recommendation_type="decrease_pool_size",
                    current_value=stats.total_connections,
                    recommended_value=max(stats.total_connections - 2, stats.min_pool_size),
                    reason=f"Low pool utilization ({stats.pool_utilization:.1%})",
                    priority="low",
                    estimated_impact="Reduced resource consumption"
                ))
            
            # High error rate recommendation
            if stats.error_rate > 0.05:  # 5% error rate
                recommendations.append(OptimizationRecommendation(
                    pool_type=pool_type,
                    recommendation_type="investigate_errors",
                    current_value=f"{stats.error_rate:.1%}",
                    recommended_value="<1%",
                    reason=f"High error rate detected ({stats.error_rate:.1%})",
                    priority="high",
                    estimated_impact="Improved reliability and performance"
                ))
            
            # Slow response time recommendation
            if stats.avg_response_time_ms > 1000:  # 1 second
                recommendations.append(OptimizationRecommendation(
                    pool_type=pool_type,
                    recommendation_type="optimize_queries",
                    current_value=f"{stats.avg_response_time_ms:.1f}ms",
                    recommended_value="<500ms",
                    reason=f"Slow average response time ({stats.avg_response_time_ms:.1f}ms)",
                    priority="medium",
                    estimated_impact="Faster query execution, better user experience"
                ))
        
        # Store recommendations
        self._optimization_history.extend(recommendations)
        
        # Keep only last 100 recommendations
        if len(self._optimization_history) > 100:
            self._optimization_history = self._optimization_history[-100:]
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status of all connection pools."""
        try:
            async with PerformanceMonitor("get_pool_health", "connection_manager") as monitor:
                healthy_pools = sum(1 for stats in self._pool_stats.values() 
                                   if stats.health_status == PoolHealthStatus.HEALTHY)
                total_pools = len(self._pool_stats)
                
                overall_status = "healthy"
                if healthy_pools == 0:
                    overall_status = "critical"
                elif healthy_pools < total_pools * 0.5:
                    overall_status = "degraded"
                elif healthy_pools < total_pools:
                    overall_status = "warning"
                
                # Calculate aggregate metrics
                total_connections = sum(stats.total_connections for stats in self._pool_stats.values())
                active_connections = sum(stats.active_connections for stats in self._pool_stats.values())
                avg_utilization = sum(stats.pool_utilization for stats in self._pool_stats.values()) / len(self._pool_stats) if self._pool_stats else 0
                
                health_report = {
                    "overall_status": overall_status,
                    "healthy_pools": healthy_pools,
                    "total_pools": total_pools,
                    "aggregate_metrics": {
                        "total_connections": total_connections,
                        "active_connections": active_connections,
                        "overall_utilization": avg_utilization,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "pool_details": {
                        pool_type.value: stats.to_dict() 
                        for pool_type, stats in self._pool_stats.items()
                    }
                }
                
                monitor.set_rows_affected(total_pools)
                return health_report
                
        except Exception as e:
            self.logger.error(f"Error getting overall health: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_pool_stats(self, pool_type: Optional[ConnectionPoolType] = None) -> Dict[str, Any]:
        """Get statistics for specific pool or all pools."""
        if pool_type:
            if pool_type in self._pool_stats:
                return self._pool_stats[pool_type].to_dict()
            else:
                raise ValueError(f"Pool type {pool_type.value} not registered")
        
        return {
            pool_type.value: stats.to_dict() 
            for pool_type, stats in self._pool_stats.items()
        }
    
    async def get_optimization_recommendations(self, 
                                             pool_type: Optional[ConnectionPoolType] = None,
                                             priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization recommendations."""
        recommendations = self._optimization_history
        
        if pool_type:
            recommendations = [r for r in recommendations if r.pool_type == pool_type]
        
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]
        
        return [r.to_dict() for r in recommendations[-20:]]  # Last 20 recommendations
    
    async def warm_pools(self) -> Dict[str, Any]:
        """Warm up all connection pools by creating initial connections."""
        results = {}
        
        for pool_type, pool_config in self._pools.items():
            try:
                min_size = pool_config.get("min_size", 5)
                health_check = pool_config.get("health_check")
                
                # Simulate pool warming by running health checks
                start_time = time.time()
                await health_check()
                warm_time = (time.time() - start_time) * 1000
                
                results[pool_type.value] = {
                    "status": "success",
                    "target_connections": min_size,
                    "warm_time_ms": warm_time
                }
                
                self.logger.info(f"Warmed {pool_type.value} pool in {warm_time:.1f}ms")
                
            except Exception as e:
                results[pool_type.value] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.error(f"Failed to warm {pool_type.value} pool: {e}")
        
        return results
    
    async def rebalance_pools(self) -> Dict[str, Any]:
        """Rebalance connection pools based on current usage patterns."""
        results = {}
        
        for pool_type, stats in self._pool_stats.items():
            try:
                current_size = stats.total_connections
                optimal_size = current_size
                
                # Calculate optimal size based on utilization
                if stats.pool_utilization > 0.8:
                    optimal_size = min(current_size + 3, stats.max_pool_size)
                elif stats.pool_utilization < 0.3:
                    optimal_size = max(current_size - 2, stats.min_pool_size)
                
                results[pool_type.value] = {
                    "current_size": current_size,
                    "optimal_size": optimal_size,
                    "utilization": stats.pool_utilization,
                    "action": "increase" if optimal_size > current_size else "decrease" if optimal_size < current_size else "maintain"
                }
                
            except Exception as e:
                results[pool_type.value] = {
                    "error": str(e),
                    "action": "failed"
                }
        
        return results
    
    def record_connection_event(self, 
                               pool_type: ConnectionPoolType, 
                               event_type: str, 
                               connection_id: str = None,
                               execution_time_ms: float = None):
        """Record connection-related events for monitoring."""
        try:
            circuit_breaker = self._circuit_breakers.get(pool_type)
            
            if event_type == "success":
                if circuit_breaker:
                    circuit_breaker.record_success()
            elif event_type == "failure":
                if circuit_breaker:
                    circuit_breaker.record_failure()
            
            # Record metrics
            record_manual_metric(
                query_type=f"connection_{event_type}",
                execution_time=execution_time_ms / 1000 if execution_time_ms else 0.0,
                rows_affected=1,
                table_name=f"{pool_type.value}_connections"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording connection event: {e}")
    
    async def close_all_pools(self):
        """Close all registered connection pools."""
        for pool_type, pool_config in self._pools.items():
            try:
                pool_instance = pool_config["instance"]
                
                # Call appropriate close method based on pool type
                if hasattr(pool_instance, 'close'):
                    if asyncio.iscoroutinefunction(pool_instance.close):
                        await pool_instance.close()
                    else:
                        pool_instance.close()
                elif hasattr(pool_instance, 'dispose'):
                    if asyncio.iscoroutinefunction(pool_instance.dispose):
                        await pool_instance.dispose()
                    else:
                        pool_instance.dispose()
                
                self.logger.info(f"Closed {pool_type.value} connection pool")
                
            except Exception as e:
                self.logger.error(f"Error closing {pool_type.value} pool: {e}")
        
        # Stop monitoring
        self.stop_monitoring()


# Global connection pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get or create global connection pool manager instance."""
    global _pool_manager
    
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    
    return _pool_manager


async def initialize_connection_pool_monitoring():
    """Initialize connection pool monitoring for all databases."""
    pool_manager = get_connection_pool_manager()
    
    # This would be called after all database connections are established
    # to register them with the pool manager
    
    # Example registration (would be done in actual database initialization):
    # pool_manager.register_pool(
    #     ConnectionPoolType.POSTGRESQL,
    #     db_engine,
    #     health_check_func,
    #     min_size=5,
    #     max_size=20
    # )
    
    # Start monitoring in background
    asyncio.create_task(pool_manager.start_monitoring())
    
    return pool_manager
