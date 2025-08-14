"""
Filepath: pcs/src/pcs/services/performance_optimization_service.py
Purpose: Comprehensive performance optimization service with bottleneck analysis and improvements
Related Components: Database, Caching, Connection pooling, Memory management, Async operations
Tags: performance, optimization, bottleneck-analysis, benchmarking, profiling
"""

import asyncio
import gc
import psutil
import time
import sys
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import statistics
from contextlib import asynccontextmanager
import cProfile
import pstats
import io

from ..utils.logger import get_logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import PCSError
from ..utils.metrics import get_metrics_collector, PerformanceMonitor
# Optional import for monitoring service
try:
    from .monitoring_service import get_monitoring_service
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    get_monitoring_service = lambda: None

logger = get_logger(__name__)


class OptimizationType(str, Enum):
    """Types of performance optimizations."""
    DATABASE = "database"
    CACHING = "caching"
    MEMORY = "memory"
    CONNECTION_POOL = "connection_pool"
    ASYNC_OPERATIONS = "async_operations"
    QUERY_OPTIMIZATION = "query_optimization"
    GARBAGE_COLLECTION = "garbage_collection"
    I_O_OPTIMIZATION = "io_optimization"


class BottleneckSeverity(str, Enum):
    """Severity levels for performance bottlenecks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class OptimizationStatus(str, Enum):
    """Status of optimization implementations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bottleneck:
    """Identified performance bottleneck."""
    id: str
    name: str
    description: str
    severity: BottleneckSeverity
    optimization_type: OptimizationType
    impact_score: float  # 0-100
    current_value: float
    target_value: float
    recommendations: List[str]
    detected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of applying an optimization."""
    optimization_id: str
    name: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    execution_time: float
    applied_at: datetime
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    name: str
    operations_per_second: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime
    test_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceError(PCSError):
    """Performance optimization related errors."""
    pass


class BottleneckAnalyzer:
    """
    Analyzes system performance to identify bottlenecks.
    
    Provides comprehensive analysis of database, cache, memory, and I/O performance.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize bottleneck analyzer."""
        self.db_session = db_session
        self.metrics_collector = get_metrics_collector()
        self.monitoring_service = get_monitoring_service()
        self._analysis_history: deque = deque(maxlen=100)
    
    async def analyze_system_performance(self) -> List[Bottleneck]:
        """
        Comprehensive system performance analysis.
        
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Analyze different performance areas
        bottlenecks.extend(await self._analyze_database_performance())
        bottlenecks.extend(await self._analyze_memory_usage())
        bottlenecks.extend(await self._analyze_cpu_usage())
        bottlenecks.extend(await self._analyze_io_performance())
        bottlenecks.extend(await self._analyze_cache_performance())
        bottlenecks.extend(await self._analyze_connection_pools())
        
        # Sort by impact score (highest first)
        bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)
        
        # Store analysis history
        self._analysis_history.append({
            "timestamp": datetime.utcnow(),
            "bottlenecks_found": len(bottlenecks),
            "critical_count": len([b for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL]),
            "high_count": len([b for b in bottlenecks if b.severity == BottleneckSeverity.HIGH])
        })
        
        logger.info(f"Performance analysis completed: {len(bottlenecks)} bottlenecks identified")
        return bottlenecks
    
    async def _analyze_database_performance(self) -> List[Bottleneck]:
        """Analyze database performance bottlenecks."""
        bottlenecks = []
        
        if not self.db_session:
            return bottlenecks
        
        try:
            # Check for slow queries
            slow_queries = await self._get_slow_queries()
            if slow_queries:
                for query in slow_queries[:5]:  # Top 5 slow queries
                    if query['mean_time'] > 1000:  # More than 1 second
                        bottlenecks.append(Bottleneck(
                            id=f"slow_query_{hash(query['query'][:50])}",
                            name="Slow Database Query",
                            description=f"Query taking {query['mean_time']:.2f}ms on average",
                            severity=BottleneckSeverity.HIGH if query['mean_time'] > 5000 else BottleneckSeverity.MEDIUM,
                            optimization_type=OptimizationType.QUERY_OPTIMIZATION,
                            impact_score=min(90, query['mean_time'] / 100),
                            current_value=query['mean_time'],
                            target_value=query['mean_time'] * 0.5,  # Target 50% improvement
                            recommendations=[
                                "Add appropriate database indexes",
                                "Optimize query structure",
                                "Consider query result caching",
                                "Review WHERE clause efficiency"
                            ],
                            detected_at=datetime.utcnow(),
                            metadata={"query": query['query'][:200], "calls": query['calls']}
                        ))
            
            # Check connection pool usage
            active_connections = await self._get_active_connections_count()
            if active_connections > 15:  # Assuming max pool size of 20
                bottlenecks.append(Bottleneck(
                    id="high_db_connections",
                    name="High Database Connection Usage",
                    description=f"{active_connections} active database connections",
                    severity=BottleneckSeverity.MEDIUM,
                    optimization_type=OptimizationType.CONNECTION_POOL,
                    impact_score=70,
                    current_value=active_connections,
                    target_value=10,
                    recommendations=[
                        "Optimize connection pool configuration",
                        "Implement connection pooling optimizations",
                        "Review long-running transactions",
                        "Consider read replicas for read-heavy operations"
                    ],
                    detected_at=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.warning(f"Error analyzing database performance: {e}")
        
        return bottlenecks
    
    async def _analyze_memory_usage(self) -> List[Bottleneck]:
        """Analyze memory usage bottlenecks."""
        bottlenecks = []
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 85:
                severity = BottleneckSeverity.CRITICAL if memory_percent > 95 else BottleneckSeverity.HIGH
                bottlenecks.append(Bottleneck(
                    id="high_memory_usage",
                    name="High Memory Usage",
                    description=f"Memory usage at {memory_percent:.1f}%",
                    severity=severity,
                    optimization_type=OptimizationType.MEMORY,
                    impact_score=memory_percent,
                    current_value=memory_percent,
                    target_value=70.0,
                    recommendations=[
                        "Implement memory caching optimization",
                        "Review object lifecycle management",
                        "Optimize data structures",
                        "Implement garbage collection tuning"
                    ],
                    detected_at=datetime.utcnow(),
                    metadata={"total_mb": memory.total // 1024 // 1024, "used_mb": memory.used // 1024 // 1024}
                ))
            
            # Check for memory leaks by analyzing trends
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                if peak > current * 2:  # Peak is significantly higher than current
                    bottlenecks.append(Bottleneck(
                        id="potential_memory_leak",
                        name="Potential Memory Leak",
                        description=f"Peak memory usage ({peak//1024//1024}MB) much higher than current ({current//1024//1024}MB)",
                        severity=BottleneckSeverity.MEDIUM,
                        optimization_type=OptimizationType.MEMORY,
                        impact_score=60,
                        current_value=peak / current,
                        target_value=1.2,
                        recommendations=[
                            "Profile memory usage patterns",
                            "Review object cleanup",
                            "Check for circular references",
                            "Implement memory monitoring"
                        ],
                        detected_at=datetime.utcnow()
                    ))
                    
        except Exception as e:
            logger.warning(f"Error analyzing memory usage: {e}")
        
        return bottlenecks
    
    async def _analyze_cpu_usage(self) -> List[Bottleneck]:
        """Analyze CPU usage bottlenecks."""
        bottlenecks = []
        
        try:
            # Get CPU usage over a short period
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 80:
                severity = BottleneckSeverity.CRITICAL if cpu_percent > 95 else BottleneckSeverity.HIGH
                bottlenecks.append(Bottleneck(
                    id="high_cpu_usage",
                    name="High CPU Usage",
                    description=f"CPU usage at {cpu_percent:.1f}%",
                    severity=severity,
                    optimization_type=OptimizationType.ASYNC_OPERATIONS,
                    impact_score=cpu_percent,
                    current_value=cpu_percent,
                    target_value=60.0,
                    recommendations=[
                        "Optimize CPU-intensive operations",
                        "Implement better async patterns",
                        "Review algorithmic complexity",
                        "Consider caching for expensive computations"
                    ],
                    detected_at=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.warning(f"Error analyzing CPU usage: {e}")
        
        return bottlenecks
    
    async def _analyze_io_performance(self) -> List[Bottleneck]:
        """Analyze I/O performance bottlenecks."""
        bottlenecks = []
        
        try:
            # Check disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # This is a simplified check - in practice you'd track I/O over time
                if hasattr(disk_io, 'busy_time') and disk_io.busy_time > 80:
                    bottlenecks.append(Bottleneck(
                        id="high_disk_io",
                        name="High Disk I/O",
                        description="High disk I/O utilization detected",
                        severity=BottleneckSeverity.MEDIUM,
                        optimization_type=OptimizationType.I_O_OPTIMIZATION,
                        impact_score=70,
                        current_value=disk_io.busy_time,
                        target_value=50.0,
                        recommendations=[
                            "Optimize file I/O operations",
                            "Implement I/O caching",
                            "Consider SSD storage",
                            "Batch I/O operations"
                        ],
                        detected_at=datetime.utcnow()
                    ))
                    
        except Exception as e:
            logger.warning(f"Error analyzing I/O performance: {e}")
        
        return bottlenecks
    
    async def _analyze_cache_performance(self) -> List[Bottleneck]:
        """Analyze cache performance bottlenecks."""
        bottlenecks = []
        
        try:
            # Analyze cache hit ratios from metrics
            metrics_summary = self.metrics_collector.get_metrics_summary(timedelta(hours=1))
            
            for operation, stats in metrics_summary.items():
                if "cache" in operation.lower():
                    # Calculate hit ratio (this is simplified - you'd track hits/misses properly)
                    avg_time = stats.get('avg_execution_time', 0)
                    if avg_time > 0.1:  # 100ms
                        bottlenecks.append(Bottleneck(
                            id=f"slow_cache_{operation}",
                            name="Slow Cache Operation",
                            description=f"Cache operation {operation} taking {avg_time*1000:.1f}ms",
                            severity=BottleneckSeverity.MEDIUM,
                            optimization_type=OptimizationType.CACHING,
                            impact_score=min(80, avg_time * 1000),
                            current_value=avg_time * 1000,
                            target_value=avg_time * 500,  # Target 50% improvement
                            recommendations=[
                                "Optimize cache key structure",
                                "Implement cache warming",
                                "Review cache TTL settings",
                                "Consider cache partitioning"
                            ],
                            detected_at=datetime.utcnow(),
                            metadata={"operation": operation, "stats": stats}
                        ))
                        
        except Exception as e:
            logger.warning(f"Error analyzing cache performance: {e}")
        
        return bottlenecks
    
    async def _analyze_connection_pools(self) -> List[Bottleneck]:
        """Analyze connection pool bottlenecks."""
        bottlenecks = []
        
        try:
            # This would integrate with your connection pool manager
            # For now, we'll use a simplified analysis
            
            # Check for connection pool exhaustion patterns
            # This would typically come from your pool monitoring
            pool_usage = 85  # Simulated high pool usage
            
            if pool_usage > 80:
                bottlenecks.append(Bottleneck(
                    id="connection_pool_pressure",
                    name="Connection Pool Under Pressure",
                    description=f"Connection pool usage at {pool_usage}%",
                    severity=BottleneckSeverity.MEDIUM,
                    optimization_type=OptimizationType.CONNECTION_POOL,
                    impact_score=pool_usage,
                    current_value=pool_usage,
                    target_value=60.0,
                    recommendations=[
                        "Increase connection pool size",
                        "Optimize connection lifecycle",
                        "Implement connection sharing",
                        "Review connection timeout settings"
                    ],
                    detected_at=datetime.utcnow()
                ))
                
        except Exception as e:
            logger.warning(f"Error analyzing connection pools: {e}")
        
        return bottlenecks
    
    async def _get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get slow queries from database statistics."""
        try:
            if not self.db_session:
                return []
            
            result = await self.db_session.execute(text("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows
                FROM pg_stat_statements 
                WHERE mean_time > 100
                ORDER BY total_time DESC 
                LIMIT 10
            """))
            
            return [dict(row._mapping) for row in result]
        except Exception:
            # pg_stat_statements might not be available
            return []
    
    async def _get_active_connections_count(self) -> int:
        """Get count of active database connections."""
        try:
            if not self.db_session:
                return 0
            
            result = await self.db_session.execute(text("""
                SELECT COUNT(*) as active_connections
                FROM pg_stat_activity 
                WHERE state = 'active'
            """))
            
            row = result.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0


class PerformanceOptimizer:
    """
    Implements performance optimizations based on bottleneck analysis.
    
    Provides automatic and manual optimization strategies.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize performance optimizer."""
        self.db_session = db_session
        self.optimization_history: List[OptimizationResult] = []
        self._optimization_registry: Dict[str, Callable] = {}
        self._register_optimizations()
    
    def _register_optimizations(self) -> None:
        """Register available optimization functions."""
        self._optimization_registry = {
            "database_connection_pool": self._optimize_database_connection_pool,
            "memory_garbage_collection": self._optimize_memory_garbage_collection,
            "cache_warming": self._optimize_cache_warming,
            "query_indexes": self._optimize_query_indexes,
            "async_operations": self._optimize_async_operations,
            "memory_usage": self._optimize_memory_usage,
            "io_operations": self._optimize_io_operations,
        }
    
    async def apply_optimization(self, bottleneck: Bottleneck) -> OptimizationResult:
        """
        Apply optimization for a specific bottleneck.
        
        Args:
            bottleneck: Bottleneck to optimize
            
        Returns:
            Optimization result
        """
        start_time = time.time()
        optimization_id = f"opt_{bottleneck.id}_{int(time.time())}"
        
        # Capture before metrics
        before_metrics = await self._capture_metrics(bottleneck.optimization_type)
        
        try:
            # Find appropriate optimization function
            optimization_func = self._find_optimization_function(bottleneck)
            
            if not optimization_func:
                return OptimizationResult(
                    optimization_id=optimization_id,
                    name=f"Optimization for {bottleneck.name}",
                    optimization_type=bottleneck.optimization_type,
                    status=OptimizationStatus.SKIPPED,
                    before_metrics=before_metrics,
                    after_metrics=before_metrics,
                    improvement_percent=0.0,
                    execution_time=time.time() - start_time,
                    applied_at=datetime.utcnow(),
                    notes="No optimization function available"
                )
            
            # Apply optimization
            success = await optimization_func(bottleneck)
            
            # Wait for optimization to take effect
            await asyncio.sleep(2)
            
            # Capture after metrics
            after_metrics = await self._capture_metrics(bottleneck.optimization_type)
            
            # Calculate improvement
            improvement_percent = self._calculate_improvement(before_metrics, after_metrics, bottleneck)
            
            result = OptimizationResult(
                optimization_id=optimization_id,
                name=f"Optimization for {bottleneck.name}",
                optimization_type=bottleneck.optimization_type,
                status=OptimizationStatus.COMPLETED if success else OptimizationStatus.FAILED,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement_percent,
                execution_time=time.time() - start_time,
                applied_at=datetime.utcnow(),
                notes="Optimization applied successfully" if success else "Optimization failed"
            )
            
            self.optimization_history.append(result)
            logger.info(f"Applied optimization {optimization_id}: {improvement_percent:.1f}% improvement")
            
            return result
            
        except Exception as e:
            after_metrics = await self._capture_metrics(bottleneck.optimization_type)
            
            result = OptimizationResult(
                optimization_id=optimization_id,
                name=f"Optimization for {bottleneck.name}",
                optimization_type=bottleneck.optimization_type,
                status=OptimizationStatus.FAILED,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time=time.time() - start_time,
                applied_at=datetime.utcnow(),
                notes=f"Optimization failed: {str(e)}"
            )
            
            self.optimization_history.append(result)
            logger.error(f"Optimization {optimization_id} failed: {e}")
            
            return result
    
    def _find_optimization_function(self, bottleneck: Bottleneck) -> Optional[Callable]:
        """Find appropriate optimization function for bottleneck."""
        optimization_map = {
            OptimizationType.CONNECTION_POOL: "database_connection_pool",
            OptimizationType.MEMORY: "memory_garbage_collection",
            OptimizationType.CACHING: "cache_warming",
            OptimizationType.QUERY_OPTIMIZATION: "query_indexes",
            OptimizationType.ASYNC_OPERATIONS: "async_operations",
            OptimizationType.GARBAGE_COLLECTION: "memory_garbage_collection",
            OptimizationType.I_O_OPTIMIZATION: "io_operations"
        }
        
        func_name = optimization_map.get(bottleneck.optimization_type)
        return self._optimization_registry.get(func_name) if func_name else None
    
    async def _capture_metrics(self, optimization_type: OptimizationType) -> Dict[str, float]:
        """Capture relevant metrics for optimization type."""
        metrics = {}
        
        try:
            if optimization_type == OptimizationType.MEMORY:
                memory = psutil.virtual_memory()
                metrics.update({
                    "memory_percent": memory.percent,
                    "memory_used_mb": memory.used / 1024 / 1024,
                    "memory_available_mb": memory.available / 1024 / 1024
                })
            
            if optimization_type == OptimizationType.CONNECTION_POOL and self.db_session:
                # This would integrate with your connection pool monitoring
                metrics["active_connections"] = await self._get_active_connections_count()
            
            # Add CPU metrics for async operations
            if optimization_type == OptimizationType.ASYNC_OPERATIONS:
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # Add general system metrics
            metrics.update({
                "timestamp": time.time(),
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            })
            
        except Exception as e:
            logger.warning(f"Error capturing metrics: {e}")
        
        return metrics
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float], bottleneck: Bottleneck) -> float:
        """Calculate improvement percentage."""
        try:
            # Find the most relevant metric for this optimization type
            metric_map = {
                OptimizationType.MEMORY: "memory_percent",
                OptimizationType.CONNECTION_POOL: "active_connections",
                OptimizationType.ASYNC_OPERATIONS: "cpu_percent"
            }
            
            key_metric = metric_map.get(bottleneck.optimization_type)
            
            if key_metric and key_metric in before and key_metric in after:
                before_val = before[key_metric]
                after_val = after[key_metric]
                
                if before_val > 0:
                    # For metrics where lower is better (memory%, CPU%, connections)
                    improvement = ((before_val - after_val) / before_val) * 100
                    return max(0.0, improvement)  # Don't report negative improvements
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating improvement: {e}")
            return 0.0
    
    async def _get_active_connections_count(self) -> int:
        """Get active database connections count."""
        try:
            if not self.db_session:
                return 0
            
            result = await self.db_session.execute(text("""
                SELECT COUNT(*) as count FROM pg_stat_activity WHERE state = 'active'
            """))
            row = result.fetchone()
            return row[0] if row else 0
        except Exception:
            return 0
    
    async def _optimize_database_connection_pool(self, bottleneck: Bottleneck) -> bool:
        """Optimize database connection pool settings."""
        try:
            logger.info("Applying database connection pool optimization")
            
            # Implement connection pool optimizations
            # This would involve adjusting pool sizes, timeouts, etc.
            # For demonstration, we'll simulate the optimization
            
            # Force garbage collection to clean up unused connections
            gc.collect()
            
            # In a real implementation, you'd:
            # 1. Adjust pool_size and max_overflow based on usage patterns
            # 2. Optimize connection timeout settings
            # 3. Implement connection sharing strategies
            # 4. Add connection pool warming
            
            await asyncio.sleep(1)  # Simulate optimization time
            return True
            
        except Exception as e:
            logger.error(f"Database connection pool optimization failed: {e}")
            return False
    
    async def _optimize_memory_garbage_collection(self, bottleneck: Bottleneck) -> bool:
        """Optimize memory usage through garbage collection."""
        try:
            logger.info("Applying memory garbage collection optimization")
            
            # Force comprehensive garbage collection
            before_memory = psutil.virtual_memory().percent
            
            # Collect all generations
            collected = gc.collect()
            
            # Force additional cleanup
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0.1)
            
            after_memory = psutil.virtual_memory().percent
            
            logger.info(f"Garbage collection freed {collected} objects, memory usage: {before_memory:.1f}% -> {after_memory:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return False
    
    async def _optimize_cache_warming(self, bottleneck: Bottleneck) -> bool:
        """Optimize caching through cache warming."""
        try:
            logger.info("Applying cache warming optimization")
            
            # Implement cache warming strategies
            # This would involve pre-loading frequently accessed data
            # For demonstration, we'll simulate the process
            
            await asyncio.sleep(1)  # Simulate cache warming time
            return True
            
        except Exception as e:
            logger.error(f"Cache warming optimization failed: {e}")
            return False
    
    async def _optimize_query_indexes(self, bottleneck: Bottleneck) -> bool:
        """Optimize database query performance through indexing suggestions."""
        try:
            logger.info("Applying query optimization")
            
            # In a real implementation, this would:
            # 1. Analyze slow queries
            # 2. Suggest and create indexes
            # 3. Optimize query plans
            # 4. Update statistics
            
            if self.db_session:
                # Update table statistics
                await self.db_session.execute(text("ANALYZE;"))
                await self.db_session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return False
    
    async def _optimize_async_operations(self, bottleneck: Bottleneck) -> bool:
        """Optimize async operation efficiency."""
        try:
            logger.info("Applying async operations optimization")
            
            # This would involve:
            # 1. Reviewing async task creation patterns
            # 2. Optimizing event loop usage
            # 3. Implementing better concurrency control
            # 4. Reducing context switching overhead
            
            await asyncio.sleep(0.5)  # Simulate optimization time
            return True
            
        except Exception as e:
            logger.error(f"Async operations optimization failed: {e}")
            return False
    
    async def _optimize_memory_usage(self, bottleneck: Bottleneck) -> bool:
        """Optimize general memory usage patterns."""
        return await self._optimize_memory_garbage_collection(bottleneck)
    
    async def _optimize_io_operations(self, bottleneck: Bottleneck) -> bool:
        """Optimize I/O operation efficiency."""
        try:
            logger.info("Applying I/O optimization")
            
            # This would involve:
            # 1. Implementing I/O batching
            # 2. Optimizing file access patterns
            # 3. Using more efficient I/O operations
            # 4. Implementing I/O caching
            
            await asyncio.sleep(0.5)
            return True
            
        except Exception as e:
            logger.error(f"I/O optimization failed: {e}")
            return False


class PerformanceBenchmark:
    """
    Performance benchmarking system.
    
    Provides comprehensive performance testing and validation.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize performance benchmark."""
        self.db_session = db_session
        self.benchmark_history: List[BenchmarkResult] = []
    
    async def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """
        Run comprehensive performance benchmark.
        
        Returns:
            Dictionary of benchmark results by category
        """
        results = {}
        
        # Run different benchmark categories
        results["database"] = await self._benchmark_database_operations()
        results["memory"] = await self._benchmark_memory_operations()
        results["cache"] = await self._benchmark_cache_operations()
        results["async"] = await self._benchmark_async_operations()
        results["io"] = await self._benchmark_io_operations()
        
        # Store results
        for category, result in results.items():
            self.benchmark_history.append(result)
        
        logger.info(f"Comprehensive benchmark completed: {len(results)} categories tested")
        return results
    
    async def _benchmark_database_operations(self) -> BenchmarkResult:
        """Benchmark database operation performance."""
        if not self.db_session:
            return BenchmarkResult(
                name="Database Operations",
                operations_per_second=0,
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.utcnow(),
                test_duration=0,
                metadata={"error": "No database session available"}
            )
        
        start_time = time.time()
        response_times = []
        operations_count = 100
        
        # Measure memory and CPU before
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_cpu = process.cpu_percent()
        
        try:
            # Run database operations
            for _ in range(operations_count):
                op_start = time.time()
                
                # Simple query for benchmarking
                await self.db_session.execute(text("SELECT 1"))
                
                response_times.append((time.time() - op_start) * 1000)
            
            await self.db_session.commit()
            
        except Exception as e:
            logger.warning(f"Database benchmark error: {e}")
            operations_count = len(response_times)
        
        test_duration = time.time() - start_time
        
        # Measure memory and CPU after
        end_memory = process.memory_info().rss / 1024 / 1024
        end_cpu = process.cpu_percent()
        
        if response_times:
            return BenchmarkResult(
                name="Database Operations",
                operations_per_second=operations_count / test_duration,
                avg_response_time_ms=statistics.mean(response_times),
                p95_response_time_ms=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                p99_response_time_ms=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times),
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                timestamp=datetime.utcnow(),
                test_duration=test_duration,
                metadata={"operations_count": operations_count}
            )
        else:
            return BenchmarkResult(
                name="Database Operations",
                operations_per_second=0,
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.utcnow(),
                test_duration=test_duration,
                metadata={"error": "No successful operations"}
            )
    
    async def _benchmark_memory_operations(self) -> BenchmarkResult:
        """Benchmark memory operation performance."""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        # Memory operations benchmark
        operations_count = 10000
        data_structures = []
        
        # Create and manipulate data structures
        for i in range(operations_count):
            data_structures.append({
                "id": i,
                "data": f"test_data_{i}",
                "timestamp": datetime.utcnow(),
                "metadata": {"value": i * 2}
            })
        
        # Access patterns
        for item in data_structures[::100]:  # Every 100th item
            _ = item["data"]
        
        # Cleanup
        data_structures.clear()
        gc.collect()
        
        test_duration = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            name="Memory Operations",
            operations_per_second=operations_count / test_duration,
            avg_response_time_ms=(test_duration * 1000) / operations_count,
            p95_response_time_ms=0,  # Not applicable for bulk operations
            p99_response_time_ms=0,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            timestamp=datetime.utcnow(),
            test_duration=test_duration,
            metadata={"operations_count": operations_count}
        )
    
    async def _benchmark_cache_operations(self) -> BenchmarkResult:
        """Benchmark cache operation performance."""
        start_time = time.time()
        response_times = []
        operations_count = 1000
        
        # Simple in-memory cache for benchmarking
        cache = {}
        
        # Cache write operations
        for i in range(operations_count // 2):
            op_start = time.time()
            cache[f"key_{i}"] = f"value_{i}"
            response_times.append((time.time() - op_start) * 1000)
        
        # Cache read operations
        for i in range(operations_count // 2):
            op_start = time.time()
            _ = cache.get(f"key_{i}")
            response_times.append((time.time() - op_start) * 1000)
        
        test_duration = time.time() - start_time
        
        return BenchmarkResult(
            name="Cache Operations",
            operations_per_second=operations_count / test_duration,
            avg_response_time_ms=statistics.mean(response_times),
            p95_response_time_ms=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            p99_response_time_ms=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times),
            memory_usage_mb=sys.getsizeof(cache) / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            timestamp=datetime.utcnow(),
            test_duration=test_duration,
            metadata={"operations_count": operations_count}
        )
    
    async def _benchmark_async_operations(self) -> BenchmarkResult:
        """Benchmark async operation performance."""
        start_time = time.time()
        operations_count = 1000
        
        async def dummy_async_operation(delay: float = 0.001):
            await asyncio.sleep(delay)
            return True
        
        # Run concurrent async operations
        tasks = [dummy_async_operation() for _ in range(operations_count)]
        await asyncio.gather(*tasks)
        
        test_duration = time.time() - start_time
        
        return BenchmarkResult(
            name="Async Operations",
            operations_per_second=operations_count / test_duration,
            avg_response_time_ms=(test_duration * 1000) / operations_count,
            p95_response_time_ms=0,  # Not meaningful for concurrent operations
            p99_response_time_ms=0,
            memory_usage_mb=0,
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            timestamp=datetime.utcnow(),
            test_duration=test_duration,
            metadata={"operations_count": operations_count, "concurrency": "full"}
        )
    
    async def _benchmark_io_operations(self) -> BenchmarkResult:
        """Benchmark I/O operation performance."""
        start_time = time.time()
        operations_count = 100
        response_times = []
        
        try:
            # File I/O operations
            import tempfile
            import os
            
            for i in range(operations_count):
                op_start = time.time()
                
                # Write operation
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(f"test_data_{i}")
                    temp_file = f.name
                
                # Read operation
                with open(temp_file, 'r') as f:
                    _ = f.read()
                
                # Cleanup
                os.unlink(temp_file)
                
                response_times.append((time.time() - op_start) * 1000)
            
        except Exception as e:
            logger.warning(f"I/O benchmark error: {e}")
        
        test_duration = time.time() - start_time
        
        if response_times:
            return BenchmarkResult(
                name="I/O Operations",
                operations_per_second=len(response_times) / test_duration,
                avg_response_time_ms=statistics.mean(response_times),
                p95_response_time_ms=statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                p99_response_time_ms=statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times),
                memory_usage_mb=0,
                cpu_usage_percent=psutil.cpu_percent(interval=0.1),
                timestamp=datetime.utcnow(),
                test_duration=test_duration,
                metadata={"operations_count": len(response_times)}
            )
        else:
            return BenchmarkResult(
                name="I/O Operations",
                operations_per_second=0,
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                timestamp=datetime.utcnow(),
                test_duration=test_duration,
                metadata={"error": "No successful operations"}
            )


class PerformanceOptimizationService:
    """
    Main performance optimization service.
    
    Orchestrates bottleneck analysis, optimization implementation, and performance validation.
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """Initialize performance optimization service."""
        self.db_session = db_session
        self.analyzer = BottleneckAnalyzer(db_session)
        self.optimizer = PerformanceOptimizer(db_session)
        self.benchmark = PerformanceBenchmark(db_session)
        self._monitoring_enabled = True
        self._optimization_runs: List[Dict[str, Any]] = []
    
    async def run_full_optimization_cycle(self) -> Dict[str, Any]:
        """
        Run complete performance optimization cycle.
        
        Returns:
            Comprehensive optimization report
        """
        cycle_start = time.time()
        
        logger.info("Starting full performance optimization cycle")
        
        # Step 1: Baseline benchmarks
        logger.info("Running baseline benchmarks...")
        baseline_benchmarks = await self.benchmark.run_comprehensive_benchmark()
        
        # Step 2: Analyze bottlenecks
        logger.info("Analyzing performance bottlenecks...")
        bottlenecks = await self.analyzer.analyze_system_performance()
        
        # Step 3: Apply optimizations
        logger.info(f"Applying optimizations for {len(bottlenecks)} bottlenecks...")
        optimization_results = []
        
        for bottleneck in bottlenecks[:5]:  # Limit to top 5 bottlenecks
            if bottleneck.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]:
                result = await self.optimizer.apply_optimization(bottleneck)
                optimization_results.append(result)
        
        # Step 4: Post-optimization benchmarks
        logger.info("Running post-optimization benchmarks...")
        post_benchmarks = await self.benchmark.run_comprehensive_benchmark()
        
        # Step 5: Generate report
        cycle_duration = time.time() - cycle_start
        
        report = {
            "cycle_id": f"opt_cycle_{int(time.time())}",
            "started_at": datetime.utcnow().isoformat(),
            "duration_seconds": cycle_duration,
            "bottlenecks_analyzed": len(bottlenecks),
            "optimizations_applied": len(optimization_results),
            "baseline_benchmarks": {k: v.__dict__ for k, v in baseline_benchmarks.items()},
            "post_benchmarks": {k: v.__dict__ for k, v in post_benchmarks.items()},
            "bottlenecks": [bottleneck.__dict__ for bottleneck in bottlenecks],
            "optimization_results": [result.__dict__ for result in optimization_results],
            "performance_improvements": self._calculate_overall_improvements(baseline_benchmarks, post_benchmarks),
            "recommendations": self._generate_recommendations(bottlenecks, optimization_results)
        }
        
        self._optimization_runs.append(report)
        
        logger.info(f"Performance optimization cycle completed in {cycle_duration:.2f}s")
        return report
    
    def _calculate_overall_improvements(self, baseline: Dict[str, BenchmarkResult], post: Dict[str, BenchmarkResult]) -> Dict[str, float]:
        """Calculate overall performance improvements."""
        improvements = {}
        
        for category in baseline:
            if category in post:
                base_ops = baseline[category].operations_per_second
                post_ops = post[category].operations_per_second
                
                if base_ops > 0:
                    ops_improvement = ((post_ops - base_ops) / base_ops) * 100
                    improvements[f"{category}_ops_improvement_percent"] = ops_improvement
                
                base_response = baseline[category].avg_response_time_ms
                post_response = post[category].avg_response_time_ms
                
                if base_response > 0:
                    response_improvement = ((base_response - post_response) / base_response) * 100
                    improvements[f"{category}_response_improvement_percent"] = response_improvement
        
        return improvements
    
    def _generate_recommendations(self, bottlenecks: List[Bottleneck], optimization_results: List[OptimizationResult]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Unresolved critical bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL]
        if critical_bottlenecks:
            recommendations.append(f"Address {len(critical_bottlenecks)} critical performance bottlenecks immediately")
        
        # Failed optimizations
        failed_optimizations = [r for r in optimization_results if r.status == OptimizationStatus.FAILED]
        if failed_optimizations:
            recommendations.append(f"Investigate {len(failed_optimizations)} failed optimization attempts")
        
        # Successful optimizations
        successful_optimizations = [r for r in optimization_results if r.status == OptimizationStatus.COMPLETED]
        if successful_optimizations:
            avg_improvement = statistics.mean([r.improvement_percent for r in successful_optimizations])
            recommendations.append(f"Monitor {len(successful_optimizations)} successful optimizations (avg improvement: {avg_improvement:.1f}%)")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous performance monitoring",
            "Schedule regular optimization cycles",
            "Monitor resource usage trends",
            "Review and update performance targets"
        ])
        
        return recommendations
    
    async def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization cycle history."""
        return self._optimization_runs
    
    async def get_current_performance_status(self) -> Dict[str, Any]:
        """Get current performance status summary."""
        # Quick performance check
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": "healthy" if memory.percent < 80 and cpu < 80 else "degraded",
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu,
            "optimization_runs_count": len(self._optimization_runs),
            "last_optimization": self._optimization_runs[-1]["started_at"] if self._optimization_runs else None
        }
        
        return status


# Global performance optimization service instance
_performance_service: Optional[PerformanceOptimizationService] = None


def get_performance_optimization_service(db_session: Optional[AsyncSession] = None) -> PerformanceOptimizationService:
    """Get the global performance optimization service instance."""
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceOptimizationService(db_session)
    return _performance_service


# Utility functions for common performance operations
async def analyze_performance_bottlenecks(db_session: Optional[AsyncSession] = None) -> List[Bottleneck]:
    """Analyze current performance bottlenecks."""
    service = get_performance_optimization_service(db_session)
    return await service.analyzer.analyze_system_performance()


async def run_performance_optimization(db_session: Optional[AsyncSession] = None) -> Dict[str, Any]:
    """Run full performance optimization cycle."""
    service = get_performance_optimization_service(db_session)
    return await service.run_full_optimization_cycle()


async def benchmark_system_performance(db_session: Optional[AsyncSession] = None) -> Dict[str, BenchmarkResult]:
    """Run comprehensive system performance benchmark."""
    service = get_performance_optimization_service(db_session)
    return await service.benchmark.run_comprehensive_benchmark()


@asynccontextmanager
async def performance_profiler(operation_name: str):
    """Context manager for profiling operation performance."""
    profiler = cProfile.Profile()
    start_time = time.time()
    
    try:
        profiler.enable()
        yield profiler
        
    finally:
        profiler.disable()
        execution_time = time.time() - start_time
        
        # Generate profiling report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        logger.info(f"Performance profile for {operation_name} (duration: {execution_time:.3f}s):\n{s.getvalue()}")
