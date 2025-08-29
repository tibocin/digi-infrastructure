"""
Filepath: pcs/src/pcs/utils/metrics.py
Purpose: Performance monitoring and metrics collection utilities for database operations
Related Components: Repository classes, Database operations, Performance monitoring
Tags: metrics, performance, monitoring, database, tracking
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueryMetric:
    """Container for individual query performance metrics."""
    query_type: str
    execution_time: float
    rows_affected: int
    timestamp: datetime
    table_name: Optional[str] = None
    query_hash: Optional[str] = None
    connection_pool_stats: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """
    Centralized metrics collection for database operations.
    
    Features:
    - Query performance tracking
    - Connection pool monitoring
    - Aggregated statistics
    - Performance trend analysis
    """
    
    def __init__(self, max_metrics: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self._metrics: List[QueryMetric] = []
        self._aggregated_stats: Dict[str, Dict[str, Any]] = {}
        self._last_aggregation: Optional[datetime] = None
    
    def record_query_metric(
        self,
        query_type: str,
        execution_time: float,
        rows_affected: int,
        table_name: Optional[str] = None,
        query_hash: Optional[str] = None,
        connection_pool_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a query performance metric."""
        metric = QueryMetric(
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=rows_affected,
            timestamp=datetime.now(UTC),
            table_name=table_name,
            query_hash=query_hash,
            connection_pool_stats=connection_pool_stats
        )
        
        self._metrics.append(metric)
        
        # Trim metrics if we exceed the limit
        if len(self._metrics) > self.max_metrics:
            self._metrics = self._metrics[-self.max_metrics:]
        
        # Update aggregated stats periodically
        self._update_aggregated_stats()
    
    def get_metrics_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get aggregated metrics summary.
        
        Args:
            time_window: Time window to filter metrics (None for all)
            
        Returns:
            Dictionary with aggregated metrics
        """
        if time_window:
            cutoff_time = datetime.now(UTC) - time_window
            filtered_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        else:
            filtered_metrics = self._metrics
        
        if not filtered_metrics:
            return {}
        
        # Group by query type
        by_query_type = {}
        for metric in filtered_metrics:
            if metric.query_type not in by_query_type:
                by_query_type[metric.query_type] = []
            by_query_type[metric.query_type].append(metric)
        
        # Calculate statistics for each query type
        summary = {}
        for query_type, metrics in by_query_type.items():
            execution_times = [m.execution_time for m in metrics]
            rows_affected = [m.rows_affected for m in metrics]
            
            summary[query_type] = {
                "count": len(metrics),
                "avg_execution_time": sum(execution_times) / len(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "total_rows_affected": sum(rows_affected),
                "avg_rows_affected": sum(rows_affected) / len(rows_affected) if rows_affected else 0,
                "p95_execution_time": self._calculate_percentile(execution_times, 95),
                "p99_execution_time": self._calculate_percentile(execution_times, 99)
            }
        
        return summary
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance trends over time.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with trends by query type
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Group by query type and hour buckets
        trends = {}
        for metric in recent_metrics:
            hour_bucket = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            
            if metric.query_type not in trends:
                trends[metric.query_type] = {}
            
            if hour_bucket not in trends[metric.query_type]:
                trends[metric.query_type][hour_bucket] = []
            
            trends[metric.query_type][hour_bucket].append(metric)
        
        # Calculate hourly averages
        result = {}
        for query_type, hourly_data in trends.items():
            result[query_type] = []
            
            for hour, metrics in sorted(hourly_data.items()):
                execution_times = [m.execution_time for m in metrics]
                avg_time = sum(execution_times) / len(execution_times)
                
                result[query_type].append({
                    "timestamp": hour.isoformat(),
                    "count": len(metrics),
                    "avg_execution_time": avg_time,
                    "total_rows_affected": sum(m.rows_affected for m in metrics)
                })
        
        return result
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._aggregated_stats.clear()
        self._last_aggregation = None
    
    def _update_aggregated_stats(self) -> None:
        """Update aggregated statistics periodically."""
        now = datetime.now(UTC)
        
        # Update every 5 minutes
        if (self._last_aggregation is None or 
            now - self._last_aggregation > timedelta(minutes=5)):
            
            self._aggregated_stats = self.get_metrics_summary(timedelta(hours=1))
            self._last_aggregation = now
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        
        if index >= len(sorted_values):
            return sorted_values[-1]
        
        return sorted_values[index]


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def track_query_performance(
    query_type: str,
    table_name: Optional[str] = None
) -> Callable:
    """
    Decorator to track query performance metrics.
    
    Args:
        query_type: Type of query being executed
        table_name: Name of the table being queried
        
    Returns:
        Decorated function with performance tracking
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                rows_affected = 0
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Try to extract rows affected from result
                    if hasattr(result, '__len__'):
                        rows_affected = len(result)
                    elif isinstance(result, int):
                        rows_affected = result
                    
                    return result
                
                finally:
                    execution_time = time.time() - start_time
                    _metrics_collector.record_query_metric(
                        query_type=query_type,
                        execution_time=execution_time,
                        rows_affected=rows_affected,
                        table_name=table_name
                    )
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                rows_affected = 0
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Try to extract rows affected from result
                    if hasattr(result, '__len__'):
                        rows_affected = len(result)
                    elif isinstance(result, int):
                        rows_affected = result
                    
                    return result
                
                finally:
                    execution_time = time.time() - start_time
                    _metrics_collector.record_query_metric(
                        query_type=query_type,
                        execution_time=execution_time,
                        rows_affected=rows_affected,
                        table_name=table_name
                    )
            
            return sync_wrapper
    
    return decorator


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


def record_manual_metric(
    query_type: str,
    execution_time: float,
    rows_affected: int,
    table_name: Optional[str] = None,
    query_hash: Optional[str] = None,
    connection_pool_stats: Optional[Dict[str, Any]] = None
) -> None:
    """
    Manually record a query performance metric.
    
    Args:
        query_type: Type of query executed
        execution_time: Execution time in seconds
        rows_affected: Number of rows affected
        table_name: Name of the table queried
        query_hash: Hash of the query for deduplication
        connection_pool_stats: Connection pool statistics
    """
    _metrics_collector.record_query_metric(
        query_type=query_type,
        execution_time=execution_time,
        rows_affected=rows_affected,
        table_name=table_name,
        query_hash=query_hash,
        connection_pool_stats=connection_pool_stats
    )


def get_performance_summary(hours: int = 1) -> Dict[str, Any]:
    """
    Get performance summary for the specified time window.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        Performance summary dictionary
    """
    return _metrics_collector.get_metrics_summary(timedelta(hours=hours))


def get_performance_trends(hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get performance trends over time.
    
    Args:
        hours: Number of hours to analyze
        
    Returns:
        Performance trends by query type
    """
    return _metrics_collector.get_performance_trends(hours)


def clear_all_metrics() -> None:
    """Clear all collected performance metrics."""
    _metrics_collector.clear_metrics()


class PerformanceMonitor:
    """
    Context manager for monitoring query performance.
    
    Usage:
        async with PerformanceMonitor("bulk_insert", "users") as monitor:
            # Perform database operations
            result = await session.execute(stmt)
            monitor.set_rows_affected(result.rowcount)
    """
    def __init__(self, query_type: str, table_name: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            query_type: Type of query being monitored
            table_name: Name of the table being queried
        """
        self.query_type = query_type
        self.table_name = table_name
        self.start_time: Optional[float] = None
        self.rows_affected: int = 0
    
    def __enter__(self):
        """Enter the context manager."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and record metrics."""
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            record_manual_metric(
                query_type=self.query_type,
                execution_time=execution_time,
                rows_affected=self.rows_affected,
                table_name=self.table_name
            )
    
    async def __aenter__(self):
        """Enter the async context manager."""
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager and record metrics."""
        if self.start_time is not None:
            execution_time = time.time() - self.start_time
            record_manual_metric(
                query_type=self.query_type,
                execution_time=execution_time,
                rows_affected=self.rows_affected,
                table_name=self.table_name
            )
    
    def set_rows_affected(self, rows: int) -> None:
        """Set the number of rows affected by the operation."""
        self.rows_affected = rows
