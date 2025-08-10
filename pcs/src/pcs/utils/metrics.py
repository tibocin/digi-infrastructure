"""
Filepath: pcs/src/pcs/utils/metrics.py
Purpose: Prometheus metrics collection and monitoring for PCS system
Related Components: Monitoring, observability, performance tracking
Tags: metrics, prometheus, monitoring, observability
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
from contextlib import asynccontextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
from prometheus_client.exposition import start_http_server
import asyncio


class PrometheusMetrics:
    """
    Prometheus metrics collection for PCS system.
    
    Provides comprehensive metrics for monitoring system performance,
    business operations, and infrastructure health.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics.
        
        Args:
            registry: Optional custom collector registry
        """
        self.registry = registry or CollectorRegistry()
        
        # HTTP request metrics
        self.request_duration = Histogram(
            'pcs_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_count = Counter(
            'pcs_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_query_duration = Histogram(
            'pcs_database_query_duration_seconds',
            'Database query duration in seconds',
            ['operation', 'database', 'table'],
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'pcs_database_connections_current',
            'Current database connections',
            ['database', 'status'],
            registry=self.registry
        )
        
        # Business metrics
        self.prompt_generation_count = Counter(
            'pcs_prompts_generated_total',
            'Total prompts generated',
            ['template_name', 'status'],
            registry=self.registry
        )
        
        self.context_operations = Counter(
            'pcs_context_operations_total',
            'Total context operations',
            ['operation_type', 'context_type', 'status'],
            registry=self.registry
        )
        
        # System metrics
        self.active_conversations = Gauge(
            'pcs_active_conversations',
            'Number of active conversations',
            ['status'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'pcs_cache_hit_ratio',
            'Cache hit ratio (0-1)',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'pcs_errors_total',
            'Total errors by type',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        # Performance metrics
        self.operation_duration = Histogram(
            'pcs_operation_duration_seconds',
            'Operation duration in seconds',
            ['operation_type', 'component'],
            registry=self.registry
        )
        
        # Memory and resource metrics
        self.memory_usage = Gauge(
            'pcs_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'pcs_cpu_usage_percent',
            'CPU usage percentage',
            ['component'],
            registry=self.registry
        )
    
    def record_request_metrics(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record HTTP request metrics.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(duration)
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
    
    def record_database_query_duration(
        self,
        operation: str,
        database: str,
        table: str,
        duration: float
    ) -> None:
        """
        Record database query performance metrics.
        
        Args:
            operation: Type of database operation
            database: Database name/type
            table: Table name
            duration: Query duration in seconds
        """
        self.database_query_duration.labels(
            operation=operation,
            database=database,
            table=table
        ).observe(duration)
    
    def update_database_connections(
        self,
        database: str,
        status: str,
        count: int
    ) -> None:
        """
        Update database connection count.
        
        Args:
            database: Database name/type
            status: Connection status (active, idle, total)
            count: Number of connections
        """
        self.database_connections.labels(
            database=database,
            status=status
        ).set(count)
    
    def record_prompt_generation(
        self,
        template_name: str,
        status: str
    ) -> None:
        """
        Record prompt generation metrics.
        
        Args:
            template_name: Name of the prompt template
            status: Generation status (success, error, cached)
        """
        self.prompt_generation_count.labels(
            template_name=template_name,
            status=status
        ).inc()
    
    def record_context_operation(
        self,
        operation_type: str,
        context_type: str,
        status: str
    ) -> None:
        """
        Record context operation metrics.
        
        Args:
            operation_type: Type of context operation
            context_type: Type of context
            status: Operation status (success, error)
        """
        self.context_operations.labels(
            operation_type=operation_type,
            context_type=context_type,
            status=status
        ).inc()
    
    def update_active_conversations(
        self,
        status: str,
        count: int
    ) -> None:
        """
        Update active conversations count.
        
        Args:
            status: Conversation status (active, pending, completed)
            count: Number of conversations
        """
        self.active_conversations.labels(status=status).set(count)
    
    def update_cache_hit_ratio(
        self,
        cache_type: str,
        ratio: float
    ) -> None:
        """
        Update cache hit ratio.
        
        Args:
            cache_type: Type of cache (redis, memory, etc.)
            ratio: Hit ratio between 0 and 1
        """
        self.cache_hit_ratio.labels(cache_type=cache_type).set(ratio)
    
    def record_error(
        self,
        error_type: str,
        component: str,
        severity: str = "error"
    ) -> None:
        """
        Record error metrics.
        
        Args:
            error_type: Type of error
            component: Component where error occurred
            severity: Error severity (info, warning, error, critical)
        """
        self.error_count.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
    
    def record_operation_duration(
        self,
        operation_type: str,
        component: str,
        duration: float
    ) -> None:
        """
        Record operation duration metrics.
        
        Args:
            operation_type: Type of operation
            component: Component performing operation
            duration: Operation duration in seconds
        """
        self.operation_duration.labels(
            operation_type=operation_type,
            component=component
        ).observe(duration)
    
    def update_memory_usage(
        self,
        component: str,
        bytes_used: int
    ) -> None:
        """
        Update memory usage metrics.
        
        Args:
            component: Component name
            bytes_used: Memory usage in bytes
        """
        self.memory_usage.labels(component=component).set(bytes_used)
    
    def update_cpu_usage(
        self,
        component: str,
        percentage: float
    ) -> None:
        """
        Update CPU usage metrics.
        
        Args:
            component: Component name
            percentage: CPU usage percentage
        """
        self.cpu_usage.labels(component=component).set(percentage)
    
    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format.
        
        Returns:
            Prometheus metrics in text format
        """
        return generate_latest(self.registry)
    
    def start_metrics_server(
        self,
        port: int = 8000,
        addr: str = "0.0.0.0"
    ) -> None:
        """
        Start HTTP server for metrics endpoint.
        
        Args:
            port: Port to serve metrics on
            addr: Address to bind to
        """
        start_http_server(port, addr, registry=self.registry)


# Global metrics instance
_metrics: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """
    Get global metrics instance.
    
    Returns:
        Configured PrometheusMetrics instance
    """
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
    return _metrics


def record_request_metrics(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    """
    Record HTTP request metrics using global metrics instance.
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    get_metrics().record_request_metrics(method, endpoint, status_code, duration)


def record_business_metrics(
    operation: str,
    component: str,
    duration: float,
    **labels: str
) -> None:
    """
    Record business operation metrics.
    
    Args:
        operation: Operation type
        component: Component name
        duration: Operation duration in seconds
        **labels: Additional label key-value pairs
    """
    metrics = get_metrics()
    metrics.record_operation_duration(operation, component, duration)
    
    # Record additional business-specific metrics
    if operation == "prompt_generation":
        template_name = labels.get("template_name", "unknown")
        status = labels.get("status", "unknown")
        metrics.record_prompt_generation(template_name, status)
    elif operation == "context_operation":
        operation_type = labels.get("operation_type", "unknown")
        context_type = labels.get("context_type", "unknown")
        status = labels.get("status", "unknown")
        metrics.record_context_operation(operation_type, context_type, status)


@asynccontextmanager
async def measure_operation(
    operation: str,
    component: str,
    **labels: str
):
    """
    Context manager for measuring operation duration.
    
    Args:
        operation: Operation type
        component: Component name
        **labels: Additional labels for metrics
    
    Yields:
        None
        
    Example:
        async with measure_operation("database_query", "postgres", table="users"):
            await db.execute(query)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        record_business_metrics(operation, component, duration, **labels)


def start_metrics_endpoint(
    port: int = 8000,
    addr: str = "0.0.0.0"
) -> None:
    """
    Start metrics endpoint server.
    
    Args:
        port: Port to serve metrics on
        addr: Address to bind to
    """
    get_metrics().start_metrics_server(port, addr)
