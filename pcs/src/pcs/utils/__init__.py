"""
Filepath: pcs/src/pcs/utils/__init__.py
Purpose: Utils package initialization and exports
Related Components: Logging, metrics, utilities
Tags: utils, logging, metrics, exports
"""

from .logger import (
    setup_logging,
    get_logger,
    LoggerAdapter,
    log_function_call,
    log_performance_metric,
    log_security_event
)

from .metrics import (
    MetricsCollector,
    QueryMetric,
    PerformanceMonitor,
    track_query_performance,
    get_metrics_collector,
    record_manual_metric,
    get_performance_summary,
    get_performance_trends,
    clear_all_metrics
)

__all__ = [
    # Logging utilities
    "setup_logging",
    "get_logger", 
    "LoggerAdapter",
    "log_function_call",
    "log_performance_metric",
    "log_security_event",
    
    # Metrics utilities
    "MetricsCollector",
    "QueryMetric",
    "PerformanceMonitor",
    "track_query_performance",
    "get_metrics_collector",
    "record_manual_metric",
    "get_performance_summary",
    "get_performance_trends",
    "clear_all_metrics"
]
