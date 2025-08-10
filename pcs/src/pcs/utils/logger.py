"""
Filepath: pcs/src/pcs/utils/logging.py
Purpose: Structured logging configuration with async support and JSON formatting
Related Components: Core logging, monitoring, metrics
Tags: logging, structured-logging, async, monitoring
"""

import logging as stdlib_logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import json
import structlog
from structlog.stdlib import LoggerFactory
from structlog.processors import JSONRenderer, TimeStamper, add_log_level
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    service_name: str = "pcs",
    environment: str = "development"
) -> None:
    """
    Setup structured logging with async support and JSON formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format (json, console)
        service_name: Name of the service for log identification
        environment: Current environment (development, staging, production)
    
    Side Effects:
        - Configures global logging configuration
        - Sets up structlog processors
        - Configures console and JSON output
    """
    
    # Configure structlog processors
    processors: list[Processor] = [
        add_log_level,
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        processors.append(JSONRenderer())
    else:
        # Console format for development
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    stdlib_logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(stdlib_logging, level.upper()),
    )
    
    # Set specific logger levels
    stdlib_logging.getLogger("uvicorn").setLevel(stdlib_logging.INFO)
    stdlib_logging.getLogger("uvicorn.access").setLevel(stdlib_logging.WARNING)
    stdlib_logging.getLogger("sqlalchemy.engine").setLevel(stdlib_logging.WARNING)
    
    # Log startup message
    logger = get_logger(__name__)
    logger.info(
        "Logging system initialized",
        service=service_name,
        environment=environment,
        log_level=level,
        log_format=log_format
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured structlog logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Operation completed", user_id=123, duration=0.5)
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """
    Logger adapter for consistent logging patterns across the application.
    
    Provides standardized logging methods with common context fields.
    """
    
    def __init__(self, name: str, **default_context: Any):
        """
        Initialize logger adapter with default context.
        
        Args:
            name: Logger name
            **default_context: Default context fields for all log entries
        """
        self.logger = get_logger(name)
        self.default_context = default_context
    
    def _merge_context(self, **context: Any) -> Dict[str, Any]:
        """Merge default context with provided context."""
        merged = self.default_context.copy()
        merged.update(context)
        return merged
    
    def info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **self._merge_context(**context))
    
    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **self._merge_context(**context))
    
    def error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **self._merge_context(**context))
    
    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **self._merge_context(**context))
    
    def exception(self, message: str, **context: Any) -> None:
        """Log exception with context and stack trace."""
        self.logger.exception(message, **self._merge_context(**context))


def log_function_call(func_name: str, **context: Any) -> None:
    """
    Log function call for debugging and monitoring.
    
    Args:
        func_name: Name of the function being called
        **context: Additional context information
    """
    logger = get_logger("function_calls")
    logger.debug(
        "Function called",
        function=func_name,
        timestamp=datetime.utcnow().isoformat(),
        **context
    )


def log_performance_metric(
    operation: str,
    duration: float,
    **context: Any
) -> None:
    """
    Log performance metrics for monitoring and optimization.
    
    Args:
        operation: Name of the operation being measured
        duration: Duration in seconds
        **context: Additional context (e.g., user_id, request_id)
    """
    logger = get_logger("performance")
    logger.info(
        "Performance metric",
        operation=operation,
        duration=duration,
        duration_ms=duration * 1000,
        timestamp=datetime.utcnow().isoformat(),
        **context
    )


def log_security_event(
    event_type: str,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log security-related events for audit and monitoring.
    
    Args:
        event_type: Type of security event
        user_id: ID of the user involved
        ip_address: IP address of the request
        **context: Additional security context
    """
    logger = get_logger("security")
    logger.warning(
        "Security event",
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        timestamp=datetime.utcnow().isoformat(),
        **context
    )
