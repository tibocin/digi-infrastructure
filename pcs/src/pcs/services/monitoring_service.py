"""
Filepath: pcs/src/pcs/services/monitoring_service.py
Purpose: Enhanced monitoring and observability service with Prometheus integration and alerting
Related Components: Metrics utilities, Background tasks, Webhooks, Rate limiting, Health checks
Tags: monitoring, observability, prometheus, alerting, metrics, health-checks
"""

import asyncio
import psutil
import time
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

from ..utils.logger import get_logger
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    multiprocess, values
)
from fastapi import Response
from pydantic import BaseModel, Field

from ..core.exceptions import PCSError
from ..utils.metrics import get_metrics_collector

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SILENCED = "silenced"


class MetricType(str, Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    severity: AlertSeverity
    duration: int = 60  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Alert:
    """Active alert instance."""
    rule: AlertRule
    value: float
    started_at: datetime
    status: AlertStatus = AlertStatus.FIRING
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration(self) -> int:
        """Get alert duration in seconds."""
        end_time = self.resolved_at or datetime.utcnow()
        return int((end_time - self.started_at).total_seconds())


class HealthCheckError(PCSError):
    """Health check related errors."""
    pass


class MonitoringError(PCSError):
    """Monitoring system related errors."""
    pass


class HealthCheck(BaseModel):
    """Health check definition."""
    name: str
    description: str
    check_function: str  # Function name to call
    timeout: int = 30
    interval: int = 60
    enabled: bool = True
    critical: bool = True  # Whether failure affects overall health
    labels: Dict[str, str] = Field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str
    duration: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrometheusExporter:
    """
    Prometheus metrics exporter.
    
    Handles collection and export of metrics in Prometheus format.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus exporter.
        
        Args:
            registry: Custom registry (None for default)
        """
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> None:
        """Set up default application metrics."""
        # Request metrics
        self.metrics["http_requests_total"] = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )
        
        self.metrics["http_request_duration_seconds"] = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        # Database metrics
        self.metrics["db_query_duration_seconds"] = Histogram(
            "db_query_duration_seconds",
            "Database query duration",
            ["query_type", "table"],
            registry=self.registry
        )
        
        self.metrics["db_connections_active"] = Gauge(
            "db_connections_active",
            "Active database connections",
            registry=self.registry
        )
        
        # Background task metrics
        self.metrics["background_tasks_total"] = Counter(
            "background_tasks_total",
            "Total background tasks",
            ["task_name", "status"],
            registry=self.registry
        )
        
        self.metrics["background_tasks_duration_seconds"] = Histogram(
            "background_tasks_duration_seconds",
            "Background task duration",
            ["task_name"],
            registry=self.registry
        )
        
        # Webhook metrics
        self.metrics["webhook_deliveries_total"] = Counter(
            "webhook_deliveries_total",
            "Total webhook deliveries",
            ["endpoint", "status"],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.metrics["rate_limit_requests_total"] = Counter(
            "rate_limit_requests_total",
            "Total rate limit checks",
            ["config", "result"],
            registry=self.registry
        )
        
        # System metrics
        self.metrics["system_cpu_usage_percent"] = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry
        )
        
        self.metrics["system_memory_usage_bytes"] = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            ["type"],
            registry=self.registry
        )
        
        self.metrics["system_disk_usage_bytes"] = Gauge(
            "system_disk_usage_bytes",
            "System disk usage in bytes",
            ["device", "type"],
            registry=self.registry
        )
        
        # Application metrics
        self.metrics["app_info"] = Info(
            "app_info",
            "Application information",
            registry=self.registry
        )
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def add_custom_metric(self, name: str, metric_type: MetricType, description: str, 
                         labels: List[str] = None) -> Any:
        """
        Add a custom metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: Label names
            
        Returns:
            Created metric object
        """
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels, registry=self.registry)
        elif metric_type == MetricType.INFO:
            metric = Info(name, description, registry=self.registry)
        else:
            raise MonitoringError(f"Unsupported metric type: {metric_type}")
        
        self.metrics[name] = metric
        return metric
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class AlertManager:
    """
    Alert management system.
    
    Handles alert rules, evaluation, and notifications.
    """
    
    def __init__(self):
        """Initialize alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        self._evaluation_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            # Resolve any active alerts for this rule
            if rule_name in self.active_alerts:
                self._resolve_alert(rule_name)
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False
    
    def add_notification_handler(self, handler: Callable) -> None:
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    async def start_evaluation(self) -> None:
        """Start alert evaluation loop."""
        if self._running:
            return
        
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Started alert evaluation loop")
    
    async def stop_evaluation(self) -> None:
        """Stop alert evaluation loop."""
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped alert evaluation loop")
    
    async def _evaluation_loop(self) -> None:
        """Main alert evaluation loop."""
        while self._running:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(10)  # Evaluate every 10 seconds
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _evaluate_rules(self) -> None:
        """Evaluate all alert rules."""
        # This is a simplified implementation
        # In a real system, you would integrate with your metrics backend
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                # Get current metric value (placeholder implementation)
                metric_value = await self._get_metric_value(rule.metric_name)
                
                if metric_value is not None:
                    should_fire = self._evaluate_condition(metric_value, rule)
                    
                    if should_fire and rule.name not in self.active_alerts:
                        await self._fire_alert(rule, metric_value)
                    elif not should_fire and rule.name in self.active_alerts:
                        self._resolve_alert(rule.name)
                        
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric (placeholder)."""
        # This would integrate with your actual metrics system
        # For now, return a placeholder value
        return 0.0
    
    def _evaluate_condition(self, value: float, rule: AlertRule) -> bool:
        """Evaluate if alert condition is met."""
        if rule.comparison == "gt":
            return value > rule.threshold
        elif rule.comparison == "lt":
            return value < rule.threshold
        elif rule.comparison == "eq":
            return value == rule.threshold
        elif rule.comparison == "gte":
            return value >= rule.threshold
        elif rule.comparison == "lte":
            return value <= rule.threshold
        else:
            return False
    
    async def _fire_alert(self, rule: AlertRule, value: float) -> None:
        """Fire a new alert."""
        alert = Alert(
            rule=rule,
            value=value,
            started_at=datetime.utcnow(),
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy()
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert fired: {rule.name} (value: {value}, threshold: {rule.threshold})")
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def _resolve_alert(self, rule_name: str) -> None:
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            del self.active_alerts[rule_name]
            
            logger.info(f"Alert resolved: {rule_name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time window."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.started_at >= cutoff]


class HealthChecker:
    """
    Health check system.
    
    Manages and executes health checks for system components.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.check_functions: Dict[str, Callable] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check_function("database_check", self._check_database)
        self.register_check_function("redis_check", self._check_redis)
        self.register_check_function("disk_space_check", self._check_disk_space)
        self.register_check_function("memory_check", self._check_memory)
        
        # Add default health checks
        self.add_check(HealthCheck(
            name="database",
            description="Database connectivity",
            check_function="database_check",
            interval=30,
            critical=True
        ))
        
        self.add_check(HealthCheck(
            name="redis",
            description="Redis connectivity",
            check_function="redis_check",
            interval=30,
            critical=True
        ))
        
        self.add_check(HealthCheck(
            name="disk_space",
            description="Disk space availability",
            check_function="disk_space_check",
            interval=60,
            critical=False
        ))
        
        self.add_check(HealthCheck(
            name="memory",
            description="Memory usage",
            check_function="memory_check",
            interval=30,
            critical=False
        ))
    
    def register_check_function(self, name: str, func: Callable) -> None:
        """Register a health check function."""
        self.check_functions[name] = func
    
    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks[check.name] = check
        logger.info(f"Added health check: {check.name}")
    
    def remove_check(self, check_name: str) -> bool:
        """Remove a health check."""
        if check_name in self.checks:
            del self.checks[check_name]
            if check_name in self.results:
                del self.results[check_name]
            logger.info(f"Removed health check: {check_name}")
            return True
        return False
    
    async def start_checks(self) -> None:
        """Start health check loop."""
        if self._running:
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Started health check loop")
    
    async def stop_checks(self) -> None:
        """Stop health check loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped health check loop")
    
    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._run_checks()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _run_checks(self) -> None:
        """Run all enabled health checks."""
        now = datetime.utcnow()
        
        for check in self.checks.values():
            if not check.enabled:
                continue
            
            # Check if it's time to run this check
            last_result = self.results.get(check.name)
            if (last_result is None or 
                (now - last_result.timestamp).total_seconds() >= check.interval):
                
                await self._execute_check(check)
    
    async def _execute_check(self, check: HealthCheck) -> None:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            if check.check_function not in self.check_functions:
                raise HealthCheckError(f"Check function not found: {check.check_function}")
            
            func = self.check_functions[check.check_function]
            
            # Execute with timeout
            result = await asyncio.wait_for(
                func(check),
                timeout=check.timeout
            )
            
            duration = time.time() - start_time
            
            if isinstance(result, HealthCheckResult):
                check_result = result
                check_result.duration = duration
            else:
                # Assume boolean result
                check_result = HealthCheckResult(
                    name=check.name,
                    healthy=bool(result),
                    message="OK" if result else "Check failed",
                    duration=duration,
                    timestamp=datetime.utcnow(),
                    labels=check.labels.copy()
                )
            
        except asyncio.TimeoutError:
            check_result = HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Check timed out after {check.timeout}s",
                duration=check.timeout,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
        except Exception as e:
            check_result = HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Check failed: {str(e)}",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
        
        self.results[check.name] = check_result
        
        if not check_result.healthy:
            logger.warning(f"Health check failed: {check.name} - {check_result.message}")
    
    async def _check_database(self, check: HealthCheck) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            # This would use your actual database connection
            # For now, simulate a check
            await asyncio.sleep(0.1)  # Simulate DB query
            
            return HealthCheckResult(
                name=check.name,
                healthy=True,
                message="Database connection OK",
                duration=0.1,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy(),
                metadata={"connection_pool_size": 10, "active_connections": 5}
            )
        except Exception as e:
            return HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Database connection failed: {e}",
                duration=0.0,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
    
    async def _check_redis(self, check: HealthCheck) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            # This would use your actual Redis connection
            # For now, simulate a check
            await asyncio.sleep(0.05)  # Simulate Redis ping
            
            return HealthCheckResult(
                name=check.name,
                healthy=True,
                message="Redis connection OK",
                duration=0.05,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy(),
                metadata={"connected_clients": 2, "used_memory": "1024KB"}
            )
        except Exception as e:
            return HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Redis connection failed: {e}",
                duration=0.0,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
    
    async def _check_disk_space(self, check: HealthCheck) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            healthy = free_percent > 10  # Alert if less than 10% free
            message = f"Disk usage: {100 - free_percent:.1f}% used, {free_percent:.1f}% free"
            
            return HealthCheckResult(
                name=check.name,
                healthy=healthy,
                message=message,
                duration=0.01,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy(),
                metadata={
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "free_bytes": disk_usage.free,
                    "free_percent": free_percent
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Disk check failed: {e}",
                duration=0.0,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
    
    async def _check_memory(self, check: HealthCheck) -> HealthCheckResult:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            healthy = used_percent < 90  # Alert if more than 90% used
            message = f"Memory usage: {used_percent:.1f}% used"
            
            return HealthCheckResult(
                name=check.name,
                healthy=healthy,
                message=message,
                duration=0.01,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy(),
                metadata={
                    "total_bytes": memory.total,
                    "used_bytes": memory.used,
                    "available_bytes": memory.available,
                    "used_percent": used_percent
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name=check.name,
                healthy=False,
                message=f"Memory check failed: {e}",
                duration=0.0,
                timestamp=datetime.utcnow(),
                labels=check.labels.copy()
            )
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.results:
            return {"status": "unknown", "checks": []}
        
        healthy_checks = sum(1 for result in self.results.values() if result.healthy)
        total_checks = len(self.results)
        
        # Check if any critical checks are failing
        critical_failing = any(
            not result.healthy and self.checks[result.name].critical
            for result in self.results.values()
            if result.name in self.checks
        )
        
        if critical_failing:
            status = "unhealthy"
        elif healthy_checks == total_checks:
            status = "healthy"
        else:
            status = "degraded"
        
        return {
            "status": status,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "checks": [
                {
                    "name": result.name,
                    "healthy": result.healthy,
                    "message": result.message,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat(),
                    "critical": self.checks.get(result.name, HealthCheck(name="", description="", check_function="")).critical,
                    "metadata": result.metadata
                }
                for result in self.results.values()
            ]
        }


class MonitoringService:
    """
    Comprehensive monitoring and observability service.
    
    Integrates Prometheus metrics, alerting, and health checks.
    """
    
    def __init__(self):
        """Initialize monitoring service."""
        self.prometheus = PrometheusExporter()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self._system_metrics_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup system metrics collection
        self._setup_system_metrics()
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                description="CPU usage is above 80%",
                metric_name="system_cpu_usage_percent",
                threshold=80.0,
                comparison="gt",
                severity=AlertSeverity.HIGH,
                duration=300  # 5 minutes
            ),
            AlertRule(
                name="high_memory_usage",
                description="Memory usage is above 90%",
                metric_name="system_memory_usage_percent",
                threshold=90.0,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                duration=180  # 3 minutes
            ),
            AlertRule(
                name="low_disk_space",
                description="Disk space is below 10%",
                metric_name="system_disk_free_percent",
                threshold=10.0,
                comparison="lt",
                severity=AlertSeverity.HIGH,
                duration=300  # 5 minutes
            ),
            AlertRule(
                name="high_request_latency",
                description="Average request latency is above 1 second",
                metric_name="http_request_duration_seconds",
                threshold=1.0,
                comparison="gt",
                severity=AlertSeverity.MEDIUM,
                duration=120  # 2 minutes
            ),
            AlertRule(
                name="background_task_failures",
                description="Background task failure rate is above 10%",
                metric_name="background_task_failure_rate",
                threshold=0.1,
                comparison="gt",
                severity=AlertSeverity.HIGH,
                duration=300  # 5 minutes
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    def _setup_system_metrics(self) -> None:
        """Setup system metrics collection."""
        # App info
        app_info = self.prometheus.get_metric("app_info")
        if app_info:
            app_info.info({
                "version": "1.0.0",
                "environment": "production",
                "service": "pcs"
            })
    
    async def start(self) -> None:
        """Start the monitoring service."""
        if self._running:
            return
        
        self._running = True
        
        # Start system metrics collection
        self._system_metrics_task = asyncio.create_task(self._collect_system_metrics())
        
        # Start health checks
        await self.health_checker.start_checks()
        
        # Start alert evaluation
        await self.alert_manager.start_evaluation()
        
        logger.info("Monitoring service started")
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        self._running = False
        
        # Stop system metrics collection
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
            try:
                await self._system_metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop health checks
        await self.health_checker.stop_checks()
        
        # Stop alert evaluation
        await self.alert_manager.stop_evaluation()
        
        logger.info("Monitoring service stopped")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics periodically."""
        while self._running:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_gauge = self.prometheus.get_metric("system_cpu_usage_percent")
                if cpu_gauge:
                    cpu_gauge.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_gauge = self.prometheus.get_metric("system_memory_usage_bytes")
                if memory_gauge:
                    memory_gauge.labels(type="used").set(memory.used)
                    memory_gauge.labels(type="available").set(memory.available)
                    memory_gauge.labels(type="total").set(memory.total)
                
                # Disk metrics
                disk_usage = psutil.disk_usage('/')
                disk_gauge = self.prometheus.get_metric("system_disk_usage_bytes")
                if disk_gauge:
                    disk_gauge.labels(device="root", type="used").set(disk_usage.used)
                    disk_gauge.labels(device="root", type="free").set(disk_usage.free)
                    disk_gauge.labels(device="root", type="total").set(disk_usage.total)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics."""
        requests_counter = self.prometheus.get_metric("http_requests_total")
        duration_histogram = self.prometheus.get_metric("http_request_duration_seconds")
        
        if requests_counter:
            requests_counter.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
        
        if duration_histogram:
            duration_histogram.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_db_query(self, query_type: str, table: str, duration: float) -> None:
        """Record database query metrics."""
        db_histogram = self.prometheus.get_metric("db_query_duration_seconds")
        if db_histogram:
            db_histogram.labels(query_type=query_type, table=table).observe(duration)
    
    def record_background_task(self, task_name: str, status: str, duration: float = None) -> None:
        """Record background task metrics."""
        tasks_counter = self.prometheus.get_metric("background_tasks_total")
        if tasks_counter:
            tasks_counter.labels(task_name=task_name, status=status).inc()
        
        if duration is not None:
            duration_histogram = self.prometheus.get_metric("background_tasks_duration_seconds")
            if duration_histogram:
                duration_histogram.labels(task_name=task_name).observe(duration)
    
    def record_webhook_delivery(self, endpoint: str, status: str) -> None:
        """Record webhook delivery metrics."""
        webhook_counter = self.prometheus.get_metric("webhook_deliveries_total")
        if webhook_counter:
            webhook_counter.labels(endpoint=endpoint, status=status).inc()
    
    def record_rate_limit_check(self, config: str, result: str) -> None:
        """Record rate limit check metrics."""
        rate_limit_counter = self.prometheus.get_metric("rate_limit_requests_total")
        if rate_limit_counter:
            rate_limit_counter.labels(config=config, result=result).inc()
    
    def set_db_connections(self, count: int) -> None:
        """Set current database connection count."""
        db_connections = self.prometheus.get_metric("db_connections_active")
        if db_connections:
            db_connections.set(count)
    
    def get_metrics_response(self) -> Response:
        """Get Prometheus metrics response."""
        metrics_output = self.prometheus.export_metrics()
        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return self.health_checker.get_overall_health()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        alerts = self.alert_manager.get_active_alerts()
        return [
            {
                "name": alert.rule.name,
                "description": alert.rule.description,
                "severity": alert.rule.severity.value,
                "value": alert.value,
                "threshold": alert.rule.threshold,
                "started_at": alert.started_at.isoformat(),
                "duration": alert.duration,
                "labels": alert.labels,
                "annotations": alert.annotations
            }
            for alert in alerts
        ]
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Context manager for monitoring operations."""
        labels = labels or {}
        start_time = time.time()
        
        try:
            yield
            # Operation succeeded
            duration = time.time() - start_time
            self._record_operation_metric(operation_name, "success", duration, labels)
            
        except Exception as e:
            # Operation failed
            duration = time.time() - start_time
            self._record_operation_metric(operation_name, "error", duration, labels)
            raise
    
    def _record_operation_metric(self, operation: str, status: str, duration: float, labels: Dict[str, str]) -> None:
        """Record operation metrics."""
        # Try to find or create appropriate metric
        metric_name = f"{operation}_duration_seconds"
        histogram = self.prometheus.get_metric(metric_name)
        
        if not histogram:
            # Create custom metric if it doesn't exist
            label_names = list(labels.keys()) + ["status"]
            histogram = self.prometheus.add_custom_metric(
                metric_name,
                MetricType.HISTOGRAM,
                f"Duration of {operation} operations",
                label_names
            )
        
        if histogram:
            all_labels = labels.copy()
            all_labels["status"] = status
            histogram.labels(**all_labels).observe(duration)


# Global monitoring service instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


async def monitoring_lifespan():
    """FastAPI lifespan context manager for monitoring service."""
    service = get_monitoring_service()
    await service.start()
    try:
        yield
    finally:
        await service.stop()


# Utility functions for common monitoring patterns
def monitor_http_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Record HTTP request metrics."""
    service = get_monitoring_service()
    service.record_http_request(method, endpoint, status_code, duration)


def monitor_db_query(query_type: str, table: str, duration: float) -> None:
    """Record database query metrics."""
    service = get_monitoring_service()
    service.record_db_query(query_type, table, duration)


def monitor_background_task(task_name: str, status: str, duration: float = None) -> None:
    """Record background task metrics."""
    service = get_monitoring_service()
    service.record_background_task(task_name, status, duration)


def monitor_webhook_delivery(endpoint: str, status: str) -> None:
    """Record webhook delivery metrics."""
    service = get_monitoring_service()
    service.record_webhook_delivery(endpoint, status)


def monitor_rate_limit_check(config: str, result: str) -> None:
    """Record rate limit check metrics."""
    service = get_monitoring_service()
    service.record_rate_limit_check(config, result)
