"""
Filepath: tests/unit/test_monitoring_service.py
Purpose: Unit tests for monitoring and observability service
Related Components: MonitoringService, PrometheusExporter, AlertManager, HealthChecker
Tags: unit-tests, monitoring, prometheus, alerting, health-checks, observability
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from typing import Any, Dict

from prometheus_client import CollectorRegistry

from pcs.services.monitoring_service import (
    MonitoringService,
    PrometheusExporter,
    AlertManager,
    HealthChecker,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertStatus,
    HealthCheck,
    HealthCheckResult,
    MetricType,
    HealthCheckError,
    MonitoringError,
    get_monitoring_service,
    monitor_http_request,
    monitor_db_query,
    monitor_background_task,
    monitor_webhook_delivery,
    monitor_rate_limit_check
)


@pytest.fixture
def prometheus_exporter():
    """Fixture providing a Prometheus exporter instance."""
    registry = CollectorRegistry()
    return PrometheusExporter(registry=registry)


@pytest.fixture
def alert_manager():
    """Fixture providing an AlertManager instance."""
    return AlertManager()


@pytest.fixture
def health_checker():
    """Fixture providing a HealthChecker instance."""
    return HealthChecker()


@pytest.fixture
def monitoring_service():
    """Fixture providing a MonitoringService instance."""
    return MonitoringService()


@pytest.fixture
def sample_alert_rule():
    """Fixture providing a sample alert rule."""
    return AlertRule(
        name="test_alert",
        description="Test alert rule",
        metric_name="test_metric",
        threshold=10.0,
        comparison="gt",
        severity=AlertSeverity.HIGH,
        duration=60,
        labels={"service": "test"},
        annotations={"runbook": "http://example.com"}
    )


@pytest.fixture
def sample_health_check():
    """Fixture providing a sample health check."""
    return HealthCheck(
        name="test_check",
        description="Test health check",
        check_function="test_function",
        timeout=30,
        interval=60,
        enabled=True,
        critical=True,
        labels={"component": "test"}
    )


class TestAlertRule:
    """Test AlertRule functionality."""
    
    def test_alert_rule_creation(self):
        """Test creating alert rule."""
        rule = AlertRule(
            name="cpu_high",
            description="High CPU usage",
            metric_name="cpu_percent",
            threshold=80.0,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            duration=300,
            labels={"env": "prod"},
            annotations={"summary": "CPU is high"}
        )
        
        assert rule.name == "cpu_high"
        assert rule.description == "High CPU usage"
        assert rule.metric_name == "cpu_percent"
        assert rule.threshold == 80.0
        assert rule.comparison == "gt"
        assert rule.severity == AlertSeverity.CRITICAL
        assert rule.duration == 300
        assert rule.labels == {"env": "prod"}
        assert rule.annotations == {"summary": "CPU is high"}
        assert rule.enabled is True


class TestAlert:
    """Test Alert functionality."""
    
    def test_alert_creation(self, sample_alert_rule):
        """Test creating alert."""
        now = datetime.utcnow()
        alert = Alert(
            rule=sample_alert_rule,
            value=15.0,
            started_at=now,
            status=AlertStatus.FIRING,
            labels={"instance": "test"}
        )
        
        assert alert.rule == sample_alert_rule
        assert alert.value == 15.0
        assert alert.started_at == now
        assert alert.status == AlertStatus.FIRING
        assert alert.labels == {"instance": "test"}
        assert alert.resolved_at is None
    
    def test_alert_duration(self, sample_alert_rule):
        """Test alert duration calculation."""
        start_time = datetime.utcnow() - timedelta(minutes=5)
        alert = Alert(
            rule=sample_alert_rule,
            value=15.0,
            started_at=start_time
        )
        
        # Duration should be approximately 5 minutes (300 seconds)
        assert 295 <= alert.duration <= 305
    
    def test_alert_duration_resolved(self, sample_alert_rule):
        """Test alert duration for resolved alert."""
        start_time = datetime.utcnow() - timedelta(minutes=10)
        resolved_time = datetime.utcnow() - timedelta(minutes=5)
        
        alert = Alert(
            rule=sample_alert_rule,
            value=15.0,
            started_at=start_time,
            status=AlertStatus.RESOLVED,
            resolved_at=resolved_time
        )
        
        # Duration should be 5 minutes (time between start and resolved)
        assert 295 <= alert.duration <= 305


class TestHealthCheck:
    """Test HealthCheck functionality."""
    
    def test_health_check_creation(self):
        """Test creating health check."""
        check = HealthCheck(
            name="database",
            description="Database connectivity",
            check_function="check_db",
            timeout=30,
            interval=60,
            enabled=True,
            critical=True,
            labels={"type": "database"}
        )
        
        assert check.name == "database"
        assert check.description == "Database connectivity"
        assert check.check_function == "check_db"
        assert check.timeout == 30
        assert check.interval == 60
        assert check.enabled is True
        assert check.critical is True
        assert check.labels == {"type": "database"}


class TestHealthCheckResult:
    """Test HealthCheckResult functionality."""
    
    def test_health_check_result_creation(self):
        """Test creating health check result."""
        now = datetime.utcnow()
        result = HealthCheckResult(
            name="test_check",
            healthy=True,
            message="All good",
            duration=0.5,
            timestamp=now,
            labels={"env": "test"},
            metadata={"connections": 5}
        )
        
        assert result.name == "test_check"
        assert result.healthy is True
        assert result.message == "All good"
        assert result.duration == 0.5
        assert result.timestamp == now
        assert result.labels == {"env": "test"}
        assert result.metadata == {"connections": 5}


class TestPrometheusExporter:
    """Test PrometheusExporter functionality."""
    
    def test_prometheus_exporter_initialization(self, prometheus_exporter):
        """Test Prometheus exporter initialization."""
        assert prometheus_exporter.registry is not None
        assert len(prometheus_exporter.metrics) > 0
        
        # Check that default metrics are created
        assert "http_requests_total" in prometheus_exporter.metrics
        assert "http_request_duration_seconds" in prometheus_exporter.metrics
        assert "db_query_duration_seconds" in prometheus_exporter.metrics
        assert "system_cpu_usage_percent" in prometheus_exporter.metrics
    
    def test_get_metric(self, prometheus_exporter):
        """Test getting metric by name."""
        metric = prometheus_exporter.get_metric("http_requests_total")
        assert metric is not None
        
        non_existent = prometheus_exporter.get_metric("non_existent_metric")
        assert non_existent is None
    
    def test_add_custom_metric_counter(self, prometheus_exporter):
        """Test adding custom counter metric."""
        metric = prometheus_exporter.add_custom_metric(
            "custom_counter",
            MetricType.COUNTER,
            "Custom counter metric",
            ["label1", "label2"]
        )
        
        assert metric is not None
        assert "custom_counter" in prometheus_exporter.metrics
        
        # Test incrementing the metric
        metric.labels(label1="value1", label2="value2").inc()
    
    def test_add_custom_metric_histogram(self, prometheus_exporter):
        """Test adding custom histogram metric."""
        metric = prometheus_exporter.add_custom_metric(
            "custom_histogram",
            MetricType.HISTOGRAM,
            "Custom histogram metric",
            ["operation"]
        )
        
        assert metric is not None
        assert "custom_histogram" in prometheus_exporter.metrics
        
        # Test observing values
        metric.labels(operation="test").observe(0.5)
    
    def test_add_custom_metric_gauge(self, prometheus_exporter):
        """Test adding custom gauge metric."""
        metric = prometheus_exporter.add_custom_metric(
            "custom_gauge",
            MetricType.GAUGE,
            "Custom gauge metric"
        )
        
        assert metric is not None
        assert "custom_gauge" in prometheus_exporter.metrics
        
        # Test setting gauge value
        metric.set(42)
    
    def test_add_custom_metric_unsupported_type(self, prometheus_exporter):
        """Test adding unsupported metric type."""
        with pytest.raises(MonitoringError, match="Unsupported metric type"):
            prometheus_exporter.add_custom_metric(
                "invalid_metric",
                "invalid_type",  # Not a valid MetricType
                "Invalid metric"
            )
    
    def test_export_metrics(self, prometheus_exporter):
        """Test exporting metrics in Prometheus format."""
        metrics_output = prometheus_exporter.export_metrics()
        
        assert isinstance(metrics_output, str)
        assert len(metrics_output) > 0
        # Should contain metric help and type information
        assert "# HELP" in metrics_output
        assert "# TYPE" in metrics_output


class TestAlertManager:
    """Test AlertManager functionality."""
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert len(alert_manager.rules) == 0
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.notification_handlers) == 0
        assert alert_manager._running is False
    
    def test_add_rule(self, alert_manager, sample_alert_rule):
        """Test adding alert rule."""
        alert_manager.add_rule(sample_alert_rule)
        
        assert sample_alert_rule.name in alert_manager.rules
        assert alert_manager.rules[sample_alert_rule.name] == sample_alert_rule
    
    def test_remove_rule(self, alert_manager, sample_alert_rule):
        """Test removing alert rule."""
        alert_manager.add_rule(sample_alert_rule)
        
        result = alert_manager.remove_rule(sample_alert_rule.name)
        
        assert result is True
        assert sample_alert_rule.name not in alert_manager.rules
    
    def test_remove_nonexistent_rule(self, alert_manager):
        """Test removing non-existent rule."""
        result = alert_manager.remove_rule("nonexistent")
        assert result is False
    
    def test_add_notification_handler(self, alert_manager):
        """Test adding notification handler."""
        handler = AsyncMock()
        alert_manager.add_notification_handler(handler)
        
        assert handler in alert_manager.notification_handlers
    
    @pytest.mark.asyncio
    async def test_start_stop_evaluation(self, alert_manager):
        """Test starting and stopping alert evaluation."""
        # Start evaluation
        await alert_manager.start_evaluation()
        assert alert_manager._running is True
        assert alert_manager._evaluation_task is not None
        
        # Stop evaluation
        await alert_manager.stop_evaluation()
        assert alert_manager._running is False
    
    def test_evaluate_condition_gt(self, alert_manager, sample_alert_rule):
        """Test evaluating greater than condition."""
        sample_alert_rule.comparison = "gt"
        sample_alert_rule.threshold = 10.0
        
        assert alert_manager._evaluate_condition(15.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(5.0, sample_alert_rule) is False
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is False
    
    def test_evaluate_condition_lt(self, alert_manager, sample_alert_rule):
        """Test evaluating less than condition."""
        sample_alert_rule.comparison = "lt"
        sample_alert_rule.threshold = 10.0
        
        assert alert_manager._evaluate_condition(5.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(15.0, sample_alert_rule) is False
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is False
    
    def test_evaluate_condition_eq(self, alert_manager, sample_alert_rule):
        """Test evaluating equal condition."""
        sample_alert_rule.comparison = "eq"
        sample_alert_rule.threshold = 10.0
        
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(5.0, sample_alert_rule) is False
        assert alert_manager._evaluate_condition(15.0, sample_alert_rule) is False
    
    def test_evaluate_condition_gte(self, alert_manager, sample_alert_rule):
        """Test evaluating greater than or equal condition."""
        sample_alert_rule.comparison = "gte"
        sample_alert_rule.threshold = 10.0
        
        assert alert_manager._evaluate_condition(15.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(5.0, sample_alert_rule) is False
    
    def test_evaluate_condition_lte(self, alert_manager, sample_alert_rule):
        """Test evaluating less than or equal condition."""
        sample_alert_rule.comparison = "lte"
        sample_alert_rule.threshold = 10.0
        
        assert alert_manager._evaluate_condition(5.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is True
        assert alert_manager._evaluate_condition(15.0, sample_alert_rule) is False
    
    def test_evaluate_condition_invalid(self, alert_manager, sample_alert_rule):
        """Test evaluating invalid condition."""
        sample_alert_rule.comparison = "invalid"
        
        assert alert_manager._evaluate_condition(10.0, sample_alert_rule) is False
    
    def test_get_active_alerts(self, alert_manager, sample_alert_rule):
        """Test getting active alerts."""
        alert = Alert(
            rule=sample_alert_rule,
            value=15.0,
            started_at=datetime.utcnow()
        )
        alert_manager.active_alerts[sample_alert_rule.name] = alert
        
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0] == alert
    
    def test_get_alert_history(self, alert_manager, sample_alert_rule):
        """Test getting alert history."""
        # Add alerts to history
        old_alert = Alert(
            rule=sample_alert_rule,
            value=15.0,
            started_at=datetime.utcnow() - timedelta(hours=25)
        )
        recent_alert = Alert(
            rule=sample_alert_rule,
            value=20.0,
            started_at=datetime.utcnow() - timedelta(hours=1)
        )
        
        alert_manager.alert_history.append(old_alert)
        alert_manager.alert_history.append(recent_alert)
        
        # Get history for last 24 hours
        history = alert_manager.get_alert_history(24)
        
        assert len(history) == 1
        assert history[0] == recent_alert


class TestHealthChecker:
    """Test HealthChecker functionality."""
    
    def test_health_checker_initialization(self, health_checker):
        """Test health checker initialization."""
        # Should have default checks registered
        assert "database" in health_checker.checks
        assert "redis" in health_checker.checks
        assert "disk_space" in health_checker.checks
        assert "memory" in health_checker.checks
        
        # Should have check functions registered
        assert "database_check" in health_checker.check_functions
        assert "redis_check" in health_checker.check_functions
    
    def test_register_check_function(self, health_checker):
        """Test registering check function."""
        test_func = AsyncMock()
        health_checker.register_check_function("test_check", test_func)
        
        assert "test_check" in health_checker.check_functions
        assert health_checker.check_functions["test_check"] == test_func
    
    def test_add_check(self, health_checker, sample_health_check):
        """Test adding health check."""
        health_checker.add_check(sample_health_check)
        
        assert sample_health_check.name in health_checker.checks
        assert health_checker.checks[sample_health_check.name] == sample_health_check
    
    def test_remove_check(self, health_checker, sample_health_check):
        """Test removing health check."""
        health_checker.add_check(sample_health_check)
        
        result = health_checker.remove_check(sample_health_check.name)
        
        assert result is True
        assert sample_health_check.name not in health_checker.checks
    
    def test_remove_nonexistent_check(self, health_checker):
        """Test removing non-existent check."""
        result = health_checker.remove_check("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_start_stop_checks(self, health_checker):
        """Test starting and stopping health checks."""
        # Start checks
        await health_checker.start_checks()
        assert health_checker._running is True
        assert health_checker._check_task is not None
        
        # Stop checks
        await health_checker.stop_checks()
        assert health_checker._running is False
    
    @pytest.mark.asyncio
    async def test_execute_check_success(self, health_checker):
        """Test executing successful health check."""
        # Mock check function
        async def mock_check(check):
            return HealthCheckResult(
                name=check.name,
                healthy=True,
                message="OK",
                duration=0.1,
                timestamp=datetime.utcnow()
            )
        
        health_checker.register_check_function("mock_check", mock_check)
        
        check = HealthCheck(
            name="test",
            description="Test check",
            check_function="mock_check"
        )
        
        await health_checker._execute_check(check)
        
        assert "test" in health_checker.results
        result = health_checker.results["test"]
        assert result.healthy is True
        assert result.message == "OK"
    
    @pytest.mark.asyncio
    async def test_execute_check_failure(self, health_checker):
        """Test executing failed health check."""
        # Mock failing check function
        async def mock_failing_check(check):
            raise Exception("Check failed")
        
        health_checker.register_check_function("mock_failing_check", mock_failing_check)
        
        check = HealthCheck(
            name="test_fail",
            description="Test failing check",
            check_function="mock_failing_check"
        )
        
        await health_checker._execute_check(check)
        
        assert "test_fail" in health_checker.results
        result = health_checker.results["test_fail"]
        assert result.healthy is False
        assert "Check failed" in result.message
    
    @pytest.mark.asyncio
    async def test_execute_check_timeout(self, health_checker):
        """Test executing health check with timeout."""
        # Mock slow check function
        async def mock_slow_check(check):
            await asyncio.sleep(2)  # Longer than timeout
            return True
        
        health_checker.register_check_function("mock_slow_check", mock_slow_check)
        
        check = HealthCheck(
            name="test_timeout",
            description="Test timeout check",
            check_function="mock_slow_check",
            timeout=1  # 1 second timeout
        )
        
        await health_checker._execute_check(check)
        
        assert "test_timeout" in health_checker.results
        result = health_checker.results["test_timeout"]
        assert result.healthy is False
        assert "timed out" in result.message
    
    @pytest.mark.asyncio
    async def test_execute_check_function_not_found(self, health_checker):
        """Test executing check with non-existent function."""
        check = HealthCheck(
            name="test_missing",
            description="Test missing function",
            check_function="non_existent_function"
        )
        
        await health_checker._execute_check(check)
        
        assert "test_missing" in health_checker.results
        result = health_checker.results["test_missing"]
        assert result.healthy is False
        assert "Check function not found" in result.message
    
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_check_disk_space_healthy(self, mock_disk_usage, health_checker):
        """Test disk space check when healthy."""
        # Mock disk usage with 20% free space
        mock_usage = MagicMock()
        mock_usage.total = 1000
        mock_usage.used = 800
        mock_usage.free = 200
        mock_disk_usage.return_value = mock_usage
        
        check = HealthCheck(name="disk", description="", check_function="")
        result = await health_checker._check_disk_space(check)
        
        assert result.healthy is True
        assert "20.0% free" in result.message
    
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_check_disk_space_unhealthy(self, mock_disk_usage, health_checker):
        """Test disk space check when unhealthy."""
        # Mock disk usage with 5% free space
        mock_usage = MagicMock()
        mock_usage.total = 1000
        mock_usage.used = 950
        mock_usage.free = 50
        mock_disk_usage.return_value = mock_usage
        
        check = HealthCheck(name="disk", description="", check_function="")
        result = await health_checker._check_disk_space(check)
        
        assert result.healthy is False  # Less than 10% free
        assert "5.0% free" in result.message
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_check_memory_healthy(self, mock_memory, health_checker):
        """Test memory check when healthy."""
        # Mock memory with 70% usage
        mock_mem = MagicMock()
        mock_mem.percent = 70.0
        mock_mem.total = 1000
        mock_mem.used = 700
        mock_mem.available = 300
        mock_memory.return_value = mock_mem
        
        check = HealthCheck(name="memory", description="", check_function="")
        result = await health_checker._check_memory(check)
        
        assert result.healthy is True
        assert "70.0% used" in result.message
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_check_memory_unhealthy(self, mock_memory, health_checker):
        """Test memory check when unhealthy."""
        # Mock memory with 95% usage
        mock_mem = MagicMock()
        mock_mem.percent = 95.0
        mock_mem.total = 1000
        mock_mem.used = 950
        mock_mem.available = 50
        mock_memory.return_value = mock_mem
        
        check = HealthCheck(name="memory", description="", check_function="")
        result = await health_checker._check_memory(check)
        
        assert result.healthy is False  # More than 90% used
        assert "95.0% used" in result.message
    
    def test_get_overall_health_healthy(self, health_checker):
        """Test overall health when all checks are healthy."""
        # Add healthy check results
        healthy_result = HealthCheckResult(
            name="test1",
            healthy=True,
            message="OK",
            duration=0.1,
            timestamp=datetime.utcnow()
        )
        health_checker.results["test1"] = healthy_result
        health_checker.checks["test1"] = HealthCheck(
            name="test1", description="", check_function="", critical=True
        )
        
        health = health_checker.get_overall_health()
        
        assert health["status"] == "healthy"
        assert health["healthy_checks"] == 1
        assert health["total_checks"] == 1
    
    def test_get_overall_health_unhealthy_critical(self, health_checker):
        """Test overall health when critical check is failing."""
        # Add unhealthy critical check result
        unhealthy_result = HealthCheckResult(
            name="critical_test",
            healthy=False,
            message="Failed",
            duration=0.1,
            timestamp=datetime.utcnow()
        )
        health_checker.results["critical_test"] = unhealthy_result
        health_checker.checks["critical_test"] = HealthCheck(
            name="critical_test", description="", check_function="", critical=True
        )
        
        health = health_checker.get_overall_health()
        
        assert health["status"] == "unhealthy"
        assert health["healthy_checks"] == 0
        assert health["total_checks"] == 1
    
    def test_get_overall_health_degraded(self, health_checker):
        """Test overall health when non-critical check is failing."""
        # Add healthy critical check
        healthy_result = HealthCheckResult(
            name="critical_test",
            healthy=True,
            message="OK",
            duration=0.1,
            timestamp=datetime.utcnow()
        )
        health_checker.results["critical_test"] = healthy_result
        health_checker.checks["critical_test"] = HealthCheck(
            name="critical_test", description="", check_function="", critical=True
        )
        
        # Add unhealthy non-critical check
        unhealthy_result = HealthCheckResult(
            name="non_critical_test",
            healthy=False,
            message="Failed",
            duration=0.1,
            timestamp=datetime.utcnow()
        )
        health_checker.results["non_critical_test"] = unhealthy_result
        health_checker.checks["non_critical_test"] = HealthCheck(
            name="non_critical_test", description="", check_function="", critical=False
        )
        
        health = health_checker.get_overall_health()
        
        assert health["status"] == "degraded"
        assert health["healthy_checks"] == 1
        assert health["total_checks"] == 2


class TestMonitoringService:
    """Test MonitoringService functionality."""
    
    def test_monitoring_service_initialization(self, monitoring_service):
        """Test monitoring service initialization."""
        assert monitoring_service.prometheus is not None
        assert monitoring_service.alert_manager is not None
        assert monitoring_service.health_checker is not None
        assert monitoring_service._running is False
        
        # Check that default alert rules are loaded
        assert len(monitoring_service.alert_manager.rules) > 0
    
    @pytest.mark.asyncio
    async def test_start_stop_service(self, monitoring_service):
        """Test starting and stopping monitoring service."""
        # Start service
        await monitoring_service.start()
        assert monitoring_service._running is True
        
        # Stop service
        await monitoring_service.stop()
        assert monitoring_service._running is False
    
    def test_record_http_request(self, monitoring_service):
        """Test recording HTTP request metrics."""
        monitoring_service.record_http_request("GET", "/api/test", 200, 0.5)
        
        # Verify metrics were recorded
        requests_counter = monitoring_service.prometheus.get_metric("http_requests_total")
        duration_histogram = monitoring_service.prometheus.get_metric("http_request_duration_seconds")
        
        assert requests_counter is not None
        assert duration_histogram is not None
    
    def test_record_db_query(self, monitoring_service):
        """Test recording database query metrics."""
        monitoring_service.record_db_query("SELECT", "users", 0.1)
        
        # Verify metrics were recorded
        db_histogram = monitoring_service.prometheus.get_metric("db_query_duration_seconds")
        assert db_histogram is not None
    
    def test_record_background_task(self, monitoring_service):
        """Test recording background task metrics."""
        monitoring_service.record_background_task("email_send", "success", 2.5)
        
        # Verify metrics were recorded
        tasks_counter = monitoring_service.prometheus.get_metric("background_tasks_total")
        duration_histogram = monitoring_service.prometheus.get_metric("background_tasks_duration_seconds")
        
        assert tasks_counter is not None
        assert duration_histogram is not None
    
    def test_record_webhook_delivery(self, monitoring_service):
        """Test recording webhook delivery metrics."""
        monitoring_service.record_webhook_delivery("http://example.com/webhook", "success")
        
        # Verify metrics were recorded
        webhook_counter = monitoring_service.prometheus.get_metric("webhook_deliveries_total")
        assert webhook_counter is not None
    
    def test_record_rate_limit_check(self, monitoring_service):
        """Test recording rate limit check metrics."""
        monitoring_service.record_rate_limit_check("api_limit", "allowed")
        
        # Verify metrics were recorded
        rate_limit_counter = monitoring_service.prometheus.get_metric("rate_limit_requests_total")
        assert rate_limit_counter is not None
    
    def test_set_db_connections(self, monitoring_service):
        """Test setting database connection count."""
        monitoring_service.set_db_connections(5)
        
        # Verify metric was set
        db_connections = monitoring_service.prometheus.get_metric("db_connections_active")
        assert db_connections is not None
    
    def test_get_metrics_response(self, monitoring_service):
        """Test getting Prometheus metrics response."""
        response = monitoring_service.get_metrics_response()
        
        assert hasattr(response, 'body') or hasattr(response, 'content')
        assert response.media_type == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_get_health_status(self, monitoring_service):
        """Test getting health status."""
        health_status = monitoring_service.get_health_status()
        
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "checks" in health_status
    
    def test_get_active_alerts(self, monitoring_service):
        """Test getting active alerts."""
        # Add a test alert
        rule = AlertRule(
            name="test_alert",
            description="Test",
            metric_name="test_metric",
            threshold=10.0,
            comparison="gt",
            severity=AlertSeverity.HIGH
        )
        alert = Alert(rule=rule, value=15.0, started_at=datetime.utcnow())
        monitoring_service.alert_manager.active_alerts["test_alert"] = alert
        
        active_alerts = monitoring_service.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0]["name"] == "test_alert"
        assert active_alerts[0]["value"] == 15.0
    
    @pytest.mark.asyncio
    async def test_monitor_operation_success(self, monitoring_service):
        """Test monitoring operation context manager for success."""
        async with monitoring_service.monitor_operation("test_operation", {"env": "test"}):
            await asyncio.sleep(0.1)  # Simulate operation
        
        # Should record success metric
        metric_name = "test_operation_duration_seconds"
        # The metric should be created dynamically
        assert metric_name in monitoring_service.prometheus.metrics
    
    @pytest.mark.asyncio
    async def test_monitor_operation_failure(self, monitoring_service):
        """Test monitoring operation context manager for failure."""
        with pytest.raises(ValueError):
            async with monitoring_service.monitor_operation("test_fail_operation"):
                raise ValueError("Test error")
        
        # Should record error metric
        metric_name = "test_fail_operation_duration_seconds"
        assert metric_name in monitoring_service.prometheus.metrics


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_monitoring_service_singleton(self):
        """Test global service instance is singleton."""
        service1 = get_monitoring_service()
        service2 = get_monitoring_service()
        
        assert service1 is service2
    
    def test_monitor_http_request(self):
        """Test HTTP request monitoring utility."""
        with patch('pcs.services.monitoring_service.get_monitoring_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            monitor_http_request("GET", "/api/test", 200, 0.5)
            
            mock_service.record_http_request.assert_called_once_with("GET", "/api/test", 200, 0.5)
    
    def test_monitor_db_query(self):
        """Test database query monitoring utility."""
        with patch('pcs.services.monitoring_service.get_monitoring_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            monitor_db_query("SELECT", "users", 0.1)
            
            mock_service.record_db_query.assert_called_once_with("SELECT", "users", 0.1)
    
    def test_monitor_background_task(self):
        """Test background task monitoring utility."""
        with patch('pcs.services.monitoring_service.get_monitoring_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            monitor_background_task("email_send", "success", 2.5)
            
            mock_service.record_background_task.assert_called_once_with("email_send", "success", 2.5)
    
    def test_monitor_webhook_delivery(self):
        """Test webhook delivery monitoring utility."""
        with patch('pcs.services.monitoring_service.get_monitoring_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            monitor_webhook_delivery("http://example.com", "success")
            
            mock_service.record_webhook_delivery.assert_called_once_with("http://example.com", "success")
    
    def test_monitor_rate_limit_check(self):
        """Test rate limit monitoring utility."""
        with patch('pcs.services.monitoring_service.get_monitoring_service') as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service
            
            monitor_rate_limit_check("api_limit", "allowed")
            
            mock_service.record_rate_limit_check.assert_called_once_with("api_limit", "allowed")


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows."""
    
    @pytest.mark.asyncio
    async def test_alert_lifecycle(self, alert_manager, sample_alert_rule):
        """Test complete alert lifecycle."""
        # Add rule
        alert_manager.add_rule(sample_alert_rule)
        
        # Start evaluation (but mock the metric fetching)
        with patch.object(alert_manager, '_get_metric_value', return_value=15.0):
            # Fire alert
            await alert_manager._fire_alert(sample_alert_rule, 15.0)
            
            # Check active alerts
            active_alerts = alert_manager.get_active_alerts()
            assert len(active_alerts) == 1
            assert active_alerts[0].rule.name == sample_alert_rule.name
            
            # Resolve alert
            alert_manager._resolve_alert(sample_alert_rule.name)
            
            # Check no active alerts
            active_alerts = alert_manager.get_active_alerts()
            assert len(active_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, health_checker):
        """Test health check integration with monitoring."""
        # Register a custom check
        async def custom_check(check):
            return HealthCheckResult(
                name=check.name,
                healthy=True,
                message="Custom check passed",
                duration=0.1,
                timestamp=datetime.utcnow(),
                metadata={"custom_data": "test"}
            )
        
        health_checker.register_check_function("custom_check", custom_check)
        
        custom_health_check = HealthCheck(
            name="custom",
            description="Custom health check",
            check_function="custom_check",
            critical=False
        )
        
        health_checker.add_check(custom_health_check)
        
        # Execute the check
        await health_checker._execute_check(custom_health_check)
        
        # Verify result
        assert "custom" in health_checker.results
        result = health_checker.results["custom"]
        assert result.healthy is True
        assert result.message == "Custom check passed"
        assert result.metadata["custom_data"] == "test"
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, mock_disk, mock_memory, mock_cpu, monitoring_service):
        """Test system metrics collection."""
        # Mock system calls
        mock_cpu.return_value = 75.0
        
        mock_mem = MagicMock()
        mock_mem.used = 8000000000
        mock_mem.available = 2000000000
        mock_mem.total = 10000000000
        mock_memory.return_value = mock_mem
        
        mock_disk_usage = MagicMock()
        mock_disk_usage.used = 500000000000
        mock_disk_usage.free = 500000000000
        mock_disk_usage.total = 1000000000000
        mock_disk.return_value = mock_disk_usage
        
        # Collect metrics once
        await monitoring_service._collect_system_metrics()
        
        # Verify metrics were set
        cpu_gauge = monitoring_service.prometheus.get_metric("system_cpu_usage_percent")
        memory_gauge = monitoring_service.prometheus.get_metric("system_memory_usage_bytes")
        disk_gauge = monitoring_service.prometheus.get_metric("system_disk_usage_bytes")
        
        assert cpu_gauge is not None
        assert memory_gauge is not None
        assert disk_gauge is not None
    
    @pytest.mark.asyncio
    async def test_notification_handler_integration(self, alert_manager, sample_alert_rule):
        """Test alert notification integration."""
        # Add notification handler
        notifications = []
        
        async def notification_handler(alert):
            notifications.append({
                "alert_name": alert.rule.name,
                "value": alert.value,
                "severity": alert.rule.severity
            })
        
        alert_manager.add_notification_handler(notification_handler)
        alert_manager.add_rule(sample_alert_rule)
        
        # Fire alert
        await alert_manager._fire_alert(sample_alert_rule, 15.0)
        
        # Check notification was sent
        assert len(notifications) == 1
        assert notifications[0]["alert_name"] == sample_alert_rule.name
        assert notifications[0]["value"] == 15.0
        assert notifications[0]["severity"] == AlertSeverity.HIGH
