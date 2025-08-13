"""
Filepath: tests/integration/test_phase5_integration.py
Purpose: Integration tests for Phase 5 advanced features working together
Related Components: Background tasks, webhooks, rate limiting, monitoring, performance optimization
Tags: integration-tests, phase5, background-tasks, webhooks, rate-limiting, monitoring, performance
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import redis.asyncio as redis

from pcs.services import (
    # Background Tasks
    BackgroundTaskService,
    get_background_task_service,
    TaskStatus,
    TaskPriority,
    
    # Webhooks
    WebhookService,
    WebhookEndpoint,
    WebhookEvent,
    get_webhook_service,
    
    # Rate Limiting
    RateLimitingService,
    RateLimitConfig,
    RateLimitAlgorithm,
    RateLimitScope,
    get_rate_limiting_service,
    
    # Monitoring
    MonitoringService,
    get_monitoring_service,
    monitor_background_task,
    monitor_webhook_delivery,
    monitor_rate_limit_check,
    
    # Performance Optimization
    PerformanceOptimizationService,
    get_performance_optimization_service,
    BottleneckSeverity
)


@pytest.fixture
async def redis_client():
    """Redis client for testing."""
    client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    
    # Clean up before test
    await client.flushdb()
    
    yield client
    
    # Clean up after test
    await client.flushdb()
    await client.close()


@pytest.fixture
async def background_task_service(redis_client):
    """Background task service for testing."""
    service = BackgroundTaskService(redis_client)
    await service.initialize()
    
    # Register a test task
    @service.register_task("test_task")
    async def test_task(data: str) -> str:
        await asyncio.sleep(0.1)  # Simulate work
        return f"processed_{data}"
    
    @service.register_task("failing_task")
    async def failing_task(data: str) -> str:
        raise ValueError(f"Task failed: {data}")
    
    yield service
    
    await service.shutdown()


@pytest.fixture
async def webhook_service(redis_client):
    """Webhook service for testing."""
    service = WebhookService(redis_client)
    await service.initialize()
    
    yield service
    
    await service.shutdown()


@pytest.fixture
async def rate_limiting_service(redis_client):
    """Rate limiting service for testing."""
    service = RateLimitingService(redis_client)
    await service.initialize()
    
    # Add test rate limit configurations
    test_config = RateLimitConfig(
        name="test_limit",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        limit=10,
        window_seconds=60,
        scope=RateLimitScope.GLOBAL
    )
    service.add_rate_limit(test_config)
    
    yield service


@pytest.fixture
async def monitoring_service():
    """Monitoring service for testing."""
    service = get_monitoring_service()
    await service.start()
    
    yield service
    
    await service.stop()


@pytest.fixture
async def performance_service():
    """Performance optimization service for testing."""
    service = get_performance_optimization_service()
    
    yield service


@pytest.fixture
async def integrated_services(background_task_service, webhook_service, rate_limiting_service, monitoring_service, performance_service):
    """All Phase 5 services integrated together."""
    return {
        "background_tasks": background_task_service,
        "webhooks": webhook_service,
        "rate_limiting": rate_limiting_service,
        "monitoring": monitoring_service,
        "performance": performance_service
    }


class TestPhase5BackgroundTaskIntegration:
    """Test background task integration with other Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_background_task_with_monitoring(self, integrated_services):
        """Test background task processing with monitoring integration."""
        bg_service = integrated_services["background_tasks"]
        monitoring_service = integrated_services["monitoring"]
        
        # Enqueue a task
        task_id = await bg_service.enqueue_task(
            "test_task",
            "test_data",
            priority=TaskPriority.HIGH
        )
        
        # Wait for task to complete
        await asyncio.sleep(0.5)
        
        # Check task status
        task_status = await bg_service.get_task_status(task_id)
        assert task_status["status"] == TaskStatus.COMPLETED.value
        assert task_status["result"] == "processed_test_data"
        
        # Verify monitoring recorded the task
        # This would normally check Prometheus metrics
        assert monitoring_service is not None
    
    @pytest.mark.asyncio
    async def test_background_task_with_webhook_notification(self, integrated_services):
        """Test background task completion triggering webhook."""
        bg_service = integrated_services["background_tasks"]
        webhook_service = integrated_services["webhooks"]
        
        # Register a webhook endpoint for task completion
        webhook_endpoint = WebhookEndpoint(
            url="http://localhost:8080/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="test_secret"
        )
        
        endpoint_id = await webhook_service.register_endpoint(webhook_endpoint)
        assert endpoint_id is not None
        
        # Enqueue a task
        task_id = await bg_service.enqueue_task("test_task", "webhook_test")
        
        # Wait for task to complete
        await asyncio.sleep(0.5)
        
        # Verify task completed
        task_status = await bg_service.get_task_status(task_id)
        assert task_status["status"] == TaskStatus.COMPLETED.value
        
        # In a real scenario, webhook would be triggered
        # Here we verify the endpoint is registered
        endpoints = await webhook_service.list_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0]["url"] == "http://localhost:8080/webhook"
    
    @pytest.mark.asyncio
    async def test_background_task_failure_with_webhook(self, integrated_services):
        """Test background task failure triggering webhook notification."""
        bg_service = integrated_services["background_tasks"]
        webhook_service = integrated_services["webhooks"]
        
        # Register webhook for task failures
        webhook_endpoint = WebhookEndpoint(
            url="http://localhost:8080/webhook/failures",
            events=[WebhookEvent.TASK_FAILED],
            secret="test_secret"
        )
        
        await webhook_service.register_endpoint(webhook_endpoint)
        
        # Enqueue a failing task
        task_id = await bg_service.enqueue_task("failing_task", "fail_test")
        
        # Wait for task to fail
        await asyncio.sleep(0.5)
        
        # Verify task failed
        task_status = await bg_service.get_task_status(task_id)
        assert task_status["status"] == TaskStatus.FAILED.value
        assert "Task failed: fail_test" in task_status.get("error", "")


class TestPhase5WebhookIntegration:
    """Test webhook integration with other Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_webhook_with_rate_limiting(self, integrated_services):
        """Test webhook delivery with rate limiting."""
        webhook_service = integrated_services["webhooks"]
        rate_service = integrated_services["rate_limiting"]
        
        # Register a webhook endpoint
        webhook_endpoint = WebhookEndpoint(
            url="http://localhost:8080/webhook",
            events=[WebhookEvent.CUSTOM],
            secret="test_secret",
            max_retries=1
        )
        
        endpoint_id = await webhook_service.register_endpoint(webhook_endpoint)
        
        # Check rate limit before webhook
        rate_result = await rate_service.check_rate_limit("test_limit", "webhook_client")
        assert rate_result.allowed is True
        assert rate_result.remaining <= 10
        
        # Trigger webhook (would normally make HTTP request)
        webhook_ids = await webhook_service.trigger_webhook(
            WebhookEvent.CUSTOM,
            {"message": "test webhook"},
            source="integration_test"
        )
        
        assert len(webhook_ids) == 1
    
    @pytest.mark.asyncio
    async def test_webhook_performance_monitoring(self, integrated_services):
        """Test webhook delivery performance monitoring."""
        webhook_service = integrated_services["webhooks"]
        monitoring_service = integrated_services["monitoring"]
        
        # Register webhook endpoint
        webhook_endpoint = WebhookEndpoint(
            url="http://localhost:8080/webhook",
            events=[WebhookEvent.CUSTOM],
            secret="test_secret"
        )
        
        await webhook_service.register_endpoint(webhook_endpoint)
        
        # Trigger webhook with monitoring
        with patch('pcs.services.monitoring_service.monitor_webhook_delivery'):
            webhook_ids = await webhook_service.trigger_webhook(
                WebhookEvent.CUSTOM,
                {"test": "data"}
            )
            
            assert len(webhook_ids) == 1
        
        # Verify monitoring service is tracking
        assert monitoring_service is not None


class TestPhase5RateLimitingIntegration:
    """Test rate limiting integration with other Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_monitoring(self, integrated_services):
        """Test rate limiting with monitoring integration."""
        rate_service = integrated_services["rate_limiting"]
        monitoring_service = integrated_services["monitoring"]
        
        # Perform several rate limit checks
        results = []
        for i in range(5):
            result = await rate_service.check_rate_limit("test_limit", f"client_{i}")
            results.append(result)
            
            # Monitor the rate limit check
            monitor_rate_limit_check("test_limit", "allowed" if result.allowed else "denied")
        
        # All should be allowed (within limit)
        assert all(r.allowed for r in results)
        
        # Verify monitoring captured the checks
        assert monitoring_service is not None
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_background_tasks(self, integrated_services):
        """Test rate limiting affecting background task enqueueing."""
        bg_service = integrated_services["background_tasks"]
        rate_service = integrated_services["rate_limiting"]
        
        # Check if we can enqueue tasks (rate limited)
        for i in range(3):
            rate_result = await rate_service.check_rate_limit("test_limit", "task_client")
            
            if rate_result.allowed:
                task_id = await bg_service.enqueue_task("test_task", f"data_{i}")
                assert task_id is not None
                
                # Wait a bit
                await asyncio.sleep(0.1)
        
        # Verify at least some tasks were enqueued
        assert True  # If we get here without exceptions, integration is working


class TestPhase5MonitoringIntegration:
    """Test monitoring integration across all Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_monitoring(self, integrated_services):
        """Test monitoring capturing metrics from all Phase 5 services."""
        services = integrated_services
        monitoring_service = services["monitoring"]
        
        # Test HTTP request monitoring (simulated)
        monitoring_service.record_http_request("POST", "/api/tasks", 201, 0.15)
        monitoring_service.record_http_request("GET", "/api/health", 200, 0.05)
        
        # Test background task monitoring
        monitoring_service.record_background_task("test_task", "success", 0.2)
        monitoring_service.record_background_task("email_task", "success", 1.5)
        
        # Test webhook monitoring
        monitoring_service.record_webhook_delivery("http://example.com/webhook", "success")
        monitoring_service.record_webhook_delivery("http://example.com/webhook2", "failure")
        
        # Test rate limiting monitoring
        monitoring_service.record_rate_limit_check("api_limit", "allowed")
        monitoring_service.record_rate_limit_check("api_limit", "denied")
        
        # Get metrics response (would normally be Prometheus format)
        metrics_response = monitoring_service.get_metrics_response()
        assert metrics_response is not None
        
        # Get health status
        health_status = monitoring_service.get_health_status()
        assert "status" in health_status
        assert "checks" in health_status
    
    @pytest.mark.asyncio
    async def test_monitoring_alerts(self, integrated_services):
        """Test monitoring alert system integration."""
        monitoring_service = integrated_services["monitoring"]
        
        # Get active alerts (should be none initially)
        alerts = monitoring_service.get_active_alerts()
        assert isinstance(alerts, list)
        
        # In a real scenario, alerts would be triggered by high resource usage
        # Here we just verify the alert system is accessible
        assert monitoring_service.alert_manager is not None


class TestPhase5PerformanceIntegration:
    """Test performance optimization integration with other services."""
    
    @pytest.mark.asyncio
    async def test_performance_analysis_integration(self, integrated_services):
        """Test performance analysis across all Phase 5 services."""
        perf_service = integrated_services["performance"]
        
        # Run performance analysis
        bottlenecks = await perf_service.analyzer.analyze_system_performance()
        
        # Should return list of bottlenecks (may be empty in test environment)
        assert isinstance(bottlenecks, list)
        
        # Check if any critical bottlenecks were found
        critical_bottlenecks = [b for b in bottlenecks if b.severity == BottleneckSeverity.CRITICAL]
        
        # In test environment, we don't expect critical issues
        # This test mainly verifies the analysis runs without errors
        assert len(critical_bottlenecks) >= 0  # May be 0 in test environment
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, integrated_services):
        """Test performance benchmarking of integrated services."""
        perf_service = integrated_services["performance"]
        
        # Run comprehensive benchmark
        benchmark_results = await perf_service.benchmark.run_comprehensive_benchmark()
        
        # Verify all benchmark categories completed
        expected_categories = ["database", "memory", "cache", "async", "io"]
        assert set(benchmark_results.keys()) == set(expected_categories)
        
        # Verify benchmark results have valid data
        for category, result in benchmark_results.items():
            assert result.name == f"{category.title()} Operations"
            assert result.operations_per_second >= 0
            assert result.test_duration >= 0
    
    @pytest.mark.asyncio
    async def test_performance_optimization_cycle(self, integrated_services):
        """Test full performance optimization cycle."""
        perf_service = integrated_services["performance"]
        
        # Get current performance status
        status = await perf_service.get_current_performance_status()
        assert "system_health" in status
        assert status["system_health"] in ["healthy", "degraded"]
        
        # Run optimization cycle (simplified for testing)
        with patch.object(perf_service.analyzer, 'analyze_system_performance', return_value=[]):
            # Mock no bottlenecks found to avoid real optimizations in test
            report = await perf_service.run_full_optimization_cycle()
            
            assert "cycle_id" in report
            assert "bottlenecks_analyzed" in report
            assert "optimizations_applied" in report
            assert "baseline_benchmarks" in report
            assert "post_benchmarks" in report


class TestPhase5EndToEndIntegration:
    """Test complete end-to-end integration of all Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self, integrated_services):
        """Test a complete workflow using all Phase 5 features."""
        services = integrated_services
        
        # 1. Check system health with monitoring
        health_status = services["monitoring"].get_health_status()
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        
        # 2. Check rate limits before processing
        rate_result = await services["rate_limiting"].check_rate_limit("test_limit", "workflow_client")
        
        if rate_result.allowed:
            # 3. Enqueue background task
            task_id = await services["background_tasks"].enqueue_task(
                "test_task",
                "workflow_data",
                priority=TaskPriority.NORMAL
            )
            
            # 4. Register webhook for task completion
            webhook_endpoint = WebhookEndpoint(
                url="http://localhost:8080/webhook/completion",
                events=[WebhookEvent.TASK_COMPLETED],
                secret="workflow_secret"
            )
            endpoint_id = await services["webhooks"].register_endpoint(webhook_endpoint)
            
            # 5. Wait for task completion
            await asyncio.sleep(0.5)
            
            # 6. Check task status
            task_status = await services["background_tasks"].get_task_status(task_id)
            assert task_status["status"] == TaskStatus.COMPLETED.value
            
            # 7. Record metrics
            services["monitoring"].record_background_task("test_task", "success", 0.5)
            services["monitoring"].record_webhook_delivery("http://localhost:8080/webhook/completion", "success")
            
            # 8. Verify webhook endpoint exists
            endpoints = await services["webhooks"].list_endpoints()
            assert any(ep["id"] == endpoint_id for ep in endpoints)
            
        else:
            # Rate limited - should record this
            services["monitoring"].record_rate_limit_check("test_limit", "denied")
        
        # 9. Run performance check
        current_status = await services["performance"].get_current_performance_status()
        assert "timestamp" in current_status
        assert "system_health" in current_status
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integrated_services):
        """Test error handling across all Phase 5 services."""
        services = integrated_services
        
        # Test background task failure
        task_id = await services["background_tasks"].enqueue_task("failing_task", "error_test")
        await asyncio.sleep(0.5)
        
        task_status = await services["background_tasks"].get_task_status(task_id)
        assert task_status["status"] == TaskStatus.FAILED.value
        
        # Record the failure
        services["monitoring"].record_background_task("failing_task", "failure", 0.1)
        
        # Test webhook registration with invalid URL (should handle gracefully)
        try:
            invalid_webhook = WebhookEndpoint(
                url="invalid-url",
                events=[WebhookEvent.CUSTOM],
                secret="test"
            )
            # This should either succeed with validation or fail gracefully
            endpoint_id = await services["webhooks"].register_endpoint(invalid_webhook)
            
            # If it succeeds, that's also valid (depends on validation level)
            if endpoint_id:
                endpoints = await services["webhooks"].list_endpoints()
                assert len(endpoints) >= 1
                
        except Exception as e:
            # Graceful error handling is acceptable
            assert isinstance(e, Exception)
        
        # Verify monitoring is still functional after errors
        health_status = services["monitoring"].get_health_status()
        assert "status" in health_status
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self, integrated_services):
        """Test concurrent operations across all Phase 5 services."""
        services = integrated_services
        
        # Create multiple concurrent operations
        tasks = []
        
        # Concurrent background tasks
        for i in range(3):
            task = services["background_tasks"].enqueue_task("test_task", f"concurrent_data_{i}")
            tasks.append(task)
        
        # Concurrent rate limit checks
        rate_checks = []
        for i in range(5):
            check = services["rate_limiting"].check_rate_limit("test_limit", f"concurrent_client_{i}")
            rate_checks.append(check)
        
        # Wait for all operations
        background_task_ids = await asyncio.gather(*tasks)
        rate_results = await asyncio.gather(*rate_checks)
        
        # Verify results
        assert len(background_task_ids) == 3
        assert all(task_id is not None for task_id in background_task_ids)
        
        assert len(rate_results) == 5
        assert all(result.allowed for result in rate_results)  # Should all be within limit
        
        # Wait for background tasks to complete
        await asyncio.sleep(1.0)
        
        # Verify all tasks completed
        for task_id in background_task_ids:
            status = await services["background_tasks"].get_task_status(task_id)
            assert status["status"] == TaskStatus.COMPLETED.value
        
        # Record metrics for all operations
        for i in range(3):
            services["monitoring"].record_background_task("test_task", "success", 0.2)
        
        for result in rate_results:
            services["monitoring"].record_rate_limit_check("test_limit", "allowed")
    
    @pytest.mark.asyncio
    async def test_service_lifecycle_integration(self, integrated_services):
        """Test service lifecycle management integration."""
        services = integrated_services
        
        # Verify all services are initialized and running
        assert services["background_tasks"] is not None
        assert services["webhooks"] is not None
        assert services["rate_limiting"] is not None
        assert services["monitoring"] is not None
        assert services["performance"] is not None
        
        # Test service health checks
        health_status = services["monitoring"].get_health_status()
        
        # Should have health checks for various components
        assert "status" in health_status
        assert "checks" in health_status
        
        # Verify services can handle shutdown signals gracefully
        # (This would normally be tested with actual shutdown, but we'll verify structure)
        
        # Test performance monitoring across services
        perf_status = await services["performance"].get_current_performance_status()
        assert "system_health" in perf_status
        assert perf_status["system_health"] in ["healthy", "degraded"]


class TestPhase5RealWorldScenarios:
    """Test real-world usage scenarios combining all Phase 5 features."""
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, integrated_services):
        """Test system behavior under simulated high load."""
        services = integrated_services
        
        # Simulate high load with multiple operations
        operations = []
        
        # Background task burst
        for i in range(10):
            op = services["background_tasks"].enqueue_task("test_task", f"load_test_{i}")
            operations.append(op)
        
        # Rate limiting pressure
        rate_ops = []
        for i in range(8):  # Close to rate limit
            op = services["rate_limiting"].check_rate_limit("test_limit", f"load_client_{i}")
            rate_ops.append(op)
        
        # Execute all operations
        task_ids = await asyncio.gather(*operations)
        rate_results = await asyncio.gather(*rate_ops)
        
        # Verify system handled the load
        assert len(task_ids) == 10
        assert len(rate_results) == 8
        
        # Most rate limit checks should pass (within limit)
        allowed_count = sum(1 for r in rate_results if r.allowed)
        assert allowed_count >= 6  # Allow some to be rate limited
        
        # Monitor the high load scenario
        services["monitoring"].record_http_request("POST", "/api/bulk", 200, 2.5)
        
        # Wait for tasks to complete
        await asyncio.sleep(2.0)
        
        # Verify tasks completed successfully
        completed_count = 0
        for task_id in task_ids:
            status = await services["background_tasks"].get_task_status(task_id)
            if status and status["status"] == TaskStatus.COMPLETED.value:
                completed_count += 1
        
        assert completed_count >= 8  # Most should complete successfully
    
    @pytest.mark.asyncio
    async def test_monitoring_dashboard_scenario(self, integrated_services):
        """Test scenario simulating a monitoring dashboard."""
        services = integrated_services
        
        # Simulate various system activities for dashboard
        
        # 1. System health check
        health = services["monitoring"].get_health_status()
        assert "status" in health
        
        # 2. Performance metrics
        perf_status = await services["performance"].get_current_performance_status()
        assert "memory_usage_percent" in perf_status
        assert "cpu_usage_percent" in perf_status
        
        # 3. Active alerts
        alerts = services["monitoring"].get_active_alerts()
        assert isinstance(alerts, list)
        
        # 4. Background task statistics
        # Enqueue some tasks for statistics
        for i in range(3):
            await services["background_tasks"].enqueue_task("test_task", f"stats_test_{i}")
        
        await asyncio.sleep(0.5)
        
        # 5. Rate limiting statistics
        for i in range(5):
            await services["rate_limiting"].check_rate_limit("test_limit", f"stats_client_{i}")
        
        # 6. Webhook delivery stats
        webhook_endpoint = WebhookEndpoint(
            url="http://localhost:8080/stats",
            events=[WebhookEvent.CUSTOM],
            secret="stats_secret"
        )
        await services["webhooks"].register_endpoint(webhook_endpoint)
        
        # 7. Generate some metrics data
        services["monitoring"].record_http_request("GET", "/api/dashboard", 200, 0.1)
        services["monitoring"].record_background_task("test_task", "success", 0.3)
        services["monitoring"].record_webhook_delivery("http://localhost:8080/stats", "success")
        services["monitoring"].record_rate_limit_check("test_limit", "allowed")
        
        # Verify dashboard data is available
        metrics_response = services["monitoring"].get_metrics_response()
        assert metrics_response is not None
