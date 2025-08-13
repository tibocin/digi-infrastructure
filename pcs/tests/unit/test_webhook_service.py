"""
Filepath: tests/unit/test_webhook_service.py
Purpose: Unit tests for webhook delivery system
Related Components: WebhookService, WebhookDeliveryEngine, Security validation, HTTP delivery
Tags: unit-tests, webhooks, http-delivery, security, async-testing
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

import httpx

from pcs.services.webhook_service import (
    WebhookService,
    WebhookDeliveryEngine,
    WebhookSecurityValidator,
    WebhookEndpoint,
    WebhookPayload,
    WebhookEvent,
    WebhookStatus,
    WebhookError,
    get_webhook_service,
    send_task_completion_webhook,
    send_task_failure_webhook,
    send_custom_webhook
)
from pcs.core.exceptions import PCSError


class TestWebhookSecurityValidator:
    """Test webhook security validation."""
    
    def test_generate_signature(self):
        """Test signature generation."""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        signature = WebhookSecurityValidator.generate_signature(payload, secret)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex length
    
    def test_verify_signature_valid(self):
        """Test signature verification with valid signature."""
        payload = '{"test": "data"}'
        secret = "test_secret"
        
        signature = WebhookSecurityValidator.generate_signature(payload, secret)
        
        assert WebhookSecurityValidator.verify_signature(payload, signature, secret) is True
    
    def test_verify_signature_invalid(self):
        """Test signature verification with invalid signature."""
        payload = '{"test": "data"}'
        secret = "test_secret"
        invalid_signature = "invalid_signature"
        
        assert WebhookSecurityValidator.verify_signature(payload, invalid_signature, secret) is False
    
    def test_validate_url_security_valid(self):
        """Test URL security validation for valid URLs."""
        valid_urls = [
            "https://example.com/webhook",
            "http://api.example.com/hooks",
            "https://webhook.service.com:8080/endpoint"
        ]
        
        for url in valid_urls:
            assert WebhookSecurityValidator.validate_url_security(url) is True
    
    def test_validate_url_security_invalid(self):
        """Test URL security validation for invalid URLs."""
        invalid_urls = [
            "http://localhost/webhook",
            "https://127.0.0.1:8080/hook",
            "http://192.168.1.1/endpoint",
            "https://10.0.0.1/webhook"
        ]
        
        for url in invalid_urls:
            assert WebhookSecurityValidator.validate_url_security(url) is False


class TestWebhookEndpoint:
    """Test WebhookEndpoint model."""
    
    def test_webhook_endpoint_creation(self):
        """Test creating webhook endpoint with valid data."""
        endpoint = WebhookEndpoint(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_FAILED]
        )
        
        assert endpoint.url == "https://example.com/webhook"
        assert len(endpoint.events) == 2
        assert endpoint.active is True
        assert endpoint.timeout == 30
        assert endpoint.max_retries == 3
    
    def test_webhook_endpoint_invalid_url(self):
        """Test webhook endpoint with invalid URL."""
        with pytest.raises(ValueError, match="URL must use http or https scheme"):
            WebhookEndpoint(
                url="ftp://example.com/webhook",
                events=[WebhookEvent.TASK_COMPLETED]
            )
    
    def test_webhook_endpoint_invalid_timeout(self):
        """Test webhook endpoint with invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be between 1 and 300 seconds"):
            WebhookEndpoint(
                url="https://example.com/webhook",
                events=[WebhookEvent.TASK_COMPLETED],
                timeout=500
            )


class TestWebhookPayload:
    """Test WebhookPayload model."""
    
    def test_webhook_payload_creation(self):
        """Test creating webhook payload."""
        payload = WebhookPayload(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123", "result": "success"}
        )
        
        assert payload.event == WebhookEvent.TASK_COMPLETED
        assert payload.data["task_id"] == "123"
        assert payload.source == "pcs"
        assert payload.version == "1.0"
        assert isinstance(payload.timestamp, datetime)
    
    def test_webhook_payload_serialization(self):
        """Test webhook payload serialization."""
        payload = WebhookPayload(
            event=WebhookEvent.CUSTOM,
            data={"custom": "data"}
        )
        
        json_data = payload.model_dump_json()
        assert isinstance(json_data, str)
        
        # Should be able to parse back
        data = json.loads(json_data)
        assert data["event"] == "custom"
        assert data["data"]["custom"] == "data"


@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    redis_mock = AsyncMock()
    
    # Mock basic Redis operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.llen = AsyncMock(return_value=0)
    redis_mock.close = AsyncMock()
    
    return redis_mock


@pytest.fixture
def mock_httpx_client():
    """Fixture providing a mocked HTTPX client."""
    client_mock = AsyncMock(spec=httpx.AsyncClient)
    
    # Mock successful response
    response_mock = AsyncMock()
    response_mock.status_code = 200
    response_mock.text = "OK"
    client_mock.post = AsyncMock(return_value=response_mock)
    client_mock.aclose = AsyncMock()
    
    return client_mock, response_mock


@pytest.fixture
def sample_endpoint():
    """Fixture providing a sample webhook endpoint."""
    return WebhookEndpoint(
        url="https://example.com/webhook",
        events=[WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_FAILED],
        secret="test_secret",
        timeout=30,
        max_retries=3
    )


@pytest.fixture
def sample_payload():
    """Fixture providing a sample webhook payload."""
    return WebhookPayload(
        event=WebhookEvent.TASK_COMPLETED,
        data={"task_id": "test_task", "result": "success"}
    )


class TestWebhookDeliveryEngine:
    """Test webhook delivery engine."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_redis):
        """Test delivery engine initialization."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        await engine.initialize()
        
        mock_redis.ping.assert_called_once()
        assert engine.redis_client == mock_redis
    
    @pytest.mark.asyncio
    async def test_initialization_redis_failure(self):
        """Test initialization with Redis failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with pytest.raises(WebhookError, match="Redis connection failed"):
            await engine.initialize()
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_success(self, mock_redis, sample_endpoint, sample_payload):
        """Test successful webhook delivery."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        # Mock HTTP client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            engine.http_client = mock_client
            
            result = await engine.deliver_webhook(sample_endpoint, sample_payload)
            
            assert result.status == WebhookStatus.DELIVERED
            assert result.status_code == 200
            assert result.delivery_time is not None
            assert result.delivered_at is not None
            mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_with_signature(self, mock_redis, sample_endpoint, sample_payload):
        """Test webhook delivery with signature generation."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            engine.http_client = mock_client
            
            await engine.deliver_webhook(sample_endpoint, sample_payload)
            
            # Check that signature header was added
            call_args = mock_client.post.call_args
            headers = call_args[1]['headers']
            assert 'X-Webhook-Signature' in headers
            assert headers['X-Webhook-Signature'].startswith('sha256=')
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_failure_status(self, mock_redis, sample_endpoint, sample_payload):
        """Test webhook delivery with failure status code."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client
            engine.http_client = mock_client
            
            result = await engine.deliver_webhook(sample_endpoint, sample_payload)
            
            assert result.status == WebhookStatus.FAILED
            assert result.status_code == 500
            assert "HTTP 500" in result.error_message
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_timeout(self, mock_redis, sample_endpoint, sample_payload):
        """Test webhook delivery timeout."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = asyncio.TimeoutError()
            mock_client_class.return_value = mock_client
            engine.http_client = mock_client
            
            result = await engine.deliver_webhook(sample_endpoint, sample_payload)
            
            assert result.status == WebhookStatus.FAILED
            assert "timeout" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_deliver_webhook_security_failure(self, mock_redis, sample_payload):
        """Test webhook delivery with security validation failure."""
        # Create endpoint with localhost URL (should fail security check)
        insecure_endpoint = WebhookEndpoint(
            url="http://localhost:8080/webhook",
            events=[WebhookEvent.TASK_COMPLETED]
        )
        
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        result = await engine.deliver_webhook(insecure_endpoint, sample_payload)
        
        assert result.status == WebhookStatus.FAILED
        assert "security validation" in result.error_message
    
    @pytest.mark.asyncio
    async def test_deliver_with_retries_success_first_attempt(self, mock_redis, sample_endpoint, sample_payload):
        """Test delivery with retries - success on first attempt."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch.object(engine, 'deliver_webhook') as mock_deliver:
            from pcs.services.webhook_service import WebhookDeliveryResult
            mock_deliver.return_value = WebhookDeliveryResult(
                webhook_id="test-webhook",
                status=WebhookStatus.DELIVERED,
                status_code=200
            )
            
            result = await engine.deliver_with_retries(sample_endpoint, sample_payload)
            
            assert result.status == WebhookStatus.DELIVERED
            mock_deliver.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deliver_with_retries_failure_all_attempts(self, mock_redis, sample_endpoint, sample_payload):
        """Test delivery with retries - failure on all attempts."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch.object(engine, 'deliver_webhook') as mock_deliver:
            from pcs.services.webhook_service import WebhookDeliveryResult
            mock_deliver.return_value = WebhookDeliveryResult(
                webhook_id="test-webhook",
                status=WebhookStatus.FAILED,
                error_message="Connection failed"
            )
            
            # Mock sleep to avoid waiting in tests
            with patch('asyncio.sleep', return_value=None):
                result = await engine.deliver_with_retries(sample_endpoint, sample_payload)
            
            assert result.status == WebhookStatus.FAILED
            assert mock_deliver.call_count == sample_endpoint.max_retries
    
    @pytest.mark.asyncio
    async def test_queue_webhook(self, mock_redis, sample_endpoint, sample_payload):
        """Test webhook queueing."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        queue_id = await engine.queue_webhook(sample_endpoint, sample_payload)
        
        assert isinstance(queue_id, str)
        mock_redis.lpush.assert_called_once()
        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_redis):
        """Test delivery engine shutdown."""
        engine = WebhookDeliveryEngine(redis_client=mock_redis)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            engine.http_client = mock_client
            
            await engine.shutdown()
            
            mock_client.aclose.assert_called_once()
            mock_redis.close.assert_called_once()


class TestWebhookService:
    """Test webhook service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_redis):
        """Test webhook service initialization."""
        service = WebhookService(redis_client=mock_redis)
        
        with patch.object(service.delivery_engine, 'initialize') as mock_init:
            await service.initialize()
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_endpoint(self, mock_redis, sample_endpoint):
        """Test endpoint registration."""
        service = WebhookService(redis_client=mock_redis)
        service.redis_client = mock_redis
        
        endpoint_id = await service.register_endpoint(sample_endpoint)
        
        assert endpoint_id == sample_endpoint.id
        assert sample_endpoint.id in service.endpoints
        mock_redis.hset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_inactive_endpoint(self, mock_redis):
        """Test registering inactive endpoint."""
        inactive_endpoint = WebhookEndpoint(
            url="https://example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            active=False
        )
        
        service = WebhookService(redis_client=mock_redis)
        
        with pytest.raises(WebhookError, match="Cannot register inactive endpoint"):
            await service.register_endpoint(inactive_endpoint)
    
    @pytest.mark.asyncio
    async def test_unregister_endpoint(self, mock_redis, sample_endpoint):
        """Test endpoint unregistration."""
        service = WebhookService(redis_client=mock_redis)
        service.redis_client = mock_redis
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        result = await service.unregister_endpoint(sample_endpoint.id)
        
        assert result is True
        assert sample_endpoint.id not in service.endpoints
        mock_redis.hdel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unregister_nonexistent_endpoint(self, mock_redis):
        """Test unregistering non-existent endpoint."""
        service = WebhookService(redis_client=mock_redis)
        
        result = await service.unregister_endpoint("nonexistent")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_endpoint(self, mock_redis, sample_endpoint):
        """Test getting endpoint by ID."""
        service = WebhookService(redis_client=mock_redis)
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        endpoint = await service.get_endpoint(sample_endpoint.id)
        
        assert endpoint == sample_endpoint
    
    @pytest.mark.asyncio
    async def test_list_endpoints(self, mock_redis, sample_endpoint):
        """Test listing all endpoints."""
        service = WebhookService(redis_client=mock_redis)
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        endpoints = await service.list_endpoints()
        
        assert len(endpoints) == 1
        assert endpoints[0] == sample_endpoint
    
    @pytest.mark.asyncio
    async def test_trigger_webhook_with_matching_endpoints(self, mock_redis, sample_endpoint):
        """Test triggering webhook with matching endpoints."""
        service = WebhookService(redis_client=mock_redis)
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        with patch.object(service.delivery_engine, 'queue_webhook') as mock_queue:
            mock_queue.return_value = "queue_id_123"
            
            result = await service.trigger_webhook(
                event=WebhookEvent.TASK_COMPLETED,
                data={"task_id": "123"}
            )
            
            assert len(result) == 1
            assert result[0] == "queue_id_123"
            mock_queue.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_webhook_no_matching_endpoints(self, mock_redis):
        """Test triggering webhook with no matching endpoints."""
        service = WebhookService(redis_client=mock_redis)
        
        result = await service.trigger_webhook(
            event=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123"}
        )
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_trigger_webhook_immediate_delivery(self, mock_redis, sample_endpoint):
        """Test triggering webhook with immediate delivery."""
        service = WebhookService(redis_client=mock_redis)
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        with patch.object(service.delivery_engine, 'deliver_with_retries') as mock_deliver:
            from pcs.services.webhook_service import WebhookDeliveryResult
            mock_deliver.return_value = WebhookDeliveryResult(
                webhook_id="webhook_id_123",
                status=WebhookStatus.DELIVERED
            )
            
            result = await service.trigger_webhook(
                event=WebhookEvent.TASK_COMPLETED,
                data={"task_id": "123"},
                immediate=True
            )
            
            assert len(result) == 1
            assert result[0] == "webhook_id_123"
            mock_deliver.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_webhook_stats(self, mock_redis, sample_endpoint):
        """Test getting webhook statistics."""
        service = WebhookService(redis_client=mock_redis)
        service.redis_client = mock_redis
        service.endpoints[sample_endpoint.id] = sample_endpoint
        
        mock_redis.llen.return_value = 5
        
        stats = await service.get_webhook_stats()
        
        assert stats["total_endpoints"] == 1
        assert stats["active_endpoints"] == 1
        assert stats["queue_length"] == 5
        assert "endpoints_by_event" in stats
    
    @pytest.mark.asyncio
    async def test_load_endpoints_success(self, mock_redis):
        """Test loading endpoints from Redis."""
        service = WebhookService(redis_client=mock_redis)
        service.redis_client = mock_redis
        
        # Mock Redis response
        endpoint_data = {
            "endpoint_1": json.dumps({
                "id": "endpoint_1",
                "url": "https://example.com/webhook",
                "events": ["task.completed"],
                "active": True,
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 60,
                "headers": {},
                "created_at": "2024-01-01T00:00:00"
            })
        }
        mock_redis.hgetall.return_value = endpoint_data
        
        await service._load_endpoints()
        
        assert len(service.endpoints) == 1
        assert "endpoint_1" in service.endpoints
    
    @pytest.mark.asyncio
    async def test_load_endpoints_failure(self, mock_redis):
        """Test loading endpoints with malformed data."""
        service = WebhookService(redis_client=mock_redis)
        service.redis_client = mock_redis
        
        # Mock Redis response with invalid JSON
        mock_redis.hgetall.return_value = {
            "endpoint_1": "invalid_json"
        }
        
        # Should not raise exception, just log error
        await service._load_endpoints()
        
        assert len(service.endpoints) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_redis):
        """Test webhook service shutdown."""
        service = WebhookService(redis_client=mock_redis)
        
        with patch.object(service.delivery_engine, 'shutdown') as mock_shutdown:
            await service.shutdown()
            mock_shutdown.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_webhook_service_singleton(self):
        """Test global webhook service instance is singleton."""
        service1 = get_webhook_service()
        service2 = get_webhook_service()
        
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_send_task_completion_webhook(self):
        """Test task completion webhook utility."""
        with patch('pcs.services.webhook_service.get_webhook_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.trigger_webhook.return_value = ["webhook_id_123"]
            mock_get_service.return_value = mock_service
            
            result = await send_task_completion_webhook("task_123", "success", 1.5)
            
            assert result == ["webhook_id_123"]
            mock_service.trigger_webhook.assert_called_once()
            
            # Check the call arguments
            args, kwargs = mock_service.trigger_webhook.call_args
            assert kwargs["event"] == WebhookEvent.TASK_COMPLETED
            assert kwargs["data"]["task_id"] == "task_123"
            assert kwargs["data"]["result"] == "success"
            assert kwargs["data"]["execution_time"] == 1.5
    
    @pytest.mark.asyncio
    async def test_send_task_failure_webhook(self):
        """Test task failure webhook utility."""
        with patch('pcs.services.webhook_service.get_webhook_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.trigger_webhook.return_value = ["webhook_id_456"]
            mock_get_service.return_value = mock_service
            
            result = await send_task_failure_webhook("task_456", "Connection error", 2.0)
            
            assert result == ["webhook_id_456"]
            mock_service.trigger_webhook.assert_called_once()
            
            # Check the call arguments
            args, kwargs = mock_service.trigger_webhook.call_args
            assert kwargs["event"] == WebhookEvent.TASK_FAILED
            assert kwargs["data"]["task_id"] == "task_456"
            assert kwargs["data"]["error"] == "Connection error"
            assert kwargs["data"]["execution_time"] == 2.0
    
    @pytest.mark.asyncio
    async def test_send_custom_webhook(self):
        """Test custom webhook utility."""
        custom_data = {"custom_field": "custom_value", "number": 42}
        
        with patch('pcs.services.webhook_service.get_webhook_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.trigger_webhook.return_value = ["webhook_id_789"]
            mock_get_service.return_value = mock_service
            
            result = await send_custom_webhook(custom_data)
            
            assert result == ["webhook_id_789"]
            mock_service.trigger_webhook.assert_called_once()
            
            # Check the call arguments
            args, kwargs = mock_service.trigger_webhook.call_args
            assert kwargs["event"] == WebhookEvent.CUSTOM
            assert kwargs["data"] == custom_data
