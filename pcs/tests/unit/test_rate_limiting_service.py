"""
Filepath: tests/unit/test_rate_limiting_service.py
Purpose: Unit tests for rate limiting and throttling system
Related Components: RateLimitingService, Rate limiting algorithms, Middleware, Redis
Tags: unit-tests, rate-limiting, algorithms, middleware, async-testing
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from pcs.services.rate_limiting_service import (
    RateLimitingService,
    RateLimitConfig,
    RateLimitResult,
    RateLimitAlgorithm,
    RateLimitScope,
    RateLimitError,
    TokenBucketAlgorithm,
    SlidingWindowAlgorithm,
    FixedWindowAlgorithm,
    LeakyBucketAlgorithm,
    RateLimitMiddleware,
    get_rate_limiting_service,
    check_ip_rate_limit,
    check_user_rate_limit
)
from pcs.core.exceptions import PCSError


@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    redis_mock = AsyncMock()
    
    # Mock basic Redis operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.eval = AsyncMock(return_value=[1, 10, time.time() + 60])  # allowed, remaining, reset_time
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.close = AsyncMock()
    
    return redis_mock


@pytest.fixture
def sample_config():
    """Fixture providing a sample rate limit configuration."""
    return RateLimitConfig(
        name="test_config",
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
        scope=RateLimitScope.PER_IP,
        limit=100,
        window=60
    )


class TestRateLimitConfig:
    """Test RateLimitConfig functionality."""
    
    def test_rate_limit_config_creation(self):
        """Test creating rate limit configuration."""
        config = RateLimitConfig(
            name="test",
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.PER_USER,
            limit=50,
            window=300,
            burst=75,
            enabled=True
        )
        
        assert config.name == "test"
        assert config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET
        assert config.scope == RateLimitScope.PER_USER
        assert config.limit == 50
        assert config.window == 300
        assert config.burst == 75
        assert config.enabled is True
    
    def test_get_cache_key_default(self, sample_config):
        """Test cache key generation with default pattern."""
        key = sample_config.get_cache_key("192.168.1.1")
        expected = "rate_limit:test_config:per_ip:192.168.1.1"
        assert key == expected
    
    def test_get_cache_key_custom_pattern(self):
        """Test cache key generation with custom pattern."""
        config = RateLimitConfig(
            name="custom",
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.PER_API_KEY,
            limit=10,
            window=60,
            key_pattern="custom:api:{identifier}:limit"
        )
        
        key = config.get_cache_key("api_key_123")
        expected = "custom:api:api_key_123:limit"
        assert key == expected


class TestRateLimitResult:
    """Test RateLimitResult functionality."""
    
    def test_rate_limit_result_allowed(self):
        """Test rate limit result when allowed."""
        result = RateLimitResult(
            allowed=True,
            limit=100,
            remaining=95,
            reset_time=time.time() + 60
        )
        
        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 95
        assert result.retry_after is None
    
    def test_rate_limit_result_denied(self):
        """Test rate limit result when denied."""
        result = RateLimitResult(
            allowed=False,
            limit=100,
            remaining=0,
            reset_time=time.time() + 60,
            retry_after=30
        )
        
        assert result.allowed is False
        assert result.limit == 100
        assert result.remaining == 0
        assert result.retry_after == 30


class TestTokenBucketAlgorithm:
    """Test token bucket rate limiting algorithm."""
    
    @pytest.mark.asyncio
    async def test_token_bucket_check_allowed(self, mock_redis, sample_config):
        """Test token bucket algorithm when request is allowed."""
        algorithm = TokenBucketAlgorithm(mock_redis)
        sample_config.algorithm = RateLimitAlgorithm.TOKEN_BUCKET
        sample_config.burst = 150
        
        # Mock Redis returning allowed request
        mock_redis.eval.return_value = [1, 99, time.time() + 60]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is True
        assert result.remaining == 99
        mock_redis.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_token_bucket_check_denied(self, mock_redis, sample_config):
        """Test token bucket algorithm when request is denied."""
        algorithm = TokenBucketAlgorithm(mock_redis)
        sample_config.algorithm = RateLimitAlgorithm.TOKEN_BUCKET
        
        reset_time = time.time() + 30
        mock_redis.eval.return_value = [0, 0, reset_time]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == int(reset_time - time.time())
    
    @pytest.mark.asyncio
    async def test_token_bucket_reset(self, mock_redis):
        """Test token bucket reset."""
        algorithm = TokenBucketAlgorithm(mock_redis)
        
        result = await algorithm.reset_rate_limit("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")


class TestSlidingWindowAlgorithm:
    """Test sliding window rate limiting algorithm."""
    
    @pytest.mark.asyncio
    async def test_sliding_window_check_allowed(self, mock_redis, sample_config):
        """Test sliding window algorithm when request is allowed."""
        algorithm = SlidingWindowAlgorithm(mock_redis)
        
        # Mock Redis returning allowed request
        mock_redis.eval.return_value = [1, 5, time.time() + 60]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is True
        assert result.remaining == 95  # limit - current_count
        mock_redis.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sliding_window_check_denied(self, mock_redis, sample_config):
        """Test sliding window algorithm when request is denied."""
        algorithm = SlidingWindowAlgorithm(mock_redis)
        
        # Mock Redis returning denied request
        mock_redis.eval.return_value = [0, 100, time.time() + 60]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == sample_config.window
    
    @pytest.mark.asyncio
    async def test_sliding_window_reset(self, mock_redis):
        """Test sliding window reset."""
        algorithm = SlidingWindowAlgorithm(mock_redis)
        
        result = await algorithm.reset_rate_limit("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")


class TestFixedWindowAlgorithm:
    """Test fixed window rate limiting algorithm."""
    
    @pytest.mark.asyncio
    async def test_fixed_window_check_allowed(self, mock_redis, sample_config):
        """Test fixed window algorithm when request is allowed."""
        algorithm = FixedWindowAlgorithm(mock_redis)
        
        # Mock Redis returning allowed request
        window_end = time.time() + 30
        mock_redis.eval.return_value = [1, 10, window_end]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is True
        assert result.remaining == 90  # limit - current_count
        assert result.reset_time == window_end
    
    @pytest.mark.asyncio
    async def test_fixed_window_check_denied(self, mock_redis, sample_config):
        """Test fixed window algorithm when request is denied."""
        algorithm = FixedWindowAlgorithm(mock_redis)
        
        # Mock Redis returning denied request
        window_end = time.time() + 30
        mock_redis.eval.return_value = [0, 100, window_end]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == int(window_end - time.time())
    
    @pytest.mark.asyncio
    async def test_fixed_window_reset(self, mock_redis):
        """Test fixed window reset."""
        algorithm = FixedWindowAlgorithm(mock_redis)
        mock_redis.keys.return_value = ["test_key:123", "test_key:456"]
        mock_redis.delete.return_value = 2
        
        result = await algorithm.reset_rate_limit("test_key")
        
        assert result is True
        mock_redis.keys.assert_called_once_with("test_key:*")
        mock_redis.delete.assert_called_once()


class TestLeakyBucketAlgorithm:
    """Test leaky bucket rate limiting algorithm."""
    
    @pytest.mark.asyncio
    async def test_leaky_bucket_check_allowed(self, mock_redis, sample_config):
        """Test leaky bucket algorithm when request is allowed."""
        algorithm = LeakyBucketAlgorithm(mock_redis)
        sample_config.algorithm = RateLimitAlgorithm.LEAKY_BUCKET
        sample_config.burst = 150
        
        # Mock Redis returning allowed request
        mock_redis.eval.return_value = [1, 140, time.time() + 60]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is True
        assert result.remaining == 140
        mock_redis.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_leaky_bucket_check_denied(self, mock_redis, sample_config):
        """Test leaky bucket algorithm when request is denied."""
        algorithm = LeakyBucketAlgorithm(mock_redis)
        sample_config.algorithm = RateLimitAlgorithm.LEAKY_BUCKET
        
        reset_time = time.time() + 30
        mock_redis.eval.return_value = [0, 0, reset_time]
        
        result = await algorithm.check_rate_limit("test_key", sample_config)
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == int(reset_time - time.time())
    
    @pytest.mark.asyncio
    async def test_leaky_bucket_reset(self, mock_redis):
        """Test leaky bucket reset."""
        algorithm = LeakyBucketAlgorithm(mock_redis)
        
        result = await algorithm.reset_rate_limit("test_key")
        
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")


class TestRateLimitingService:
    """Test rate limiting service functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_redis):
        """Test service initialization."""
        service = RateLimitingService(redis_client=mock_redis)
        
        await service.initialize()
        
        mock_redis.ping.assert_called_once()
        assert len(service.algorithms) == 4  # All algorithms initialized
        assert len(service.configs) > 0  # Default configs loaded
    
    @pytest.mark.asyncio
    async def test_initialization_redis_failure(self):
        """Test initialization with Redis failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        service = RateLimitingService(redis_client=mock_redis)
        
        with pytest.raises(RateLimitError, match="Redis connection failed"):
            await service.initialize()
    
    def test_add_rate_limit(self, mock_redis, sample_config):
        """Test adding rate limit configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        
        service.add_rate_limit(sample_config)
        
        assert sample_config.name in service.configs
        assert service.configs[sample_config.name] == sample_config
    
    def test_remove_rate_limit(self, mock_redis, sample_config):
        """Test removing rate limit configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        
        result = service.remove_rate_limit(sample_config.name)
        
        assert result is True
        assert sample_config.name not in service.configs
    
    def test_remove_nonexistent_rate_limit(self, mock_redis):
        """Test removing non-existent rate limit configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        
        result = service.remove_rate_limit("nonexistent")
        
        assert result is False
    
    def test_get_rate_limit(self, mock_redis, sample_config):
        """Test getting rate limit configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        
        config = service.get_rate_limit(sample_config.name)
        
        assert config == sample_config
    
    def test_list_rate_limits(self, mock_redis, sample_config):
        """Test listing rate limit configurations."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        
        configs = service.list_rate_limits()
        
        assert len(configs) == 1
        assert configs[0] == sample_config
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, mock_redis, sample_config):
        """Test checking rate limit when allowed."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        service.algorithms[sample_config.algorithm] = AsyncMock()
        
        # Mock algorithm returning allowed
        expected_result = RateLimitResult(
            allowed=True,
            limit=100,
            remaining=95,
            reset_time=time.time() + 60
        )
        service.algorithms[sample_config.algorithm].check_rate_limit.return_value = expected_result
        
        result = await service.check_rate_limit(sample_config.name, "192.168.1.1")
        
        assert result.allowed is True
        assert result.remaining == 95
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_disabled_config(self, mock_redis, sample_config):
        """Test checking rate limit with disabled configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        sample_config.enabled = False
        service.configs[sample_config.name] = sample_config
        
        result = await service.check_rate_limit(sample_config.name, "192.168.1.1")
        
        assert result.allowed is True
        assert result.limit == sample_config.limit
        assert result.remaining == sample_config.limit
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_config_not_found(self, mock_redis):
        """Test checking rate limit with non-existent configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        
        with pytest.raises(RateLimitError, match="Rate limit configuration 'nonexistent' not found"):
            await service.check_rate_limit("nonexistent", "192.168.1.1")
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_algorithm_failure(self, mock_redis, sample_config):
        """Test checking rate limit when algorithm fails."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        service.algorithms[sample_config.algorithm] = AsyncMock()
        
        # Mock algorithm failure
        service.algorithms[sample_config.algorithm].check_rate_limit.side_effect = Exception("Redis error")
        
        # Should fail open and allow request
        result = await service.check_rate_limit(sample_config.name, "192.168.1.1")
        
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit_success(self, mock_redis, sample_config):
        """Test resetting rate limit successfully."""
        service = RateLimitingService(redis_client=mock_redis)
        service.configs[sample_config.name] = sample_config
        service.algorithms[sample_config.algorithm] = AsyncMock()
        service.algorithms[sample_config.algorithm].reset_rate_limit.return_value = True
        
        result = await service.reset_rate_limit(sample_config.name, "192.168.1.1")
        
        assert result is True
        service.algorithms[sample_config.algorithm].reset_rate_limit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset_rate_limit_config_not_found(self, mock_redis):
        """Test resetting rate limit with non-existent configuration."""
        service = RateLimitingService(redis_client=mock_redis)
        
        result = await service.reset_rate_limit("nonexistent", "192.168.1.1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_rate_limit_stats(self, mock_redis):
        """Test getting rate limit statistics."""
        service = RateLimitingService(redis_client=mock_redis)
        
        # Add some test configs
        config1 = RateLimitConfig("test1", RateLimitAlgorithm.TOKEN_BUCKET, RateLimitScope.PER_IP, 100, 60)
        config2 = RateLimitConfig("test2", RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.PER_USER, 200, 60, enabled=False)
        service.configs["test1"] = config1
        service.configs["test2"] = config2
        
        stats = await service.get_rate_limit_stats()
        
        assert stats["total_configs"] == 2
        assert stats["enabled_configs"] == 1
        assert len(stats["algorithms_used"]) == 2
        assert len(stats["configs"]) == 2
    
    @pytest.mark.asyncio
    async def test_shutdown(self, mock_redis):
        """Test service shutdown."""
        service = RateLimitingService(redis_client=mock_redis)
        
        await service.shutdown()
        
        mock_redis.close.assert_called_once()


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""
    
    def test_middleware_initialization(self, mock_redis):
        """Test middleware initialization."""
        app = FastAPI()
        service = RateLimitingService(redis_client=mock_redis)
        
        middleware = RateLimitMiddleware(app, service)
        
        assert middleware.rate_limit_service == service
        assert len(middleware.config_mapping) > 0
    
    def test_get_config_for_request_exact_match(self, mock_redis):
        """Test getting configuration for exact path match."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.url.path = "/api/v1/prompts/generate"
        
        config_name = middleware._get_config_for_request(request)
        
        assert config_name == "per_endpoint_heavy"
    
    def test_get_config_for_request_prefix_match(self, mock_redis):
        """Test getting configuration for prefix match."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.url.path = "/api/v1/contexts/list"
        
        config_name = middleware._get_config_for_request(request)
        
        # The path "/api/v1/contexts/list" matches "/api/v1/" prefix first, so returns per_ip_api
        assert config_name == "per_ip_api"
    
    def test_get_config_for_request_v1_fallback(self, mock_redis):
        """Test getting configuration for v1 API fallback."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.url.path = "/api/v1/unknown"
        
        config_name = middleware._get_config_for_request(request)
        
        # The path "/api/v1/unknown" matches "/api/v1/" prefix first, so returns per_ip_api
        assert config_name == "per_ip_api"
    
    def test_get_config_for_request_global_api(self, mock_redis):
        """Test getting configuration for global API fallback."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.url.path = "/api/v2/unknown"
        
        config_name = middleware._get_config_for_request(request)
        
        # The path "/api/v2/unknown" doesn't match v1 prefixes, so falls back to global_api
        assert config_name == "global_api"
    
    def test_get_config_for_request_no_match(self, mock_redis):
        """Test getting configuration when no match."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.url.path = "/health"
        
        config_name = middleware._get_config_for_request(request)
        
        assert config_name is None
    
    def test_get_client_ip_forwarded_for(self, mock_redis):
        """Test getting client IP from X-Forwarded-For header."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.side_effect = lambda h: "192.168.1.1, 10.0.0.1" if h == "X-Forwarded-For" else None
        
        ip = middleware._get_client_ip(request)
        
        assert ip == "192.168.1.1"
    
    def test_get_client_ip_real_ip(self, mock_redis):
        """Test getting client IP from X-Real-IP header."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.side_effect = lambda h: "192.168.1.1" if h == "X-Real-IP" else None
        
        ip = middleware._get_client_ip(request)
        
        assert ip == "192.168.1.1"
    
    def test_get_client_ip_direct(self, mock_redis):
        """Test getting client IP directly."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.return_value = None
        request.client.host = "192.168.1.1"
        
        ip = middleware._get_client_ip(request)
        
        assert ip == "192.168.1.1"
    
    @pytest.mark.asyncio
    async def test_get_user_id_with_auth(self, mock_redis):
        """Test getting user ID from authorization header."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.side_effect = lambda h: "Bearer token123" if h == "Authorization" else None
        
        user_id = await middleware._get_user_id(request)
        
        assert user_id == "user_placeholder"
    
    @pytest.mark.asyncio
    async def test_get_user_id_no_auth(self, mock_redis):
        """Test getting user ID without authorization header."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.return_value = None
        
        user_id = await middleware._get_user_id(request)
        
        assert user_id is None
    
    @pytest.mark.asyncio
    async def test_get_api_key(self, mock_redis):
        """Test getting API key from header."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request
        request = MagicMock()
        request.headers.get.side_effect = lambda h: "api_key_123" if h == "X-API-Key" else None
        
        api_key = await middleware._get_api_key(request)
        
        assert api_key == "api_key_123"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_rate_limiting_service_singleton(self):
        """Test global service instance is singleton."""
        service1 = get_rate_limiting_service()
        service2 = get_rate_limiting_service()
        
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_check_ip_rate_limit(self):
        """Test IP rate limit utility function."""
        with patch('pcs.services.rate_limiting_service.get_rate_limiting_service') as mock_get_service:
            mock_service = AsyncMock()
            # get_rate_limit is synchronous, not async
            mock_service.get_rate_limit = MagicMock(return_value=None)  # Force config creation
            mock_service.add_rate_limit = MagicMock()
            mock_service.check_rate_limit.return_value = RateLimitResult(
                allowed=True, limit=100, remaining=99, reset_time=time.time() + 60
            )
            mock_get_service.return_value = mock_service
            
            result = await check_ip_rate_limit("192.168.1.1", 100, 60)
            
            assert result.allowed is True
            mock_service.add_rate_limit.assert_called_once()
            mock_service.check_rate_limit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_user_rate_limit(self):
        """Test user rate limit utility function."""
        with patch('pcs.services.rate_limiting_service.get_rate_limiting_service') as mock_get_service:
            mock_service = AsyncMock()
            # get_rate_limit is synchronous, not async
            mock_service.get_rate_limit = MagicMock(return_value=None)  # Force config creation
            mock_service.add_rate_limit = MagicMock()
            mock_service.check_rate_limit.return_value = RateLimitResult(
                allowed=True, limit=500, remaining=499, reset_time=time.time() + 60
            )
            mock_get_service.return_value = mock_service
            
            result = await check_user_rate_limit("user_123", 500, 60)
            
            assert result.allowed is True
            mock_service.add_rate_limit.assert_called_once()
            mock_service.check_rate_limit.assert_called_once()


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_multiple_algorithms_same_service(self, mock_redis):
        """Test using multiple algorithms in the same service."""
        service = RateLimitingService(redis_client=mock_redis)
        await service.initialize()
        
        # Add configs with different algorithms
        token_config = RateLimitConfig("token", RateLimitAlgorithm.TOKEN_BUCKET, RateLimitScope.PER_IP, 10, 60)
        sliding_config = RateLimitConfig("sliding", RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.PER_USER, 20, 60)
        
        service.add_rate_limit(token_config)
        service.add_rate_limit(sliding_config)
        
        # Mock different results for each algorithm
        mock_redis.eval.side_effect = [
            [1, 9, time.time() + 60],  # Token bucket result
            [1, 5, time.time() + 60]   # Sliding window result
        ]
        
        result1 = await service.check_rate_limit("token", "192.168.1.1")
        result2 = await service.check_rate_limit("sliding", "user_123")
        
        assert result1.allowed is True
        assert result1.remaining == 9
        assert result2.allowed is True
        assert result2.remaining == 15  # 20 - 5
    
    @pytest.mark.asyncio
    async def test_rate_limit_burst_behavior(self, mock_redis):
        """Test burst behavior in token bucket algorithm."""
        service = RateLimitingService(redis_client=mock_redis)
        await service.initialize()
        
        config = RateLimitConfig(
            "burst_test",
            RateLimitAlgorithm.TOKEN_BUCKET,
            RateLimitScope.PER_IP,
            limit=10,
            window=60,
            burst=20
        )
        service.add_rate_limit(config)
        
        # Simulate burst consumption
        mock_redis.eval.return_value = [1, 15, time.time() + 60]  # 5 tokens consumed from burst
        
        result = await service.check_rate_limit("burst_test", "192.168.1.1")
        
        assert result.allowed is True
        assert result.remaining == 15  # Still have tokens from burst capacity
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_checks(self, mock_redis):
        """Test concurrent rate limit checks."""
        service = RateLimitingService(redis_client=mock_redis)
        await service.initialize()
        
        config = RateLimitConfig("concurrent", RateLimitAlgorithm.SLIDING_WINDOW, RateLimitScope.PER_IP, 5, 60)
        service.add_rate_limit(config)
        
        # Mock decreasing remaining counts
        mock_redis.eval.side_effect = [
            [1, 1, time.time() + 60],  # 1st request
            [1, 2, time.time() + 60],  # 2nd request
            [1, 3, time.time() + 60],  # 3rd request
        ]
        
        # Run concurrent checks
        tasks = [
            service.check_rate_limit("concurrent", "192.168.1.1"),
            service.check_rate_limit("concurrent", "192.168.1.1"),
            service.check_rate_limit("concurrent", "192.168.1.1")
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert all(result.allowed for result in results)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_rate_limit_key_patterns(self, mock_redis):
        """Test custom key patterns."""
        service = RateLimitingService(redis_client=mock_redis)
        await service.initialize()
        
        config = RateLimitConfig(
            "custom_key",
            RateLimitAlgorithm.FIXED_WINDOW,
            RateLimitScope.PER_API_KEY,
            limit=100,
            window=3600,
            key_pattern="api:v2:{identifier}:hourly"
        )
        service.add_rate_limit(config)
        
        mock_redis.eval.return_value = [1, 10, time.time() + 3600]
        
        await service.check_rate_limit("custom_key", "key_abc123")
        
        # Verify the algorithm was called with the custom key pattern
        call_args = mock_redis.eval.call_args[0]
        # The actual key would be passed to the Lua script
        assert "api:v2:key_abc123:hourly" in str(call_args)
