"""
Filepath: src/pcs/services/rate_limiting_service.py
Purpose: Rate limiting and throttling system with multiple algorithms and monitoring
Related Components: Redis, FastAPI middleware, Configuration, Metrics
Tags: rate-limiting, throttling, redis, middleware, algorithms, monitoring
"""

import asyncio
import time
import json
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import redis.asyncio as aioredis
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from pcs.core.config import get_settings
from pcs.core.exceptions import PCSError
from pcs.utils.logger import get_logger
from pcs.utils.metrics import get_metrics_collector, record_manual_metric

logger = get_logger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(str, Enum):
    """Rate limiting scopes."""
    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    PER_ENDPOINT = "per_endpoint"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    name: str
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    limit: int  # requests per window
    window: int  # window size in seconds
    burst: Optional[int] = None  # burst capacity for token bucket
    key_pattern: Optional[str] = None  # custom key pattern
    enabled: bool = True
    
    def get_cache_key(self, identifier: str) -> str:
        """Generate cache key for rate limit."""
        if self.key_pattern:
            return self.key_pattern.format(identifier=identifier)
        return f"rate_limit:{self.name}:{self.scope.value}:{identifier}"


@dataclass
class RateLimitResult:
    """Rate limiting result."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: float  # Unix timestamp
    retry_after: Optional[int] = None  # seconds to wait


class RateLimitError(PCSError):
    """Rate limiting related errors."""
    pass


class RateLimitAlgorithmBase(ABC):
    """Base class for rate limiting algorithms."""
    
    def __init__(self, redis_client: aioredis.Redis):
        """Initialize algorithm with Redis client."""
        self.redis = redis_client
    
    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """
        Check if request should be rate limited.
        
        Args:
            key: Rate limit key
            config: Rate limit configuration
            current_time: Current timestamp (optional)
            
        Returns:
            Rate limit result
        """
        pass
    
    @abstractmethod
    async def reset_rate_limit(self, key: str) -> bool:
        """
        Reset rate limit for key.
        
        Args:
            key: Rate limit key
            
        Returns:
            True if reset successful
        """
        pass


class TokenBucketAlgorithm(RateLimitAlgorithmBase):
    """
    Token bucket rate limiting algorithm.
    
    Allows burst traffic up to bucket capacity, then limits to steady rate.
    """
    
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        if current_time is None:
            current_time = time.time()
        
        bucket_capacity = config.burst or config.limit
        refill_rate = config.limit / config.window  # tokens per second
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local tokens_requested = tonumber(ARGV[4])
        
        local bucket = redis.call('hmget', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Calculate tokens to add based on time elapsed
        local time_elapsed = current_time - last_refill
        local tokens_to_add = time_elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        local allowed = tokens >= tokens_requested
        if allowed then
            tokens = tokens - tokens_requested
        end
        
        -- Update bucket state
        redis.call('hmset', key, 'tokens', tokens, 'last_refill', current_time)
        redis.call('expire', key, math.ceil(capacity / refill_rate) + 60)
        
        return {allowed and 1 or 0, tokens, current_time + (capacity - tokens) / refill_rate}
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            bucket_capacity,
            refill_rate,
            current_time,
            1  # requesting 1 token
        )
        
        allowed, remaining, reset_time = result
        
        return RateLimitResult(
            allowed=bool(allowed),
            limit=config.limit,
            remaining=int(remaining),
            reset_time=reset_time,
            retry_after=int(reset_time - current_time) if not allowed else None
        )
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset token bucket."""
        result = await self.redis.delete(key)
        return result > 0


class SlidingWindowAlgorithm(RateLimitAlgorithmBase):
    """
    Sliding window rate limiting algorithm.
    
    Maintains precise request count over a sliding time window.
    """
    
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        if current_time is None:
            current_time = time.time()
        
        window_start = current_time - config.window
        
        # Lua script for atomic sliding window operation
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local current_time = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window_size = tonumber(ARGV[4])
        
        -- Remove expired entries
        redis.call('zremrangebyscore', key, '-inf', window_start)
        
        -- Count current requests in window
        local current_count = redis.call('zcard', key)
        
        local allowed = current_count < limit
        if allowed then
            -- Add current request
            redis.call('zadd', key, current_time, current_time)
            current_count = current_count + 1
        end
        
        -- Set expiration
        redis.call('expire', key, window_size + 60)
        
        return {allowed and 1 or 0, current_count, current_time + window_size}
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            window_start,
            current_time,
            config.limit,
            config.window
        )
        
        allowed, current_count, reset_time = result
        remaining = max(0, config.limit - current_count)
        
        return RateLimitResult(
            allowed=bool(allowed),
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=config.window if not allowed else None
        )
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset sliding window."""
        result = await self.redis.delete(key)
        return result > 0


class FixedWindowAlgorithm(RateLimitAlgorithmBase):
    """
    Fixed window rate limiting algorithm.
    
    Resets counter at fixed intervals.
    """
    
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        if current_time is None:
            current_time = time.time()
        
        # Calculate window start
        window_start = int(current_time // config.window) * config.window
        window_key = f"{key}:{window_start}"
        
        # Lua script for atomic fixed window operation
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_size = tonumber(ARGV[2])
        local window_end = tonumber(ARGV[3])
        
        local current_count = tonumber(redis.call('get', key)) or 0
        local allowed = current_count < limit
        
        if allowed then
            current_count = redis.call('incr', key)
            if current_count == 1 then
                redis.call('expire', key, window_size)
            end
        end
        
        return {allowed and 1 or 0, current_count, window_end}
        """
        
        window_end = window_start + config.window
        
        result = await self.redis.eval(
            lua_script,
            1,
            window_key,
            config.limit,
            config.window,
            window_end
        )
        
        allowed, current_count, reset_time = result
        remaining = max(0, config.limit - current_count)
        
        return RateLimitResult(
            allowed=bool(allowed),
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(reset_time - current_time) if not allowed else None
        )
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset fixed window."""
        # For fixed window, we need to find and delete all window keys
        # This is a simplified implementation
        pattern = f"{key}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            result = await self.redis.delete(*keys)
            return result > 0
        return True


class LeakyBucketAlgorithm(RateLimitAlgorithmBase):
    """
    Leaky bucket rate limiting algorithm.
    
    Processes requests at a steady rate, dropping excess requests.
    """
    
    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """Check rate limit using leaky bucket algorithm."""
        if current_time is None:
            current_time = time.time()
        
        bucket_capacity = config.burst or config.limit
        leak_rate = config.limit / config.window  # requests per second
        
        # Lua script for atomic leaky bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local leak_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        
        local bucket = redis.call('hmget', key, 'volume', 'last_leak')
        local volume = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or current_time
        
        -- Calculate volume leaked since last check
        local time_elapsed = current_time - last_leak
        local leaked_volume = time_elapsed * leak_rate
        volume = math.max(0, volume - leaked_volume)
        
        local allowed = volume < capacity
        if allowed then
            volume = volume + 1
        end
        
        -- Update bucket state
        redis.call('hmset', key, 'volume', volume, 'last_leak', current_time)
        redis.call('expire', key, math.ceil(capacity / leak_rate) + 60)
        
        local time_to_leak = volume / leak_rate
        return {allowed and 1 or 0, capacity - volume, current_time + time_to_leak}
        """
        
        result = await self.redis.eval(
            lua_script,
            1,
            key,
            bucket_capacity,
            leak_rate,
            current_time
        )
        
        allowed, remaining, reset_time = result
        
        return RateLimitResult(
            allowed=bool(allowed),
            limit=config.limit,
            remaining=int(remaining),
            reset_time=reset_time,
            retry_after=int(reset_time - current_time) if not allowed else None
        )
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset leaky bucket."""
        result = await self.redis.delete(key)
        return result > 0


class RateLimitingService:
    """
    Rate limiting service with multiple algorithms and configurations.
    
    Features:
    - Multiple rate limiting algorithms
    - Configurable rate limits per scope
    - Redis-based distributed rate limiting
    - Monitoring and metrics integration
    - Middleware for FastAPI integration
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize rate limiting service.
        
        Args:
            redis_client: Redis client instance (optional)
        """
        self.settings = get_settings()
        self.redis_client = redis_client
        self.metrics_collector = get_metrics_collector()
        
        # Rate limit configurations
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Algorithm implementations
        self.algorithms: Dict[RateLimitAlgorithm, RateLimitAlgorithmBase] = {}
        
        logger.info("Rate limiting service initialized")
    
    async def initialize(self) -> None:
        """Initialize Redis connection and algorithms."""
        if not self.redis_client:
            self.redis_client = aioredis.from_url(
                self.settings.redis.url,
                decode_responses=True,
                max_connections=self.settings.redis.max_connections,
                socket_timeout=self.settings.redis.socket_timeout,
                socket_connect_timeout=self.settings.redis.connection_timeout
            )
        
        try:
            await self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise RateLimitError("Redis connection failed for rate limiting")
        
        # Initialize algorithms
        self.algorithms = {
            RateLimitAlgorithm.TOKEN_BUCKET: TokenBucketAlgorithm(self.redis_client),
            RateLimitAlgorithm.SLIDING_WINDOW: SlidingWindowAlgorithm(self.redis_client),
            RateLimitAlgorithm.FIXED_WINDOW: FixedWindowAlgorithm(self.redis_client),
            RateLimitAlgorithm.LEAKY_BUCKET: LeakyBucketAlgorithm(self.redis_client),
        }
        
        # Load default configurations
        await self._load_default_configs()
        
        logger.info("Rate limiting service initialization complete")
    
    async def shutdown(self) -> None:
        """Shutdown the rate limiting service."""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Rate limiting service shutdown complete")
    
    def add_rate_limit(self, config: RateLimitConfig) -> None:
        """
        Add a rate limit configuration.
        
        Args:
            config: Rate limit configuration
        """
        self.configs[config.name] = config
        logger.info(f"Added rate limit config: {config.name}")
    
    def remove_rate_limit(self, name: str) -> bool:
        """
        Remove a rate limit configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            True if configuration was removed
        """
        if name in self.configs:
            del self.configs[name]
            logger.info(f"Removed rate limit config: {name}")
            return True
        return False
    
    def get_rate_limit(self, name: str) -> Optional[RateLimitConfig]:
        """Get rate limit configuration by name."""
        return self.configs.get(name)
    
    def list_rate_limits(self) -> List[RateLimitConfig]:
        """List all rate limit configurations."""
        return list(self.configs.values())
    
    async def check_rate_limit(
        self,
        config_name: str,
        identifier: str,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """
        Check rate limit for a request.
        
        Args:
            config_name: Name of rate limit configuration
            identifier: Request identifier (IP, user ID, etc.)
            current_time: Current timestamp (optional)
            
        Returns:
            Rate limit result
        """
        config = self.configs.get(config_name)
        if not config:
            raise RateLimitError(f"Rate limit configuration '{config_name}' not found")
        
        if not config.enabled:
            # Rate limiting disabled for this config
            return RateLimitResult(
                allowed=True,
                limit=config.limit,
                remaining=config.limit,
                reset_time=time.time() + config.window
            )
        
        algorithm = self.algorithms.get(config.algorithm)
        if not algorithm:
            raise RateLimitError(f"Rate limit algorithm '{config.algorithm}' not implemented")
        
        cache_key = config.get_cache_key(identifier)
        
        try:
            result = await algorithm.check_rate_limit(cache_key, config, current_time)
            
            # Record metrics
            record_manual_metric(
                query_type="rate_limit_check",
                execution_time=0.0,
                rows_affected=1 if not result.allowed else 0,
                table_name="rate_limits"
            )
            
            if not result.allowed:
                logger.debug(f"Rate limit exceeded for {config_name}:{identifier}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                limit=config.limit,
                remaining=config.limit,
                reset_time=time.time() + config.window
            )
    
    async def reset_rate_limit(
        self,
        config_name: str,
        identifier: str
    ) -> bool:
        """
        Reset rate limit for identifier.
        
        Args:
            config_name: Name of rate limit configuration
            identifier: Request identifier
            
        Returns:
            True if reset successful
        """
        config = self.configs.get(config_name)
        if not config:
            return False
        
        algorithm = self.algorithms.get(config.algorithm)
        if not algorithm:
            return False
        
        cache_key = config.get_cache_key(identifier)
        
        try:
            result = await algorithm.reset_rate_limit(cache_key)
            logger.info(f"Reset rate limit for {config_name}:{identifier}")
            return result
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
    
    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        stats = {
            "total_configs": len(self.configs),
            "enabled_configs": len([c for c in self.configs.values() if c.enabled]),
            "algorithms_used": list(set(c.algorithm for c in self.configs.values())),
            "scopes_used": list(set(c.scope for c in self.configs.values())),
            "configs": []
        }
        
        for config in self.configs.values():
            config_stats = {
                "name": config.name,
                "algorithm": config.algorithm.value,
                "scope": config.scope.value,
                "limit": config.limit,
                "window": config.window,
                "enabled": config.enabled
            }
            stats["configs"].append(config_stats)
        
        return stats
    
    async def _load_default_configs(self) -> None:
        """Load default rate limit configurations."""
        default_configs = [
            RateLimitConfig(
                name="global_api",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
                limit=1000,
                window=60  # 1000 requests per minute globally
            ),
            RateLimitConfig(
                name="per_ip_api",
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                scope=RateLimitScope.PER_IP,
                limit=100,
                window=60,
                burst=150  # 100 req/min with burst of 150
            ),
            RateLimitConfig(
                name="per_user_api",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                scope=RateLimitScope.PER_USER,
                limit=500,
                window=60  # 500 requests per minute per user
            ),
            RateLimitConfig(
                name="per_endpoint_heavy",
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                scope=RateLimitScope.PER_ENDPOINT,
                limit=10,
                window=60,
                burst=20  # Heavy endpoints: 10 req/min with burst of 20
            )
        ]
        
        for config in default_configs:
            self.add_rate_limit(config)
        
        logger.info(f"Loaded {len(default_configs)} default rate limit configurations")


# Global rate limiting service instance
_rate_limiting_service: Optional[RateLimitingService] = None


def get_rate_limiting_service() -> RateLimitingService:
    """Get the global rate limiting service instance."""
    global _rate_limiting_service
    if _rate_limiting_service is None:
        _rate_limiting_service = RateLimitingService()
    return _rate_limiting_service


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Applies rate limits based on configured rules and adds headers.
    """
    
    def __init__(self, app, rate_limit_service: Optional[RateLimitingService] = None):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            rate_limit_service: Rate limiting service instance
        """
        super().__init__(app)
        self.rate_limit_service = rate_limit_service or get_rate_limiting_service()
        self.config_mapping = {
            # Map endpoint patterns to rate limit configurations
            "/api/v1/": "per_ip_api",
            "/api/v1/prompts/generate": "per_endpoint_heavy",
            "/api/v1/contexts/": "per_user_api",
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Determine rate limit configuration
        config_name = self._get_config_for_request(request)
        if not config_name:
            # No rate limiting for this endpoint
            return await call_next(request)
        
        # Extract identifier based on scope
        identifier = await self._get_identifier(request, config_name)
        if not identifier:
            # Cannot identify request, skip rate limiting
            return await call_next(request)
        
        try:
            # Check rate limit
            result = await self.rate_limit_service.check_rate_limit(
                config_name, identifier
            )
            
            # Create response with rate limit headers
            if result.allowed:
                response = await call_next(request)
            else:
                # Rate limit exceeded
                response = Response(
                    content=json.dumps({
                        "error": "Rate limit exceeded",
                        "retry_after": result.retry_after
                    }),
                    status_code=429,
                    media_type="application/json"
                )
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(result.reset_time))
            
            if result.retry_after:
                response.headers["Retry-After"] = str(result.retry_after)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Fail open - allow request if rate limiting fails
            return await call_next(request)
    
    def _get_config_for_request(self, request: Request) -> Optional[str]:
        """Determine rate limit configuration for request."""
        path = request.url.path
        
        # Check exact matches first
        if path in self.config_mapping:
            return self.config_mapping[path]
        
        # Check prefix matches
        for pattern, config_name in self.config_mapping.items():
            if path.startswith(pattern):
                return config_name
        
        # Default to global API rate limit
        if path.startswith("/api/"):
            return "global_api"
        
        return None
    
    async def _get_identifier(self, request: Request, config_name: str) -> Optional[str]:
        """Extract identifier for rate limiting."""
        config = self.rate_limit_service.get_rate_limit(config_name)
        if not config:
            return None
        
        if config.scope == RateLimitScope.GLOBAL:
            return "global"
        elif config.scope == RateLimitScope.PER_IP:
            return self._get_client_ip(request)
        elif config.scope == RateLimitScope.PER_USER:
            # Extract user ID from auth header or session
            return await self._get_user_id(request)
        elif config.scope == RateLimitScope.PER_API_KEY:
            return await self._get_api_key(request)
        elif config.scope == RateLimitScope.PER_ENDPOINT:
            return f"{request.method}:{request.url.path}"
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # This would typically integrate with your auth system
        # For now, return a placeholder
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT token here
            return "user_placeholder"
        return None
    
    async def _get_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        return request.headers.get("X-API-Key")


# Utility functions for common rate limiting patterns
async def check_ip_rate_limit(ip_address: str, limit: int = 100, window: int = 60) -> RateLimitResult:
    """Check rate limit for IP address."""
    service = get_rate_limiting_service()
    
    # Create temporary config if not exists
    config_name = f"temp_ip_{limit}_{window}"
    if not service.get_rate_limit(config_name):
        config = RateLimitConfig(
            name=config_name,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=RateLimitScope.PER_IP,
            limit=limit,
            window=window
        )
        service.add_rate_limit(config)
    
    return await service.check_rate_limit(config_name, ip_address)


async def check_user_rate_limit(user_id: str, limit: int = 500, window: int = 60) -> RateLimitResult:
    """Check rate limit for user."""
    service = get_rate_limiting_service()
    
    # Create temporary config if not exists
    config_name = f"temp_user_{limit}_{window}"
    if not service.get_rate_limit(config_name):
        config = RateLimitConfig(
            name=config_name,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.PER_USER,
            limit=limit,
            window=window,
            burst=int(limit * 1.5)
        )
        service.add_rate_limit(config)
    
    return await service.check_rate_limit(config_name, user_id)
