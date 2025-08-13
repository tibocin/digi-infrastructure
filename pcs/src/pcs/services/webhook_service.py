"""
Filepath: src/pcs/services/webhook_service.py
Purpose: Webhook delivery system with retry logic, security validation, and monitoring
Related Components: HTTP clients, Redis for queuing, Background tasks, Security validation
Tags: webhooks, http-delivery, retry-logic, security, async, monitoring
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel, Field, field_validator

from pcs.core.config import get_settings
from pcs.core.exceptions import PCSError, ValidationError
from pcs.utils.logger import get_logger
from pcs.utils.metrics import get_metrics_collector, record_manual_metric

logger = get_logger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class WebhookEvent(str, Enum):
    """Webhook event types."""
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    PROMPT_GENERATED = "prompt.generated"
    CONTEXT_UPDATED = "context.updated"
    SYSTEM_ALERT = "system.alert"
    CUSTOM = "custom"


@dataclass
class WebhookDeliveryResult:
    """Container for webhook delivery result."""
    webhook_id: str
    status: WebhookStatus
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    delivery_time: Optional[float] = None
    delivered_at: Optional[datetime] = None


class WebhookEndpoint(BaseModel):
    """Webhook endpoint configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str = Field(..., description="Webhook delivery URL")
    events: List[WebhookEvent] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(default=None, description="Secret for signature validation")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=60, description="Initial retry delay in seconds")
    active: bool = Field(default=True, description="Whether endpoint is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        """Validate URL format."""
        parsed = urlparse(v)
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("URL must use http or https scheme")
        if not parsed.netloc:
            raise ValueError("URL must have a valid hostname")
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout range."""
        if v <= 0 or v > 300:
            raise ValueError("Timeout must be between 1 and 300 seconds")
        return v


class WebhookPayload(BaseModel):
    """Webhook payload structure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event: WebhookEvent = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="pcs", description="Event source")
    version: str = Field(default="1.0", description="Payload version")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebhookError(PCSError):
    """Webhook-related errors."""
    pass


class WebhookSecurityValidator:
    """Security validation for webhooks."""
    
    @staticmethod
    def generate_signature(payload: str, secret: str, algorithm: str = "sha256") -> str:
        """
        Generate HMAC signature for webhook payload.
        
        Args:
            payload: JSON payload string
            secret: Signing secret
            algorithm: HMAC algorithm
            
        Returns:
            Hex-encoded signature
        """
        mac = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            getattr(hashlib, algorithm)
        )
        return mac.hexdigest()
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str, algorithm: str = "sha256") -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: JSON payload string
            signature: Received signature
            secret: Signing secret
            algorithm: HMAC algorithm
            
        Returns:
            True if signature is valid
        """
        expected_signature = WebhookSecurityValidator.generate_signature(
            payload, secret, algorithm
        )
        return hmac.compare_digest(signature, expected_signature)
    
    @staticmethod
    def validate_url_security(url: str) -> bool:
        """
        Validate URL for security issues.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is considered safe
        """
        parsed = urlparse(url)
        
        # Block localhost and private IP ranges in production
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False
        
        # Block private IP ranges (simplified check)
        if parsed.hostname and (
            parsed.hostname.startswith('192.168.') or
            parsed.hostname.startswith('10.') or
            parsed.hostname.startswith('172.')
        ):
            return False
        
        return True


class WebhookDeliveryEngine:
    """
    Webhook delivery engine with retry logic and monitoring.
    
    Features:
    - Asynchronous HTTP delivery
    - Exponential backoff retry logic
    - Signature generation and validation
    - Delivery monitoring and metrics
    - Security validation
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize webhook delivery engine.
        
        Args:
            redis_client: Redis client for queuing (optional)
        """
        self.settings = get_settings()
        self.redis_client = redis_client
        self.metrics_collector = get_metrics_collector()
        self.security_validator = WebhookSecurityValidator()
        
        # HTTP client configuration
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={
                "User-Agent": f"PCS-Webhook/{self.settings.version}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("Webhook delivery engine initialized")
    
    async def initialize(self) -> None:
        """Initialize Redis connection if needed."""
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
            logger.info("Redis connection established for webhook delivery")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise WebhookError("Redis connection failed for webhook service")
    
    async def shutdown(self) -> None:
        """Shutdown the webhook delivery engine."""
        logger.info("Shutting down webhook delivery engine...")
        
        if self.http_client:
            await self.http_client.aclose()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Webhook delivery engine shutdown complete")
    
    async def deliver_webhook(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload,
        attempt: int = 1
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook to endpoint with retry logic.
        
        Args:
            endpoint: Webhook endpoint configuration
            payload: Webhook payload
            attempt: Current attempt number
            
        Returns:
            Delivery result
        """
        start_time = time.time()
        webhook_id = f"{endpoint.id}-{payload.id}"
        
        # Security validation
        if not self.security_validator.validate_url_security(endpoint.url):
            logger.warning(f"Webhook URL failed security validation: {endpoint.url}")
            return WebhookDeliveryResult(
                webhook_id=webhook_id,
                status=WebhookStatus.FAILED,
                error_message="URL failed security validation"
            )
        
        try:
            # Prepare payload
            payload_json = payload.model_dump_json()
            
            # Prepare headers
            headers = endpoint.headers.copy()
            headers["X-Webhook-Event"] = payload.event.value
            headers["X-Webhook-ID"] = payload.id
            headers["X-Webhook-Timestamp"] = str(int(payload.timestamp.timestamp()))
            headers["X-Webhook-Attempt"] = str(attempt)
            
            # Add signature if secret is provided
            if endpoint.secret:
                signature = self.security_validator.generate_signature(
                    payload_json, endpoint.secret
                )
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            # Make HTTP request
            logger.info(f"Delivering webhook {webhook_id} to {endpoint.url} (attempt {attempt})")
            
            response = await self.http_client.post(
                endpoint.url,
                content=payload_json,
                headers=headers,
                timeout=endpoint.timeout
            )
            
            delivery_time = time.time() - start_time
            
            # Check response status
            if response.status_code >= 200 and response.status_code < 300:
                # Success
                logger.info(f"Webhook {webhook_id} delivered successfully in {delivery_time:.2f}s")
                
                # Record metrics
                record_manual_metric(
                    query_type="webhook_delivered",
                    execution_time=delivery_time,
                    rows_affected=1,
                    table_name="webhooks"
                )
                
                return WebhookDeliveryResult(
                    webhook_id=webhook_id,
                    status=WebhookStatus.DELIVERED,
                    status_code=response.status_code,
                    response_body=response.text[:1000],  # Limit response body
                    delivery_time=delivery_time,
                    delivered_at=datetime.utcnow()
                )
            else:
                # Non-success status code
                error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
                logger.warning(f"Webhook {webhook_id} failed with status {response.status_code}")
                
                # Record metrics
                record_manual_metric(
                    query_type="webhook_failed",
                    execution_time=delivery_time,
                    rows_affected=1,
                    table_name="webhooks"
                )
                
                return WebhookDeliveryResult(
                    webhook_id=webhook_id,
                    status=WebhookStatus.FAILED,
                    status_code=response.status_code,
                    response_body=response.text[:1000],
                    error_message=error_msg,
                    delivery_time=delivery_time
                )
                
        except asyncio.TimeoutError:
            delivery_time = time.time() - start_time
            error_msg = f"Request timeout after {endpoint.timeout}s"
            logger.warning(f"Webhook {webhook_id} timed out after {delivery_time:.2f}s")
            
            return WebhookDeliveryResult(
                webhook_id=webhook_id,
                status=WebhookStatus.FAILED,
                error_message=error_msg,
                delivery_time=delivery_time
            )
            
        except Exception as e:
            delivery_time = time.time() - start_time
            error_msg = f"Delivery error: {str(e)}"
            logger.error(f"Webhook {webhook_id} delivery failed: {error_msg}")
            
            return WebhookDeliveryResult(
                webhook_id=webhook_id,
                status=WebhookStatus.FAILED,
                error_message=error_msg,
                delivery_time=delivery_time
            )
    
    async def deliver_with_retries(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook with retry logic.
        
        Args:
            endpoint: Webhook endpoint configuration
            payload: Webhook payload
            
        Returns:
            Final delivery result
        """
        last_result = None
        
        for attempt in range(1, endpoint.max_retries + 1):
            result = await self.deliver_webhook(endpoint, payload, attempt)
            
            if result.status == WebhookStatus.DELIVERED:
                return result
            
            last_result = result
            
            # If not the last attempt, wait before retrying
            if attempt < endpoint.max_retries:
                # Exponential backoff
                delay = endpoint.retry_delay * (2 ** (attempt - 1))
                logger.info(f"Webhook {result.webhook_id} attempt {attempt} failed, retrying in {delay}s")
                await asyncio.sleep(delay)
        
        # All attempts failed
        logger.error(f"Webhook {last_result.webhook_id} failed after {endpoint.max_retries} attempts")
        return last_result
    
    async def queue_webhook(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> str:
        """
        Queue webhook for background delivery.
        
        Args:
            endpoint: Webhook endpoint configuration
            payload: Webhook payload
            
        Returns:
            Webhook queue ID
        """
        if not self.redis_client:
            await self.initialize()
        
        webhook_data = {
            "endpoint": endpoint.model_dump_json(),
            "payload": payload.model_dump_json(),
            "queued_at": datetime.utcnow().isoformat()
        }
        
        queue_id = str(uuid.uuid4())
        
        # Add to Redis queue
        await self.redis_client.lpush(
            "pcs:webhooks:queue",
            json.dumps(webhook_data)
        )
        
        # Store metadata
        await self.redis_client.hset(
            f"pcs:webhooks:meta:{queue_id}",
            mapping={
                "status": WebhookStatus.PENDING.value,
                "endpoint_url": endpoint.url,
                "event": payload.event.value,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        # Set expiration (7 days)
        await self.redis_client.expire(f"pcs:webhooks:meta:{queue_id}", 7 * 24 * 60 * 60)
        
        logger.info(f"Webhook queued for delivery: {queue_id}")
        return queue_id


class WebhookService:
    """
    Webhook management service.
    
    Features:
    - Endpoint registration and management
    - Event subscription handling
    - Webhook delivery coordination
    - Monitoring and statistics
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize webhook service.
        
        Args:
            redis_client: Redis client instance (optional)
        """
        self.settings = get_settings()
        self.redis_client = redis_client
        self.delivery_engine = WebhookDeliveryEngine(redis_client)
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        
        logger.info("Webhook service initialized")
    
    async def initialize(self) -> None:
        """Initialize the webhook service."""
        if not self.redis_client:
            self.redis_client = aioredis.from_url(
                self.settings.redis.url,
                decode_responses=True
            )
        
        await self.delivery_engine.initialize()
        await self._load_endpoints()
        
        logger.info("Webhook service initialization complete")
    
    async def shutdown(self) -> None:
        """Shutdown the webhook service."""
        await self.delivery_engine.shutdown()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Webhook service shutdown complete")
    
    async def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """
        Register a webhook endpoint.
        
        Args:
            endpoint: Webhook endpoint configuration
            
        Returns:
            Endpoint ID
        """
        # Validate endpoint
        if not endpoint.active:
            raise WebhookError("Cannot register inactive endpoint")
        
        # Store in memory and Redis
        self.endpoints[endpoint.id] = endpoint
        
        if self.redis_client:
            await self.redis_client.hset(
                "pcs:webhooks:endpoints",
                endpoint.id,
                endpoint.model_dump_json()
            )
        
        logger.info(f"Registered webhook endpoint: {endpoint.id} -> {endpoint.url}")
        return endpoint.id
    
    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister a webhook endpoint.
        
        Args:
            endpoint_id: Endpoint ID
            
        Returns:
            True if endpoint was removed
        """
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            
            if self.redis_client:
                await self.redis_client.hdel("pcs:webhooks:endpoints", endpoint_id)
            
            logger.info(f"Unregistered webhook endpoint: {endpoint_id}")
            return True
        
        return False
    
    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID."""
        return self.endpoints.get(endpoint_id)
    
    async def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all registered webhook endpoints."""
        return list(self.endpoints.values())
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        source: str = "pcs",
        immediate: bool = False
    ) -> List[str]:
        """
        Trigger webhooks for an event.
        
        Args:
            event: Event type
            data: Event data
            source: Event source
            immediate: Whether to deliver immediately or queue
            
        Returns:
            List of webhook queue/delivery IDs
        """
        # Find matching endpoints
        matching_endpoints = [
            endpoint for endpoint in self.endpoints.values()
            if endpoint.active and event in endpoint.events
        ]
        
        if not matching_endpoints:
            logger.debug(f"No webhook endpoints registered for event: {event}")
            return []
        
        # Create payload
        payload = WebhookPayload(
            event=event,
            data=data,
            source=source
        )
        
        results = []
        
        for endpoint in matching_endpoints:
            try:
                if immediate:
                    # Deliver immediately
                    result = await self.delivery_engine.deliver_with_retries(
                        endpoint, payload
                    )
                    results.append(result.webhook_id)
                else:
                    # Queue for background delivery
                    queue_id = await self.delivery_engine.queue_webhook(
                        endpoint, payload
                    )
                    results.append(queue_id)
                    
            except Exception as e:
                logger.error(f"Failed to trigger webhook for endpoint {endpoint.id}: {e}")
        
        logger.info(f"Triggered {len(results)} webhooks for event: {event}")
        return results
    
    async def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics."""
        if not self.redis_client:
            return {}
        
        stats = {
            "total_endpoints": len(self.endpoints),
            "active_endpoints": len([e for e in self.endpoints.values() if e.active]),
            "queue_length": await self.redis_client.llen("pcs:webhooks:queue"),
            "endpoints_by_event": {}
        }
        
        # Count endpoints by event type
        for event in WebhookEvent:
            count = len([
                e for e in self.endpoints.values()
                if event in e.events and e.active
            ])
            stats["endpoints_by_event"][event.value] = count
        
        return stats
    
    async def _load_endpoints(self) -> None:
        """Load webhook endpoints from Redis."""
        if not self.redis_client:
            return
        
        try:
            endpoints_data = await self.redis_client.hgetall("pcs:webhooks:endpoints")
            
            for endpoint_id, endpoint_json in endpoints_data.items():
                try:
                    endpoint = WebhookEndpoint.model_validate_json(endpoint_json)
                    self.endpoints[endpoint_id] = endpoint
                except Exception as e:
                    logger.error(f"Failed to load webhook endpoint {endpoint_id}: {e}")
            
            logger.info(f"Loaded {len(self.endpoints)} webhook endpoints")
            
        except Exception as e:
            logger.warning(f"Failed to load webhook endpoints: {e}")


# Global webhook service instance
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get the global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service


# Utility functions for common webhook patterns
async def send_task_completion_webhook(task_id: str, result: Any, execution_time: float) -> List[str]:
    """Send webhook for task completion."""
    service = get_webhook_service()
    return await service.trigger_webhook(
        event=WebhookEvent.TASK_COMPLETED,
        data={
            "task_id": task_id,
            "result": result,
            "execution_time": execution_time,
            "completed_at": datetime.utcnow().isoformat()
        }
    )


async def send_task_failure_webhook(task_id: str, error: str, execution_time: float) -> List[str]:
    """Send webhook for task failure."""
    service = get_webhook_service()
    return await service.trigger_webhook(
        event=WebhookEvent.TASK_FAILED,
        data={
            "task_id": task_id,
            "error": error,
            "execution_time": execution_time,
            "failed_at": datetime.utcnow().isoformat()
        }
    )


async def send_custom_webhook(event_data: Dict[str, Any]) -> List[str]:
    """Send custom webhook with arbitrary data."""
    service = get_webhook_service()
    return await service.trigger_webhook(
        event=WebhookEvent.CUSTOM,
        data=event_data
    )
