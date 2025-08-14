"""
Filepath: src/pcs/sdk/python/config.py
Purpose: Configuration classes for PCS Python SDK client setup and behavior
Related Components: PCS Client, Authentication, Retry logic, HTTP settings
Tags: config, client, retry, timeout, authentication
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import os


class LogLevel(str, Enum):
    """Logging levels for SDK operations."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior with exponential backoff.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        retry_on_connection_errors: Retry on connection errors
        retry_on_timeout: Retry on timeout errors
        retry_on_rate_limit: Retry on rate limit errors (429)
        retry_on_server_errors: Retry on server errors (5xx)
    """
    
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_connection_errors: bool = True
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_errors: bool = True
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the error should be retried
        """
        from .exceptions import (
            PCSConnectionError, PCSTimeoutError, 
            PCSRateLimitError, PCSServerError
        )
        
        if self.retry_on_connection_errors and isinstance(exception, PCSConnectionError):
            return True
        if self.retry_on_timeout and isinstance(exception, PCSTimeoutError):
            return True
        if self.retry_on_rate_limit and isinstance(exception, PCSRateLimitError):
            return True
        if self.retry_on_server_errors and isinstance(exception, PCSServerError):
            return True
            
        return False


@dataclass
class HTTPConfig:
    """
    Configuration for HTTP client behavior.
    
    Attributes:
        timeout: Request timeout in seconds
        connect_timeout: Connection timeout in seconds
        read_timeout: Read timeout in seconds
        pool_connections: Number of connection pools
        pool_maxsize: Maximum pool size per connection
        max_retries_per_connection: Max retries for connection-level errors
        keep_alive: Enable HTTP keep-alive
        verify_ssl: Verify SSL certificates
        ssl_ca_bundle: Path to SSL CA bundle
        user_agent: User-Agent header value
        headers: Additional default headers
    """
    
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    pool_connections: int = 10
    pool_maxsize: int = 20
    max_retries_per_connection: int = 3
    keep_alive: bool = True
    verify_ssl: bool = True
    ssl_ca_bundle: Optional[str] = None
    user_agent: str = "PCS-Python-SDK/1.0.0"
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass 
class PCSClientConfig:
    """
    Main configuration for PCS Python SDK client.
    
    This class consolidates all configuration options and provides
    sensible defaults for production use while allowing full customization.
    
    Attributes:
        base_url: Base URL for PCS API
        api_key: API key for authentication
        jwt_token: JWT token for authentication (alternative to API key)
        api_version: API version to use
        environment: Environment name (for logging/debugging)
        
        retry_config: Retry behavior configuration
        http_config: HTTP client configuration
        
        log_level: Logging level for SDK operations
        log_requests: Whether to log HTTP requests
        log_responses: Whether to log HTTP responses
        
        rate_limit_tracking: Enable client-side rate limit tracking
        cache_enabled: Enable response caching (for read operations)
        cache_ttl: Cache TTL in seconds
        
        validate_ssl: Validate SSL certificates
        debug_mode: Enable debug mode with verbose logging
    """
    
    # Connection settings
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    api_version: str = "v1"
    environment: str = "production"
    
    # Nested configurations
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    http_config: HTTPConfig = field(default_factory=HTTPConfig)
    
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_requests: bool = False
    log_responses: bool = False
    
    # Feature flags
    rate_limit_tracking: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 300  # 5 minutes
    
    # Development settings
    validate_ssl: bool = True
    debug_mode: bool = False
    
    def __post_init__(self) -> None:
        """Post-initialization validation and setup."""
        # Normalize base URL
        if self.base_url.endswith("/"):
            self.base_url = self.base_url.rstrip("/")
            
        # Add API version to URL if not present
        if not self.base_url.endswith(f"/api/{self.api_version}"):
            self.base_url = f"{self.base_url}/api/{self.api_version}"
            
        # Validate authentication
        if not self.api_key and not self.jwt_token:
            # Try to get from environment
            self.api_key = os.getenv("PCS_API_KEY")
            self.jwt_token = os.getenv("PCS_JWT_TOKEN")
            
        # Set debug configurations
        if self.debug_mode:
            self.log_level = LogLevel.DEBUG
            self.log_requests = True
            self.log_responses = True
            self.http_config.verify_ssl = False
            
        # Update HTTP config based on settings
        self.http_config.verify_ssl = self.validate_ssl
        
        if self.debug_mode:
            self.http_config.user_agent += " (debug)"
            
    @classmethod
    def from_environment(cls) -> "PCSClientConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            PCS_BASE_URL: Base URL for PCS API
            PCS_API_KEY: API key for authentication
            PCS_JWT_TOKEN: JWT token for authentication
            PCS_API_VERSION: API version (default: v1)
            PCS_ENVIRONMENT: Environment name
            PCS_DEBUG: Enable debug mode
            PCS_LOG_LEVEL: Logging level
            PCS_TIMEOUT: Request timeout in seconds
            PCS_MAX_RETRIES: Maximum retry attempts
            
        Returns:
            Configured PCSClientConfig instance
        """
        return cls(
            base_url=os.getenv("PCS_BASE_URL", "http://localhost:8000"),
            api_key=os.getenv("PCS_API_KEY"),
            jwt_token=os.getenv("PCS_JWT_TOKEN"),
            api_version=os.getenv("PCS_API_VERSION", "v1"),
            environment=os.getenv("PCS_ENVIRONMENT", "production"),
            debug_mode=os.getenv("PCS_DEBUG", "").lower() in ("true", "1", "yes"),
            log_level=LogLevel(os.getenv("PCS_LOG_LEVEL", "INFO")),
            retry_config=RetryConfig(
                max_retries=int(os.getenv("PCS_MAX_RETRIES", "3")),
            ),
            http_config=HTTPConfig(
                timeout=float(os.getenv("PCS_TIMEOUT", "30.0")),
            ),
        )
    
    @property
    def full_api_url(self) -> str:
        """Get the full API URL with version."""
        return self.base_url
        
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for requests.
        
        Returns:
            Dictionary of headers for authentication
            
        Raises:
            ValueError: If no authentication method is configured
        """
        headers = {}
        
        if self.jwt_token:
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key
        else:
            raise ValueError(
                "No authentication configured. Set either api_key or jwt_token."
            )
            
        return headers
    
    def is_authenticated(self) -> bool:
        """Check if any authentication method is configured."""
        return bool(self.api_key or self.jwt_token)
