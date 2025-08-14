"""
Filepath: src/pcs/sdk/python/exceptions.py
Purpose: Exception hierarchy for PCS Python SDK with detailed error handling
Related Components: PCS Client, HTTP responses, API errors
Tags: exceptions, error-handling, http, client, sdk
"""

from typing import Optional, Dict, Any, List
import json


class PCSSDKError(Exception):
    """
    Base exception for all PCS SDK errors.
    
    Provides common error handling functionality including:
    - Error details and context
    - HTTP status codes 
    - Request/response information
    - Retry indicators
    """
    
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        is_retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        self.request_id = request_id
        self.is_retryable = is_retryable
    
    def __str__(self) -> str:
        """Return a formatted error message with all available context."""
        parts = [self.message]
        
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
            
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
            
        return " | ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "details": self.details,
            "request_id": self.request_id,
            "is_retryable": self.is_retryable,
        }


class PCSAuthenticationError(PCSSDKError):
    """
    Authentication failed - invalid API key, expired JWT, etc.
    
    This error is never retryable as it indicates a fundamental
    authentication problem that requires user intervention.
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 401)
        kwargs.setdefault("is_retryable", False)
        super().__init__(message, **kwargs)


class PCSAuthorizationError(PCSSDKError):
    """
    Authorization failed - authenticated but lacking permissions.
    
    This error is not retryable as it indicates insufficient permissions.
    """
    
    def __init__(
        self,
        message: str = "Authorization failed - insufficient permissions",
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 403)
        kwargs.setdefault("is_retryable", False)
        super().__init__(message, **kwargs)


class PCSValidationError(PCSSDKError):
    """
    Request validation failed - invalid data, missing fields, etc.
    
    Contains detailed validation error information.
    Not retryable unless the request is modified.
    """
    
    def __init__(
        self,
        message: str = "Request validation failed",
        *,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 422)
        kwargs.setdefault("is_retryable", False)
        if validation_errors:
            kwargs.setdefault("details", {}).update({"validation_errors": validation_errors})
        super().__init__(message, **kwargs)
    
    @property
    def validation_errors(self) -> List[Dict[str, Any]]:
        """Get the validation errors list."""
        return self.details.get("validation_errors", [])


class PCSRateLimitError(PCSSDKError):
    """
    Rate limit exceeded.
    
    Contains retry-after information and is retryable with backoff.
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[int] = None,
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 429)
        kwargs.setdefault("is_retryable", True)
        if retry_after:
            kwargs.setdefault("details", {}).update({"retry_after": retry_after})
        super().__init__(message, **kwargs)
    
    @property
    def retry_after(self) -> Optional[int]:
        """Get retry-after time in seconds."""
        return self.details.get("retry_after")


class PCSConnectionError(PCSSDKError):
    """
    Network connection error - DNS, timeout, connection refused, etc.
    
    Generally retryable with exponential backoff.
    """
    
    def __init__(
        self,
        message: str = "Connection error",
        **kwargs
    ) -> None:
        kwargs.setdefault("is_retryable", True)
        super().__init__(message, **kwargs)


class PCSTimeoutError(PCSSDKError):
    """
    Request timeout error.
    
    Retryable with increased timeout or backoff.
    """
    
    def __init__(
        self,
        message: str = "Request timeout",
        *,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 408)
        kwargs.setdefault("is_retryable", True)
        if timeout_seconds:
            kwargs.setdefault("details", {}).update({"timeout_seconds": timeout_seconds})
        super().__init__(message, **kwargs)


class PCSServerError(PCSSDKError):
    """
    Server error (5xx status codes).
    
    Generally retryable as server issues are often transient.
    """
    
    def __init__(
        self,
        message: str = "Server error",
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 500)
        kwargs.setdefault("is_retryable", True)
        super().__init__(message, **kwargs)


class PCSNotFoundError(PCSSDKError):
    """
    Resource not found error.
    
    Not retryable as the resource doesn't exist.
    """
    
    def __init__(
        self,
        message: str = "Resource not found",
        *,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 404)
        kwargs.setdefault("is_retryable", False)
        if resource_type or resource_id:
            details = kwargs.setdefault("details", {})
            if resource_type:
                details["resource_type"] = resource_type
            if resource_id:
                details["resource_id"] = resource_id
        super().__init__(message, **kwargs)


class PCSConflictError(PCSSDKError):
    """
    Resource conflict error - duplicate names, version conflicts, etc.
    
    Not retryable without resolving the conflict.
    """
    
    def __init__(
        self,
        message: str = "Resource conflict",
        **kwargs
    ) -> None:
        kwargs.setdefault("status_code", 409)
        kwargs.setdefault("is_retryable", False)
        super().__init__(message, **kwargs)


def create_exception_from_response(
    status_code: int,
    response_data: Dict[str, Any],
    request_id: Optional[str] = None,
) -> PCSSDKError:
    """
    Create appropriate exception from HTTP response.
    
    Args:
        status_code: HTTP status code
        response_data: Parsed response body
        request_id: Optional request ID for tracking
        
    Returns:
        Appropriate PCSSDKError subclass instance
    """
    message = response_data.get("message", "API request failed")
    error_code = response_data.get("error_code")
    details = response_data.get("details", {})
    
    common_kwargs = {
        "status_code": status_code,
        "error_code": error_code,
        "details": details,
        "request_id": request_id,
    }
    
    # Authentication/Authorization errors
    if status_code == 401:
        return PCSAuthenticationError(message, **common_kwargs)
    elif status_code == 403:
        return PCSAuthorizationError(message, **common_kwargs)
    
    # Client errors
    elif status_code == 404:
        return PCSNotFoundError(message, **common_kwargs)
    elif status_code == 409:
        return PCSConflictError(message, **common_kwargs)
    elif status_code == 422:
        validation_errors = response_data.get("validation_errors", [])
        return PCSValidationError(
            message, 
            validation_errors=validation_errors,
            **common_kwargs
        )
    elif status_code == 429:
        retry_after = response_data.get("retry_after")
        return PCSRateLimitError(
            message,
            retry_after=retry_after,
            **common_kwargs
        )
    
    # Server errors
    elif 500 <= status_code < 600:
        return PCSServerError(message, **common_kwargs)
    
    # Timeout-like errors
    elif status_code == 408:
        return PCSTimeoutError(message, **common_kwargs)
    
    # Generic client errors
    elif 400 <= status_code < 500:
        return PCSValidationError(message, **common_kwargs)
    
    # Fallback to generic error
    else:
        return PCSSDKError(message, **common_kwargs)