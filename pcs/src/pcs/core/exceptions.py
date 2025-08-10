"""
Filepath: src/pcs/core/exceptions.py
Purpose: Custom exception classes for the PCS application
Related Components: All application layers, Error handling, Logging
Tags: exceptions, error-handling, custom-errors, application-errors
"""

from typing import Any, Dict, Optional


class PCSError(Exception):
    """
    Base exception class for all PCS-related errors.
    
    This provides a consistent interface for all application exceptions
    with optional error codes and additional details.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class DatabaseError(PCSError):
    """Database-related errors."""
    pass


class ConnectionError(PCSError):
    """Connection-related errors (database, cache, external services)."""
    pass


class ConfigurationError(PCSError):
    """Configuration-related errors."""
    pass


class ValidationError(PCSError):
    """Data validation errors."""
    pass


class AuthenticationError(PCSError):
    """Authentication-related errors."""
    pass


class AuthorizationError(PCSError):
    """Authorization/permission-related errors."""
    pass


class NotFoundError(PCSError):
    """Resource not found errors."""
    pass


class ServiceError(PCSError):
    """General service/business logic errors."""
    pass


class TemplateError(PCSError):
    """Template processing errors."""
    pass


class RuleEngineError(PCSError):
    """Rule engine processing errors."""
    pass
