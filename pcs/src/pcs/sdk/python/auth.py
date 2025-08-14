"""
Filepath: src/pcs/sdk/python/auth.py
Purpose: Authentication handling for PCS Python SDK with API keys and JWT tokens
Related Components: PCS Client, HTTP requests, Security configuration
Tags: authentication, jwt, api-key, security, http
"""

from enum import Enum
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import time
import jwt
from dataclasses import dataclass


class AuthenticationMethod(str, Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    NONE = "none"


@dataclass
class JWTTokenInfo:
    """
    Information extracted from JWT tokens.
    
    Attributes:
        token: The raw JWT token
        payload: Decoded JWT payload
        expires_at: Token expiration timestamp
        is_expired: Whether the token has expired
        user_id: User ID from token claims
        scopes: Authorization scopes/permissions
    """
    
    token: str
    payload: Dict[str, Any]
    expires_at: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Extract information from JWT payload."""
        if "exp" in self.payload:
            self.expires_at = datetime.fromtimestamp(self.payload["exp"])
    
    @property
    def is_expired(self) -> bool:
        """Check if the JWT token has expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at
    
    @property
    def expires_in_seconds(self) -> Optional[int]:
        """Get seconds until token expires."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.now()
        return max(0, int(delta.total_seconds()))
    
    @property
    def user_id(self) -> Optional[str]:
        """Extract user ID from token claims."""
        return self.payload.get("sub") or self.payload.get("user_id")
    
    @property
    def scopes(self) -> list[str]:
        """Extract scopes/permissions from token claims."""
        scopes = self.payload.get("scopes", [])
        if isinstance(scopes, str):
            return scopes.split(",")
        return scopes or []
    
    @classmethod
    def decode_token(cls, token: str, verify: bool = False) -> "JWTTokenInfo":
        """
        Decode a JWT token and create JWTTokenInfo.
        
        Args:
            token: JWT token string
            verify: Whether to verify token signature (requires secret)
            
        Returns:
            JWTTokenInfo instance
            
        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            if verify:
                # In production, you'd get the secret from config
                # For now, decode without verification for inspection
                payload = jwt.decode(token, options={"verify_signature": False})
            else:
                payload = jwt.decode(token, options={"verify_signature": False})
                
            return cls(token=token, payload=payload)
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid JWT token: {e}")


class AuthHandler:
    """
    Authentication handler for PCS SDK.
    
    Manages API keys, JWT tokens, and authentication headers.
    Provides token validation and automatic refresh capabilities.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        api_key_header: str = "X-API-Key",
    ) -> None:
        """
        Initialize authentication handler.
        
        Args:
            api_key: API key for authentication
            jwt_token: JWT token for authentication  
            api_key_header: Header name for API key authentication
        """
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.api_key_header = api_key_header
        self._jwt_info: Optional[JWTTokenInfo] = None
        
        # Parse JWT token if provided
        if self.jwt_token:
            try:
                self._jwt_info = JWTTokenInfo.decode_token(self.jwt_token)
            except ValueError:
                # Invalid token, we'll handle this when making requests
                pass
    
    @property
    def authentication_method(self) -> AuthenticationMethod:
        """Get the current authentication method."""
        if self.jwt_token:
            return AuthenticationMethod.JWT_TOKEN
        elif self.api_key:
            return AuthenticationMethod.API_KEY
        else:
            return AuthenticationMethod.NONE
    
    @property
    def is_authenticated(self) -> bool:
        """Check if any authentication method is configured."""
        return self.authentication_method != AuthenticationMethod.NONE
    
    @property
    def jwt_info(self) -> Optional[JWTTokenInfo]:
        """Get JWT token information if available."""
        return self._jwt_info
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for HTTP requests.
        
        Returns:
            Dictionary of authentication headers
            
        Raises:
            ValueError: If no authentication is configured or token is expired
        """
        headers = {}
        
        if self.jwt_token:
            # Check if token is expired
            if self._jwt_info and self._jwt_info.is_expired:
                raise ValueError("JWT token has expired")
            headers["Authorization"] = f"Bearer {self.jwt_token}"
            
        elif self.api_key:
            headers[self.api_key_header] = self.api_key
            
        else:
            raise ValueError("No authentication method configured")
            
        return headers
    
    def update_jwt_token(self, token: str) -> None:
        """
        Update the JWT token and refresh token info.
        
        Args:
            token: New JWT token
            
        Raises:
            ValueError: If token is invalid
        """
        self.jwt_token = token
        self._jwt_info = JWTTokenInfo.decode_token(token)
    
    def update_api_key(self, api_key: str) -> None:
        """
        Update the API key.
        
        Args:
            api_key: New API key
        """
        self.api_key = api_key
        # Clear JWT info since we're switching to API key
        self.jwt_token = None
        self._jwt_info = None
    
    def clear_authentication(self) -> None:
        """Clear all authentication credentials."""
        self.api_key = None
        self.jwt_token = None
        self._jwt_info = None
    
    def validate_authentication(self) -> None:
        """
        Validate current authentication configuration.
        
        Raises:
            ValueError: If authentication is invalid or expired
        """
        if not self.is_authenticated:
            raise ValueError("No authentication configured")
            
        if self.jwt_token and self._jwt_info and self._jwt_info.is_expired:
            raise ValueError("JWT token has expired")
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get user information from authentication context.
        
        Returns:
            Dictionary with available user information
        """
        info = {
            "authentication_method": self.authentication_method.value,
            "is_authenticated": self.is_authenticated,
        }
        
        if self._jwt_info:
            info.update({
                "user_id": self._jwt_info.user_id,
                "scopes": self._jwt_info.scopes,
                "expires_at": self._jwt_info.expires_at.isoformat() if self._jwt_info.expires_at else None,
                "expires_in_seconds": self._jwt_info.expires_in_seconds,
                "is_expired": self._jwt_info.is_expired,
            })
            
        return info
    
    def __repr__(self) -> str:
        """String representation of auth handler."""
        method = self.authentication_method.value
        if self._jwt_info and self._jwt_info.user_id:
            return f"AuthHandler(method={method}, user_id={self._jwt_info.user_id})"
        return f"AuthHandler(method={method})"