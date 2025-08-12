"""
Filepath: src/pcs/api/dependencies.py
Purpose: FastAPI dependencies for authentication, database, and validation
Related Components: Database sessions, Authentication, Request validation
Tags: fastapi, dependencies, auth, database, validation
"""

from typing import AsyncGenerator, Dict, Any, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings, get_settings
from ..core.database import get_db_session
from ..core.exceptions import AuthenticationError, ValidationError


# Security scheme for JWT authentication
security = HTTPBearer(auto_error=False)


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async for session in get_db_session():
        yield session


def get_settings_dependency() -> Settings:
    """
    FastAPI dependency to get application settings.
    
    Returns:
        Settings: Application settings
    """
    return get_settings()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings_dependency)
) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        settings: Application settings
        
    Returns:
        User information if authenticated, None otherwise
        
    Note:
        This is a placeholder implementation. Real JWT validation will be
        implemented in Step 6: Authentication & Security.
    """
    if not credentials:
        return None
    
    # TODO: Implement JWT token validation
    # For now, return a mock user for development
    return {
        "id": "mock-user-id",
        "username": "mock-user",
        "email": "mock@example.com"
    }


async def require_authentication(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require user authentication.
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        User information
        
    Raises:
        HTTPException: If user is not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return current_user


async def require_admin(
    current_user: Dict[str, Any] = Depends(require_authentication)
) -> Dict[str, Any]:
    """
    Require admin privileges.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information if admin
        
    Raises:
        HTTPException: If user is not an admin
        
    Note:
        This is a placeholder implementation. Real role checking will be
        implemented in Step 6: Authentication & Security.
    """
    # TODO: Check if user has admin role
    # For now, allow all authenticated users
    return current_user


class PaginationParams:
    """Pagination parameters for list endpoints."""
    
    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Number of items to skip"),
        limit: int = Query(100, ge=1, le=1000, description="Number of items to return")
    ):
        self.skip = skip
        self.limit = limit
        self.offset = skip  # Alias for offset
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "skip": self.skip,
            "limit": self.limit,
            "offset": self.offset
        }


def get_pagination(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return")
) -> PaginationParams:
    """
    FastAPI dependency for pagination parameters.
    
    Args:
        skip: Number of items to skip
        limit: Number of items to return
        
    Returns:
        PaginationParams: Pagination parameters
    """
    return PaginationParams(skip=skip, limit=limit)


class RequestMetadata:
    """Request metadata for tracking and logging."""
    
    def __init__(self, request: Request):
        self.method = request.method
        self.url = str(request.url)
        self.headers = dict(request.headers)
        self.client_ip = request.client.host if request.client else None
        self.user_agent = request.headers.get("user-agent")
        self.request_id = request.headers.get("x-request-id")


def get_request_metadata(request: Request) -> RequestMetadata:
    """
    FastAPI dependency to extract request metadata.
    
    Args:
        request: FastAPI request object
        
    Returns:
        RequestMetadata: Request metadata
    """
    return RequestMetadata(request)


class CommonQueryParams:
    """Common query parameters for filtering and searching."""
    
    def __init__(
        self,
        q: Optional[str] = Query(None, description="Search query"),
        sort_by: Optional[str] = Query(None, description="Field to sort by"),
        sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order"),
        include_deleted: bool = Query(False, description="Include deleted items"),
        created_after: Optional[str] = Query(None, description="Filter by creation date (ISO format)"),
        created_before: Optional[str] = Query(None, description="Filter by creation date (ISO format)")
    ):
        self.q = q
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.include_deleted = include_deleted
        self.created_after = created_after
        self.created_before = created_before


def get_common_params(
    q: Optional[str] = Query(None, description="Search query"),
    sort_by: Optional[str] = Query(None, description="Field to sort by"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="Sort order"),
    include_deleted: bool = Query(False, description="Include deleted items"),
    created_after: Optional[str] = Query(None, description="Filter by creation date (ISO format)"),
    created_before: Optional[str] = Query(None, description="Filter by creation date (ISO format)")
) -> CommonQueryParams:
    """
    FastAPI dependency for common query parameters.
    
    Returns:
        CommonQueryParams: Common query parameters
    """
    return CommonQueryParams(
        q=q,
        sort_by=sort_by,
        sort_order=sort_order,
        include_deleted=include_deleted,
        created_after=created_after,
        created_before=created_before
    )


# Placeholder dependencies for future features
async def rate_limit_dependency() -> None:
    """
    Rate limiting dependency.
    
    Note:
        This is a placeholder. Real rate limiting will be implemented
        in Step 8: Logging & Metrics Setup.
    """
    pass


async def cache_dependency() -> Optional[Any]:
    """
    Cache dependency for storing/retrieving cached data.
    
    Note:
        This is a placeholder. Real caching will be implemented
        when Redis integration is added.
    """
    return None


def validate_uuid(uuid_str: str) -> UUID:
    """
    Validate and convert string to UUID.
    
    Args:
        uuid_str: String representation of UUID
        
    Returns:
        UUID: Validated UUID object
        
    Raises:
        HTTPException: If UUID is invalid
    """
    try:
        return UUID(uuid_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format: {uuid_str}"
        )
