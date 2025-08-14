"""
Filepath: src/pcs/sdk/python/client.py
Purpose: Main PCS Python SDK client with async HTTP operations and comprehensive API coverage
Related Components: PCS API endpoints, HTTP client, Authentication, Retry logic
Tags: client, async, http, api, authentication, retry, sdk
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from uuid import UUID
from contextlib import asynccontextmanager

import httpx
from httpx import AsyncClient, Response

from .config import PCSClientConfig, RetryConfig
from .auth import AuthHandler
from .exceptions import (
    PCSSDKError, PCSConnectionError, PCSTimeoutError,
    create_exception_from_response
)
from .models import (
    # Base models
    PaginatedResponse, HealthResponse, MetricsResponse,
    
    # Prompt models
    PromptCreate, PromptUpdate, PromptResponse, PromptVersionResponse,
    GeneratePromptRequest, GeneratedPromptResponse,
    
    # Context models
    ContextCreate, ContextUpdate, ContextResponse,
    ContextMergeRequest, ContextMergeResponse,
    
    # Conversation models
    ConversationCreate, ConversationUpdate, ConversationResponse,
    MessageCreate, MessageResponse,
)


class PCSClient:
    """
    Asynchronous Python client for the Prompt and Context Service (PCS).
    
    Provides comprehensive access to all PCS API endpoints with:
    - Async/await support for high performance
    - Automatic retry with exponential backoff
    - Connection pooling and keep-alive
    - Type-safe request/response handling
    - Structured error handling
    - JWT and API key authentication
    
    Example:
        ```python
        async with PCSClient("https://api.pcs.example.com", api_key="your-key") as client:
            # Create a prompt
            prompt = await client.create_prompt(PromptCreate(
                name="greeting",
                content="Hello {{name}}!",
                variables={"name": "World"}
            ))
            
            # Generate from template
            result = await client.generate_prompt("greeting", {"name": "Alice"})
            print(result.generated_prompt)  # "Hello Alice!"
        ```
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        config: Optional[PCSClientConfig] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """
        Initialize PCS client.
        
        Args:
            base_url: Base URL for PCS API (e.g., "https://api.pcs.example.com")
            api_key: API key for authentication
            jwt_token: JWT token for authentication (alternative to API key)
            config: Complete client configuration (overrides individual args)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            
        Raises:
            ValueError: If no authentication is provided or configuration is invalid
        """
        # Use provided config or create from args
        if config:
            self.config = config
        else:
            self.config = PCSClientConfig(
                base_url=base_url or "http://localhost:8000",
                api_key=api_key,
                jwt_token=jwt_token,
            )
            
            # Apply timeout/retry overrides
            if timeout is not None:
                self.config.http_config.timeout = timeout
            if max_retries is not None:
                self.config.retry_config.max_retries = max_retries
        
        # Initialize components
        self.auth_handler = AuthHandler(
            api_key=self.config.api_key,
            jwt_token=self.config.jwt_token,
        )
        
        # HTTP client will be initialized in __aenter__
        self._http_client: Optional[AsyncClient] = None
        self._closed = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.value))
        
    async def __aenter__(self) -> "PCSClient":
        """Async context manager entry."""
        await self._ensure_http_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_http_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            # Build HTTP client configuration
            limits = httpx.Limits(
                max_keepalive_connections=self.config.http_config.pool_maxsize,
                max_connections=self.config.http_config.pool_connections,
                keepalive_expiry=300,  # 5 minutes
            )
            
            timeout = httpx.Timeout(
                connect=self.config.http_config.connect_timeout,
                read=self.config.http_config.read_timeout,
                write=30.0,
                pool=None,
            )
            
            headers = {
                "User-Agent": self.config.http_config.user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
                **self.config.http_config.headers,
            }
            
            self._http_client = AsyncClient(
                base_url=self.config.full_api_url,
                timeout=timeout,
                limits=limits,
                verify=self.config.http_config.verify_ssl,
                headers=headers,
                follow_redirects=True,
            )
    
    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._http_client and not self._closed:
            await self._http_client.aclose()
            self._closed = True
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body
            headers: Additional headers
            
        Returns:
            Parsed JSON response
            
        Raises:
            PCSSDKError: On API errors or network issues
        """
        await self._ensure_http_client()
        
        # Prepare headers with authentication
        request_headers = {}
        if headers:
            request_headers.update(headers)
            
        try:
            auth_headers = self.auth_handler.get_auth_headers()
            request_headers.update(auth_headers)
        except ValueError as e:
            raise PCSSDKError(f"Authentication error: {e}")
        
        # Prepare request
        url = endpoint.lstrip("/")
        
        # Retry logic with exponential backoff
        retry_config = self.config.retry_config
        last_exception = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Log request if enabled
                if self.config.log_requests:
                    self.logger.debug(
                        f"Making {method} request to {url}",
                        extra={
                            "method": method,
                            "url": url,
                            "params": params,
                            "attempt": attempt + 1,
                        }
                    )
                
                # Make HTTP request
                response = await self._http_client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=request_headers,
                )
                
                # Log response if enabled
                if self.config.log_responses:
                    elapsed_ms = (time.time() - start_time) * 1000
                    self.logger.debug(
                        f"Response: {response.status_code} in {elapsed_ms:.1f}ms",
                        extra={
                            "status_code": response.status_code,
                            "elapsed_ms": elapsed_ms,
                            "attempt": attempt + 1,
                        }
                    )
                
                # Handle response
                if response.is_success:
                    return response.json()
                else:
                    # Parse error response
                    try:
                        error_data = response.json()
                    except Exception:
                        error_data = {"message": response.text or "Unknown error"}
                    
                    request_id = response.headers.get("X-Request-ID")
                    exception = create_exception_from_response(
                        response.status_code,
                        error_data,
                        request_id,
                    )
                    
                    # Check if we should retry
                    if attempt < retry_config.max_retries and exception.is_retryable:
                        last_exception = exception
                        await self._wait_for_retry(attempt, retry_config, exception)
                        continue
                    else:
                        raise exception
                        
            except httpx.ConnectError as e:
                exception = PCSConnectionError(f"Connection failed: {e}")
                last_exception = exception
                
                if attempt < retry_config.max_retries and retry_config.retry_on_connection_errors:
                    await self._wait_for_retry(attempt, retry_config, exception)
                    continue
                else:
                    raise exception
                    
            except httpx.TimeoutException as e:
                exception = PCSTimeoutError(f"Request timeout: {e}")
                last_exception = exception
                
                if attempt < retry_config.max_retries and retry_config.retry_on_timeout:
                    await self._wait_for_retry(attempt, retry_config, exception)
                    continue
                else:
                    raise exception
                    
            except Exception as e:
                # Unexpected error
                exception = PCSSDKError(f"Unexpected error: {e}")
                raise exception
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise PCSSDKError("Request failed after all retry attempts")
    
    async def _wait_for_retry(
        self,
        attempt: int,
        retry_config: RetryConfig,
        exception: Exception,
    ) -> None:
        """
        Wait before retrying request with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-based)
            retry_config: Retry configuration
            exception: Exception that caused the retry
        """
        from .exceptions import PCSRateLimitError
        
        # Calculate delay
        if isinstance(exception, PCSRateLimitError) and exception.retry_after:
            # Use server-specified retry delay
            delay = exception.retry_after
        else:
            # Exponential backoff with jitter
            delay = min(
                retry_config.base_delay * (retry_config.backoff_factor ** attempt),
                retry_config.max_delay
            )
            
            if retry_config.jitter:
                # Add up to 25% jitter
                jitter = delay * 0.25 * random.random()
                delay += jitter
        
        self.logger.debug(
            f"Retrying in {delay:.1f}s after {type(exception).__name__}",
            extra={"delay": delay, "attempt": attempt + 1}
        )
        
        await asyncio.sleep(delay)
    
    # ========================================================================
    # Health and System Endpoints
    # ========================================================================
    
    async def get_health(self) -> HealthResponse:
        """
        Get basic system health status.
        
        Returns:
            Health status information
        """
        response = await self._make_request("GET", "/health")
        return HealthResponse(**response)
    
    async def get_metrics(self) -> MetricsResponse:
        """
        Get system metrics and performance data.
        
        Returns:
            System metrics
            
        Raises:
            PCSAuthorizationError: If not authorized for metrics
        """
        response = await self._make_request("GET", "/admin/metrics")
        return MetricsResponse(**response)
    
    # ========================================================================
    # Prompt Template Management
    # ========================================================================
    
    async def create_prompt(self, prompt_data: PromptCreate) -> PromptResponse:
        """
        Create a new prompt template.
        
        Args:
            prompt_data: Prompt template data
            
        Returns:
            Created prompt template
            
        Raises:
            PCSValidationError: If prompt data is invalid
            PCSConflictError: If prompt name already exists
        """
        response = await self._make_request(
            "POST",
            "/prompts",
            json_data=prompt_data.model_dump(exclude_unset=True),
        )
        return PromptResponse(**response)
    
    async def get_prompt(self, prompt_id: UUID) -> PromptResponse:
        """
        Get a prompt template by ID.
        
        Args:
            prompt_id: Prompt template ID
            
        Returns:
            Prompt template details
            
        Raises:
            PCSNotFoundError: If prompt doesn't exist
        """
        response = await self._make_request("GET", f"/prompts/{prompt_id}")
        return PromptResponse(**response)
    
    async def get_prompt_by_name(self, name: str) -> PromptResponse:
        """
        Get a prompt template by name.
        
        Args:
            name: Prompt template name
            
        Returns:
            Prompt template details
            
        Raises:
            PCSNotFoundError: If prompt doesn't exist
        """
        response = await self._make_request("GET", f"/prompts/by-name/{name}")
        return PromptResponse(**response)
    
    async def update_prompt(
        self,
        prompt_id: UUID,
        updates: PromptUpdate,
    ) -> PromptResponse:
        """
        Update a prompt template.
        
        Args:
            prompt_id: Prompt template ID
            updates: Fields to update
            
        Returns:
            Updated prompt template
            
        Raises:
            PCSNotFoundError: If prompt doesn't exist
            PCSValidationError: If update data is invalid
        """
        response = await self._make_request(
            "PUT",
            f"/prompts/{prompt_id}",
            json_data=updates.model_dump(exclude_unset=True),
        )
        return PromptResponse(**response)
    
    async def delete_prompt(self, prompt_id: UUID) -> bool:
        """
        Delete a prompt template.
        
        Args:
            prompt_id: Prompt template ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            PCSNotFoundError: If prompt doesn't exist
        """
        await self._make_request("DELETE", f"/prompts/{prompt_id}")
        return True
    
    async def list_prompts(
        self,
        *,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
        include_versions: bool = False,
        include_rules: bool = False,
    ) -> PaginatedResponse:
        """
        List prompt templates with filtering and pagination.
        
        Args:
            page: Page number (1-based)
            size: Page size (1-100)
            category: Filter by category
            status: Filter by status
            search: Search in name and description
            include_versions: Include version data
            include_rules: Include rule data
            
        Returns:
            Paginated list of prompts
        """
        params = {
            "page": page,
            "size": size,
        }
        
        if category:
            params["category"] = category
        if status:
            params["status"] = status
        if search:
            params["search"] = search
        if include_versions:
            params["include_versions"] = include_versions
        if include_rules:
            params["include_rules"] = include_rules
        
        response = await self._make_request("GET", "/prompts", params=params)
        
        # Convert items to PromptResponse objects
        items = [PromptResponse(**item) for item in response["items"]]
        response["items"] = items
        
        return PaginatedResponse(**response)
    
    async def generate_prompt(
        self,
        template_name: Optional[str] = None,
        *,
        template_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None,
        context_ids: Optional[List[str]] = None,
        optimization_level: str = "balanced",
    ) -> GeneratedPromptResponse:
        """
        Generate a prompt from a template with context.
        
        Args:
            template_name: Template name to use
            template_id: Template ID to use (alternative to name)
            context: Context variables for generation
            context_ids: Context IDs to merge
            optimization_level: Generation optimization level
            
        Returns:
            Generated prompt result
            
        Raises:
            PCSNotFoundError: If template doesn't exist
            PCSValidationError: If generation parameters are invalid
        """
        request_data = GeneratePromptRequest(
            template_name=template_name,
            template_id=template_id,
            context=context or {},
            context_ids=context_ids or [],
            optimization_level=optimization_level,
        )
        
        response = await self._make_request(
            "POST",
            "/prompts/generate",
            json_data=request_data.model_dump(exclude_unset=True),
        )
        return GeneratedPromptResponse(**response)
    
    # ========================================================================
    # Context Management
    # ========================================================================
    
    async def create_context(self, context_data: ContextCreate) -> ContextResponse:
        """
        Create a new context.
        
        Args:
            context_data: Context data
            
        Returns:
            Created context
            
        Raises:
            PCSValidationError: If context data is invalid
        """
        response = await self._make_request(
            "POST",
            "/contexts",
            json_data=context_data.model_dump(exclude_unset=True),
        )
        return ContextResponse(**response)
    
    async def get_context(self, context_id: UUID) -> ContextResponse:
        """
        Get a context by ID.
        
        Args:
            context_id: Context ID
            
        Returns:
            Context details
            
        Raises:
            PCSNotFoundError: If context doesn't exist
        """
        response = await self._make_request("GET", f"/contexts/{context_id}")
        return ContextResponse(**response)
    
    async def update_context(
        self,
        context_id: UUID,
        updates: ContextUpdate,
    ) -> ContextResponse:
        """
        Update a context.
        
        Args:
            context_id: Context ID
            updates: Fields to update
            
        Returns:
            Updated context
            
        Raises:
            PCSNotFoundError: If context doesn't exist
            PCSValidationError: If update data is invalid
        """
        response = await self._make_request(
            "PUT",
            f"/contexts/{context_id}",
            json_data=updates.model_dump(exclude_unset=True),
        )
        return ContextResponse(**response)
    
    async def delete_context(self, context_id: UUID) -> bool:
        """
        Delete a context.
        
        Args:
            context_id: Context ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            PCSNotFoundError: If context doesn't exist
        """
        await self._make_request("DELETE", f"/contexts/{context_id}")
        return True
    
    async def merge_contexts(
        self,
        merge_request: ContextMergeRequest,
    ) -> ContextMergeResponse:
        """
        Merge multiple contexts into one.
        
        Args:
            merge_request: Context merge configuration
            
        Returns:
            Merge result with merged context
            
        Raises:
            PCSNotFoundError: If any context doesn't exist
            PCSValidationError: If merge configuration is invalid
        """
        response = await self._make_request(
            "POST",
            "/contexts/merge",
            json_data=merge_request.model_dump(exclude_unset=True),
        )
        return ContextMergeResponse(**response)
    
    # ========================================================================
    # Conversation Management  
    # ========================================================================
    
    async def create_conversation(
        self,
        conversation_data: ConversationCreate,
    ) -> ConversationResponse:
        """
        Create a new conversation.
        
        Args:
            conversation_data: Conversation data
            
        Returns:
            Created conversation
            
        Raises:
            PCSValidationError: If conversation data is invalid
        """
        response = await self._make_request(
            "POST",
            "/conversations",
            json_data=conversation_data.model_dump(exclude_unset=True),
        )
        return ConversationResponse(**response)
    
    async def get_conversation(self, conversation_id: UUID) -> ConversationResponse:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation details
            
        Raises:
            PCSNotFoundError: If conversation doesn't exist
        """
        response = await self._make_request("GET", f"/conversations/{conversation_id}")
        return ConversationResponse(**response)
    
    async def update_conversation(
        self,
        conversation_id: UUID,
        updates: ConversationUpdate,
    ) -> ConversationResponse:
        """
        Update a conversation.
        
        Args:
            conversation_id: Conversation ID
            updates: Fields to update
            
        Returns:
            Updated conversation
            
        Raises:
            PCSNotFoundError: If conversation doesn't exist
            PCSValidationError: If update data is invalid
        """
        response = await self._make_request(
            "PUT",
            f"/conversations/{conversation_id}",
            json_data=updates.model_dump(exclude_unset=True),
        )
        return ConversationResponse(**response)
    
    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if deleted successfully
            
        Raises:
            PCSNotFoundError: If conversation doesn't exist
        """
        await self._make_request("DELETE", f"/conversations/{conversation_id}")
        return True
    
    async def add_message(
        self,
        conversation_id: UUID,
        message: MessageCreate,
    ) -> MessageResponse:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            message: Message data
            
        Returns:
            Created message
            
        Raises:
            PCSNotFoundError: If conversation doesn't exist
            PCSValidationError: If message data is invalid
        """
        response = await self._make_request(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json_data=message.model_dump(exclude_unset=True),
        )
        return MessageResponse(**response)
    
    async def get_conversation_history(
        self,
        conversation_id: UUID,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include_deleted: bool = False,
    ) -> List[MessageResponse]:
        """
        Get conversation message history.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages
            offset: Number of messages to skip
            include_deleted: Include deleted messages
            
        Returns:
            List of messages in chronological order
            
        Raises:
            PCSNotFoundError: If conversation doesn't exist
        """
        params = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if include_deleted:
            params["include_deleted"] = include_deleted
        
        response = await self._make_request(
            "GET",
            f"/conversations/{conversation_id}/messages",
            params=params,
        )
        
        return [MessageResponse(**msg) for msg in response]
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_auth_info(self) -> Dict[str, Any]:
        """
        Get authentication information.
        
        Returns:
            Dictionary with auth details
        """
        return self.auth_handler.get_user_info()
    
    def update_authentication(
        self,
        *,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
    ) -> None:
        """
        Update authentication credentials.
        
        Args:
            api_key: New API key
            jwt_token: New JWT token
        """
        if api_key is not None:
            self.auth_handler.update_api_key(api_key)
            self.config.api_key = api_key
        elif jwt_token is not None:
            self.auth_handler.update_jwt_token(jwt_token)
            self.config.jwt_token = jwt_token
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return f"PCSClient(base_url='{self.config.base_url}', auth={self.auth_handler})"
