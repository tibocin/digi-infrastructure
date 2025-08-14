"""
Filepath: src/pcs/sdk/python/__init__.py
Purpose: Python SDK package initialization with all public exports
Related Components: PCS Client, Models, Exceptions, Utils
Tags: python-sdk, client, async, api
"""

# Main client
from .client import PCSClient

# Models for type hints and request/response handling
from .models import (
    # Base models
    PCSResponse,
    PaginatedResponse,
    HealthResponse,
    MetricsResponse,
    
    # Prompt models
    PromptCreate,
    PromptUpdate,
    PromptResponse,
    PromptVersionResponse,
    GeneratePromptRequest,
    GeneratedPromptResponse,
    
    # Context models  
    ContextCreate,
    ContextUpdate,
    ContextResponse,
    ContextMergeRequest,
    ContextMergeResponse,
    
    # Conversation models
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
)

# Exceptions
from .exceptions import (
    PCSSDKError,
    PCSAuthenticationError,
    PCSValidationError,
    PCSRateLimitError,
    PCSConnectionError,
    PCSTimeoutError,
    PCSServerError,
)

# Configuration and utilities
from .config import PCSClientConfig, RetryConfig
from .auth import AuthenticationMethod

__version__ = "1.0.0"

__all__ = [
    # Main client
    "PCSClient",
    
    # Base models
    "PCSResponse",
    "PaginatedResponse", 
    "HealthResponse",
    "MetricsResponse",
    
    # Prompt models
    "PromptCreate",
    "PromptUpdate", 
    "PromptResponse",
    "PromptVersionResponse",
    "GeneratePromptRequest",
    "GeneratedPromptResponse",
    
    # Context models
    "ContextCreate",
    "ContextUpdate",
    "ContextResponse", 
    "ContextMergeRequest",
    "ContextMergeResponse",
    
    # Conversation models
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationResponse",
    "MessageCreate", 
    "MessageResponse",
    
    # Exceptions
    "PCSSDKError",
    "PCSAuthenticationError",
    "PCSValidationError", 
    "PCSRateLimitError",
    "PCSConnectionError",
    "PCSTimeoutError",
    "PCSServerError",
    
    # Configuration
    "PCSClientConfig",
    "RetryConfig",
    "AuthenticationMethod",
]
