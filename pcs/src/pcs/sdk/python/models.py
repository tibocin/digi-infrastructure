"""
Filepath: src/pcs/sdk/python/models.py
Purpose: Pydantic models for PCS Python SDK request/response handling
Related Components: PCS API endpoints, Database models, Client requests
Tags: models, pydantic, api, requests, responses, validation
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic.types import EmailStr

# Import existing enums from PCS models
try:
    from ...models.prompts import PromptStatus, RulePriority
    from ...models.contexts import ContextTypeEnum, ContextScope, RelationshipType
    from ...models.conversations import ConversationStatus, ConversationPriority, MessageRole, MessageType
except ImportError:
    # Fallback definitions for when models aren't available
    class PromptStatus(str, Enum):
        DRAFT = "draft"
        ACTIVE = "active"
        ARCHIVED = "archived"
        DEPRECATED = "deprecated"
    
    class RulePriority(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class ContextTypeEnum(str, Enum):
        SYSTEM = "system"
        USER = "user"
        SESSION = "session"
        PROJECT = "project"
        CUSTOM = "custom"
    
    class ContextScope(str, Enum):
        GLOBAL = "global"
        ORGANIZATION = "organization"
        PROJECT = "project"
        USER = "user"
        SESSION = "session"
    
    class RelationshipType(str, Enum):
        DEPENDS_ON = "depends_on"
        EXTENDS = "extends"
        INCLUDES = "includes"
        REFERENCES = "references"
    
    class ConversationStatus(str, Enum):
        ACTIVE = "active"
        PAUSED = "paused"
        COMPLETED = "completed"
        ARCHIVED = "archived"
    
    class ConversationPriority(str, Enum):
        LOW = "low"
        NORMAL = "normal"
        HIGH = "high"
        URGENT = "urgent"
    
    class MessageRole(str, Enum):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
    
    class MessageType(str, Enum):
        TEXT = "text"
        CODE = "code"
        ERROR = "error"
        SYSTEM_NOTIFICATION = "system_notification"


# ============================================================================
# Base Response Models
# ============================================================================

class PCSResponse(BaseModel):
    """Base response model with common fields."""
    
    id: UUID = Field(..., description="Unique identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        validate_assignment=True,
    )


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., ge=0, description="Total number of items")
    page: int = Field(..., ge=1, description="Current page number")
    size: int = Field(..., ge=1, le=100, description="Page size")
    pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")
    
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment name")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    
    model_config = ConfigDict(from_attributes=True)


class MetricsResponse(BaseModel):
    """System metrics response."""
    
    requests_total: int = Field(..., description="Total requests processed")
    requests_per_second: float = Field(..., description="Current RPS")
    avg_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    active_connections: int = Field(..., description="Active connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Prompt Models
# ============================================================================

class PromptCreate(BaseModel):
    """Schema for creating a new prompt template."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Unique prompt name")
    description: Optional[str] = Field(None, max_length=1000, description="Prompt description")
    category: Optional[str] = Field(None, max_length=100, description="Category for organization")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering")
    author: Optional[str] = Field(None, max_length=255, description="Author name")
    content: str = Field(..., min_length=1, description="Template content with variables")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variable definitions")
    rules: List[Dict[str, Any]] = Field(default_factory=list, description="Processing rules")
    is_system: bool = Field(default=False, description="System template flag")
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Template content cannot be empty")
        return v


class PromptUpdate(BaseModel):
    """Schema for updating a prompt template."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    author: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = Field(None, min_length=1)
    variables: Optional[Dict[str, Any]] = None
    rules: Optional[List[Dict[str, Any]]] = None
    status: Optional[PromptStatus] = None


class PromptResponse(PCSResponse):
    """Response model for prompt templates."""
    
    name: str = Field(..., description="Prompt name")
    description: Optional[str] = Field(None, description="Prompt description")
    category: Optional[str] = Field(None, description="Category")
    tags: List[str] = Field(default_factory=list, description="Tags")
    author: Optional[str] = Field(None, description="Author")
    status: PromptStatus = Field(..., description="Current status")
    is_system: bool = Field(..., description="System template flag")
    version_count: int = Field(..., description="Number of versions")
    current_version: Optional["PromptVersionResponse"] = Field(None, description="Current active version")


class PromptVersionResponse(PCSResponse):
    """Response model for prompt versions."""
    
    template_id: UUID = Field(..., description="Parent template ID")
    version_number: int = Field(..., description="Version number")
    content: str = Field(..., description="Template content")
    variables: Dict[str, Any] = Field(..., description="Variable definitions")
    rules: List[Dict[str, Any]] = Field(..., description="Processing rules")
    changelog: Optional[str] = Field(None, description="Version changelog")
    is_active: bool = Field(..., description="Active version flag")
    

class GeneratePromptRequest(BaseModel):
    """Request to generate a prompt from template."""
    
    template_name: Optional[str] = Field(None, description="Template name to use")
    template_id: Optional[UUID] = Field(None, description="Template ID to use")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context variables")
    context_ids: List[str] = Field(default_factory=list, description="Context IDs to merge")
    optimization_level: str = Field(default="balanced", description="Optimization level")
    
    @field_validator('template_name', 'template_id')
    @classmethod
    def validate_template_identifier(cls, v, info):
        # At least one of template_name or template_id must be provided
        if info.data.get('template_name') is None and info.data.get('template_id') is None:
            raise ValueError("Either template_name or template_id must be provided")
        return v


class GeneratedPromptResponse(BaseModel):
    """Response from prompt generation."""
    
    generated_prompt: str = Field(..., description="Generated prompt text")
    template_id: UUID = Field(..., description="Source template ID")
    template_name: str = Field(..., description="Source template name")
    context_applied: Dict[str, Any] = Field(..., description="Applied context variables")
    rules_applied: List[Dict[str, Any]] = Field(..., description="Applied rules")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    cache_hit: bool = Field(..., description="Whether result was cached")
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Context Models
# ============================================================================

class ContextCreate(BaseModel):
    """Schema for creating a new context."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    context_type_id: UUID = Field(..., description="Context type reference")
    scope: ContextScope = Field(..., description="Context scope")
    owner_id: Optional[str] = Field(None, max_length=255, description="Owner ID")
    project_id: Optional[str] = Field(None, max_length=255, description="Project ID")
    context_data: Dict[str, Any] = Field(..., description="Context data payload")
    context_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: int = Field(default=0, description="Context priority")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    
    @field_validator('context_data')
    @classmethod
    def validate_context_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("Context data cannot be empty")
        return v


class ContextUpdate(BaseModel):
    """Schema for updating a context."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    scope: Optional[ContextScope] = None
    owner_id: Optional[str] = Field(None, max_length=255)
    project_id: Optional[str] = Field(None, max_length=255)
    context_data: Optional[Dict[str, Any]] = None
    context_metadata: Optional[Dict[str, Any]] = None
    priority: Optional[int] = None
    vector_embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    is_active: Optional[bool] = None


class ContextResponse(PCSResponse):
    """Response model for contexts."""
    
    name: str = Field(..., description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    context_type_id: UUID = Field(..., description="Context type reference")
    scope: ContextScope = Field(..., description="Context scope")
    owner_id: Optional[str] = Field(None, description="Owner ID")
    project_id: Optional[str] = Field(None, description="Project ID")
    context_data: Dict[str, Any] = Field(..., description="Context data")
    context_metadata: Dict[str, Any] = Field(..., description="Context metadata")
    priority: int = Field(..., description="Context priority")
    is_active: bool = Field(..., description="Active status")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    embedding_model: Optional[str] = Field(None, description="Embedding model")
    access_count: int = Field(..., description="Number of times accessed")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access time")


class ContextMergeRequest(BaseModel):
    """Request to merge multiple contexts."""
    
    base_context_id: UUID = Field(..., description="Base context to merge into")
    merge_context_ids: List[UUID] = Field(..., description="Contexts to merge")
    merge_strategy: str = Field(default="deep", description="Merge strategy")
    conflict_resolution: str = Field(default="prioritize_base", description="Conflict resolution")
    preserve_metadata: bool = Field(default=True, description="Preserve metadata")
    
    @field_validator('merge_context_ids')
    @classmethod
    def validate_merge_contexts(cls, v: List[UUID]) -> List[UUID]:
        if not v:
            raise ValueError("At least one context to merge is required")
        if len(v) > 10:
            raise ValueError("Cannot merge more than 10 contexts at once")
        return v


class ContextMergeResponse(BaseModel):
    """Response from context merge operation."""
    
    merged_context: ContextResponse = Field(..., description="Resulting merged context")
    merge_summary: Dict[str, Any] = Field(..., description="Merge operation summary")
    conflicts_resolved: List[Dict[str, Any]] = Field(..., description="Conflicts that were resolved")
    merge_time_ms: int = Field(..., description="Time taken to merge")
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Conversation Models
# ============================================================================

class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    
    title: str = Field(..., min_length=1, max_length=255, description="Conversation title")
    description: Optional[str] = Field(None, description="Conversation description")
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    project_id: Optional[str] = Field(None, max_length=255, description="Project ID")
    session_id: Optional[str] = Field(None, max_length=255, description="Session ID")
    priority: ConversationPriority = Field(default=ConversationPriority.NORMAL, description="Priority level")
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Conversation settings")
    context_ids: List[str] = Field(default_factory=list, description="Associated context IDs")
    active_prompt_template_id: Optional[UUID] = Field(None, description="Active prompt template")


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[ConversationStatus] = None
    priority: Optional[ConversationPriority] = None
    conversation_metadata: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    context_ids: Optional[List[str]] = None
    active_prompt_template_id: Optional[UUID] = None


class ConversationResponse(PCSResponse):
    """Response model for conversations."""
    
    title: str = Field(..., description="Conversation title")
    description: Optional[str] = Field(None, description="Conversation description")
    user_id: str = Field(..., description="User ID")
    project_id: Optional[str] = Field(None, description="Project ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    status: ConversationStatus = Field(..., description="Current status")
    priority: ConversationPriority = Field(..., description="Priority level")
    conversation_metadata: Dict[str, Any] = Field(..., description="Conversation metadata")
    settings: Dict[str, Any] = Field(..., description="Conversation settings")
    context_ids: List[str] = Field(..., description="Associated context IDs")
    active_prompt_template_id: Optional[UUID] = Field(None, description="Active prompt template")
    started_at: datetime = Field(..., description="Start time")
    last_activity_at: datetime = Field(..., description="Last activity time")
    ended_at: Optional[datetime] = Field(None, description="End time")
    message_count: int = Field(..., description="Number of messages")
    total_tokens: int = Field(..., description="Total tokens used")


class MessageCreate(BaseModel):
    """Schema for creating a conversation message."""
    
    role: MessageRole = Field(..., description="Message sender role")
    content: str = Field(..., min_length=1, description="Message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Message type")
    raw_content: Optional[str] = Field(None, description="Raw content before processing")
    message_metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    prompt_template_id: Optional[UUID] = Field(None, description="Prompt template used")
    context_ids: List[str] = Field(default_factory=list, description="Context IDs used")
    parent_message_id: Optional[UUID] = Field(None, description="Parent message for threading")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class MessageResponse(PCSResponse):
    """Response model for conversation messages."""
    
    conversation_id: UUID = Field(..., description="Parent conversation ID")
    sequence_number: int = Field(..., description="Message sequence number")
    role: MessageRole = Field(..., description="Message sender role")
    message_type: MessageType = Field(..., description="Message type")
    content: str = Field(..., description="Message content")
    raw_content: Optional[str] = Field(None, description="Raw content")
    message_metadata: Dict[str, Any] = Field(..., description="Message metadata")
    prompt_template_id: Optional[UUID] = Field(None, description="Prompt template used")
    context_ids: List[str] = Field(..., description="Context IDs used")
    input_tokens: Optional[int] = Field(None, description="Input tokens")
    output_tokens: Optional[int] = Field(None, description="Output tokens")
    total_tokens: Optional[int] = Field(None, description="Total tokens")
    cost: Optional[float] = Field(None, description="Processing cost")
    model_used: Optional[str] = Field(None, description="AI model used")
    processing_time_ms: Optional[int] = Field(None, description="Processing time")
    is_edited: bool = Field(..., description="Edit status")
    is_deleted: bool = Field(..., description="Delete status")
    parent_message_id: Optional[UUID] = Field(None, description="Parent message ID")


# Update forward references
PromptResponse.model_rebuild()
ContextMergeResponse.model_rebuild()