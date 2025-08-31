"""
Filepath: src/pcs/api/v1/conversations.py
Purpose: REST API endpoints for conversation management with lifecycle and history tracking
Related Components: Conversation models, Message models, Context integration, Prompt templates
Tags: api, conversations, messages, chat, history, lifecycle, fastapi
"""

import time
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ...core.exceptions import PCSError, ValidationError
from ...models.conversations import (
    Conversation, ConversationMessage, 
    ConversationStatus, ConversationPriority, 
    MessageRole, MessageType
)
from ...repositories.postgres_repo import PostgreSQLRepository
from ..dependencies import (
    get_database_session,
    get_current_user,
    get_current_app,
    validate_pagination
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


# Pydantic schemas for request/response validation
class ConversationBase(BaseModel):
    """Base schema for conversation operations."""
    title: str = Field(..., min_length=1, max_length=255, description="Title of the conversation")
    description: Optional[str] = Field(None, description="Description of the conversation's purpose")
    project_id: Optional[str] = Field(None, max_length=255, description="Project this conversation belongs to")
    session_id: Optional[str] = Field(None, max_length=255, description="Session ID for grouping conversations")
    priority: ConversationPriority = Field(default=ConversationPriority.NORMAL, description="Priority level")
    conversation_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Conversation settings")
    context_ids: Optional[List[str]] = Field(default_factory=list, description="Associated context IDs")
    active_prompt_template_id: Optional[UUID] = Field(None, description="Active prompt template")


class ConversationCreate(ConversationBase):
    """Schema for creating a new conversation."""
    pass


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    project_id: Optional[str] = Field(None, max_length=255)
    session_id: Optional[str] = Field(None, max_length=255)
    status: Optional[ConversationStatus] = None
    priority: Optional[ConversationPriority] = None
    conversation_metadata: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    context_ids: Optional[List[str]] = None
    active_prompt_template_id: Optional[UUID] = None


class MessageBase(BaseModel):
    """Base schema for message operations."""
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="The actual message content")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message content")
    raw_content: Optional[str] = Field(None, description="Raw content before processing")
    message_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")
    prompt_template_id: Optional[UUID] = Field(None, description="Prompt template used")
    context_ids: Optional[List[str]] = Field(default_factory=list, description="Context IDs used")
    input_tokens: Optional[int] = Field(None, ge=0, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, ge=0, description="Number of output tokens")
    
    @field_validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Message content cannot be empty")
        return v


class MessageCreate(MessageBase):
    """Schema for creating a new message."""
    pass


class MessageUpdate(BaseModel):
    """Schema for updating a message."""
    content: Optional[str] = Field(None, min_length=1)
    raw_content: Optional[str] = None
    message_metadata: Optional[Dict[str, Any]] = None
    input_tokens: Optional[int] = Field(None, ge=0)
    output_tokens: Optional[int] = Field(None, ge=0)


class ConversationSearchRequest(BaseModel):
    """Schema for conversation search operations."""
    query: Optional[str] = Field(None, description="Text query for searching")
    user_ids: Optional[List[str]] = Field(None, description="Filter by user IDs")
    project_ids: Optional[List[str]] = Field(None, description="Filter by project IDs")
    session_ids: Optional[List[str]] = Field(None, description="Filter by session IDs")
    statuses: Optional[List[ConversationStatus]] = Field(None, description="Filter by statuses")
    priorities: Optional[List[ConversationPriority]] = Field(None, description="Filter by priorities")
    date_from: Optional[datetime] = Field(None, description="Filter by start date from")
    date_to: Optional[datetime] = Field(None, description="Filter by start date to")
    min_messages: Optional[int] = Field(None, ge=0, description="Minimum message count")
    max_messages: Optional[int] = Field(None, ge=0, description="Maximum message count")
    include_archived: bool = Field(default=False, description="Include archived conversations")


class ConversationExportRequest(BaseModel):
    """Schema for conversation export operations."""
    format: str = Field(..., pattern="^(json|csv|txt|markdown)$", description="Export format")
    include_metadata: bool = Field(default=True, description="Include metadata in export")
    include_context: bool = Field(default=False, description="Include context information")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range for export")


# Response schemas
class MessageResponse(BaseModel):
    """Response schema for conversation message."""
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sequence_number: int
    prompt_template_id: Optional[UUID] = None
    context_ids: Optional[List[str]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    """Response schema for conversation."""
    id: UUID
    title: str
    description: Optional[str] = None
    user_id: str
    is_active: bool = True
    total_messages: int = 0
    total_tokens: int = 0
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Nested responses
    messages: Optional[List[MessageResponse]] = None
    latest_message: Optional[MessageResponse] = None
    
    # Timestamps
    started_at: datetime
    last_activity_at: datetime
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ConversationStatsResponse(BaseModel):
    """Response schema for conversation statistics."""
    conversation_id: UUID
    total_messages: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: Optional[float] = None
    average_response_time: Optional[float] = None
    active_contexts: int
    prompt_templates_used: int
    
    model_config = ConfigDict(from_attributes=True)


class PaginatedConversationsResponse(BaseModel):
    """Paginated response for conversations."""
    items: List[ConversationResponse]
    total: int
    page: int
    size: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


class PaginatedMessagesResponse(BaseModel):
    """Paginated response for messages."""
    items: List[MessageResponse]
    total: int
    page: int
    size: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


# Conversation Management Endpoints

@router.get("/", response_model=PaginatedConversationsResponse)
async def list_conversations(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    status: Optional[ConversationStatus] = Query(None, description="Filter by status"),
    priority: Optional[ConversationPriority] = Query(None, description="Filter by priority"),
    search: Optional[str] = Query(None, description="Search in title and description"),
    include_messages: bool = Query(False, description="Include latest messages"),
    include_archived: bool = Query(False, description="Include archived conversations"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedConversationsResponse:
    """
    List conversations with filtering and pagination.
    
    **Features:**
    - Comprehensive filtering options
    - Search functionality
    - Optional inclusion of latest messages
    - Access control based on user permissions
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        
        # Build filter criteria
        filters = {}
        if user_id:
            filters['user_id'] = user_id
        elif not (current_user.get('is_admin') if current_user else False):
            # Non-admin users can only see their own conversations
            filters['user_id'] = current_user.get('id') if current_user else None
        
        if project_id:
            filters['project_id'] = project_id
        if session_id:
            filters['session_id'] = session_id
        if status:
            filters['status'] = status
        if priority:
            filters['priority'] = priority
        
        # Exclude archived unless explicitly requested
        if not include_archived:
            filters['status__ne'] = ConversationStatus.ARCHIVED
        
        # Get results
        conversations = await repository.find_by_criteria(**filters)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            conversations = [
                c for c in conversations 
                if (search_lower in c.title.lower() if c.title else False) or 
                   (search_lower in c.description.lower() if c.description else False)
            ]
        
        # Sort by last activity (most recent first)
        conversations.sort(key=lambda x: x.last_activity_at, reverse=True)
        
        # Apply pagination
        total = len(conversations)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_conversations = conversations[start_idx:end_idx]
        
        # Convert to response schema
        response_conversations = []
        for conversation in paginated_conversations:
            conversation_data = ConversationResponse.model_validate(conversation)
            
            # Include latest message if requested
            if include_messages and conversation.messages:
                # Get the latest message
                latest_message = max(conversation.messages, key=lambda x: x.sequence_number)
                conversation_data.latest_message = MessageResponse.model_validate(latest_message)
            
            response_conversations.append(conversation_data)
        
        return PaginatedConversationsResponse(
            items=response_conversations,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list conversations: {str(e)}"
        )


@router.post("/", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    conversation_data: ConversationCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ConversationResponse:
    """
    Create a new conversation.
    
    **Features:**
    - Sets user association and timestamps
    - Validates metadata and settings
    - Initializes conversation state
    - Returns created conversation with metadata
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        
        # Create conversation
        new_conversation = Conversation(
            title=conversation_data.title,
            description=conversation_data.description,
            user_id=current_user.get('id') if current_user else None,
            project_id=conversation_data.project_id,
            session_id=conversation_data.session_id,
            status=ConversationStatus.ACTIVE,
            priority=conversation_data.priority,
            conversation_metadata=conversation_data.conversation_metadata or {},
            settings=conversation_data.settings or {},
            context_ids=conversation_data.context_ids or [],
            active_prompt_template_id=conversation_data.active_prompt_template_id,
            started_at=datetime.now(timezone.utc),
            last_activity_at=datetime.now(timezone.utc),
            message_count=0,
            total_tokens=0
        )
        
        created_conversation = await repository.create(new_conversation)
        return ConversationResponse.model_validate(created_conversation)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    include_messages: bool = Query(False, description="Include all messages"),
    include_stats: bool = Query(False, description="Include conversation statistics"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ConversationResponse:
    """
    Get a specific conversation by ID.
    
    **Features:**
    - Access control validation
    - Optional inclusion of all messages
    - Optional statistics inclusion
    - Activity tracking update
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        conversation = await repository.get_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        conversation_data = ConversationResponse.model_validate(conversation)
        
        # Include messages if requested
        if include_messages and conversation.messages:
            conversation_data.messages = [
                MessageResponse.model_validate(msg) for msg in conversation.messages
            ]
        
        return conversation_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation: {str(e)}"
        )


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    conversation_updates: ConversationUpdate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ConversationResponse:
    """
    Update a conversation.
    
    **Features:**
    - Partial updates supported
    - Access control validation
    - Activity timestamp updates
    - Status transition validation
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        
        # Check if conversation exists and user has access
        existing_conversation = await repository.get_by_id(conversation_id)
        if not existing_conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if existing_conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Prepare update data
        update_data = conversation_updates.model_dump(exclude_unset=True)
        
        # Update last activity timestamp
        update_data['last_activity_at'] = datetime.now(timezone.utc)
        
        # Handle status transitions
        if 'status' in update_data:
            new_status = update_data['status']
            if new_status == ConversationStatus.COMPLETED and not existing_conversation.ended_at:
                update_data['ended_at'] = datetime.now(timezone.utc)
            elif new_status == ConversationStatus.ACTIVE and existing_conversation.ended_at:
                update_data['ended_at'] = None
        
        # Update conversation
        updated_conversation = await repository.update(conversation_id, update_data)
        return ConversationResponse.model_validate(updated_conversation)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update conversation: {str(e)}"
        )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: UUID,
    force: bool = Query(False, description="Force delete even if conversation is active"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a conversation.
    
    **Features:**
    - Access control validation
    - Cascading deletion of messages
    - Protection for active conversations (unless forced)
    - Soft delete option (archive instead of delete)
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        
        # Check if conversation exists and user has access
        existing_conversation = await repository.get_by_id(conversation_id)
        if not existing_conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if existing_conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Prevent deletion of active conversations unless forced
        if existing_conversation.status == ConversationStatus.ACTIVE and not force:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active conversation. Complete or archive it first, or use force=true"
            )
        
        # Delete conversation (cascades to messages)
        deleted = await repository.delete(conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete conversation"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {str(e)}"
        )


# Message Management Endpoints

@router.get("/{conversation_id}/messages", response_model=PaginatedMessagesResponse)
async def list_messages(
    conversation_id: UUID,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=200, description="Page size"),
    role: Optional[MessageRole] = Query(None, description="Filter by message role"),
    message_type: Optional[MessageType] = Query(None, description="Filter by message type"),
    search: Optional[str] = Query(None, description="Search in message content"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedMessagesResponse:
    """
    List messages in a conversation with filtering and pagination.
    
    **Features:**
    - Chronological ordering
    - Role and type filtering
    - Content search
    - Access control validation
    """
    try:
        conv_repository = PostgreSQLRepository(db, Conversation)
        msg_repository = PostgreSQLRepository(db, ConversationMessage)
        
        # Verify conversation exists and user has access
        conversation = await conv_repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Build filter criteria
        filters = {'conversation_id': conversation_id}
        if role:
            filters['role'] = role
        if message_type:
            filters['message_type'] = message_type
        
        # Get messages
        messages = await msg_repository.find_by_criteria(**filters)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            messages = [
                m for m in messages 
                if search_lower in m.content.lower()
            ]
        
        # Sort by sequence number
        messages.sort(key=lambda x: x.sequence_number)
        
        # Apply pagination
        total = len(messages)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_messages = messages[start_idx:end_idx]
        
        # Convert to response schema
        response_messages = [MessageResponse.model_validate(msg) for msg in paginated_messages]
        
        return PaginatedMessagesResponse(
            items=response_messages,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list messages: {str(e)}"
        )


@router.post("/{conversation_id}/messages", response_model=MessageResponse, status_code=201)
async def create_message(
    conversation_id: UUID,
    message_data: MessageCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MessageResponse:
    """
    Add a new message to a conversation.
    
    **Features:**
    - Automatic sequence numbering
    - Token calculation and tracking
    - Conversation statistics updates
    - Activity timestamp updates
    """
    try:
        conv_repository = PostgreSQLRepository(db, Conversation)
        msg_repository = PostgreSQLRepository(db, ConversationMessage)
        
        # Verify conversation exists and user has access
        conversation = await conv_repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Check if conversation is active
        if conversation.status not in [ConversationStatus.ACTIVE, ConversationStatus.PAUSED]:
            raise HTTPException(
                status_code=400,
                detail="Cannot add messages to completed or archived conversations"
            )
        
        # Calculate total tokens if provided
        total_tokens = None
        if message_data.input_tokens is not None and message_data.output_tokens is not None:
            total_tokens = message_data.input_tokens + message_data.output_tokens
        elif message_data.input_tokens is not None:
            total_tokens = message_data.input_tokens
        elif message_data.output_tokens is not None:
            total_tokens = message_data.output_tokens
        
        # Create message
        new_message = ConversationMessage(
            conversation_id=conversation_id,
            sequence_number=conversation.message_count + 1,
            role=message_data.role,
            message_type=message_data.message_type,
            content=message_data.content,
            raw_content=message_data.raw_content,
            message_metadata=message_data.message_metadata or {},
            prompt_template_id=message_data.prompt_template_id,
            context_ids=message_data.context_ids or [],
            input_tokens=message_data.input_tokens,
            output_tokens=message_data.output_tokens,
            total_tokens=total_tokens,
            processing_time_ms=0.0  # Would be calculated by the service
        )
        
        created_message = await msg_repository.create(new_message)
        
        # Update conversation statistics
        conversation_updates = {
            'message_count': conversation.message_count + 1,
            'last_activity_at': datetime.now(timezone.utc)
        }
        
        if total_tokens:
            conversation_updates['total_tokens'] = conversation.total_tokens + total_tokens
        
        await conv_repository.update(conversation_id, conversation_updates)
        
        return MessageResponse.model_validate(created_message)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create message: {str(e)}"
        )


@router.get("/{conversation_id}/messages/{message_id}", response_model=MessageResponse)
async def get_message(
    conversation_id: UUID,
    message_id: UUID,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MessageResponse:
    """Get a specific message by ID."""
    try:
        conv_repository = PostgreSQLRepository(db, Conversation)
        msg_repository = PostgreSQLRepository(db, ConversationMessage)
        
        # Verify conversation exists and user has access
        conversation = await conv_repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Get message
        message = await msg_repository.get_by_id(message_id)
        if not message or message.conversation_id != conversation_id:
            raise HTTPException(
                status_code=404,
                detail=f"Message with ID {message_id} not found in this conversation"
            )
        
        return MessageResponse.model_validate(message)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get message: {str(e)}"
        )


# Conversation Statistics and Analytics

@router.get("/{conversation_id}/stats", response_model=ConversationStatsResponse)
async def get_conversation_stats(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ConversationStatsResponse:
    """
    Get detailed statistics for a conversation.
    
    **Features:**
    - Message and token statistics
    - Activity analysis
    - Role-based breakdowns
    - Timeline information
    """
    try:
        conv_repository = PostgreSQLRepository(db, Conversation)
        
        # Verify conversation exists and user has access
        conversation = await conv_repository.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation with ID {conversation_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id') if current_user else None
        if conversation.user_id != user_id and not (current_user.get('is_admin') if current_user else False):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this conversation"
            )
        
        # Calculate statistics
        messages = conversation.messages or []
        
        # Basic counts
        user_messages = len([m for m in messages if m.role == MessageRole.USER])
        assistant_messages = len([m for m in messages if m.role == MessageRole.ASSISTANT])
        system_messages = len([m for m in messages if m.role == MessageRole.SYSTEM])
        
        # Message types breakdown
        message_types = {}
        for msg_type in MessageType:
            count = len([m for m in messages if m.message_type == msg_type])
            if count > 0:
                message_types[msg_type.value] = count
        
        # Token usage by role
        token_usage_by_role = {}
        for role in MessageRole:
            role_tokens = sum(
                m.total_tokens or 0 
                for m in messages 
                if m.role == role and m.total_tokens
            )
            if role_tokens > 0:
                token_usage_by_role[role.value] = role_tokens
        
        # Duration calculation
        duration_minutes = None
        if conversation.ended_at:
            duration = conversation.ended_at - conversation.started_at
            duration_minutes = duration.total_seconds() / 60
        
        # Average tokens per message
        avg_tokens = 0.0
        if conversation.message_count > 0 and conversation.total_tokens > 0:
            avg_tokens = conversation.total_tokens / conversation.message_count
        
        # Activity timeline (simplified)
        activity_timeline = []
        for i, message in enumerate(messages[:10]):  # Last 10 messages
            activity_timeline.append({
                "sequence": message.sequence_number,
                "timestamp": message.created_at.isoformat(),
                "role": message.role.value,
                "type": message.message_type.value,
                "tokens": message.total_tokens or 0
            })
        
        return ConversationStatsResponse(
            conversation_id=conversation_id,
            message_count=conversation.message_count,
            total_tokens=conversation.total_tokens,
            average_tokens_per_message=avg_tokens,
            duration_minutes=duration_minutes,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            system_messages=system_messages,
            message_types=message_types,
            token_usage_by_role=token_usage_by_role,
            activity_timeline=activity_timeline
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation stats: {str(e)}"
        )


# Conversation Search and Export

@router.post("/search", response_model=PaginatedConversationsResponse)
async def search_conversations(
    search_request: ConversationSearchRequest,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedConversationsResponse:
    """
    Advanced conversation search with multiple filters.
    
    **Features:**
    - Text search across titles and content
    - Multiple filter combinations
    - Date range filtering
    - Message count filtering
    - Status and priority filtering
    """
    try:
        repository = PostgreSQLRepository(db, Conversation)
        
        # Start with user's conversations (unless admin)
        if not (current_user.get('is_admin') if current_user else False):
            user_conversations = await repository.find_by_criteria(user_id=current_user.get('id') if current_user else None)
        else:
            user_conversations = await repository.find_by_criteria()
        
        conversations = user_conversations
        
        # Apply filters
        if search_request.user_ids and (current_user.get('is_admin') if current_user else False):
            conversations = [c for c in conversations if c.user_id in search_request.user_ids]
        
        if search_request.project_ids:
            conversations = [c for c in conversations if c.project_id in search_request.project_ids]
        
        if search_request.session_ids:
            conversations = [c for c in conversations if c.session_id in search_request.session_ids]
        
        if search_request.statuses:
            conversations = [c for c in conversations if c.status in search_request.statuses]
        
        if search_request.priorities:
            conversations = [c for c in conversations if c.priority in search_request.priorities]
        
        if not search_request.include_archived:
            conversations = [c for c in conversations if c.status != ConversationStatus.ARCHIVED]
        
        # Apply text search
        if search_request.query:
            query_lower = search_request.query.lower()
            conversations = [
                c for c in conversations
                if (query_lower in c.title.lower() if c.title else False) or
                   (query_lower in c.description.lower() if c.description else False)
            ]
        
        # Apply date filtering
        if search_request.date_from:
            conversations = [c for c in conversations if c.started_at >= search_request.date_from]
        
        if search_request.date_to:
            conversations = [c for c in conversations if c.started_at <= search_request.date_to]
        
        # Apply message count filtering
        if search_request.min_messages is not None:
            conversations = [c for c in conversations if c.message_count >= search_request.min_messages]
        
        if search_request.max_messages is not None:
            conversations = [c for c in conversations if c.message_count <= search_request.max_messages]
        
        # Sort by last activity (most recent first)
        conversations.sort(key=lambda x: x.last_activity_at, reverse=True)
        
        # Apply pagination
        total = len(conversations)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_conversations = conversations[start_idx:end_idx]
        
        # Convert to response schema
        response_conversations = [ConversationResponse.model_validate(c) for c in paginated_conversations]
        
        return PaginatedConversationsResponse(
            items=response_conversations,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search conversations: {str(e)}"
        )
