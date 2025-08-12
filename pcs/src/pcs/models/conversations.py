"""
Filepath: src/pcs/models/conversations.py
Purpose: Data models for conversations and messages
Related Components: Chat interface, Message history, User interactions
Tags: conversations, messages, chat, history, sqlalchemy
"""

import enum
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Text, JSON, Enum, Boolean, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .base import BaseModel, GUID


class ConversationStatus(enum.Enum):
    """Status of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


class ConversationPriority(enum.Enum):
    """Priority levels for conversations."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageRole(enum.Enum):
    """Role of the message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class MessageType(enum.Enum):
    """Type of message content."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FILE = "file"
    COMMAND = "command"
    ERROR = "error"
    SYSTEM_NOTIFICATION = "system_notification"


class Conversation(BaseModel):
    """
    Conversation model for tracking chat sessions and context.
    """
    __tablename__ = "conversations"

    # Core identification
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Title of the conversation"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of the conversation's purpose"
    )

    # User and project association
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="ID of the user who started the conversation"
    )
    
    project_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Project this conversation belongs to"
    )
    
    session_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Session ID for grouping related conversations"
    )

    # Status and metadata
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(ConversationStatus),
        default=ConversationStatus.ACTIVE,
        nullable=False,
        index=True,
        comment="Current status of the conversation"
    )
    
    priority: Mapped[ConversationPriority] = mapped_column(
        Enum(ConversationPriority),
        default=ConversationPriority.NORMAL,
        nullable=False,
        index=True,
        comment="Priority level of the conversation"
    )
    
    # Configuration
    conversation_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about the conversation"
    )
    
    settings: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Conversation-specific settings and preferences"
    )
    
    # Context and prompts
    context_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of context IDs associated with this conversation"
    )
    
    active_prompt_template_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(),
        nullable=True,
        comment="Currently active prompt template for this conversation"
    )
    
    # Timing information
    started_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="When the conversation was started"
    )
    
    last_activity_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Last activity timestamp"
    )
    
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="When the conversation ended"
    )
    
    # Statistics
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of messages in this conversation"
    )
    
    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Total tokens used in this conversation"
    )

    # Relationships
    messages: Mapped[List["ConversationMessage"]] = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.sequence_number"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_conversations_user_status", "user_id", "status"),
        Index("idx_conversations_project_status", "project_id", "status"),
        Index("idx_conversations_session_status", "session_id", "status"),
        Index("idx_conversations_started_at", "started_at"),
    )

    def add_message(
        self,
        role: MessageRole,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        **kwargs
    ) -> "ConversationMessage":
        """Add a new message to this conversation."""
        message = ConversationMessage(
            conversation_id=self.id,
            role=role,
            content=content,
            message_type=message_type,
            sequence_number=self.message_count + 1,
            **kwargs
        )
        
        self.messages.append(message)
        self.message_count += 1
        self.last_activity_at = datetime.now(timezone.utc)
        
        return message


class ConversationMessage(BaseModel):
    """
    Individual messages within a conversation.
    """
    __tablename__ = "conversation_messages"

    # Foreign key
    conversation_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the conversation"
    )

    # Message identification
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Sequential number of this message in the conversation"
    )
    
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole),
        nullable=False,
        index=True,
        comment="Role of the message sender"
    )
    
    message_type: Mapped[MessageType] = mapped_column(
        Enum(MessageType),
        default=MessageType.TEXT,
        nullable=False,
        index=True,
        comment="Type of message content"
    )

    # Content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The actual message content"
    )
    
    raw_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Raw/original content before processing"
    )
    
    # Metadata
    message_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about the message"
    )
    
    # Processing information
    prompt_template_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(),
        nullable=True,
        comment="Prompt template used to generate this message"
    )
    
    context_ids: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Context IDs used for this message"
    )
    
    # Token and cost tracking
    input_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of input tokens for this message"
    )
    
    output_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of output tokens for this message"
    )
    
    total_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total tokens used for this message"
    )
    
    cost: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Cost associated with processing this message"
    )
    
    # Processing metadata
    model_used: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="AI model used to generate this message"
    )
    
    processing_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Time taken to process this message in milliseconds"
    )
    
    # Status flags
    is_edited: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this message has been edited"
    )
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this message has been deleted"
    )
    
    # Parent message for threading
    parent_message_id: Mapped[Optional[UUID]] = mapped_column(
        GUID(),
        ForeignKey("conversation_messages.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent message ID for threaded conversations"
    )

    # Relationships
    conversation: Mapped[Conversation] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    # Self-referential relationship for threading
    parent_message: Mapped[Optional["ConversationMessage"]] = relationship(
        "ConversationMessage",
        remote_side="ConversationMessage.id",
        back_populates="child_messages"
    )
    
    child_messages: Mapped[List["ConversationMessage"]] = relationship(
        "ConversationMessage",
        back_populates="parent_message",
        cascade="all, delete-orphan"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_conversation_messages_conv_seq", "conversation_id", "sequence_number"),
        Index("idx_conversation_messages_role_type", "role", "message_type"),
        Index("idx_conversation_messages_parent", "parent_message_id"),
        Index("idx_conversation_messages_created", "created_at"),
    )
