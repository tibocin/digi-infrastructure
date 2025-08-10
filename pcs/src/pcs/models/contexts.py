"""
Filepath: src/pcs/models/contexts.py
Purpose: Data models for context types, instances, and relationships
Related Components: Context management, Hierarchical contexts, Vector storage
Tags: contexts, relationships, hierarchy, sqlalchemy
"""

import enum
from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Text, JSON, Enum, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .base import BaseModel, GUID


class ContextTypeEnum(enum.Enum):
    """Types of contexts in the system."""
    PROJECT = "project"
    FILE = "file" 
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    SYSTEM_STATE = "system_state"
    CUSTOM = "custom"


class ContextScope(enum.Enum):
    """Scope/visibility of context."""
    GLOBAL = "global"
    PROJECT = "project"
    USER = "user"
    SESSION = "session"
    PRIVATE = "private"


class RelationshipType(enum.Enum):
    """Types of relationships between contexts."""
    PARENT_CHILD = "parent_child"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    CONFLICTS_WITH = "conflicts_with"
    ENHANCES = "enhances"
    REPLACES = "replaces"
    REFERENCES = "references"


class ContextType(BaseModel):
    """
    Defines types of contexts with their schemas and validation rules.
    """
    __tablename__ = "context_types"

    # Core fields
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique name for the context type"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of this context type"
    )
    
    type_enum: Mapped[ContextTypeEnum] = mapped_column(
        Enum(ContextTypeEnum),
        nullable=False,
        index=True,
        comment="Enum value for this context type"
    )
    
    # Schema and validation
    schema_definition: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="JSON schema for validating context data"
    )
    
    default_scope: Mapped[ContextScope] = mapped_column(
        Enum(ContextScope),
        default=ContextScope.USER,
        nullable=False,
        comment="Default scope for contexts of this type"
    )
    
    # Configuration
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this context type is active"
    )
    
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this is a system-defined type"
    )
    
    # Vector storage configuration
    supports_vectors: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether contexts of this type support vector embeddings"
    )
    
    vector_dimension: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Dimension of vector embeddings for this type"
    )

    # Relationships
    contexts: Mapped[List["Context"]] = relationship(
        "Context",
        back_populates="context_type",
        cascade="all, delete-orphan"
    )


class Context(BaseModel):
    """
    Individual context instances with data and relationships.
    """
    __tablename__ = "contexts"

    # Foreign key
    context_type_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("context_types.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the context type"
    )

    # Core identification
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Name of this context instance"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of this context"
    )
    
    # Scope and ownership
    scope: Mapped[ContextScope] = mapped_column(
        Enum(ContextScope),
        nullable=False,
        index=True,
        comment="Scope/visibility of this context"
    )
    
    owner_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="ID of the user/entity that owns this context"
    )
    
    project_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Project this context belongs to"
    )

    # Data and metadata
    context_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="The actual context data"
    )
    
    context_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about the context"
    )
    
    # Status and configuration
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this context is active"
    )
    
    priority: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Priority/weight of this context"
    )
    
    # Vector embedding data
    vector_embedding: Mapped[Optional[List[float]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Vector embedding for semantic search"
    )
    
    embedding_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Model used to generate the embedding"
    )
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
        comment="Number of times this context has been used"
    )

    # Relationships
    context_type: Mapped[ContextType] = relationship(
        "ContextType",
        back_populates="contexts"
    )
    
    # Self-referential relationships for hierarchy
    parent_relationships: Mapped[List["ContextRelationship"]] = relationship(
        "ContextRelationship",
        foreign_keys="ContextRelationship.child_context_id",
        back_populates="child_context",
        cascade="all, delete-orphan"
    )
    
    child_relationships: Mapped[List["ContextRelationship"]] = relationship(
        "ContextRelationship",
        foreign_keys="ContextRelationship.parent_context_id", 
        back_populates="parent_context",
        cascade="all, delete-orphan"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_contexts_type_scope", "context_type_id", "scope"),
        Index("idx_contexts_owner_project", "owner_id", "project_id"),
        Index("idx_contexts_name_active", "name", "is_active"),
    )


class ContextRelationship(BaseModel):
    """
    Relationships between context instances.
    """
    __tablename__ = "context_relationships"

    # Foreign keys
    parent_context_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("contexts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent context in the relationship"
    )
    
    child_context_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("contexts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Child context in the relationship"
    )

    # Relationship definition
    relationship_type: Mapped[RelationshipType] = mapped_column(
        Enum(RelationshipType),
        nullable=False,
        index=True,
        comment="Type of relationship between contexts"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of this relationship"
    )
    
    # Configuration
    strength: Mapped[float] = mapped_column(
        default=1.0,
        nullable=False,
        comment="Strength/weight of this relationship (0.0-1.0)"
    )
    
    is_bidirectional: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="Whether this relationship works in both directions"
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this relationship is active"
    )
    
    # Metadata
    relationship_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Additional metadata about the relationship"
    )

    # Relationships
    parent_context: Mapped[Context] = relationship(
        "Context",
        foreign_keys=[parent_context_id],
        back_populates="child_relationships"
    )
    
    child_context: Mapped[Context] = relationship(
        "Context", 
        foreign_keys=[child_context_id],
        back_populates="parent_relationships"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_context_relationships_parent_type", "parent_context_id", "relationship_type"),
        Index("idx_context_relationships_child_type", "child_context_id", "relationship_type"),
        Index("idx_context_relationships_type_active", "relationship_type", "is_active"),
    )
