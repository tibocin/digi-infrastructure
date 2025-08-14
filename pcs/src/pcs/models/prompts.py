"""
Filepath: src/pcs/models/prompts.py
Purpose: Data models for prompt templates, versions, and rules
Related Components: Template engine, Rule engine, Context management
Tags: prompts, templates, versioning, rules, sqlalchemy
"""

import enum
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import String, Text, Integer, Boolean, Enum, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .base import BaseModel, GUID


class PromptStatus(enum.Enum):
    """Status of a prompt template."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class RulePriority(enum.Enum):
    """Priority levels for prompt rules."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PromptTemplate(BaseModel):
    """
    Main prompt template model storing template metadata and relationships.
    """
    __tablename__ = "prompt_templates"

    # Core fields
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Unique name for the prompt template"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of the prompt template's purpose"
    )
    
    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Category for grouping templates"
    )
    
    tags: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Tags for searching and filtering"
    )
    
    status: Mapped[PromptStatus] = mapped_column(
        Enum(PromptStatus),
        default=PromptStatus.DRAFT,
        nullable=False,
        index=True,
        comment="Current status of the template"
    )
    
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this is a system template"
    )
    
    # Metadata
    author: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Author of the template"
    )
    
    version_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of versions for this template"
    )

    # Relationships
    versions: Mapped[List["PromptVersion"]] = relationship(
        "PromptVersion",
        back_populates="template_rel",
        cascade="all, delete-orphan",
        order_by="PromptVersion.version_number.desc()"
    )
    
    rules: Mapped[List["PromptRule"]] = relationship(
        "PromptRule",
        back_populates="template",
        cascade="all, delete-orphan"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_prompt_templates_name_status", "name", "status"),
        Index("idx_prompt_templates_category_status", "category", "status"),
    )

    @property
    def current_version(self) -> Optional["PromptVersion"]:
        """Get the current active version of this template."""
        for version in self.versions:
            if version.is_active:
                return version
        return None

    @property
    def latest_version(self) -> Optional["PromptVersion"]:
        """Get the latest version by version number."""
        if self.versions:
            return self.versions[0]  # Already ordered by version_number desc
        return None


class PromptVersion(BaseModel):
    """
    Version of a prompt template with actual template content.
    """
    __tablename__ = "prompt_versions"

    # Foreign key
    template_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the prompt template"
    )

    # Version info
    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Version number (incremental)"
    )
    
    version_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Human-readable version name"
    )

    # Template content
    template: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The actual prompt template content"
    )
    
    variables: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of variables used in the template"
    )
    
    # Status and metadata
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether this version is currently active"
    )
    
    change_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Notes about changes in this version"
    )
    
    # Performance metadata
    usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times this version has been used"
    )
    
    success_rate: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Success rate of this template version"
    )

    # Relationships
    template_rel: Mapped[PromptTemplate] = relationship(
        "PromptTemplate",
        back_populates="versions"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_prompt_versions_template_version", "template_id", "version_number"),
        Index("idx_prompt_versions_template_active", "template_id", "is_active"),
    )


class PromptRule(BaseModel):
    """
    Rules that control when and how prompt templates are used.
    """
    __tablename__ = "prompt_rules"

    # Foreign key
    template_id: Mapped[UUID] = mapped_column(
        GUID(),
        ForeignKey("prompt_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the prompt template"
    )

    # Rule definition
    rule_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name of the rule"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Description of what the rule does"
    )
    
    conditions: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Conditions that trigger this rule"
    )
    
    actions: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Actions to take when rule matches"
    )
    
    priority: Mapped[RulePriority] = mapped_column(
        Enum(RulePriority),
        default=RulePriority.MEDIUM,
        nullable=False,
        index=True,
        comment="Priority of this rule"
    )
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        comment="Whether this rule is currently active"
    )
    
    # Execution metadata
    execution_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times this rule has been executed"
    )
    
    last_executed: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="When this rule was last executed"
    )

    # Relationships
    template: Mapped[PromptTemplate] = relationship(
        "PromptTemplate",
        back_populates="rules"
    )

    # Table indexes
    __table_args__ = (
        Index("idx_prompt_rules_template_active", "template_id", "is_active"),
        Index("idx_prompt_rules_priority_active", "priority", "is_active"),
    )
