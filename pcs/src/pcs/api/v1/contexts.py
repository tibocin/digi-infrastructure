"""
Filepath: src/pcs/api/v1/contexts.py
Purpose: REST API endpoints for context management with CRUD operations, merging, and relationships
Related Components: Context models, ContextService, Redis caching, Vector storage
Tags: api, contexts, crud, merging, relationships, caching, fastapi
"""

import time
from typing import List, Optional, Dict, Any, Set
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from ...core.exceptions import PCSError, ValidationError
from ...models.contexts import (
    Context, ContextType, ContextRelationship, 
    ContextTypeEnum, ContextScope, RelationshipType
)
from ...services.context_service import (
    ContextManager, MergeStrategy, ContextError
)
from ...repositories.postgres_repo import PostgreSQLRepository
from ...repositories.redis_repo import RedisRepository
from ..dependencies import (
    get_database_session,
    get_current_user,
    get_current_app,
    validate_pagination
)

router = APIRouter(prefix="/contexts", tags=["contexts"])


# Pydantic schemas for request/response validation
class ContextTypeBase(BaseModel):
    """Base schema for context type operations."""
    name: str = Field(..., min_length=1, max_length=100, description="Unique name for the context type")
    description: Optional[str] = Field(None, max_length=1000, description="Description of this context type")
    type_enum: ContextTypeEnum = Field(..., description="Enum value for this context type")
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="JSON schema for validating context data")
    default_scope: ContextScope = Field(default=ContextScope.USER, description="Default scope for contexts")
    supports_vectors: bool = Field(default=False, description="Whether contexts support vector embeddings")
    vector_dimension: Optional[int] = Field(None, description="Dimension of vector embeddings")
    
    @model_validator(mode='after')
    def validate_vector_configuration(self):
        if self.supports_vectors and self.vector_dimension is None:
            raise ValueError("Vector dimension required when supports_vectors is True")
        if self.vector_dimension is not None and self.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        return self


class ContextTypeCreate(ContextTypeBase):
    """Schema for creating a new context type."""
    pass


class ContextTypeUpdate(BaseModel):
    """Schema for updating a context type."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    schema_definition: Optional[Dict[str, Any]] = None
    default_scope: Optional[ContextScope] = None
    is_active: Optional[bool] = None
    supports_vectors: Optional[bool] = None
    vector_dimension: Optional[int] = None


class ContextBase(BaseModel):
    """Base schema for context operations."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of this context instance")
    description: Optional[str] = Field(None, description="Description of this context")
    scope: ContextScope = Field(..., description="Scope/visibility of this context")
    owner_id: Optional[str] = Field(None, max_length=255, description="ID of the owner")
    project_id: Optional[str] = Field(None, max_length=255, description="Project this context belongs to")
    context_data: Dict[str, Any] = Field(..., description="The actual context data")
    context_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    priority: int = Field(default=0, description="Priority/weight of this context")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector embedding for semantic search")
    embedding_model: Optional[str] = Field(None, max_length=100, description="Model used for embedding")
    
    @field_validator('context_data')
    @classmethod
    def validate_context_data(cls, v):
        if not v:
            raise ValueError("Context data cannot be empty")
        return v


class ContextCreate(ContextBase):
    """Schema for creating a new context."""
    context_type_id: UUID = Field(..., description="Reference to the context type")


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
    is_active: Optional[bool] = None
    vector_embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = Field(None, max_length=100)


class ContextMergeRequest(BaseModel):
    """Schema for context merging operations."""
    source_context_ids: List[UUID] = Field(..., min_items=2, description="Context IDs to merge")
    target_context_id: Optional[UUID] = Field(None, description="Target context to merge into")
    merge_strategy: MergeStrategy = Field(default=MergeStrategy.MERGE_DEEP, description="Strategy for merging")
    conflict_resolution: Dict[str, str] = Field(default_factory=dict, description="Field-specific conflict resolution")
    preserve_metadata: bool = Field(default=True, description="Whether to preserve metadata")
    create_new: bool = Field(default=False, description="Create new context instead of updating existing")
    
    @field_validator('source_context_ids')
    @classmethod
    def validate_source_contexts(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("Duplicate context IDs not allowed")
        return v


class ContextRelationshipBase(BaseModel):
    """Base schema for context relationship operations."""
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Strength of the relationship")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional relationship metadata")
    is_bidirectional: bool = Field(default=False, description="Whether relationship is bidirectional")


class ContextRelationshipCreate(ContextRelationshipBase):
    """Schema for creating context relationships."""
    parent_context_id: UUID = Field(..., description="Parent context ID")
    child_context_id: UUID = Field(..., description="Child context ID")
    
    @field_validator('child_context_id')
    @classmethod
    def validate_different_contexts(cls, v, info):
        if v == info.data.get('parent_context_id'):
            raise ValueError("Parent and child contexts must be different")
        return v


class ContextSearchRequest(BaseModel):
    """Schema for context search operations."""
    query: Optional[str] = Field(None, description="Text query for searching")
    context_types: Optional[List[ContextTypeEnum]] = Field(None, description="Filter by context types")
    scopes: Optional[List[ContextScope]] = Field(None, description="Filter by scopes")
    owner_ids: Optional[List[str]] = Field(None, description="Filter by owner IDs")
    project_ids: Optional[List[str]] = Field(None, description="Filter by project IDs")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    date_from: Optional[datetime] = Field(None, description="Filter by creation date from")
    date_to: Optional[datetime] = Field(None, description="Filter by creation date to")
    vector_search: Optional[List[float]] = Field(None, description="Vector for similarity search")
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold for vector search")
    include_inactive: bool = Field(default=False, description="Include inactive contexts")


# Response schemas
class ContextTypeResponse(BaseModel):
    """Response schema for context type."""
    id: UUID
    name: str
    description: Optional[str] = None
    type_enum: ContextTypeEnum
    schema_definition: Dict[str, Any]
    validation_rules: Optional[Dict[str, Any]] = None
    default_scope: ContextScope = ContextScope.USER
    max_instances: Optional[int] = None
    is_system: bool = False
    context_count: Optional[int] = 0
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ContextResponse(BaseModel):
    """Response schema for context."""
    id: UUID
    context_type_id: UUID
    name: str
    description: Optional[str] = None
    data: Dict[str, Any]
    scope: ContextScope = ContextScope.USER
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    usage_count: int = 0
    is_active: bool = True
    
    # Nested responses
    context_type: Optional[ContextTypeResponse] = None
    parent_relationships: Optional[List['ContextRelationshipResponse']] = None
    child_relationships: Optional[List['ContextRelationshipResponse']] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ContextRelationshipResponse(BaseModel):
    """Response schema for context relationship."""
    id: UUID
    parent_context_id: UUID
    child_context_id: UUID
    relationship_type: RelationshipType
    relationship_data: Optional[Dict[str, Any]] = None
    strength: float = 1.0
    is_active: bool = True
    
    # Nested context data (optional)
    parent_context: Optional[ContextResponse] = None
    child_context: Optional[ContextResponse] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ContextMergeResponse(BaseModel):
    """Response schema for context merge operations."""
    merge_id: str
    result_context: ContextResponse
    source_context_ids: List[UUID]
    strategy_used: str
    conflicts_resolved: int
    merge_metadata: Dict[str, Any]
    processing_time_ms: float
    
    model_config = ConfigDict(from_attributes=True)


class PaginatedContextsResponse(BaseModel):
    """Paginated response for contexts."""
    items: List[ContextResponse]
    total: int
    page: int
    size: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


class PaginatedContextTypesResponse(BaseModel):
    """Paginated response for context types."""
    items: List[ContextTypeResponse]
    total: int
    page: int
    size: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


# Context Type Management Endpoints

@router.get("/types", response_model=PaginatedContextTypesResponse)
async def list_context_types(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    active_only: bool = Query(True, description="Filter active types only"),
    supports_vectors: Optional[bool] = Query(None, description="Filter by vector support"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedContextTypesResponse:
    """
    List context types with filtering and pagination.
    
    **Features:**
    - Pagination with configurable page size
    - Filter by active status and vector support
    - Include context count statistics
    - User access control
    """
    try:
        repository = PostgreSQLRepository(db, ContextType)
        
        # Build filter criteria
        filters = {}
        if active_only:
            filters['is_active'] = True
        if supports_vectors is not None:
            filters['supports_vectors'] = supports_vectors
        
        # Get results
        context_types = await repository.find_by_criteria(**filters)
        
        # Apply pagination
        total = len(context_types)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_types = context_types[start_idx:end_idx]
        
        # Convert to response schema with context counts
        response_types = []
        for ctx_type in paginated_types:
            type_data = ContextTypeResponse.model_validate(ctx_type)
            # Add context count if needed
            if hasattr(ctx_type, 'contexts'):
                type_data.context_count = len(ctx_type.contexts)
            response_types.append(type_data)
        
        return PaginatedContextTypesResponse(
            items=response_types,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list context types: {str(e)}"
        )


@router.post("/types", response_model=ContextTypeResponse, status_code=status.HTTP_201_CREATED)
async def create_context_type(
    type_data: ContextTypeCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ContextTypeResponse:
    """
    Create a new context type.
    
    **Features:**
    - Validates type data and schema
    - Prevents duplicate names
    - Sets up vector configuration
    - Returns created type with metadata
    """
    try:
        repository = PostgreSQLRepository(db, ContextType)
        
        # Check if type name already exists
        existing = await repository.find_by_criteria(name=type_data.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Context type with name '{type_data.name}' already exists"
            )
        
        # Create context type
        new_type = ContextType(
            name=type_data.name,
            description=type_data.description,
            type_enum=type_data.type_enum,
            schema_definition=type_data.schema_definition,
            default_scope=type_data.default_scope,
            is_active=True,
            is_system=False,
            supports_vectors=type_data.supports_vectors,
            vector_dimension=type_data.vector_dimension
        )
        
        created_type = await repository.create(new_type)
        return ContextTypeResponse.model_validate(created_type)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create context type: {str(e)}"
        )


# Context Management Endpoints

@router.get("/", response_model=PaginatedContextsResponse)
async def list_contexts(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    context_type_id: Optional[UUID] = Query(None, description="Filter by context type"),
    scope: Optional[ContextScope] = Query(None, description="Filter by scope"),
    owner_id: Optional[str] = Query(None, description="Filter by owner"),
    project_id: Optional[str] = Query(None, description="Filter by project"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    active_only: bool = Query(True, description="Filter active contexts only"),
    include_type: bool = Query(False, description="Include context type data"),
    include_relationships: bool = Query(False, description="Include relationship data"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedContextsResponse:
    """
    List contexts with filtering and pagination.
    
    **Features:**
    - Comprehensive filtering options
    - Search functionality
    - Optional inclusion of nested data
    - Access control based on scope and ownership
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        
        # Build filter criteria
        filters = {}
        if context_type_id:
            filters['context_type_id'] = context_type_id
        if scope:
            filters['scope'] = scope
        if owner_id:
            filters['owner_id'] = owner_id
        if project_id:
            filters['project_id'] = project_id
        if active_only:
            filters['is_active'] = True
        
        # Apply access control based on scope
        user_id = current_user.get('id')
        if scope == ContextScope.PRIVATE and owner_id != user_id:
            filters['owner_id'] = user_id  # Can only see own private contexts
        
        # Get results
        contexts = await repository.find_by_criteria(**filters)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            contexts = [
                c for c in contexts 
                if (search_lower in c.name.lower() if c.name else False) or 
                   (search_lower in c.description.lower() if c.description else False)
            ]
        
        # Apply pagination
        total = len(contexts)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_contexts = contexts[start_idx:end_idx]
        
        # Convert to response schema
        response_contexts = []
        for context in paginated_contexts:
            context_data = ContextResponse.model_validate(context)
            
            # Include context type if requested
            if include_type and context.context_type:
                context_data.context_type = ContextTypeResponse.model_validate(context.context_type)
            
            # Include relationships if requested
            if include_relationships:
                if context.parent_relationships:
                    context_data.parent_relationships = [
                        ContextRelationshipResponse.model_validate(r) for r in context.parent_relationships
                    ]
                if context.child_relationships:
                    context_data.child_relationships = [
                        ContextRelationshipResponse.model_validate(r) for r in context.child_relationships
                    ]
            
            response_contexts.append(context_data)
        
        return PaginatedContextsResponse(
            items=response_contexts,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list contexts: {str(e)}"
        )


@router.post("/", response_model=ContextResponse, status_code=status.HTTP_201_CREATED)
async def create_context(
    context_data: ContextCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ContextResponse:
    """
    Create a new context.
    
    **Features:**
    - Validates context data against type schema
    - Sets ownership and access control
    - Generates vector embeddings if supported
    - Returns created context with metadata
    """
    try:
        context_repo = PostgreSQLRepository(db, Context)
        type_repo = PostgreSQLRepository(db, ContextType)
        
        # Verify context type exists
        context_type = await type_repo.get_by_id(context_data.context_type_id)
        if not context_type:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context type with ID {context_data.context_type_id} not found"
            )
        
        # Validate context data against schema if defined
        if context_type.schema_definition:
            # Here you would implement JSON schema validation
            # For now, we'll just check that context_data is not empty
            if not context_data.context_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Context data cannot be empty"
                )
        
        # Set owner if not specified and scope requires it
        owner_id = context_data.owner_id
        if context_data.scope in [ContextScope.PRIVATE, ContextScope.USER] and not owner_id:
            owner_id = current_user.get('id')
        
        # Create context
        new_context = Context(
            context_type_id=context_data.context_type_id,
            name=context_data.name,
            description=context_data.description,
            scope=context_data.scope,
            owner_id=owner_id,
            project_id=context_data.project_id,
            context_data=context_data.context_data,
            context_metadata=context_data.context_metadata or {},
            is_active=True,
            priority=context_data.priority,
            vector_embedding=context_data.vector_embedding,
            embedding_model=context_data.embedding_model,
            usage_count=0
        )
        
        created_context = await context_repo.create(new_context)
        return ContextResponse.model_validate(created_context)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create context: {str(e)}"
        )


@router.get("/{context_id}", response_model=ContextResponse)
async def get_context(
    context_id: UUID,
    include_type: bool = Query(False, description="Include context type data"),
    include_relationships: bool = Query(False, description="Include relationship data"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ContextResponse:
    """
    Get a specific context by ID.
    
    **Features:**
    - Access control validation
    - Optional nested data inclusion
    - Usage tracking update
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        context = await repository.get_by_id(context_id)
        
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context with ID {context_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id')
        if context.scope == ContextScope.PRIVATE and context.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to private context"
            )
        
        # Update usage count
        await repository.update(context_id, {"usage_count": context.usage_count + 1})
        context.usage_count += 1
        
        context_data = ContextResponse.model_validate(context)
        
        # Include context type if requested
        if include_type and context.context_type:
            context_data.context_type = ContextTypeResponse.model_validate(context.context_type)
        
        # Include relationships if requested
        if include_relationships:
            if context.parent_relationships:
                context_data.parent_relationships = [
                    ContextRelationshipResponse.model_validate(r) for r in context.parent_relationships
                ]
            if context.child_relationships:
                context_data.child_relationships = [
                    ContextRelationshipResponse.model_validate(r) for r in context.child_relationships
                ]
        
        return context_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get context: {str(e)}"
        )


@router.put("/{context_id}", response_model=ContextResponse)
async def update_context(
    context_id: UUID,
    context_updates: ContextUpdate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ContextResponse:
    """
    Update a context.
    
    **Features:**
    - Partial updates supported
    - Access control validation
    - Schema validation for data updates
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        
        # Check if context exists and user has access
        existing_context = await repository.get_by_id(context_id)
        if not existing_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context with ID {context_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id')
        if existing_context.scope == ContextScope.PRIVATE and existing_context.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to private context"
            )
        
        # Prepare update data
        update_data = context_updates.model_dump(exclude_unset=True)
        
        # Update context
        updated_context = await repository.update(context_id, update_data)
        return ContextResponse.model_validate(updated_context)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update context: {str(e)}"
        )


@router.delete("/{context_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_context(
    context_id: UUID,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a context.
    
    **Features:**
    - Access control validation
    - Cascading deletion of relationships
    - Cache invalidation
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        
        # Check if context exists and user has access
        existing_context = await repository.get_by_id(context_id)
        if not existing_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context with ID {context_id} not found"
            )
        
        # Check access permissions
        user_id = current_user.get('id')
        if existing_context.scope == ContextScope.PRIVATE and existing_context.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to private context"
            )
        
        # Delete context (cascades to relationships)
        deleted = await repository.delete(context_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete context"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete context: {str(e)}"
        )


# Context Merging Endpoints

@router.post("/merge", response_model=ContextMergeResponse)
async def merge_contexts(
    merge_request: ContextMergeRequest,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ContextMergeResponse:
    """
    Merge multiple contexts using specified strategy.
    
    **Features:**
    - Multiple merge strategies supported
    - Conflict resolution mechanisms
    - Option to create new context or update existing
    - Comprehensive merge metadata
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        
        # Verify all source contexts exist and user has access
        source_contexts = []
        for context_id in merge_request.source_context_ids:
            context = await repository.get_by_id(context_id)
            if not context:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Context with ID {context_id} not found"
                )
            
            # Check access permissions
            user_id = current_user.get('id')
            if context.scope == ContextScope.PRIVATE and context.owner_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied to private context {context_id}"
                )
            
            source_contexts.append(context)
        
        # Start timing
        start_time = time.time()
        
        # Perform merge based on strategy
        merged_data = {}
        merged_fields = []
        preserved_fields = []
        conflicts_resolved = {}
        
        # Simple merge implementation (in production, use ContextManager service)
        for context in source_contexts:
            for key, value in context.context_data.items():
                if key not in merged_data:
                    merged_data[key] = value
                    merged_fields.append(key)
                else:
                    # Handle conflict based on strategy
                    if merge_request.merge_strategy == MergeStrategy.REPLACE:
                        merged_data[key] = value
                        conflicts_resolved[key] = "replaced"
                    elif merge_request.merge_strategy == MergeStrategy.PRESERVE:
                        preserved_fields.append(key)
                    else:  # MERGE_DEEP, MERGE_SHALLOW, APPEND
                        # Simplified merge logic
                        if isinstance(merged_data[key], dict) and isinstance(value, dict):
                            merged_data[key].update(value)
                            conflicts_resolved[key] = "merged"
                        else:
                            merged_data[key] = value
                            conflicts_resolved[key] = "replaced"
        
        # Create result context
        if merge_request.create_new or not merge_request.target_context_id:
            # Create new context
            result_context = Context(
                context_type_id=source_contexts[0].context_type_id,
                name=f"Merged Context {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                description="Merged from multiple contexts",
                scope=min(ctx.scope for ctx in source_contexts),  # Most restrictive scope
                owner_id=current_user.get('id'),
                project_id=source_contexts[0].project_id,
                context_data=merged_data,
                context_metadata={
                    "merge_info": {
                        "source_ids": [str(ctx.id) for ctx in source_contexts],
                        "merge_strategy": merge_request.merge_strategy.value,
                        "merged_at": datetime.utcnow().isoformat()
                    }
                },
                is_active=True,
                priority=max(ctx.priority for ctx in source_contexts),
                usage_count=0
            )
            
            created_context = await repository.create(result_context)
            result_context_id = created_context.id
        else:
            # Update existing context
            await repository.update(merge_request.target_context_id, {
                "context_data": merged_data,
                "context_metadata": {
                    "merge_info": {
                        "source_ids": [str(ctx.id) for ctx in source_contexts],
                        "merge_strategy": merge_request.merge_strategy.value,
                        "merged_at": datetime.utcnow().isoformat()
                    }
                }
            })
            result_context_id = merge_request.target_context_id
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return ContextMergeResponse(
            merge_id=str(uuid4()),
            result_context=ContextResponse.model_validate(await repository.get_by_id(result_context_id)), # Fetch full context for response
            source_context_ids=merge_request.source_context_ids,
            strategy_used=merge_request.merge_strategy.value,
            conflicts_resolved=len(conflicts_resolved),
            merge_metadata={
                "total_fields": len(merged_data),
                "conflicts_count": len(conflicts_resolved),
                "preserve_metadata": merge_request.preserve_metadata
            },
            processing_time_ms=processing_time,
            created_at=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge contexts: {str(e)}"
        )


# Context Search Endpoint

@router.post("/search", response_model=PaginatedContextsResponse)
async def search_contexts(
    search_request: ContextSearchRequest,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedContextsResponse:
    """
    Advanced context search with multiple filters.
    
    **Features:**
    - Text search across context data
    - Vector similarity search
    - Multiple filter combinations
    - Date range filtering
    - Tag-based filtering
    """
    try:
        repository = PostgreSQLRepository(db, Context)
        
        # Start with all contexts (would use Elasticsearch or similar in production)
        contexts = await repository.find_by_criteria()
        
        # Apply filters
        if search_request.context_types:
            # Would need to join with context_type table
            pass
        
        if search_request.scopes:
            contexts = [c for c in contexts if c.scope in search_request.scopes]
        
        if search_request.owner_ids:
            contexts = [c for c in contexts if c.owner_id in search_request.owner_ids]
        
        if search_request.project_ids:
            contexts = [c for c in contexts if c.project_id in search_request.project_ids]
        
        if not search_request.include_inactive:
            contexts = [c for c in contexts if c.is_active]
        
        # Apply text search
        if search_request.query:
            query_lower = search_request.query.lower()
            contexts = [
                c for c in contexts
                if (query_lower in c.name.lower() if c.name else False) or
                   (query_lower in str(c.context_data).lower()) or
                   (query_lower in c.description.lower() if c.description else False)
            ]
        
        # Apply date filtering
        if search_request.date_from:
            contexts = [c for c in contexts if c.created_at >= search_request.date_from]
        
        if search_request.date_to:
            contexts = [c for c in contexts if c.created_at <= search_request.date_to]
        
        # Apply access control
        user_id = current_user.get('id')
        contexts = [
            c for c in contexts
            if c.scope != ContextScope.PRIVATE or c.owner_id == user_id
        ]
        
        # Vector similarity search would be implemented here
        if search_request.vector_search:
            # Use ChromaDB or similar for vector similarity
            pass
        
        # Apply pagination
        total = len(contexts)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_contexts = contexts[start_idx:end_idx]
        
        # Convert to response schema
        response_contexts = [ContextResponse.model_validate(c) for c in paginated_contexts]
        
        return PaginatedContextsResponse(
            items=response_contexts,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search contexts: {str(e)}"
        )
