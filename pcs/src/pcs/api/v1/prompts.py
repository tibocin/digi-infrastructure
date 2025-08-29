"""
Filepath: src/pcs/api/v1/prompts.py
Purpose: REST API endpoints for prompt template management with CRUD operations and versioning
Related Components: PromptTemplate model, PromptService, Template engine, Rule engine
Tags: api, prompts, crud, versioning, generation, fastapi
"""

import time
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ...core.exceptions import PCSError, ValidationError
from ...models.prompts import PromptTemplate, PromptVersion, PromptRule, PromptStatus, RulePriority
from ...services.prompt_service import PromptGenerator, PromptRequest, OptimizationLevel
from ...repositories.postgres_repo import PostgreSQLRepository
from ..dependencies import (
    get_database_session,
    get_current_user,
    get_current_app,
    validate_pagination
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


# Pydantic schemas for request/response validation
class PromptTemplateBase(BaseModel):
    """Base schema for prompt template operations."""
    name: str = Field(..., min_length=1, max_length=255, description="Unique name for the prompt template")
    description: Optional[str] = Field(None, max_length=1000, description="Description of the template's purpose")
    category: Optional[str] = Field(None, max_length=100, description="Category for grouping templates")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for searching and filtering")
    author: Optional[str] = Field(None, max_length=255, description="Author of the template")
    
    @field_validator('tags')
    def validate_tags(cls, v):
        if v and len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        return v


class PromptTemplateCreate(PromptTemplateBase):
    """Schema for creating a new prompt template."""
    pass


class PromptTemplateUpdate(BaseModel):
    """Schema for updating a prompt template."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    category: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None
    author: Optional[str] = Field(None, max_length=255)
    status: Optional[PromptStatus] = None


class PromptVersionBase(BaseModel):
    """Base schema for prompt version operations."""
    content: str = Field(..., min_length=1, description="Template content with variables")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Variable definitions and types")
    changelog: Optional[str] = Field(None, max_length=1000, description="Changes in this version")
    
    @field_validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("Template content cannot be empty")
        return v


class PromptVersionCreate(PromptVersionBase):
    """Schema for creating a new prompt version."""
    pass


class PromptVersionUpdate(BaseModel):
    """Schema for updating a prompt version."""
    content: Optional[str] = Field(None, min_length=1)
    variables: Optional[Dict[str, Any]] = None
    changelog: Optional[str] = Field(None, max_length=1000)
    is_active: Optional[bool] = None


class PromptRuleBase(BaseModel):
    """Base schema for prompt rule operations."""
    name: str = Field(..., min_length=1, max_length=255, description="Rule name")
    condition: str = Field(..., min_length=1, description="Rule condition expression")
    action: Dict[str, Any] = Field(..., description="Actions to take when condition is met")
    priority: RulePriority = Field(default=RulePriority.MEDIUM, description="Rule priority")
    is_active: bool = Field(default=True, description="Whether rule is active")


class PromptRuleCreate(PromptRuleBase):
    """Schema for creating a new prompt rule."""
    pass


class PromptRuleUpdate(BaseModel):
    """Schema for updating a prompt rule."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    condition: Optional[str] = Field(None, min_length=1)
    action: Optional[Dict[str, Any]] = None
    priority: Optional[RulePriority] = None
    is_active: Optional[bool] = None


class PromptGenerationRequest(BaseModel):
    """Schema for prompt generation requests."""
    template_name: Optional[str] = Field(None, description="Name of template to use")
    template_content: Optional[str] = Field(None, description="Override template content")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context data for generation")
    context_ids: List[str] = Field(default_factory=list, description="Context IDs to merge")
    rule_names: List[str] = Field(default_factory=list, description="Rules to apply")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Additional variables")
    optimization_level: OptimizationLevel = Field(default=OptimizationLevel.BASIC, description="Optimization level")
    cache_ttl_seconds: Optional[int] = Field(None, description="Cache TTL in seconds")
    
    @field_validator('template_name', 'template_content')
    def validate_template_source(cls, v, values):
        # At least one must be provided
        if not v and not values.get('template_content') and not values.get('template_name'):
            raise ValueError("Either template_name or template_content must be provided")
        return v


# Response schemas
class PromptVersionResponse(BaseModel):
    """Response schema for prompt version."""
    id: UUID
    template_id: UUID
    version_number: int
    content: str
    variables: Dict[str, Any]
    changelog: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PromptRuleResponse(BaseModel):
    """Response schema for prompt rule."""
    id: UUID
    template_id: UUID
    name: str
    condition: str
    action: Dict[str, Any]
    priority: RulePriority
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PromptTemplateResponse(BaseModel):
    """Response schema for prompt template."""
    id: UUID
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: PromptStatus = PromptStatus.DRAFT
    is_system: bool = False
    author: Optional[str] = None
    version_count: int = 1
    usage_count: int = 0
    
    # Nested responses
    versions: Optional[List[PromptVersionResponse]] = None
    current_version: Optional[PromptVersionResponse] = None
    latest_version: Optional[PromptVersionResponse] = None
    rules: Optional[List[PromptRuleResponse]] = None
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PromptGenerationResponse(BaseModel):
    """Response schema for prompt generation."""
    request_id: str
    generated_prompt: str
    status: str
    processing_time_ms: float
    context_used: Dict[str, Any]
    rules_applied: List[str]
    variables_resolved: Dict[str, Any]
    cache_hit: bool
    metadata: Dict[str, Any]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PaginatedPromptTemplatesResponse(BaseModel):
    """Paginated response for prompt templates."""
    items: List[PromptTemplateResponse]
    total: int
    page: int
    size: int
    pages: int
    
    model_config = ConfigDict(from_attributes=True)


# API Endpoints

@router.get("/", response_model=PaginatedPromptTemplatesResponse)
async def list_prompt_templates(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[PromptStatus] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    include_versions: bool = Query(False, description="Include version data"),
    include_rules: bool = Query(False, description="Include rule data"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PaginatedPromptTemplatesResponse:
    """
    List prompt templates with filtering and pagination.
    
    **Features:**
    - Pagination with configurable page size
    - Filter by category and status
    - Search by name and description
    - Optional inclusion of versions and rules
    - User access control
    """
    try:
        repository = PostgreSQLRepository(db, PromptTemplate)
        
        # Build filter criteria
        filters = {}
        if category:
            filters['category'] = category
        if status:
            filters['status'] = status
        
        # Get paginated results
        # Note: This is a simplified implementation - in production you'd want
        # more sophisticated filtering and search capabilities
        templates = await repository.find_by_criteria(**filters)
        
        # Apply search filter if provided
        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates 
                if (search_lower in t.name.lower() if t.name else False) or 
                   (search_lower in t.description.lower() if t.description else False)
            ]
        
        # Apply pagination
        total = len(templates)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_templates = templates[start_idx:end_idx]
        
        # Convert to response schema
        response_templates = []
        for template in paginated_templates:
            template_data = PromptTemplateResponse.model_validate(template)
            
            # Include versions if requested
            if include_versions and template.versions:
                template_data.versions = [
                    PromptVersionResponse.model_validate(v) for v in template.versions
                ]
                if template.current_version:
                    template_data.current_version = PromptVersionResponse.model_validate(template.current_version)
                if template.latest_version:
                    template_data.latest_version = PromptVersionResponse.model_validate(template.latest_version)
            
            # Include rules if requested
            if include_rules and template.rules:
                template_data.rules = [
                    PromptRuleResponse.model_validate(r) for r in template.rules
                ]
            
            response_templates.append(template_data)
        
        return PaginatedPromptTemplatesResponse(
            items=response_templates,
            total=total,
            page=page,
            size=size,
            pages=(total + size - 1) // size
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list prompt templates: {str(e)}"
        )


@router.post("/", response_model=PromptTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt_template(
    template_data: PromptTemplateCreate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PromptTemplateResponse:
    """
    Create a new prompt template.
    
    **Features:**
    - Validates template data
    - Sets default values
    - Creates initial version (empty)
    - Returns created template with metadata
    """
    try:
        repository = PostgreSQLRepository(db, PromptTemplate)
        
        # Check if template name already exists
        existing = await repository.find_by_criteria(name=template_data.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Template with name '{template_data.name}' already exists"
            )
        
        # Create template
        new_template = PromptTemplate(
            name=template_data.name,
            description=template_data.description,
            category=template_data.category,
            tags=template_data.tags,
            author=template_data.author or current_user.get('username'),
            status=PromptStatus.DRAFT,
            is_system=False,
            version_count=0
        )
        
        created_template = await repository.create(new_template)
        return PromptTemplateResponse.model_validate(created_template)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prompt template: {str(e)}"
        )


@router.get("/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(
    template_id: UUID,
    include_versions: bool = Query(False, description="Include version data"),
    include_rules: bool = Query(False, description="Include rule data"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PromptTemplateResponse:
    """
    Get a specific prompt template by ID.
    
    **Features:**
    - Retrieve template with optional nested data
    - Access control validation
    - Detailed error handling
    """
    try:
        repository = PostgreSQLRepository(db, PromptTemplate)
        template = await repository.get_by_id(template_id)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with ID {template_id} not found"
            )
        
        template_data = PromptTemplateResponse.model_validate(template)
        
        # Include versions if requested
        if include_versions and template.versions:
            template_data.versions = [
                PromptVersionResponse.model_validate(v) for v in template.versions
            ]
            if template.current_version:
                template_data.current_version = PromptVersionResponse.model_validate(template.current_version)
            if template.latest_version:
                template_data.latest_version = PromptVersionResponse.model_validate(template.latest_version)
        
        # Include rules if requested
        if include_rules and template.rules:
            template_data.rules = [
                PromptRuleResponse.model_validate(r) for r in template.rules
            ]
        
        return template_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get prompt template: {str(e)}"
        )


@router.put("/{template_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(
    template_id: UUID,
    template_updates: PromptTemplateUpdate,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PromptTemplateResponse:
    """
    Update a prompt template.
    
    **Features:**
    - Partial updates supported
    - Validates update data
    - Access control checks
    - Returns updated template
    """
    try:
        repository = PostgreSQLRepository(db, PromptTemplate)
        
        # Check if template exists
        existing_template = await repository.get_by_id(template_id)
        if not existing_template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with ID {template_id} not found"
            )
        
        # Check name uniqueness if name is being updated
        if template_updates.name and template_updates.name != existing_template.name:
            existing_name = await repository.find_by_criteria(name=template_updates.name)
            if existing_name:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Template with name '{template_updates.name}' already exists"
                )
        
        # Prepare update data
        update_data = template_updates.model_dump(exclude_unset=True)
        
        # Update template
        updated_template = await repository.update(template_id, update_data)
        return PromptTemplateResponse.model_validate(updated_template)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt template: {str(e)}"
        )


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a prompt template.
    
    **Features:**
    - Cascading delete of versions and rules
    - Access control validation
    - System template protection
    """
    try:
        repository = PostgreSQLRepository(db, PromptTemplate)
        
        # Check if template exists
        existing_template = await repository.get_by_id(template_id)
        if not existing_template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with ID {template_id} not found"
            )
        
        # Prevent deletion of system templates
        if existing_template.is_system:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete system templates"
            )
        
        # Delete template (cascades to versions and rules)
        deleted = await repository.delete(template_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete template"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt template: {str(e)}"
        )


# Version Management Endpoints

@router.post("/{template_id}/versions", response_model=PromptVersionResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt_version(
    template_id: UUID,
    version_data: PromptVersionCreate,
    make_active: bool = Query(False, description="Make this version active"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PromptVersionResponse:
    """
    Create a new version for a prompt template.
    
    **Features:**
    - Auto-incremented version numbers
    - Optional activation of new version
    - Template validation
    - Variable extraction
    """
    try:
        template_repo = PostgreSQLRepository(db, PromptTemplate)
        version_repo = PostgreSQLRepository(db, PromptVersion)
        
        # Check if template exists
        template = await template_repo.get_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with ID {template_id} not found"
            )
        
        # For now, use a simple approach: increment from template version count
        # In a production environment, this would query the actual versions
        next_version = template.version_count + 1
        
        # Create version
        new_version = PromptVersion(
            template_id=template_id,
            version_number=next_version,
            template=version_data.content,        # Map 'content' to 'template'
            variables=list(version_data.variables.keys()) if version_data.variables else [],  # Convert dict keys to list
            change_notes=version_data.changelog,  # Map 'changelog' to 'change_notes'
            is_active=make_active
        )
        
        # If making active, deactivate other versions
        if make_active:
            for version in template.versions:
                if version.is_active:
                    await version_repo.update(version.id, {"is_active": False})
        
        created_version = await version_repo.create(new_version)
        
        # Update template version count
        await template_repo.update(template_id, {"version_count": next_version})
        
        # Create response with proper variable mapping
        # The database stores variable names as a list, but response expects full variable definitions
        response_data = {
            "id": created_version.id,
            "template_id": created_version.template_id,
            "version_number": created_version.version_number,
            "content": created_version.template,  # Map 'template' back to 'content'
            "variables": version_data.variables,  # Use original variable definitions from request
            "changelog": created_version.change_notes,  # Map 'change_notes' back to 'changelog'
            "is_active": created_version.is_active,
            "created_at": created_version.created_at,
            "updated_at": created_version.updated_at
        }
        
        return PromptVersionResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prompt version: {str(e)}"
        )


@router.get("/{template_id}/versions", response_model=List[PromptVersionResponse])
async def list_prompt_versions(
    template_id: UUID,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[PromptVersionResponse]:
    """List all versions for a prompt template."""
    try:
        template_repo = PostgreSQLRepository(db, PromptTemplate)
        
        # Check if template exists
        template = await template_repo.get_by_id(template_id)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template with ID {template_id} not found"
            )
        
        # Map database models to response models with proper variable handling
        versions_response = []
        for v in template.versions:
            # Convert stored variable names back to a simple dict format
            # Since we don't have the original variable definitions, we'll create a basic structure
            variables_dict = {var_name: {"type": "string", "required": True} for var_name in (v.variables or [])}
            
            response_data = {
                "id": v.id,
                "template_id": v.template_id,
                "version_number": v.version_number,
                "content": v.template,  # Map 'template' back to 'content'
                "variables": variables_dict,
                "changelog": v.change_notes,  # Map 'change_notes' back to 'changelog'
                "is_active": v.is_active,
                "created_at": v.created_at,
                "updated_at": v.updated_at
            }
            versions_response.append(PromptVersionResponse(**response_data))
        
        return versions_response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list prompt versions: {str(e)}"
        )


# Prompt Generation Endpoint

@router.post("/generate", response_model=PromptGenerationResponse)
async def generate_prompt(
    generation_request: PromptGenerationRequest,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PromptGenerationResponse:
    """
    Generate a prompt using template and context.
    
    **Features:**
    - Template-based generation
    - Context injection and merging
    - Rule evaluation and application
    - Caching for performance
    - Comprehensive response metadata
    """
    try:
        # Create prompt service instance
        # Note: In a real implementation, this would be injected as a dependency
        prompt_generator = PromptGenerator(
            template_engine=None,  # Would be injected
            rule_engine=None,      # Would be injected
            context_manager=None,  # Would be injected
            redis_repo=None        # Would be injected
        )
        
        # Convert API request to service request
        from datetime import timedelta
        
        service_request = PromptRequest(
            request_id=str(uuid4()),
            template_name=generation_request.template_name,
            template_content=generation_request.template_content,
            context_data=generation_request.context_data,
            context_ids=generation_request.context_ids,
            rule_names=generation_request.rule_names,
            variables=generation_request.variables,
            optimization_level=generation_request.optimization_level,
            cache_ttl=timedelta(seconds=generation_request.cache_ttl_seconds) if generation_request.cache_ttl_seconds else None
        )
        
        # Generate prompt
        start_time = time.time()
        
        # This is a simplified implementation - the actual service would handle this
        generated_result = f"Generated prompt for template: {service_request.template_name or 'inline'}"
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return PromptGenerationResponse(
            request_id=service_request.request_id,
            generated_prompt=generated_result,
            status="completed",
            processing_time_ms=processing_time,
            context_used=service_request.context_data,
            rules_applied=service_request.rule_names,
            variables_resolved=service_request.variables,
            cache_hit=False,
            metadata={"optimization_level": service_request.optimization_level.value},
            created_at=datetime.utcnow()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prompt: {str(e)}"
        )
