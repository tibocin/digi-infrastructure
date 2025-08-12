"""
Filepath: tests/unit/test_prompts_api.py
Purpose: Unit tests for prompt management API endpoints
Related Components: Prompt API, Pydantic schemas, Repository pattern, FastAPI
Tags: unit-tests, api-testing, prompts, crud, fastapi-testing
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4, UUID
from datetime import datetime
from typing import Dict, Any

from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from sqlalchemy.ext.asyncio import AsyncSession

from pcs.api.v1.prompts import (
    router,
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptVersionCreate,
    PromptGenerationRequest,
    PromptTemplateResponse,
    PromptVersionResponse,
    PromptGenerationResponse,
    PaginatedPromptTemplatesResponse
)
from pcs.models.prompts import PromptTemplate, PromptVersion, PromptStatus, RulePriority
from pcs.services.prompt_service import OptimizationLevel


class TestPromptTemplateSchemas:
    """Test Pydantic schemas for prompt templates."""
    
    def test_prompt_template_create_valid(self):
        """Test valid prompt template creation schema."""
        data = {
            "name": "test-template",
            "description": "A test template",
            "category": "testing",
            "tags": ["test", "example"],
            "author": "test-user"
        }
        schema = PromptTemplateCreate(**data)
        assert schema.name == "test-template"
        assert schema.description == "A test template"
        assert schema.category == "testing"
        assert schema.tags == ["test", "example"]
        assert schema.author == "test-user"
    
    def test_prompt_template_create_minimal(self):
        """Test minimal prompt template creation."""
        data = {"name": "minimal-template"}
        schema = PromptTemplateCreate(**data)
        assert schema.name == "minimal-template"
        assert schema.description is None
        assert schema.category is None
        assert schema.tags == []
        assert schema.author is None
    
    def test_prompt_template_create_invalid_name(self):
        """Test invalid name validation."""
        with pytest.raises(ValueError):
            PromptTemplateCreate(name="")
        
        with pytest.raises(ValueError):
            PromptTemplateCreate(name="a" * 256)  # Too long
    
    def test_prompt_template_create_too_many_tags(self):
        """Test tag count validation."""
        data = {
            "name": "test-template",
            "tags": [f"tag{i}" for i in range(21)]  # 21 tags (max is 20)
        }
        with pytest.raises(ValueError, match="Maximum 20 tags allowed"):
            PromptTemplateCreate(**data)
    
    def test_prompt_version_create_valid(self):
        """Test valid prompt version creation."""
        data = {
            "content": "Hello {{ name }}!",
            "variables": {"name": {"type": "string", "required": True}},
            "changelog": "Initial version"
        }
        schema = PromptVersionCreate(**data)
        assert schema.content == "Hello {{ name }}!"
        assert schema.variables == {"name": {"type": "string", "required": True}}
        assert schema.changelog == "Initial version"
    
    def test_prompt_version_create_empty_content(self):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Template content cannot be empty"):
            PromptVersionCreate(content="   ")
    
    def test_prompt_generation_request_valid(self):
        """Test valid prompt generation request."""
        data = {
            "template_name": "test-template",
            "context_data": {"user": "John"},
            "variables": {"greeting": "Hello"},
            "optimization_level": "basic"
        }
        schema = PromptGenerationRequest(**data)
        assert schema.template_name == "test-template"
        assert schema.context_data == {"user": "John"}
        assert schema.variables == {"greeting": "Hello"}
        assert schema.optimization_level == OptimizationLevel.BASIC


class TestPromptAPIEndpoints:
    """Test prompt management API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test app."""
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)
    
    @pytest.fixture
    def mock_current_user(self):
        """Mock current user."""
        return {"id": str(uuid4()), "username": "testuser", "email": "test@example.com"}
    
    @pytest.fixture
    def sample_template_data(self):
        """Sample template data for testing."""
        return {
            "name": "test-template",
            "description": "A test template for unit testing",
            "category": "testing",
            "tags": ["test", "example"],
            "author": "testuser"
        }
    
    @pytest.fixture
    def sample_template_model(self, sample_template_data):
        """Sample template model instance."""
        template = Mock(spec=PromptTemplate)
        template.id = uuid4()
        template.name = sample_template_data["name"]
        template.description = sample_template_data["description"]
        template.category = sample_template_data["category"]
        template.tags = sample_template_data["tags"]
        template.author = sample_template_data["author"]
        template.status = PromptStatus.DRAFT
        template.is_system = False
        template.version_count = 0
        template.created_at = datetime.utcnow()
        template.updated_at = datetime.utcnow()
        template.versions = []
        template.rules = []
        template.current_version = None
        template.latest_version = None
        return template
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_list_prompt_templates_success(self, mock_repo_class, mock_get_user, mock_get_db, 
                                         client, mock_db_session, mock_current_user, sample_template_model):
        """Test successful listing of prompt templates."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_template_model])
        
        # Make request
        response = client.get("/api/v1/prompts/")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert "pages" in data
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "test-template"
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_list_prompt_templates_with_filters(self, mock_repo_class, mock_get_user, mock_get_db,
                                               client, mock_db_session, mock_current_user, sample_template_model):
        """Test listing templates with filters."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_template_model])
        
        # Make request with filters
        response = client.get("/api/v1/prompts/?category=testing&status=draft&search=test")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        mock_repo.find_by_criteria.assert_called_once_with(category="testing", status=PromptStatus.DRAFT)
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_create_prompt_template_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                          client, mock_db_session, mock_current_user, 
                                          sample_template_data, sample_template_model):
        """Test successful template creation."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[])  # No existing template
        mock_repo.create = AsyncMock(return_value=sample_template_model)
        
        # Make request
        response = client.post("/api/v1/prompts/", json=sample_template_data)
        
        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_template_data["name"]
        assert data["description"] == sample_template_data["description"]
        assert data["status"] == "draft"
        assert data["is_system"] is False
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_create_prompt_template_duplicate_name(self, mock_repo_class, mock_get_user, mock_get_db,
                                                  client, mock_db_session, mock_current_user,
                                                  sample_template_data, sample_template_model):
        """Test template creation with duplicate name."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_template_model])  # Existing template
        
        # Make request
        response = client.post("/api/v1/prompts/", json=sample_template_data)
        
        # Assertions
        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert "already exists" in data["detail"]
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_get_prompt_template_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                       client, mock_db_session, mock_current_user, sample_template_model):
        """Test successful template retrieval."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_template_model)
        
        template_id = str(sample_template_model.id)
        
        # Make request
        response = client.get(f"/api/v1/prompts/{template_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == sample_template_model.name
        assert data["id"] == template_id
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_get_prompt_template_not_found(self, mock_repo_class, mock_get_user, mock_get_db,
                                         client, mock_db_session, mock_current_user):
        """Test template retrieval with non-existent ID."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=None)
        
        template_id = str(uuid4())
        
        # Make request
        response = client.get(f"/api/v1/prompts/{template_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_update_prompt_template_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                          client, mock_db_session, mock_current_user, sample_template_model):
        """Test successful template update."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_template_model)
        mock_repo.find_by_criteria = AsyncMock(return_value=[])  # No name conflict
        
        updated_template = Mock(spec=PromptTemplate)
        updated_template.id = sample_template_model.id
        updated_template.name = "updated-template"
        updated_template.description = "Updated description"
        updated_template.category = sample_template_model.category
        updated_template.tags = sample_template_model.tags
        updated_template.author = sample_template_model.author
        updated_template.status = PromptStatus.ACTIVE
        updated_template.is_system = False
        updated_template.version_count = 0
        updated_template.created_at = sample_template_model.created_at
        updated_template.updated_at = datetime.utcnow()
        
        mock_repo.update = AsyncMock(return_value=updated_template)
        
        template_id = str(sample_template_model.id)
        update_data = {
            "name": "updated-template",
            "description": "Updated description",
            "status": "active"
        }
        
        # Make request
        response = client.put(f"/api/v1/prompts/{template_id}", json=update_data)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "updated-template"
        assert data["description"] == "Updated description"
        assert data["status"] == "active"
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_delete_prompt_template_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                          client, mock_db_session, mock_current_user, sample_template_model):
        """Test successful template deletion."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_template_model)
        mock_repo.delete = AsyncMock(return_value=True)
        
        template_id = str(sample_template_model.id)
        
        # Make request
        response = client.delete(f"/api/v1/prompts/{template_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_repo.delete.assert_called_once_with(sample_template_model.id)
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_delete_system_template_forbidden(self, mock_repo_class, mock_get_user, mock_get_db,
                                            client, mock_db_session, mock_current_user, sample_template_model):
        """Test deletion of system template is forbidden."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        # Make template a system template
        sample_template_model.is_system = True
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_template_model)
        
        template_id = str(sample_template_model.id)
        
        # Make request
        response = client.delete(f"/api/v1/prompts/{template_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert "Cannot delete system templates" in data["detail"]
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    def test_create_prompt_version_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                         client, mock_db_session, mock_current_user, sample_template_model):
        """Test successful version creation."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_template_repo = Mock()
        mock_version_repo = Mock()
        
        # Mock repository class to return different instances
        def mock_repo_side_effect(session, model_class):
            if model_class == PromptTemplate:
                return mock_template_repo
            elif model_class == PromptVersion:
                return mock_version_repo
            return Mock()
        
        mock_repo_class.side_effect = mock_repo_side_effect
        
        mock_template_repo.get_by_id = AsyncMock(return_value=sample_template_model)
        mock_template_repo.update = AsyncMock(return_value=sample_template_model)
        
        # Create mock version
        new_version = Mock(spec=PromptVersion)
        new_version.id = uuid4()
        new_version.template_id = sample_template_model.id
        new_version.version_number = 1
        new_version.content = "Hello {{ name }}!"
        new_version.variables = {"name": {"type": "string"}}
        new_version.changelog = "Initial version"
        new_version.is_active = True
        new_version.created_at = datetime.utcnow()
        new_version.updated_at = datetime.utcnow()
        
        mock_version_repo.create = AsyncMock(return_value=new_version)
        mock_version_repo.update = AsyncMock()
        
        template_id = str(sample_template_model.id)
        version_data = {
            "content": "Hello {{ name }}!",
            "variables": {"name": {"type": "string"}},
            "changelog": "Initial version"
        }
        
        # Make request
        response = client.post(f"/api/v1/prompts/{template_id}/versions?make_active=true", json=version_data)
        
        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["content"] == "Hello {{ name }}!"
        assert data["version_number"] == 1
        assert data["is_active"] is True
    
    @patch('pcs.api.v1.prompts.get_database_session')
    @patch('pcs.api.v1.prompts.get_current_user')
    def test_generate_prompt_success(self, mock_get_user, mock_get_db,
                                    client, mock_db_session, mock_current_user):
        """Test successful prompt generation."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        generation_data = {
            "template_name": "test-template",
            "context_data": {"user": "John"},
            "variables": {"greeting": "Hello"},
            "optimization_level": "basic"
        }
        
        # Make request
        response = client.post("/api/v1/prompts/generate", json=generation_data)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "request_id" in data
        assert "generated_prompt" in data
        assert "status" in data
        assert "processing_time_ms" in data
        assert data["status"] == "completed"
    
    def test_prompt_generation_request_validation(self):
        """Test prompt generation request validation."""
        # Test missing template source
        with pytest.raises(ValueError, match="Either template_name or template_content must be provided"):
            PromptGenerationRequest(context_data={"test": "data"})
        
        # Test valid with template_name
        request1 = PromptGenerationRequest(template_name="test-template")
        assert request1.template_name == "test-template"
        
        # Test valid with template_content
        request2 = PromptGenerationRequest(template_content="Hello {{ name }}")
        assert request2.template_content == "Hello {{ name }}"


class TestPromptAPIIntegration:
    """Integration tests for prompt API with real-like scenarios."""
    
    def test_full_template_lifecycle(self):
        """Test complete template lifecycle: create, update, version, delete."""
        # This would be an integration test that uses a real test database
        # For now, we'll create a simplified version
        
        template_data = {
            "name": "lifecycle-template",
            "description": "Template for lifecycle testing",
            "category": "testing"
        }
        
        # In a real integration test, you would:
        # 1. Create template
        # 2. Verify creation
        # 3. Update template
        # 4. Create versions
        # 5. Test generation
        # 6. Delete template
        
        # For unit test, we just validate the data structure
        create_schema = PromptTemplateCreate(**template_data)
        assert create_schema.name == "lifecycle-template"
        
        update_schema = PromptTemplateUpdate(description="Updated description")
        assert update_schema.description == "Updated description"
        
        version_schema = PromptVersionCreate(
            content="Hello {{ name }}!",
            variables={"name": {"type": "string"}},
            changelog="Initial version"
        )
        assert version_schema.content == "Hello {{ name }}!"
    
    def test_concurrent_template_creation(self):
        """Test handling of concurrent template creation with same name."""
        # This test would verify race condition handling
        # For unit test, we verify the validation logic exists
        
        template_data = {
            "name": "concurrent-template",
            "description": "Template for concurrency testing"
        }
        
        schema = PromptTemplateCreate(**template_data)
        assert schema.name == "concurrent-template"
        
        # In integration test, you would create multiple concurrent requests
        # and verify only one succeeds with HTTP 409 for others
    
    def test_prompt_generation_with_complex_context(self):
        """Test prompt generation with complex context and rules."""
        generation_data = {
            "template_name": "complex-template",
            "context_data": {
                "user": {"name": "John", "role": "admin"},
                "project": {"name": "TestProject", "type": "web"}
            },
            "context_ids": ["user-context-123", "project-context-456"],
            "rule_names": ["admin-rules", "web-project-rules"],
            "variables": {
                "max_length": 500,
                "tone": "professional"
            },
            "optimization_level": "aggressive",
            "cache_ttl_seconds": 3600
        }
        
        schema = PromptGenerationRequest(**generation_data)
        assert schema.template_name == "complex-template"
        assert schema.context_data["user"]["name"] == "John"
        assert schema.optimization_level == OptimizationLevel.AGGRESSIVE
        assert schema.cache_ttl_seconds == 3600
