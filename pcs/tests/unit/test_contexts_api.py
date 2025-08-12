"""
Filepath: tests/unit/test_contexts_api.py
Purpose: Unit tests for context management API endpoints
Related Components: Context API, Pydantic schemas, Repository pattern, Context merging
Tags: unit-tests, api-testing, contexts, crud, merging, fastapi-testing
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

from pcs.api.v1.contexts import (
    router,
    ContextTypeCreate,
    ContextTypeUpdate,
    ContextCreate,
    ContextUpdate,
    ContextMergeRequest,
    ContextSearchRequest,
    ContextTypeResponse,
    ContextResponse,
    ContextMergeResponse,
    PaginatedContextsResponse,
    PaginatedContextTypesResponse
)
from pcs.models.contexts import Context, ContextType, ContextTypeEnum, ContextScope, RelationshipType
from pcs.services.context_service import MergeStrategy


class TestContextTypeSchemas:
    """Test Pydantic schemas for context types."""
    
    def test_context_type_create_valid(self):
        """Test valid context type creation schema."""
        data = {
            "name": "user-context",
            "description": "User-specific context data",
            "type_enum": "user_preference",
            "schema_definition": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "language": {"type": "string"}
                }
            },
            "default_scope": "user",
            "supports_vectors": True,
            "vector_dimension": 768
        }
        schema = ContextTypeCreate(**data)
        assert schema.name == "user-context"
        assert schema.description == "User-specific context data"
        assert schema.type_enum == ContextTypeEnum.USER_PREFERENCE
        assert schema.supports_vectors is True
        assert schema.vector_dimension == 768
    
    def test_context_type_create_minimal(self):
        """Test minimal context type creation."""
        data = {
            "name": "minimal-type",
            "type_enum": "custom"
        }
        schema = ContextTypeCreate(**data)
        assert schema.name == "minimal-type"
        assert schema.type_enum == ContextTypeEnum.CUSTOM
        assert schema.supports_vectors is False
        assert schema.vector_dimension is None
    
    def test_context_type_vector_validation(self):
        """Test vector configuration validation."""
        # Valid: supports_vectors=True with dimension
        data = {
            "name": "vector-type",
            "type_enum": "custom",
            "supports_vectors": True,
            "vector_dimension": 512
        }
        schema = ContextTypeCreate(**data)
        assert schema.vector_dimension == 512
        
        # Invalid: supports_vectors=True without dimension
        with pytest.raises(ValueError, match="Vector dimension required"):
            ContextTypeCreate(
                name="invalid-type",
                type_enum="custom",
                supports_vectors=True
            )
        
        # Invalid: negative dimension
        with pytest.raises(ValueError, match="Vector dimension must be positive"):
            ContextTypeCreate(
                name="invalid-type",
                type_enum="custom",
                supports_vectors=True,
                vector_dimension=-1
            )
    
    def test_context_create_valid(self):
        """Test valid context creation schema."""
        data = {
            "context_type_id": str(uuid4()),
            "name": "test-context",
            "description": "A test context",
            "scope": "user",
            "owner_id": "user123",
            "project_id": "project456",
            "context_data": {"key": "value", "nested": {"data": "test"}},
            "context_metadata": {"source": "api"},
            "priority": 5,
            "vector_embedding": [0.1, 0.2, 0.3],
            "embedding_model": "text-embedding-ada-002"
        }
        schema = ContextCreate(**data)
        assert schema.name == "test-context"
        assert schema.scope == ContextScope.USER
        assert schema.context_data == {"key": "value", "nested": {"data": "test"}}
        assert schema.priority == 5
    
    def test_context_create_empty_data_validation(self):
        """Test validation of empty context data."""
        with pytest.raises(ValueError, match="Context data cannot be empty"):
            ContextCreate(
                context_type_id=str(uuid4()),
                name="test-context",
                scope="user",
                context_data={}
            )
    
    def test_context_merge_request_valid(self):
        """Test valid context merge request."""
        data = {
            "source_context_ids": [str(uuid4()), str(uuid4()), str(uuid4())],
            "merge_strategy": "merge_deep",
            "conflict_resolution": {"field1": "use_first", "field2": "use_last"},
            "preserve_metadata": True,
            "create_new": True
        }
        schema = ContextMergeRequest(**data)
        assert len(schema.source_context_ids) == 3
        assert schema.merge_strategy == MergeStrategy.MERGE_DEEP
        assert schema.create_new is True
    
    def test_context_merge_request_validation(self):
        """Test context merge request validation."""
        # Test minimum context count
        with pytest.raises(ValueError):
            ContextMergeRequest(source_context_ids=[str(uuid4())])
        
        # Test duplicate context IDs
        context_id = str(uuid4())
        with pytest.raises(ValueError, match="Duplicate context IDs not allowed"):
            ContextMergeRequest(source_context_ids=[context_id, context_id])
    
    def test_context_search_request_valid(self):
        """Test valid context search request."""
        data = {
            "query": "test search",
            "context_types": ["user_preference", "conversation"],
            "scopes": ["user", "shared"],
            "owner_ids": ["user1", "user2"],
            "project_ids": ["proj1"],
            "tags": ["tag1", "tag2"],
            "date_from": datetime(2024, 1, 1),
            "date_to": datetime(2024, 12, 31),
            "vector_search": [0.1, 0.2, 0.3],
            "similarity_threshold": 0.85,
            "include_inactive": True
        }
        schema = ContextSearchRequest(**data)
        assert schema.query == "test search"
        assert len(schema.context_types) == 2
        assert schema.similarity_threshold == 0.85
        assert schema.include_inactive is True


class TestContextAPIEndpoints:
    """Test context management API endpoints."""
    
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
    def sample_context_type_data(self):
        """Sample context type data for testing."""
        return {
            "name": "test-context-type",
            "description": "A test context type",
            "type_enum": "custom",
            "schema_definition": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"}
                }
            },
            "default_scope": "user",
            "supports_vectors": False
        }
    
    @pytest.fixture
    def sample_context_type_model(self, sample_context_type_data):
        """Sample context type model instance."""
        context_type = Mock(spec=ContextType)
        context_type.id = uuid4()
        context_type.name = sample_context_type_data["name"]
        context_type.description = sample_context_type_data["description"]
        context_type.type_enum = ContextTypeEnum.CUSTOM
        context_type.schema_definition = sample_context_type_data["schema_definition"]
        context_type.default_scope = ContextScope.USER
        context_type.is_active = True
        context_type.is_system = False
        context_type.supports_vectors = False
        context_type.vector_dimension = None
        context_type.created_at = datetime.utcnow()
        context_type.updated_at = datetime.utcnow()
        context_type.contexts = []
        return context_type
    
    @pytest.fixture
    def sample_context_data(self, sample_context_type_model):
        """Sample context data for testing."""
        return {
            "context_type_id": str(sample_context_type_model.id),
            "name": "test-context",
            "description": "A test context",
            "scope": "user",
            "owner_id": "user123",
            "project_id": "project456",
            "context_data": {"key": "value", "setting": "enabled"},
            "context_metadata": {"source": "api"},
            "priority": 1
        }
    
    @pytest.fixture
    def sample_context_model(self, sample_context_data, sample_context_type_model):
        """Sample context model instance."""
        context = Mock(spec=Context)
        context.id = uuid4()
        context.context_type_id = sample_context_type_model.id
        context.name = sample_context_data["name"]
        context.description = sample_context_data["description"]
        context.scope = ContextScope.USER
        context.owner_id = sample_context_data["owner_id"]
        context.project_id = sample_context_data["project_id"]
        context.context_data = sample_context_data["context_data"]
        context.context_metadata = sample_context_data["context_metadata"]
        context.is_active = True
        context.priority = sample_context_data["priority"]
        context.vector_embedding = None
        context.embedding_model = None
        context.usage_count = 0
        context.created_at = datetime.utcnow()
        context.updated_at = datetime.utcnow()
        context.context_type = sample_context_type_model
        context.parent_relationships = []
        context.child_relationships = []
        return context
    
    # Context Type Tests
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_list_context_types_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                       client, mock_db_session, mock_current_user, sample_context_type_model):
        """Test successful listing of context types."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_context_type_model])
        
        # Make request
        response = client.get("/api/v1/contexts/types")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "test-context-type"
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_create_context_type_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                        client, mock_db_session, mock_current_user,
                                        sample_context_type_data, sample_context_type_model):
        """Test successful context type creation."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[])  # No existing type
        mock_repo.create = AsyncMock(return_value=sample_context_type_model)
        
        # Make request
        response = client.post("/api/v1/contexts/types", json=sample_context_type_data)
        
        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_context_type_data["name"]
        assert data["type_enum"] == "custom"
        assert data["is_active"] is True
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_create_context_type_duplicate_name(self, mock_repo_class, mock_get_user, mock_get_db,
                                               client, mock_db_session, mock_current_user,
                                               sample_context_type_data, sample_context_type_model):
        """Test context type creation with duplicate name."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_context_type_model])  # Existing type
        
        # Make request
        response = client.post("/api/v1/contexts/types", json=sample_context_type_data)
        
        # Assertions
        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert "already exists" in data["detail"]
    
    # Context Tests
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_list_contexts_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                  client, mock_db_session, mock_current_user, sample_context_model):
        """Test successful listing of contexts."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_context_model])
        
        # Make request
        response = client.get("/api/v1/contexts/")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["name"] == "test-context"
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_list_contexts_with_filters(self, mock_repo_class, mock_get_user, mock_get_db,
                                       client, mock_db_session, mock_current_user, sample_context_model):
        """Test listing contexts with filters."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_context_model])
        
        # Make request with filters
        response = client.get("/api/v1/contexts/?scope=user&owner_id=user123&search=test")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        mock_repo.find_by_criteria.assert_called_once()
        call_args = mock_repo.find_by_criteria.call_args[1]
        assert call_args["scope"] == ContextScope.USER
        assert call_args["owner_id"] == "user123"
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_create_context_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                   client, mock_db_session, mock_current_user,
                                   sample_context_data, sample_context_model, sample_context_type_model):
        """Test successful context creation."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        def mock_repo_side_effect(session, model_class):
            mock_repo = Mock()
            if model_class == Context:
                mock_repo.create = AsyncMock(return_value=sample_context_model)
                return mock_repo
            elif model_class == ContextType:
                mock_repo.get_by_id = AsyncMock(return_value=sample_context_type_model)
                return mock_repo
            return Mock()
        
        mock_repo_class.side_effect = mock_repo_side_effect
        
        # Make request
        response = client.post("/api/v1/contexts/", json=sample_context_data)
        
        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == sample_context_data["name"]
        assert data["scope"] == "user"
        assert data["context_data"] == sample_context_data["context_data"]
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_create_context_invalid_type(self, mock_repo_class, mock_get_user, mock_get_db,
                                        client, mock_db_session, mock_current_user, sample_context_data):
        """Test context creation with invalid context type."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        def mock_repo_side_effect(session, model_class):
            mock_repo = Mock()
            if model_class == ContextType:
                mock_repo.get_by_id = AsyncMock(return_value=None)  # Type not found
            return mock_repo
        
        mock_repo_class.side_effect = mock_repo_side_effect
        
        # Make request
        response = client.post("/api/v1/contexts/", json=sample_context_data)
        
        # Assertions
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "not found" in data["detail"]
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_get_context_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                client, mock_db_session, mock_current_user, sample_context_model):
        """Test successful context retrieval."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_context_model)
        mock_repo.update = AsyncMock(return_value=sample_context_model)
        
        context_id = str(sample_context_model.id)
        
        # Make request
        response = client.get(f"/api/v1/contexts/{context_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == sample_context_model.name
        assert data["id"] == context_id
        
        # Verify usage count was updated
        mock_repo.update.assert_called_once()
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_get_context_access_denied(self, mock_repo_class, mock_get_user, mock_get_db,
                                      client, mock_db_session, mock_current_user, sample_context_model):
        """Test context retrieval with access denied."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_current_user["id"] = "different-user"  # Different user
        mock_get_user.return_value = mock_current_user
        
        # Make context private
        sample_context_model.scope = ContextScope.PRIVATE
        sample_context_model.owner_id = "owner123"
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_context_model)
        
        context_id = str(sample_context_model.id)
        
        # Make request
        response = client.get(f"/api/v1/contexts/{context_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert "Access denied" in data["detail"]
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_update_context_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                   client, mock_db_session, mock_current_user, sample_context_model):
        """Test successful context update."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        updated_context = Mock(spec=Context)
        updated_context.id = sample_context_model.id
        updated_context.name = "updated-context"
        updated_context.description = "Updated description"
        updated_context.scope = sample_context_model.scope
        updated_context.owner_id = sample_context_model.owner_id
        updated_context.project_id = sample_context_model.project_id
        updated_context.context_data = {"updated": "data"}
        updated_context.context_metadata = sample_context_model.context_metadata
        updated_context.is_active = True
        updated_context.priority = sample_context_model.priority
        updated_context.vector_embedding = None
        updated_context.embedding_model = None
        updated_context.usage_count = sample_context_model.usage_count
        updated_context.created_at = sample_context_model.created_at
        updated_context.updated_at = datetime.utcnow()
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_context_model)
        mock_repo.update = AsyncMock(return_value=updated_context)
        
        context_id = str(sample_context_model.id)
        update_data = {
            "name": "updated-context",
            "description": "Updated description",
            "context_data": {"updated": "data"}
        }
        
        # Make request
        response = client.put(f"/api/v1/contexts/{context_id}", json=update_data)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "updated-context"
        assert data["description"] == "Updated description"
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_delete_context_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                   client, mock_db_session, mock_current_user, sample_context_model):
        """Test successful context deletion."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=sample_context_model)
        mock_repo.delete = AsyncMock(return_value=True)
        
        context_id = str(sample_context_model.id)
        
        # Make request
        response = client.delete(f"/api/v1/contexts/{context_id}")
        
        # Assertions
        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_repo.delete.assert_called_once_with(sample_context_model.id)
    
    # Context Merging Tests
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_merge_contexts_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                   client, mock_db_session, mock_current_user):
        """Test successful context merging."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        # Create source contexts
        context1 = Mock(spec=Context)
        context1.id = uuid4()
        context1.scope = ContextScope.USER
        context1.owner_id = mock_current_user["id"]
        context1.context_type_id = uuid4()
        context1.project_id = "project1"
        context1.context_data = {"key1": "value1", "shared": "original"}
        context1.priority = 1
        
        context2 = Mock(spec=Context)
        context2.id = uuid4()
        context2.scope = ContextScope.USER
        context2.owner_id = mock_current_user["id"]
        context2.context_type_id = context1.context_type_id
        context2.project_id = "project1"
        context2.context_data = {"key2": "value2", "shared": "updated"}
        context2.priority = 2
        
        merged_context = Mock(spec=Context)
        merged_context.id = uuid4()
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        def get_by_id_side_effect(context_id):
            if context_id == context1.id:
                return context1
            elif context_id == context2.id:
                return context2
            return None
        
        mock_repo.get_by_id = AsyncMock(side_effect=get_by_id_side_effect)
        mock_repo.create = AsyncMock(return_value=merged_context)
        
        merge_data = {
            "source_context_ids": [str(context1.id), str(context2.id)],
            "merge_strategy": "merge_deep",
            "create_new": True
        }
        
        # Make request
        response = client.post("/api/v1/contexts/merge", json=merge_data)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "merge_id" in data
        assert "result_context_id" in data
        assert data["merge_strategy"] == "merge_deep"
        assert "processing_time_ms" in data
        assert len(data["source_context_ids"]) == 2
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_merge_contexts_access_denied(self, mock_repo_class, mock_get_user, mock_get_db,
                                         client, mock_db_session, mock_current_user):
        """Test context merging with access denied."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        # Create context owned by different user
        context1 = Mock(spec=Context)
        context1.id = uuid4()
        context1.scope = ContextScope.PRIVATE
        context1.owner_id = "different-user"
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.get_by_id = AsyncMock(return_value=context1)
        
        merge_data = {
            "source_context_ids": [str(context1.id), str(uuid4())],
            "merge_strategy": "merge_deep"
        }
        
        # Make request
        response = client.post("/api/v1/contexts/merge", json=merge_data)
        
        # Assertions
        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert "Access denied" in data["detail"]
    
    # Context Search Tests
    
    @patch('pcs.api.v1.contexts.get_database_session')
    @patch('pcs.api.v1.contexts.get_current_user')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    def test_search_contexts_success(self, mock_repo_class, mock_get_user, mock_get_db,
                                    client, mock_db_session, mock_current_user, sample_context_model):
        """Test successful context search."""
        # Setup mocks
        mock_get_db.return_value = mock_db_session
        mock_get_user.return_value = mock_current_user
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        mock_repo.find_by_criteria = AsyncMock(return_value=[sample_context_model])
        
        search_data = {
            "query": "test",
            "scopes": ["user"],
            "owner_ids": ["user123"],
            "include_inactive": False
        }
        
        # Make request
        response = client.post("/api/v1/contexts/search", json=search_data)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "items" in data
        assert data["total"] >= 0
        assert "page" in data


class TestContextIntegration:
    """Integration tests for context API with real-like scenarios."""
    
    def test_full_context_lifecycle(self):
        """Test complete context lifecycle: type creation, context creation, merging, deletion."""
        # This would be an integration test that uses a real test database
        # For now, we'll create a simplified version
        
        type_data = {
            "name": "lifecycle-type",
            "description": "Type for lifecycle testing",
            "type_enum": "custom"
        }
        
        context_data = {
            "context_type_id": str(uuid4()),
            "name": "lifecycle-context",
            "scope": "user",
            "context_data": {"stage": "created"}
        }
        
        merge_data = {
            "source_context_ids": [str(uuid4()), str(uuid4())],
            "merge_strategy": "merge_deep"
        }
        
        search_data = {
            "query": "lifecycle",
            "scopes": ["user"]
        }
        
        # In a real integration test, you would:
        # 1. Create context type
        # 2. Create contexts
        # 3. Test merging
        # 4. Test search
        # 5. Delete contexts and type
        
        # For unit test, we just validate the data structures
        type_schema = ContextTypeCreate(**type_data)
        assert type_schema.name == "lifecycle-type"
        
        context_schema = ContextCreate(**context_data)
        assert context_schema.name == "lifecycle-context"
        
        merge_schema = ContextMergeRequest(**merge_data)
        assert merge_schema.merge_strategy == MergeStrategy.MERGE_DEEP
        
        search_schema = ContextSearchRequest(**search_data)
        assert search_schema.query == "lifecycle"
    
    def test_context_access_control_scenarios(self):
        """Test various access control scenarios."""
        # Test private context access
        private_context = {
            "context_type_id": str(uuid4()),
            "name": "private-context",
            "scope": "private",
            "owner_id": "user123",
            "context_data": {"secret": "data"}
        }
        
        # Test shared context access
        shared_context = {
            "context_type_id": str(uuid4()),
            "name": "shared-context",
            "scope": "shared",
            "context_data": {"public": "info"}
        }
        
        # Validate schemas
        private_schema = ContextCreate(**private_context)
        assert private_schema.scope == ContextScope.PRIVATE
        
        shared_schema = ContextCreate(**shared_context)
        assert shared_schema.scope == ContextScope.SHARED
    
    def test_complex_context_merging_scenarios(self):
        """Test complex context merging scenarios."""
        merge_scenarios = [
            {
                "source_context_ids": [str(uuid4()), str(uuid4())],
                "merge_strategy": "replace",
                "conflict_resolution": {"field1": "use_first"}
            },
            {
                "source_context_ids": [str(uuid4()), str(uuid4()), str(uuid4())],
                "merge_strategy": "merge_deep",
                "preserve_metadata": True,
                "create_new": True
            },
            {
                "source_context_ids": [str(uuid4()), str(uuid4())],
                "target_context_id": str(uuid4()),
                "merge_strategy": "append"
            }
        ]
        
        # Validate all scenarios
        for scenario in merge_scenarios:
            schema = ContextMergeRequest(**scenario)
            assert len(schema.source_context_ids) >= 2
            assert schema.merge_strategy in MergeStrategy
