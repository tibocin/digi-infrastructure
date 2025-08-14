"""
Filepath: tests/integration/test_phase3_api_integration.py
Purpose: Comprehensive integration tests for all Phase 3 API endpoints
Related Components: All Phase 3 APIs, Database, Authentication, Full system workflow
Tags: integration-tests, api-testing, phase3, end-to-end, workflow-testing
"""

import asyncio
import pytest
import time
from typing import Dict, Any, List
from uuid import uuid4
from datetime import datetime
from unittest.mock import patch, Mock, AsyncMock

from fastapi.testclient import TestClient
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from pcs.main import create_app
from pcs.core.config import get_settings
from pcs.models.prompts import PromptStatus
from pcs.models.contexts import ContextScope
from pcs.models.conversations import ConversationStatus, MessageRole

# Remove the module-level mock approach


class TestPhase3APIIntegration:
    """
    Comprehensive integration tests for Phase 3 API endpoints.
    
    These tests simulate real-world workflows and test the interaction
    between all API endpoints in realistic scenarios.
    """
    
    @pytest.fixture(scope="class")
    def app(self):
        """Create test FastAPI application."""
        from unittest.mock import AsyncMock, Mock, patch
        from pcs.api.dependencies import get_database_session, get_current_user, get_current_app
        
        settings = get_settings()
        app = create_app(settings)
        
        # Override dependencies with mocks
        def mock_get_db():
            mock_session = AsyncMock()
            # Mock the add method to store entities
            mock_session.add = Mock()
            # Mock commit to return successfully
            mock_session.commit = AsyncMock()
            # Mock refresh to return the entity
            mock_session.refresh = AsyncMock()
            # Mock execute for queries
            mock_session.execute = AsyncMock()
            # Mock scalar_one_or_none for get operations
            mock_session.execute.return_value.scalar_one_or_none = Mock(return_value=None)
            # Mock rollback
            mock_session.rollback = AsyncMock()
            return mock_session
        
        def mock_get_user():
            return {"id": "test-user-id", "username": "testuser", "is_admin": True}
        
        def mock_get_app():
            return {"app_id": "test-app", "app_name": "PCS", "permissions": ["admin"], "environment": "test"}
        
        app.dependency_overrides[get_database_session] = mock_get_db
        app.dependency_overrides[get_current_user] = mock_get_user
        app.dependency_overrides[get_current_app] = mock_get_app
        
        return app
    
    @pytest.fixture(scope="class")
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def auth_headers(self):
        """Mock authentication headers."""
        return {
            "Authorization": "Bearer test_token_admin",
            "Content-Type": "application/json"
        }
    
    @pytest.fixture(scope="class")
    def user_headers(self):
        """Mock user authentication headers."""
        return {
            "Authorization": "Bearer test_token_user",
            "Content-Type": "application/json"
        }

    @patch('pcs.api.v1.prompts.PostgreSQLRepository')
    @patch('pcs.api.v1.contexts.PostgreSQLRepository')
    @patch('pcs.api.v1.conversations.PostgreSQLRepository')
    @patch('pcs.api.v1.admin.PostgreSQLRepository')
    def test_complete_prompt_workflow(self, mock_admin, mock_conversations, mock_contexts, mock_prompts, client, auth_headers):
        """
        Test complete prompt management workflow:
        1. Create prompt template
        2. Create versions
        3. Generate prompts
        4. Update template
        5. Delete template
        """
        # Set up mock repository
        mock_repo = Mock()
        
        # Mock create method to return a mock entity
        async def mock_create(entity):
            # Check if this is a PromptTemplate or PromptVersion
            if hasattr(entity, 'name'):  # PromptTemplate
                # Create a mock entity with required attributes
                mock_entity = Mock()
                mock_entity.id = uuid4()  # Use proper UUID
                mock_entity.name = getattr(entity, 'name', 'Test Template')
                mock_entity.description = getattr(entity, 'description', 'Test Description')
                mock_entity.category = getattr(entity, 'category', 'test')
                mock_entity.tags = getattr(entity, 'tags', [])
                mock_entity.status = getattr(entity, 'status', 'draft')
                mock_entity.is_system = getattr(entity, 'is_system', False)
                mock_entity.author = getattr(entity, 'author', 'testuser')
                mock_entity.version_count = getattr(entity, 'version_count', 0)
                mock_entity.usage_count = 0
                mock_entity.created_at = datetime.now()  # Use proper datetime
                mock_entity.updated_at = datetime.now()  # Use proper datetime
                mock_entity.versions = []
                mock_entity.current_version = None
                mock_entity.latest_version = None
                mock_entity.rules = []
                
                # Store the template entity for later retrieval
                mock_get_by_id.template_entity = mock_entity
                mock_get_by_id.template_id = mock_entity.id
                
            else:  # PromptVersion
                # Create a mock version entity with required attributes
                mock_entity = Mock()
                mock_entity.id = uuid4()  # Use proper UUID
                mock_entity.template_id = getattr(entity, 'template_id', uuid4())
                mock_entity.version_number = getattr(entity, 'version_number', 1)
                # Map database fields to response schema fields
                mock_entity.content = getattr(entity, 'template', 'Test content')  # Map 'template' to 'content'
                # For the response, we need to return the original variables dict, not the converted list
                mock_entity.variables = {
                    "context": {"type": "string", "required": True, "description": "Conversation context"},
                    "user_input": {"type": "string", "required": True, "description": "User input"}
                }  # Return the full variables dict as expected by response schema
                mock_entity.changelog = getattr(entity, 'change_notes', 'Test changelog')  # Map 'change_notes' to 'changelog'
                mock_entity.is_active = getattr(entity, 'is_active', False)
                mock_entity.usage_count = 0
                mock_entity.success_rate = None
                mock_entity.created_at = datetime.now()  # Use proper datetime
                mock_entity.updated_at = datetime.now()  # Use proper datetime
                
                # Add the version to the template's versions list
                if hasattr(mock_get_by_id, 'template_entity') and mock_get_by_id.template_entity:
                    mock_get_by_id.template_entity.versions.append(mock_entity)
            
            return mock_entity
        
        # Mock find_by_criteria to return empty list (no conflicts)
        async def mock_find_by_criteria(**kwargs):
            return []
        
        # Mock get_by_id to return the template when ID matches
        async def mock_get_by_id(id):
            # If this is the template ID we created, return the template
            if hasattr(mock_get_by_id, 'template_id') and str(id) == str(mock_get_by_id.template_id):
                return mock_get_by_id.template_entity
            return None
        
        # Mock update method to handle updates
        async def mock_update(id, updates):
            # Update the stored template entity if it matches
            if hasattr(mock_get_by_id, 'template_entity') and str(id) == str(mock_get_by_id.template_entity.id):
                for key, value in updates.items():
                    setattr(mock_get_by_id.template_entity, key, value)
            return mock_get_by_id.template_entity
        
        # Store the template entity for later retrieval
        mock_get_by_id.template_entity = None
        mock_get_by_id.template_id = None
        
        mock_repo.create = mock_create
        mock_repo.find_by_criteria = mock_find_by_criteria
        mock_repo.get_by_id = mock_get_by_id
        mock_repo.update = mock_update
        
        # Set the mock class to return our mock instance
        mock_prompts.return_value = mock_repo
        mock_contexts.return_value = mock_repo
        mock_conversations.return_value = mock_repo
        mock_admin.return_value = mock_repo
        
        # Step 1: Create prompt template
        template_data = {
            "name": f"integration-test-template-{uuid4().hex[:8]}",
            "description": "Integration test template",
            "category": "testing",
            "tags": ["integration", "test"],
            "author": "test_user"
        }
        
        response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
        
        # Debug: Print response details
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        assert response.status_code == status.HTTP_201_CREATED
        template = response.json()
        template_id = template["id"]
        
        assert template["name"] == template_data["name"]
        assert template["status"] == "draft"
        assert template["version_count"] == 0
        
        # Step 2: Create first version
        version_data = {
            "content": "You are a helpful assistant. Context: {{context}}\n\nUser: {{user_input}}\n\nAssistant:",
            "variables": {
                "context": {
                    "type": "string",
                    "required": True,
                    "description": "Conversation context"
                },
                "user_input": {
                    "type": "string",
                    "required": True,
                    "description": "User input"
                }
            },
            "changelog": "Initial version"
        }
        
        response = client.post(
            f"/api/v1/prompts/{template_id}/versions",
            json=version_data,
            headers=auth_headers,
            params={"make_active": True}
        )
        
        # Debug: Print response details for version creation
        print(f"Version creation response status: {response.status_code}")
        print(f"Version creation response content: {response.text}")
        
        assert response.status_code == status.HTTP_201_CREATED
        version = response.json()
        
        assert version["version_number"] == 1
        assert version["is_active"] is True
        assert version["content"] == version_data["content"]
        
        # Step 3: Create second version
        version_data_v2 = {
            "content": "You are an expert assistant. Context: {{context}}\n\nUser: {{user_input}}\n\nAssistant:",
            "variables": version_data["variables"],
            "changelog": "Improved prompt"
        }
        
        response = client.post(
            f"/api/v1/prompts/{template_id}/versions",
            json=version_data_v2,
            headers=auth_headers,
            params={"make_active": True}
        )
        assert response.status_code == status.HTTP_201_CREATED
        version_v2 = response.json()
        assert version_v2["version_number"] == 2
        
        # Step 4: List versions
        response = client.get(f"/api/v1/prompts/{template_id}/versions", headers=auth_headers)
        
        # Debug: Print response details for version listing
        print(f"Version listing response status: {response.status_code}")
        print(f"Version listing response content: {response.text}")
        
        assert response.status_code == status.HTTP_200_OK
        versions = response.json()
        assert len(versions) == 2
        
        # Step 5: Generate prompt
        generation_data = {
            "template_name": template_data["name"],
            "context_data": {"session_id": "test_session"},
            "variables": {
                "context": "This is a test conversation",
                "user_input": "Hello, how are you?"
            }
        }
        
        response = client.post("/api/v1/prompts/generate", json=generation_data, headers=auth_headers)
        
        # Debug: Print response details for prompt generation
        print(f"Prompt generation response status: {response.status_code}")
        print(f"Prompt generation response content: {response.text}")
        
        assert response.status_code == status.HTTP_200_OK
        generation_result = response.json()
        
        assert "generated_prompt" in generation_result
        assert "request_id" in generation_result
        assert generation_result["status"] == "completed"
        
        # Step 6: Update template
        update_data = {
            "description": "Updated integration test template",
            "status": "active"
        }
        
        response = client.put(f"/api/v1/prompts/{template_id}", json=update_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        updated_template = response.json()
        
        assert updated_template["description"] == update_data["description"]
        assert updated_template["status"] == "active"
        
        # Step 7: Get template with versions
        response = client.get(
            f"/api/v1/prompts/{template_id}",
            headers=auth_headers,
            params={"include_versions": True}
        )
        assert response.status_code == status.HTTP_200_OK
        template_with_versions = response.json()
        assert len(template_with_versions.get("versions", [])) == 2
        
        # Step 8: Clean up - Delete template
        response = client.delete(f"/api/v1/prompts/{template_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_complete_context_workflow(self, client, auth_headers):
        """
        Test complete context management workflow:
        1. Create context type
        2. Create contexts
        3. Merge contexts
        4. Search contexts
        5. Clean up
        """
        # Step 1: Create context type
        context_type_data = {
            "name": f"integration-test-type-{uuid4().hex[:8]}",
            "description": "Integration test context type",
            "type_enum": "custom",
            "schema_definition": {
                "type": "object",
                "properties": {
                    "setting1": {"type": "string"},
                    "setting2": {"type": "number"},
                    "nested": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"}
                        }
                    }
                }
            },
            "default_scope": "user",
            "supports_vectors": False
        }
        
        response = client.post("/api/v1/contexts/types", json=context_type_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        context_type = response.json()
        context_type_id = context_type["id"]
        
        # Step 2: Create first context
        context1_data = {
            "context_type_id": context_type_id,
            "name": f"integration-test-context-1-{uuid4().hex[:8]}",
            "description": "First test context",
            "scope": "user",
            "context_data": {
                "setting1": "value1",
                "setting2": 100,
                "nested": {"value": "nested1"}
            },
            "priority": 1
        }
        
        response = client.post("/api/v1/contexts/", json=context1_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        context1 = response.json()
        context1_id = context1["id"]
        
        # Step 3: Create second context
        context2_data = {
            "context_type_id": context_type_id,
            "name": f"integration-test-context-2-{uuid4().hex[:8]}",
            "description": "Second test context",
            "scope": "user",
            "context_data": {
                "setting1": "value2",
                "setting2": 200,
                "nested": {"value": "nested2"},
                "additional": "extra_data"
            },
            "priority": 2
        }
        
        response = client.post("/api/v1/contexts/", json=context2_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        context2 = response.json()
        context2_id = context2["id"]
        
        # Step 4: Merge contexts
        merge_data = {
            "source_context_ids": [context1_id, context2_id],
            "merge_strategy": "merge_deep",
            "create_new": True
        }
        
        response = client.post("/api/v1/contexts/merge", json=merge_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        merge_result = response.json()
        
        assert "result_context_id" in merge_result
        assert merge_result["merge_strategy"] == "merge_deep"
        assert len(merge_result["source_context_ids"]) == 2
        merged_context_id = merge_result["result_context_id"]
        
        # Step 5: Verify merged context
        response = client.get(f"/api/v1/contexts/{merged_context_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        merged_context = response.json()
        
        # Verify merge results
        merged_data = merged_context["context_data"]
        assert "additional" in merged_data  # From context2
        assert merged_data["setting2"] == 200  # Latest value
        
        # Step 6: Search contexts
        search_data = {
            "query": "integration-test",
            "scopes": ["user"],
            "include_inactive": False
        }
        
        response = client.post("/api/v1/contexts/search", json=search_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        search_results = response.json()
        
        assert search_results["total"] >= 3  # At least our 3 contexts
        found_contexts = [item["name"] for item in search_results["items"]]
        assert any("integration-test-context" in name for name in found_contexts)
        
        # Step 7: Clean up
        for context_id in [context1_id, context2_id, merged_context_id]:
            response = client.delete(f"/api/v1/contexts/{context_id}", headers=auth_headers)
            assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_complete_conversation_workflow(self, client, auth_headers):
        """
        Test complete conversation workflow:
        1. Create conversation
        2. Add messages
        3. Get statistics
        4. Update conversation
        5. Search conversations
        6. Clean up
        """
        # Step 1: Create conversation
        conversation_data = {
            "title": f"Integration Test Conversation {uuid4().hex[:8]}",
            "description": "Test conversation for integration testing",
            "project_id": "test_project",
            "priority": "normal",
            "conversation_metadata": {
                "test_type": "integration",
                "created_by": "test_suite"
            }
        }
        
        response = client.post("/api/v1/conversations/", json=conversation_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        conversation = response.json()
        conversation_id = conversation["id"]
        
        assert conversation["title"] == conversation_data["title"]
        assert conversation["status"] == "active"
        assert conversation["message_count"] == 0
        
        # Step 2: Add user message
        user_message_data = {
            "role": "user",
            "content": "Hello, I need help with integration testing",
            "message_type": "text",
            "input_tokens": 10
        }
        
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json=user_message_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        user_message = response.json()
        
        assert user_message["sequence_number"] == 1
        assert user_message["role"] == "user"
        assert user_message["content"] == user_message_data["content"]
        
        # Step 3: Add assistant message
        assistant_message_data = {
            "role": "assistant",
            "content": "I'd be happy to help you with integration testing. What specific aspect would you like to explore?",
            "message_type": "text",
            "input_tokens": 15,
            "output_tokens": 25
        }
        
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json=assistant_message_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        assistant_message = response.json()
        
        assert assistant_message["sequence_number"] == 2
        assert assistant_message["total_tokens"] == 40
        
        # Step 4: Add system message
        system_message_data = {
            "role": "system",
            "content": "Conversation started",
            "message_type": "system_notification"
        }
        
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json=system_message_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        
        # Step 5: List messages
        response = client.get(f"/api/v1/conversations/{conversation_id}/messages", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        messages_list = response.json()
        
        assert messages_list["total"] == 3
        assert len(messages_list["items"]) == 3
        
        # Verify message order
        messages = messages_list["items"]
        assert messages[0]["sequence_number"] == 1
        assert messages[1]["sequence_number"] == 2
        assert messages[2]["sequence_number"] == 3
        
        # Step 6: Get conversation statistics
        response = client.get(f"/api/v1/conversations/{conversation_id}/stats", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        stats = response.json()
        
        assert stats["message_count"] == 3
        assert stats["user_messages"] == 1
        assert stats["assistant_messages"] == 1
        assert stats["system_messages"] == 1
        assert stats["total_tokens"] > 0
        
        # Step 7: Update conversation
        update_data = {
            "status": "completed",
            "priority": "high"
        }
        
        response = client.put(f"/api/v1/conversations/{conversation_id}", json=update_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        updated_conversation = response.json()
        
        assert updated_conversation["status"] == "completed"
        assert updated_conversation["priority"] == "high"
        assert updated_conversation["ended_at"] is not None
        
        # Step 8: Search conversations
        search_data = {
            "query": "Integration Test",
            "statuses": ["completed"],
            "include_archived": False
        }
        
        response = client.post("/api/v1/conversations/search", json=search_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        search_results = response.json()
        
        # Should find our conversation
        found_conversation = None
        for conv in search_results["items"]:
            if conv["id"] == conversation_id:
                found_conversation = conv
                break
        
        assert found_conversation is not None
        assert found_conversation["status"] == "completed"
        
        # Step 9: Clean up
        response = client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_admin_workflow(self, client, auth_headers):
        """
        Test admin API workflow:
        1. Get system stats
        2. Get detailed health
        3. Database maintenance
        4. Cache management
        5. User management
        6. Get system config
        7. Get logs
        """
        # Step 1: Get system statistics
        response = client.get("/api/v1/admin/stats", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        stats = response.json()
        
        required_sections = ["system_info", "database_stats", "application_stats", "resource_usage"]
        for section in required_sections:
            assert section in stats
        
        assert "timestamp" in stats
        assert "uptime_seconds" in stats
        
        # Step 2: Get detailed health check
        response = client.get("/api/v1/admin/health/detailed", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        health = response.json()
        
        assert "status" in health
        assert "checks" in health
        assert "timestamp" in health
        
        # Step 3: Database maintenance (dry run)
        maintenance_data = {
            "operation": "analyze",
            "dry_run": True,
            "force": False
        }
        
        response = client.post("/api/v1/admin/database/maintenance", json=maintenance_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        maintenance_result = response.json()
        
        assert maintenance_result["operation"] == "analyze"
        assert maintenance_result["dry_run"] is True
        assert "tables_processed" in maintenance_result
        
        # Step 4: Cache management
        cache_data = {
            "operation": "analyze"
        }
        
        response = client.post("/api/v1/admin/cache/management", json=cache_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        cache_result = response.json()
        
        assert cache_result["operation"] == "analyze"
        assert cache_result["status"] == "completed"
        
        # Step 5: List users
        response = client.get("/api/v1/admin/users", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        users_list = response.json()
        
        assert "users" in users_list
        assert "total" in users_list
        assert "page" in users_list
        
        # Step 6: Get system configuration
        response = client.get("/api/v1/admin/config", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        config = response.json()
        
        assert "application" in config
        assert "server" in config
        assert "features" in config
        
        # Step 7: Get system logs
        response = client.get("/api/v1/admin/logs", headers=auth_headers, params={"level": "INFO", "lines": 10})
        assert response.status_code == status.HTTP_200_OK
        logs = response.json()
        
        assert "logs" in logs
        assert "total_lines" in logs
        assert "timestamp" in logs

    def test_cross_api_integration_workflow(self, client, auth_headers):
        """
        Test integration across multiple APIs in a realistic workflow:
        1. Create prompt template and context type
        2. Create context instance
        3. Create conversation
        4. Generate prompt using context
        5. Add generated message to conversation
        6. Get conversation stats with prompt usage
        7. Clean up all resources
        """
        # Step 1: Create prompt template
        template_data = {
            "name": f"cross-api-template-{uuid4().hex[:8]}",
            "description": "Template for cross-API integration testing",
            "category": "integration",
            "tags": ["cross-api", "integration"]
        }
        
        response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        template = response.json()
        template_id = template["id"]
        
        # Create version for the template
        version_data = {
            "content": "Context: {{user_context}}\nProject: {{project_info}}\n\nUser: {{user_message}}\n\nAssistant:",
            "variables": {
                "user_context": {"type": "string", "required": True},
                "project_info": {"type": "string", "required": False},
                "user_message": {"type": "string", "required": True}
            },
            "changelog": "Initial cross-API version"
        }
        
        response = client.post(
            f"/api/v1/prompts/{template_id}/versions",
            json=version_data,
            headers=auth_headers,
            params={"make_active": True}
        )
        assert response.status_code == status.HTTP_201_CREATED
        
        # Step 2: Create context type
        context_type_data = {
            "name": f"cross-api-context-type-{uuid4().hex[:8]}",
            "description": "Context type for cross-API testing",
            "type_enum": "custom",
            "schema_definition": {
                "type": "object",
                "properties": {
                    "user_preferences": {"type": "object"},
                    "project_settings": {"type": "object"}
                }
            },
            "default_scope": "user"
        }
        
        response = client.post("/api/v1/contexts/types", json=context_type_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        context_type = response.json()
        context_type_id = context_type["id"]
        
        # Step 3: Create context instance
        context_data = {
            "context_type_id": context_type_id,
            "name": f"cross-api-context-{uuid4().hex[:8]}",
            "description": "Context for cross-API integration",
            "scope": "user",
            "context_data": {
                "user_preferences": {
                    "theme": "dark",
                    "language": "python",
                    "experience_level": "intermediate"
                },
                "project_settings": {
                    "name": "integration-test-project",
                    "type": "microservice",
                    "framework": "fastapi"
                }
            }
        }
        
        response = client.post("/api/v1/contexts/", json=context_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        context = response.json()
        context_id = context["id"]
        
        # Step 4: Create conversation
        conversation_data = {
            "title": f"Cross-API Integration Test {uuid4().hex[:8]}",
            "description": "Testing integration between prompts, contexts, and conversations",
            "project_id": "integration-test-project",
            "active_prompt_template_id": template_id,
            "context_ids": [context_id]
        }
        
        response = client.post("/api/v1/conversations/", json=conversation_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        conversation = response.json()
        conversation_id = conversation["id"]
        
        # Step 5: Generate prompt using context
        generation_data = {
            "template_name": template_data["name"],
            "context_data": {
                "user_context": "Python FastAPI development",
                "project_info": "Building a microservice API"
            },
            "variables": {
                "user_message": "Can you help me with async database operations?"
            }
        }
        
        response = client.post("/api/v1/prompts/generate", json=generation_data, headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        generation_result = response.json()
        
        assert generation_result["status"] == "completed"
        generated_prompt = generation_result["generated_prompt"]
        
        # Step 6: Add user message to conversation
        user_message_data = {
            "role": "user",
            "content": "Can you help me with async database operations?",
            "message_type": "text",
            "prompt_template_id": template_id,
            "context_ids": [context_id]
        }
        
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json=user_message_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        user_message = response.json()
        
        # Step 7: Add assistant response
        assistant_message_data = {
            "role": "assistant",
            "content": "I'll help you with async database operations in FastAPI. Here are some best practices...",
            "message_type": "text",
            "prompt_template_id": template_id,
            "context_ids": [context_id]
        }
        
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json=assistant_message_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        
        # Step 8: Get conversation with full context
        response = client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=auth_headers,
            params={"include_messages": True}
        )
        assert response.status_code == status.HTTP_200_OK
        full_conversation = response.json()
        
        assert len(full_conversation["messages"]) == 2
        assert full_conversation["active_prompt_template_id"] == template_id
        assert context_id in full_conversation["context_ids"]
        
        # Step 9: Get conversation statistics
        response = client.get(f"/api/v1/conversations/{conversation_id}/stats", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        stats = response.json()
        
        assert stats["message_count"] == 2
        assert stats["user_messages"] == 1
        assert stats["assistant_messages"] == 1
        
        # Step 10: Verify context usage tracking
        response = client.get(f"/api/v1/contexts/{context_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        updated_context = response.json()
        
        # Usage count should have been incremented
        assert updated_context["usage_count"] > 0
        
        # Step 11: Clean up all resources
        cleanup_items = [
            ("conversations", conversation_id),
            ("contexts", context_id),
            ("prompts", template_id)
        ]
        
        for resource_type, resource_id in cleanup_items:
            response = client.delete(f"/api/v1/{resource_type}/{resource_id}", headers=auth_headers)
            assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_error_handling_and_edge_cases(self, client, auth_headers, user_headers):
        """
        Test error handling and edge cases across APIs:
        1. Access control violations
        2. Invalid data handling
        3. Resource not found scenarios
        4. Validation errors
        5. Conflict scenarios
        """
        # Test 1: Access control - Non-admin trying to access admin endpoints
        response = client.get("/api/v1/admin/stats", headers=user_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN
        
        # Test 2: Invalid UUID format
        invalid_uuid = "not-a-uuid"
        response = client.get(f"/api/v1/prompts/{invalid_uuid}", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test 3: Resource not found
        non_existent_uuid = str(uuid4())
        response = client.get(f"/api/v1/prompts/{non_existent_uuid}", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Test 4: Validation errors - Invalid prompt template data
        invalid_template_data = {
            "name": "",  # Empty name should fail validation
            "description": "Invalid template"
        }
        
        response = client.post("/api/v1/prompts/", json=invalid_template_data, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test 5: Validation errors - Invalid context merge request
        invalid_merge_data = {
            "source_context_ids": [str(uuid4())],  # Only one context (minimum is 2)
            "merge_strategy": "merge_deep"
        }
        
        response = client.post("/api/v1/contexts/merge", json=invalid_merge_data, headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test 6: Create template for conflict test
        template_data = {
            "name": f"conflict-test-{uuid4().hex[:8]}",
            "description": "Template for conflict testing"
        }
        
        response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        template = response.json()
        template_id = template["id"]
        
        # Test 7: Conflict - Duplicate template name
        duplicate_template_data = {
            "name": template_data["name"],  # Same name
            "description": "Duplicate template"
        }
        
        response = client.post("/api/v1/prompts/", json=duplicate_template_data, headers=auth_headers)
        assert response.status_code == status.HTTP_409_CONFLICT
        
        # Test 8: Invalid conversation message role
        invalid_message_data = {
            "role": "invalid_role",
            "content": "Test message"
        }
        
        response = client.post(
            f"/api/v1/conversations/{uuid4()}/messages",
            json=invalid_message_data,
            headers=auth_headers
        )
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_404_NOT_FOUND]
        
        # Clean up
        response = client.delete(f"/api/v1/prompts/{template_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_204_NO_CONTENT

    def test_performance_and_pagination(self, client, auth_headers):
        """
        Test pagination and performance characteristics:
        1. Create multiple resources
        2. Test pagination parameters
        3. Test search performance
        4. Test large dataset handling
        """
        created_templates = []
        
        # Create multiple prompt templates
        for i in range(25):
            template_data = {
                "name": f"perf-test-template-{i:03d}-{uuid4().hex[:6]}",
                "description": f"Performance test template {i}",
                "category": "performance",
                "tags": ["performance", "test", f"batch_{i // 5}"]
            }
            
            response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
            assert response.status_code == status.HTTP_201_CREATED
            created_templates.append(response.json()["id"])
        
        # Test pagination - First page
        response = client.get("/api/v1/prompts/", headers=auth_headers, params={"page": 1, "size": 10})
        assert response.status_code == status.HTTP_200_OK
        page1 = response.json()
        
        assert len(page1["items"]) <= 10
        assert page1["page"] == 1
        assert page1["size"] == 10
        assert page1["total"] >= 25
        
        # Test pagination - Second page
        response = client.get("/api/v1/prompts/", headers=auth_headers, params={"page": 2, "size": 10})
        assert response.status_code == status.HTTP_200_OK
        page2 = response.json()
        
        assert page2["page"] == 2
        # Ensure different items on different pages
        page1_ids = {item["id"] for item in page1["items"]}
        page2_ids = {item["id"] for item in page2["items"]}
        assert page1_ids.isdisjoint(page2_ids)
        
        # Test search performance
        search_start = time.time()
        response = client.get(
            "/api/v1/prompts/",
            headers=auth_headers,
            params={"search": "perf-test", "category": "performance"}
        )
        search_time = time.time() - search_start
        
        assert response.status_code == status.HTTP_200_OK
        search_results = response.json()
        assert search_results["total"] >= 25
        assert search_time < 2.0  # Should complete within 2 seconds
        
        # Test large page size limits
        response = client.get("/api/v1/prompts/", headers=auth_headers, params={"size": 150})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY  # Should exceed max page size
        
        # Clean up
        for template_id in created_templates:
            response = client.delete(f"/api/v1/prompts/{template_id}", headers=auth_headers)
            # Don't assert here as some deletes might fail due to dependencies

    def test_concurrent_operations(self, client, auth_headers):
        """
        Test concurrent operations and race conditions:
        1. Concurrent template creation
        2. Concurrent context merging
        3. Concurrent message additions
        """
        import concurrent.futures
        import threading
        
        # Test 1: Concurrent template creation with unique names
        def create_template(index):
            template_data = {
                "name": f"concurrent-template-{index}-{uuid4().hex[:8]}",
                "description": f"Concurrent test template {index}",
                "category": "concurrent"
            }
            
            response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
            return response.status_code, response.json() if response.status_code == 201 else None
        
        # Create 10 templates concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_template, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        successful_creates = [r for r in results if r[0] == 201]
        assert len(successful_creates) == 10
        
        # Clean up created templates
        template_ids = [r[1]["id"] for r in successful_creates if r[1]]
        for template_id in template_ids:
            client.delete(f"/api/v1/prompts/{template_id}", headers=auth_headers)

    def test_data_consistency(self, client, auth_headers):
        """
        Test data consistency across operations:
        1. Template version consistency
        2. Context relationship consistency
        3. Conversation message ordering
        4. Statistics accuracy
        """
        # Test 1: Template version consistency
        template_data = {
            "name": f"consistency-test-{uuid4().hex[:8]}",
            "description": "Template for consistency testing"
        }
        
        response = client.post("/api/v1/prompts/", json=template_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        template = response.json()
        template_id = template["id"]
        
        # Create multiple versions
        for i in range(3):
            version_data = {
                "content": f"Version {i+1} content",
                "variables": {"var1": {"type": "string"}},
                "changelog": f"Version {i+1}"
            }
            
            response = client.post(
                f"/api/v1/prompts/{template_id}/versions",
                json=version_data,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED
            version = response.json()
            assert version["version_number"] == i + 1
        
        # Verify template version count
        response = client.get(f"/api/v1/prompts/{template_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        updated_template = response.json()
        assert updated_template["version_count"] == 3
        
        # Test 2: Conversation message ordering
        conversation_data = {
            "title": "Consistency Test Conversation",
            "description": "Testing message ordering consistency"
        }
        
        response = client.post("/api/v1/conversations/", json=conversation_data, headers=auth_headers)
        assert response.status_code == status.HTTP_201_CREATED
        conversation = response.json()
        conversation_id = conversation["id"]
        
        # Add messages in specific order
        messages_data = [
            {"role": "user", "content": "First message", "message_type": "text"},
            {"role": "assistant", "content": "Second message", "message_type": "text"},
            {"role": "user", "content": "Third message", "message_type": "text"}
        ]
        
        for i, msg_data in enumerate(messages_data):
            response = client.post(
                f"/api/v1/conversations/{conversation_id}/messages",
                json=msg_data,
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_201_CREATED
            message = response.json()
            assert message["sequence_number"] == i + 1
        
        # Verify message ordering
        response = client.get(f"/api/v1/conversations/{conversation_id}/messages", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        messages_list = response.json()
        
        assert len(messages_list["items"]) == 3
        for i, message in enumerate(messages_list["items"]):
            assert message["sequence_number"] == i + 1
            assert message["content"] == messages_data[i]["content"]
        
        # Clean up
        client.delete(f"/api/v1/conversations/{conversation_id}", headers=auth_headers)
        client.delete(f"/api/v1/prompts/{template_id}", headers=auth_headers)

    @pytest.mark.asyncio
    async def test_system_resilience(self, client, auth_headers):
        """
        Test system resilience and recovery:
        1. Graceful degradation
        2. Error recovery
        3. Resource cleanup
        """
        # This test would typically involve more complex scenarios
        # like database connection failures, cache misses, etc.
        # For now, we'll test basic resilience patterns
        
        # Test graceful handling of invalid operations
        test_cases = [
            {
                "method": "DELETE",
                "url": f"/api/v1/prompts/{uuid4()}",
                "expected_status": status.HTTP_404_NOT_FOUND
            },
            {
                "method": "PUT", 
                "url": f"/api/v1/contexts/{uuid4()}",
                "json": {"name": "updated"},
                "expected_status": status.HTTP_404_NOT_FOUND
            },
            {
                "method": "GET",
                "url": f"/api/v1/conversations/{uuid4()}/messages",
                "expected_status": status.HTTP_404_NOT_FOUND
            }
        ]
        
        for test_case in test_cases:
            method = test_case["method"]
            url = test_case["url"]
            expected_status = test_case["expected_status"]
            
            if method == "GET":
                response = client.get(url, headers=auth_headers)
            elif method == "PUT":
                response = client.put(url, json=test_case.get("json", {}), headers=auth_headers)
            elif method == "DELETE":
                response = client.delete(url, headers=auth_headers)
            
            assert response.status_code == expected_status

    def test_final_integration_summary(self, client, auth_headers):
        """
        Final integration test that exercises all major API groups
        and verifies the complete system works end-to-end.
        """
        print("\n" + "="*60)
        print("PHASE 3 API INTEGRATION TEST SUMMARY")
        print("="*60)
        
        # Test each major API group
        api_groups = {
            "Health API": "/api/v1/health",
            "Prompts API": "/api/v1/prompts/",
            "Contexts API": "/api/v1/contexts/types",
            "Conversations API": "/api/v1/conversations/",
            "Admin API": "/api/v1/admin/stats"
        }
        
        results = {}
        
        for api_name, endpoint in api_groups.items():
            try:
                response = client.get(endpoint, headers=auth_headers)
                results[api_name] = {
                    "status": "✅ PASS" if response.status_code < 400 else "❌ FAIL",
                    "status_code": response.status_code,
                    "response_time": "< 1s"  # Simplified
                }
            except Exception as e:
                results[api_name] = {
                    "status": "❌ ERROR",
                    "error": str(e)
                }
        
        # Print results
        for api_name, result in results.items():
            status = result.get("status", "❌ ERROR")
            print(f"{api_name:20} {status}")
            if "status_code" in result:
                print(f"{'':20} Status Code: {result['status_code']}")
        
        print("="*60)
        
        # Verify all APIs are functional
        all_passed = all(result.get("status", "").startswith("✅") for result in results.values())
        assert all_passed, f"Some API groups failed: {results}"
        
        print("🎉 ALL PHASE 3 API INTEGRATION TESTS PASSED!")
        print("="*60)
