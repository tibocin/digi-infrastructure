#!/usr/bin/env python3
"""
Filepath: tests/unit/test_prompt_versioning.py
Purpose: Unit tests for prompt version numbering logic
Related Components: PromptVersion, version numbering, template management
Tags: testing, unit-tests, prompt-versioning, version-numbering
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timezone

from pcs.models.prompts import PromptTemplate, PromptVersion, PromptStatus
from pcs.repositories.postgres_repo import PostgreSQLRepository


class TestPromptVersionNumbering:
    """Test prompt version numbering logic."""
    
    def test_version_number_increment(self):
        """Test that version numbers increment correctly."""
        template_id = uuid4()
        
        # Create first version
        version1 = PromptVersion(
            id=uuid4(),
            template_id=template_id,
            version_number=1,
            template="First version content",
            variables=["var1", "var2"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Create second version
        version2 = PromptVersion(
            id=uuid4(),
            template_id=template_id,
            version_number=2,
            template="Second version content",
            variables=["var1", "var2", "var3"],
            is_active=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        assert version1.version_number == 1
        assert version2.version_number == 2
        assert version1.template_id == version2.template_id
    
    def test_version_number_calculation(self):
        """Test version number calculation logic."""
        # Simulate counting existing versions
        existing_versions = [1, 2, 3]  # Version numbers 1, 2, 3
        next_version = len(existing_versions) + 1
        assert next_version == 4
        
        # Test with no existing versions
        existing_versions = []
        next_version = len(existing_versions) + 1
        assert next_version == 1
    
    def test_template_version_count_update(self):
        """Test that template version count is updated correctly."""
        template = PromptTemplate(
            id=uuid4(),
            name="Test Template",
            description="Test description",
            status=PromptStatus.DRAFT,
            version_count=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Simulate creating a version
        template.version_count += 1
        assert template.version_count == 1
        
        # Create another version
        template.version_count += 1
        assert template.version_count == 2
    
    def test_version_creation_sequence(self):
        """Test the complete sequence of version creation."""
        template_id = uuid4()
        versions = []
        
        # Create versions in sequence
        for i in range(1, 4):
            version = PromptVersion(
                id=uuid4(),
                template_id=template_id,
                version_number=i,
                template=f"Version {i} content",
                variables=[f"var{i}"],
                is_active=(i == 3),  # Only last version is active
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            versions.append(version)
        
        # Verify version numbers
        assert len(versions) == 3
        assert versions[0].version_number == 1
        assert versions[1].version_number == 2
        assert versions[2].version_number == 3
        
        # Verify only last version is active
        assert not versions[0].is_active
        assert not versions[1].is_active
        assert versions[2].is_active
        
        # Verify all versions belong to same template
        for version in versions:
            assert version.template_id == template_id


class TestPromptVersionRepository:
    """Test prompt version repository operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.add = Mock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        return session
    
    @pytest.fixture
    def version_repo(self, mock_session):
        """Create a version repository with mock session."""
        return PostgreSQLRepository(mock_session, PromptVersion)
    
    @pytest.mark.asyncio
    async def test_create_version(self, version_repo, mock_session):
        """Test creating a new version."""
        template_id = uuid4()
        version_data = {
            "template_id": template_id,
            "version_number": 1,
            "template": "Test content",
            "variables": ["var1", "var2"],
            "is_active": True
        }
        
        # Mock the entity creation
        mock_version = Mock()
        mock_version.id = uuid4()
        mock_version.template_id = template_id
        mock_version.version_number = 1
        mock_version.template = "Test content"
        mock_version.variables = ["var1", "var2"]
        mock_version.is_active = True
        mock_version.created_at = datetime.now(timezone.utc)
        mock_version.updated_at = datetime.now(timezone.utc)
        
        # Mock the session operations
        mock_session.add.return_value = None
        mock_session.refresh.return_value = None
        
        # Create the version
        result = await version_repo.create(mock_version)
        
        # Verify the session was called correctly
        mock_session.add.assert_called_once_with(mock_version)
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once_with(mock_version)
        
        assert result == mock_version
        assert result.version_number == 1
        assert result.template_id == template_id


if __name__ == "__main__":
    pytest.main([__file__])
