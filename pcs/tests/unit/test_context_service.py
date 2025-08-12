"""
Unit tests for Context Management Service components.

Tests the ContextManager, ContextCache, ContextMerger, and ContextValidator
to ensure proper context operations, caching, and validation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import sys
from pathlib import Path

# Add src to path for direct imports without triggering main app initialization
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pcs.services.context_service import (
    ContextManager,
    ContextCache,
    ContextMerger,
    ContextValidator,
    Context,
    ContextMetadata,
    ContextType,
    ContextScope,
    MergeStrategy,
    ContextError
)
from pcs.repositories.redis_repo import RedisRepository


class TestContextMetadata:
    """Test ContextMetadata functionality."""
    
    def test_create_metadata(self):
        """Test creating context metadata."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(
            created_at=now,
            updated_at=now,
            tags={"test", "metadata"}
        )
        
        assert metadata.created_at == now
        assert metadata.updated_at == now
        assert metadata.tags == {"test", "metadata"}
        assert metadata.access_count == 0
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        metadata = ContextMetadata(
            created_at=now,
            updated_at=now,
            expires_at=expires,
            tags={"test", "example"}
        )
        
        data = metadata.to_dict()
        
        assert data["created_at"] == now.isoformat()
        assert data["expires_at"] == expires.isoformat()
        assert set(data["tags"]) == {"test", "example"}
        assert data["access_count"] == 0
    
    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": None,
            "access_count": 5,
            "tags": ["test", "example"]
        }
        
        metadata = ContextMetadata.from_dict(data)
        
        assert metadata.created_at == now
        assert metadata.updated_at == now
        assert metadata.expires_at is None
        assert metadata.access_count == 5
        assert metadata.tags == {"test", "example"}


class TestContext:
    """Test Context functionality."""
    
    def test_create_context(self):
        """Test creating a context."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test-123",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95},
            metadata=metadata
        )
        
        assert context.context_id == "test-123"
        assert context.context_type == ContextType.USER
        assert context.scope == ContextScope.PRIVATE
        assert context.data["name"] == "Alice"
    
    def test_context_is_expired(self):
        """Test context expiration check."""
        # Not expired context
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        metadata_future = ContextMetadata(
            created_at=now,
            updated_at=now,
            expires_at=future
        )
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},
            metadata=metadata_future
        )
        
        assert context.is_expired() is False
        
        # Expired context
        past = now - timedelta(hours=1)
        metadata_past = ContextMetadata(
            created_at=now,
            updated_at=now,
            expires_at=past
        )
        
        context_expired = Context(
            context_id="test2",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},
            metadata=metadata_past
        )
        
        assert context_expired.is_expired() is True
    
    def test_context_update_access(self):
        """Test updating access tracking."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},
            metadata=metadata
        )
        
        initial_count = context.metadata.access_count
        context.update_access()
        
        assert context.metadata.access_count == initial_count + 1
        assert context.metadata.last_accessed is not None
    
    def test_context_serialization(self):
        """Test context to_dict and from_dict."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        original_context = Context(
            context_id="test-123",
            context_type=ContextType.SESSION,
            scope=ContextScope.SHARED,
            data={"user_id": 42, "settings": {"theme": "dark"}},
            metadata=metadata,
            parent_id="parent-456"
        )
        
        # Convert to dict and back
        context_dict = original_context.to_dict()
        restored_context = Context.from_dict(context_dict)
        
        assert restored_context.context_id == original_context.context_id
        assert restored_context.context_type == original_context.context_type
        assert restored_context.scope == original_context.scope
        assert restored_context.data == original_context.data
        assert restored_context.parent_id == original_context.parent_id


class TestContextMerger:
    """Test ContextMerger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = ContextMerger()
        
        now = datetime.now(timezone.utc)
        self.base_metadata = ContextMetadata(created_at=now, updated_at=now)
        self.update_metadata = ContextMetadata(created_at=now, updated_at=now)
    
    def test_merge_replace_strategy(self):
        """Test replace merge strategy."""
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95, "level": 5},
            metadata=self.base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Bob", "age": 30},
            metadata=self.update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.REPLACE)
        
        # Should completely replace data
        assert merged.data == {"name": "Bob", "age": 30}
        assert "score" not in merged.data
        assert "level" not in merged.data
    
    def test_merge_deep_strategy(self):
        """Test deep merge strategy."""
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "user": {"name": "Alice", "settings": {"theme": "light", "lang": "en"}},
                "score": 95
            },
            metadata=self.base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "user": {"settings": {"theme": "dark", "notifications": True}},
                "level": 5
            },
            metadata=self.update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.MERGE_DEEP)
        
        # Should deep merge nested dictionaries
        expected_data = {
            "user": {
                "name": "Alice",  # Preserved from base
                "settings": {
                    "theme": "dark",  # Updated
                    "lang": "en",  # Preserved from base
                    "notifications": True  # Added from update
                }
            },
            "score": 95,  # Preserved from base
            "level": 5  # Added from update
        }
        
        assert merged.data == expected_data
    
    def test_merge_shallow_strategy(self):
        """Test shallow merge strategy."""
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "user": {"name": "Alice", "age": 30},
                "score": 95
            },
            metadata=self.base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "user": {"name": "Bob"},  # Will completely replace user object
                "level": 5
            },
            metadata=self.update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.MERGE_SHALLOW)
        
        # Should only merge top-level keys
        assert merged.data["user"] == {"name": "Bob"}  # Completely replaced
        assert merged.data["score"] == 95  # Preserved
        assert merged.data["level"] == 5  # Added
    
    def test_merge_append_strategy(self):
        """Test append merge strategy."""
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "tags": ["python", "web"],
                "score": 95
            },
            metadata=self.base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "tags": ["api", "testing"],
                "level": 5
            },
            metadata=self.update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.APPEND)
        
        # Should append to lists
        assert merged.data["tags"] == ["python", "web", "api", "testing"]
        assert merged.data["score"] == 95  # Non-list values are replaced
        assert merged.data["level"] == 5
    
    def test_merge_preserve_strategy(self):
        """Test preserve merge strategy."""
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95},
            metadata=self.base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Bob", "level": 5},
            metadata=self.update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.PRESERVE)
        
        # Should preserve existing values, only add new ones
        assert merged.data["name"] == "Alice"  # Preserved from base
        assert merged.data["score"] == 95  # Preserved from base
        assert merged.data["level"] == 5  # Added from update
    
    def test_merge_metadata_combination(self):
        """Test that metadata is properly combined during merge."""
        base_metadata = ContextMetadata(
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            access_count=5,
            tags={"base", "original"},
            priority=1
        )
        
        update_metadata = ContextMetadata(
            created_at=datetime(2024, 1, 2),
            updated_at=datetime(2024, 1, 2),
            expires_at=datetime(2024, 1, 3),
            tags={"update", "new"},
            priority=3
        )
        
        base_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},
            metadata=base_metadata
        )
        
        update_context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},
            metadata=update_metadata
        )
        
        merged = self.merger.merge_contexts(base_context, update_context, MergeStrategy.MERGE_DEEP)
        
        # Check metadata combination
        assert merged.metadata.created_at == base_metadata.created_at  # From base
        assert merged.metadata.expires_at == update_metadata.expires_at  # From update
        assert merged.metadata.access_count == base_metadata.access_count  # From base
        assert merged.metadata.tags == {"base", "original", "update", "new"}  # Union
        assert merged.metadata.priority == 3  # Max priority


class TestContextValidator:
    """Test ContextValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ContextValidator()
    
    def test_validate_valid_context(self):
        """Test validation of a valid context."""
        now = datetime.now(timezone.utc)
        future = now + timedelta(hours=1)
        metadata = ContextMetadata(created_at=now, updated_at=now, expires_at=future)
        
        context = Context(
            context_id="valid-context",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95},
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) == 0
    
    def test_validate_missing_context_id(self):
        """Test validation with missing context ID."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="",  # Empty ID
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"test": "data"},
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("context id" in error.lower() for error in errors)
    
    def test_validate_empty_data(self):
        """Test validation with empty data."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={},  # Empty data
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("empty" in error.lower() for error in errors)
    
    def test_validate_expired_context(self):
        """Test validation of expired context."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        metadata = ContextMetadata(created_at=now, updated_at=now, expires_at=past)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"test": "data"},
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("expired" in error.lower() for error in errors)
    
    def test_validate_restricted_keys(self):
        """Test validation with restricted keys in data."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={
                "normal_key": "value",
                "__proto__": "malicious",  # Restricted key
                "constructor": "also_bad"  # Restricted key
            },
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) >= 2  # At least 2 restricted key errors
        assert any("__proto__" in error for error in errors)
        assert any("constructor" in error for error in errors)
    
    def test_validate_deeply_nested_data(self):
        """Test validation with deeply nested data."""
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        # Create deeply nested structure beyond limit
        nested_data = {"level0": {}}
        current = nested_data["level0"]
        for i in range(1, 15):  # Create 15 levels (beyond limit of 10)
            current[f"level{i}"] = {}
            current = current[f"level{i}"]
        current["final"] = "value"
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data=nested_data,
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("deeply nested" in error.lower() for error in errors)
    
    @patch('json.dumps')
    def test_validate_large_data_size(self, mock_json_dumps):
        """Test validation with data that's too large."""
        # Mock json.dumps to return a very large string
        mock_json_dumps.return_value = "x" * (1024 * 1024 + 1)  # Just over 1MB
        
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"large": "data"},
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("too large" in error.lower() for error in errors)
    
    @patch('json.dumps')
    def test_validate_non_serializable_data(self, mock_json_dumps):
        """Test validation with non-JSON-serializable data."""
        # Mock json.dumps to raise an exception
        mock_json_dumps.side_effect = TypeError("Object not serializable")
        
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context = Context(
            context_id="test",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"test": "data"},
            metadata=metadata
        )
        
        errors = self.validator.validate_context(context)
        assert len(errors) > 0
        assert any("serializable" in error.lower() for error in errors)


class TestContextManager:
    """Test ContextManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Redis repository
        self.mock_redis_repo = Mock(spec=RedisRepository)
        self.context_manager = ContextManager(self.mock_redis_repo)
        
        # Mock the cache methods
        self.context_manager.cache.set_context = AsyncMock(return_value=True)
        self.context_manager.cache.get_context = AsyncMock(return_value=None)
        self.context_manager.cache.delete_context = AsyncMock(return_value=True)
    
    @pytest.mark.asyncio
    async def test_create_context(self):
        """Test creating a new context."""
        data = {"name": "Alice", "score": 95}
        ttl = timedelta(hours=1)
        tags = {"test", "user"}
        
        context = await self.context_manager.create_context(
            ContextType.USER,
            ContextScope.PRIVATE,
            data,
            ttl=ttl,
            tags=tags
        )
        
        assert context.context_type == ContextType.USER
        assert context.scope == ContextScope.PRIVATE
        assert context.data == data
        assert context.metadata.tags == tags
        assert context.metadata.expires_at is not None
        assert context.context_id is not None
        
        # Verify cache was called
        self.context_manager.cache.set_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_context_validation_failure(self):
        """Test creating context with validation failure."""
        # Empty data should cause validation failure
        data = {}
        
        with pytest.raises(ContextError) as exc_info:
            await self.context_manager.create_context(
                ContextType.USER,
                ContextScope.PRIVATE,
                data
            )
        
        assert "validation failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_get_context(self):
        """Test getting an existing context."""
        # Setup mock to return a context
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        mock_context = Context(
            context_id="test-123",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice"},
            metadata=metadata
        )
        
        self.context_manager.cache.get_context.return_value = mock_context
        
        result = await self.context_manager.get_context("test-123")
        
        assert result == mock_context
        self.context_manager.cache.get_context.assert_called_once_with("test-123")
    
    @pytest.mark.asyncio
    async def test_get_context_not_found(self):
        """Test getting a non-existent context."""
        self.context_manager.cache.get_context.return_value = None
        
        result = await self.context_manager.get_context("non-existent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_context(self):
        """Test updating an existing context."""
        # Setup existing context
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        existing_context = Context(
            context_id="test-123",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95},
            metadata=metadata
        )
        
        # Mock get_context to return existing context
        self.context_manager.cache.get_context.return_value = existing_context
        
        updates = {"score": 100, "level": 5}
        
        result = await self.context_manager.update_context("test-123", updates)
        
        assert result is not None
        assert result.data["name"] == "Alice"  # Preserved
        assert result.data["score"] == 100  # Updated
        assert result.data["level"] == 5  # Added
        
        # Verify cache operations
        self.context_manager.cache.get_context.assert_called_once_with("test-123")
        self.context_manager.cache.set_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_context_not_found(self):
        """Test updating a non-existent context."""
        self.context_manager.cache.get_context.return_value = None
        
        result = await self.context_manager.update_context("non-existent", {"key": "value"})
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_context(self):
        """Test deleting a context."""
        result = await self.context_manager.delete_context("test-123")
        
        assert result is True
        self.context_manager.cache.delete_context.assert_called_once_with("test-123")
    
    @pytest.mark.asyncio
    async def test_merge_contexts(self):
        """Test merging multiple contexts."""
        # Setup multiple contexts
        now = datetime.now(timezone.utc)
        metadata = ContextMetadata(created_at=now, updated_at=now)
        
        context1 = Context(
            context_id="ctx1",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"name": "Alice", "score": 95},
            metadata=metadata
        )
        
        context2 = Context(
            context_id="ctx2",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"level": 5, "achievements": ["first", "second"]},
            metadata=metadata
        )
        
        # Mock get_context to return appropriate contexts
        def mock_get_context(context_id):
            if context_id == "ctx1":
                return context1
            elif context_id == "ctx2":
                return context2
            return None
        
        self.context_manager.cache.get_context.side_effect = mock_get_context
        
        result = await self.context_manager.merge_contexts(["ctx1", "ctx2"])
        
        assert result is not None
        assert result.data["name"] == "Alice"  # From context1
        assert result.data["score"] == 95  # From context1
        assert result.data["level"] == 5  # From context2
        assert result.data["achievements"] == ["first", "second"]  # From context2
    
    @pytest.mark.asyncio
    async def test_merge_contexts_empty_list(self):
        """Test merging with empty context list."""
        result = await self.context_manager.merge_contexts([])
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired contexts."""
        # Mock cache cleanup
        self.context_manager.cache.cleanup_expired = AsyncMock(return_value=3)
        
        result = await self.context_manager.cleanup_expired()
        
        assert result == 3
        self.context_manager.cache.cleanup_expired.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
