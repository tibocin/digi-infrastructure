"""
Unit tests for Repository layer components.

Tests the abstract repository pattern and specific repository implementations
to ensure proper data access operations and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from uuid import uuid4

import sys
from pathlib import Path

# Add src to path for direct imports without triggering main app initialization
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pcs.repositories.base import (
    AbstractRepository,
    BaseRepository,
    RepositoryError
)
from pcs.repositories.redis_repo import RedisRepository
from pcs.models.base import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from typing import Optional


# Mock model for testing
class MockModel(BaseModel):
    """Mock model for repository testing."""
    __tablename__ = "test_mock_models"
    
    # Additional test columns beyond the base columns (id, created_at, updated_at)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    value: Mapped[Optional[int]] = mapped_column(nullable=True)
    
    def __init__(self, id=None, name=None, value=None, **kwargs):
        super().__init__(**kwargs)
        if id is not None:
            self.id = id
        self.name = name
        self.value = value


class TestAbstractRepository:
    """Test AbstractRepository interface."""
    
    def test_abstract_repository_interface(self):
        """Test that AbstractRepository cannot be instantiated."""
        with pytest.raises(TypeError):
            AbstractRepository()
    
    def test_abstract_methods_defined(self):
        """Test that all abstract methods are defined."""
        abstract_methods = AbstractRepository.__abstractmethods__
        
        expected_methods = {
            'create', 'get_by_id', 'update', 'delete', 
            'find_by_criteria', 'count', 'exists'
        }
        
        assert abstract_methods == expected_methods


class TestBaseRepository:
    """Test BaseRepository implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_session = AsyncMock()
        # Configure common async methods
        self.mock_session.execute = AsyncMock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.rollback = AsyncMock()
        self.mock_session.close = AsyncMock()
        self.mock_session.refresh = AsyncMock()
        self.mock_session.add = Mock()  # add() is not async
        
        self.repository = BaseRepository(self.mock_session, MockModel)
    
    @pytest.mark.asyncio
    async def test_create_success(self):
        """Test successful entity creation."""
        entity = MockModel(name="test", value=42)
        
        # Mock session operations
        self.mock_session.add = Mock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.refresh = AsyncMock()
        
        result = await self.repository.create(entity)
        
        assert result == entity
        self.mock_session.add.assert_called_once_with(entity)
        self.mock_session.commit.assert_called_once()
        self.mock_session.refresh.assert_called_once_with(entity)
    
    @pytest.mark.asyncio
    async def test_create_integrity_error(self):
        """Test creation with integrity constraint violation."""
        from sqlalchemy.exc import IntegrityError
        
        entity = MockModel(name="duplicate", value=42)
        
        # Mock session to raise IntegrityError
        self.mock_session.add = Mock()
        self.mock_session.commit = AsyncMock(side_effect=IntegrityError("", "", ""))
        self.mock_session.rollback = AsyncMock()
        
        with pytest.raises(IntegrityError):
            await self.repository.create(entity)
        
        self.mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_general_error(self):
        """Test creation with general SQL error."""
        from sqlalchemy.exc import SQLAlchemyError
        
        entity = MockModel(name="test", value=42)
        
        # Mock session to raise SQLAlchemyError
        self.mock_session.add = Mock()
        self.mock_session.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        self.mock_session.rollback = AsyncMock()
        
        with pytest.raises(RepositoryError) as exc_info:
            await self.repository.create(entity)
        
        assert "Failed to create" in str(exc_info.value)
        self.mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        """Test getting entity by ID when found."""
        entity_id = uuid4()
        mock_entity = MockModel(id=entity_id, name="found")
        
        # Mock query result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_entity
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.get_by_id(entity_id)
        
        assert result == mock_entity
        self.mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """Test getting entity by ID when not found."""
        entity_id = uuid4()
        
        # Mock query result returning None
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.get_by_id(entity_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_success(self):
        """Test successful entity update."""
        entity_id = uuid4()
        updates = {"name": "updated", "value": 100}
        mock_entity = MockModel(id=entity_id, name="original")
        updated_entity = MockModel(id=entity_id, name="updated", value=100)
        
        # Mock get_by_id to return entity (called twice)
        self.repository.get_by_id = AsyncMock(side_effect=[mock_entity, updated_entity])
        
        # Mock update execution
        self.mock_session.execute = AsyncMock()
        self.mock_session.commit = AsyncMock()
        
        result = await self.repository.update(entity_id, updates)
        
        assert result == updated_entity
        assert result.name == "updated"
        assert result.value == 100
    
    @pytest.mark.asyncio
    async def test_update_not_found(self):
        """Test updating non-existent entity."""
        entity_id = uuid4()
        updates = {"name": "updated"}
        
        # Mock get_by_id to return None
        self.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await self.repository.update(entity_id, updates)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_success(self):
        """Test successful entity deletion."""
        entity_id = uuid4()
        mock_entity = MockModel(id=entity_id, name="to_delete")
        
        # Mock get_by_id to return entity
        self.repository.get_by_id = AsyncMock(return_value=mock_entity)
        
        # Mock delete execution
        mock_result = Mock()
        mock_result.rowcount = 1
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        self.mock_session.commit = AsyncMock()
        
        result = await self.repository.delete(entity_id)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        """Test deleting non-existent entity."""
        entity_id = uuid4()
        
        # Mock get_by_id to return None
        self.repository.get_by_id = AsyncMock(return_value=None)
        
        result = await self.repository.delete(entity_id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_find_by_criteria(self):
        """Test finding entities by criteria."""
        criteria = {"name": "test", "active": True}
        mock_entities = [
            MockModel(name="test", value=1),
            MockModel(name="test", value=2)
        ]
        
        # Mock query result
        mock_scalars = Mock()
        mock_scalars.all.return_value = mock_entities
        mock_result = Mock()
        mock_result.scalars.return_value = mock_scalars
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.find_by_criteria(**criteria)
        
        assert result == mock_entities
    
    @pytest.mark.asyncio
    async def test_count(self):
        """Test counting entities with criteria."""
        criteria = {"active": True}
        
        # Mock count result
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.count(**criteria)
        
        assert result == 5
    
    @pytest.mark.asyncio
    async def test_count_none_result(self):
        """Test counting with None result."""
        # Mock count result returning None
        mock_result = Mock()
        mock_result.scalar.return_value = None
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.count()
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Test entity existence check returning True."""
        entity_id = uuid4()
        
        # Mock count result
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.exists(entity_id)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Test entity existence check returning False."""
        entity_id = uuid4()
        
        # Mock count result
        mock_result = Mock()
        mock_result.scalar.return_value = 0
        self.mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await self.repository.exists(entity_id)
        
        assert result is False


class TestRedisRepository:
    """Test RedisRepository functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Redis client
        self.mock_redis_client = Mock()
        self.repository = RedisRepository(self.mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_set_with_serialization(self):
        """Test setting value with JSON serialization."""
        key = "test:key"
        value = {"name": "Alice", "score": 95}
        ttl = timedelta(hours=1)
        
        self.mock_redis_client.set = AsyncMock(return_value=True)
        
        result = await self.repository.set(key, value, ttl=ttl, serialize=True)
        
        assert result is True
        
        # Check call arguments
        call_args = self.mock_redis_client.set.call_args
        assert call_args[0][0] == key
        # Value should be JSON serialized
        import json
        assert json.loads(call_args[0][1]) == value
        assert call_args[1]['ex'] == int(ttl.total_seconds())
    
    @pytest.mark.asyncio
    async def test_set_without_serialization(self):
        """Test setting value without serialization."""
        key = "test:key"
        value = "plain string"
        
        self.mock_redis_client.set = AsyncMock(return_value=True)
        
        result = await self.repository.set(key, value, serialize=False)
        
        assert result is True
        
        # Check call arguments
        call_args = self.mock_redis_client.set.call_args
        assert call_args[0][1] == value  # Should not be serialized
    
    @pytest.mark.asyncio
    async def test_get_with_deserialization(self):
        """Test getting value with JSON deserialization."""
        key = "test:key"
        stored_value = '{"name": "Alice", "score": 95}'
        expected_value = {"name": "Alice", "score": 95}
        
        self.mock_redis_client.get = AsyncMock(return_value=stored_value)
        
        result = await self.repository.get(key, deserialize=True)
        
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_get_not_found(self):
        """Test getting non-existent key."""
        key = "nonexistent:key"
        
        self.mock_redis_client.get = AsyncMock(return_value=None)
        
        result = await self.repository.get(key)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_invalid_json(self):
        """Test getting value with invalid JSON."""
        key = "test:key"
        invalid_json = "not valid json"
        
        self.mock_redis_client.get = AsyncMock(return_value=invalid_json)
        
        result = await self.repository.get(key, deserialize=True)
        
        # Should return the raw string when JSON parsing fails
        assert result == invalid_json
    
    @pytest.mark.asyncio
    async def test_delete_single_key(self):
        """Test deleting a single key."""
        key = "test:key"
        
        self.mock_redis_client.delete = AsyncMock(return_value=1)
        
        result = await self.repository.delete(key)
        
        assert result == 1
        self.mock_redis_client.delete.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_delete_multiple_keys(self):
        """Test deleting multiple keys."""
        keys = ["key1", "key2", "key3"]
        
        self.mock_redis_client.delete = AsyncMock(return_value=3)
        
        result = await self.repository.delete(*keys)
        
        assert result == 3
        self.mock_redis_client.delete.assert_called_once_with(*keys)
    
    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Test key existence check returning True."""
        key = "existing:key"
        
        self.mock_redis_client.exists = AsyncMock(return_value=1)
        
        result = await self.repository.exists(key)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Test key existence check returning False."""
        key = "nonexistent:key"
        
        self.mock_redis_client.exists = AsyncMock(return_value=0)
        
        result = await self.repository.exists(key)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_expire_with_timedelta(self):
        """Test setting TTL with timedelta."""
        key = "test:key"
        ttl = timedelta(minutes=30)
        
        self.mock_redis_client.expire = AsyncMock(return_value=True)
        
        result = await self.repository.expire(key, ttl)
        
        assert result is True
        self.mock_redis_client.expire.assert_called_once_with(key, 1800)  # 30 minutes in seconds
    
    @pytest.mark.asyncio
    async def test_ttl(self):
        """Test getting TTL of a key."""
        key = "test:key"
        expected_ttl = 3600  # 1 hour
        
        self.mock_redis_client.ttl = AsyncMock(return_value=expected_ttl)
        
        result = await self.repository.ttl(key)
        
        assert result == expected_ttl
    
    @pytest.mark.asyncio
    async def test_hset_with_serialization(self):
        """Test setting hash fields with serialization."""
        key = "test:hash"
        mapping = {
            "name": "Alice",
            "data": {"score": 95, "level": 5},
            "count": 42
        }
        
        self.mock_redis_client.hset = AsyncMock(return_value=3)
        
        result = await self.repository.hset(key, mapping, serialize=True)
        
        assert result == 3
        
        # Check that complex data was serialized
        call_args = self.mock_redis_client.hset.call_args
        assert call_args[0][0] == key
        mapping_arg = call_args[1]['mapping']
        
        # Simple values should remain unchanged
        assert mapping_arg["name"] == "Alice"
        assert mapping_arg["count"] == 42
        
        # Complex data should be JSON serialized
        import json
        assert json.loads(mapping_arg["data"]) == {"score": 95, "level": 5}
    
    @pytest.mark.asyncio
    async def test_hget_with_deserialization(self):
        """Test getting hash field with deserialization."""
        key = "test:hash"
        field = "data"
        stored_value = '{"score": 95}'
        expected_value = {"score": 95}
        
        self.mock_redis_client.hget = AsyncMock(return_value=stored_value)
        
        result = await self.repository.hget(key, field, deserialize=True)
        
        assert result == expected_value
    
    @pytest.mark.asyncio
    async def test_hgetall(self):
        """Test getting all hash fields."""
        key = "test:hash"
        hash_data = {
            b"name": b"Alice",
            b"score": b"95",
            b"data": b'{"level": 5}'
        }
        
        self.mock_redis_client.hgetall = AsyncMock(return_value=hash_data)
        
        result = await self.repository.hgetall(key, deserialize=True)
        
        assert result["name"] == "Alice"
        assert result["score"] == "95"  # Non-JSON string
        assert result["data"] == {"level": 5}  # JSON deserialized
    
    @pytest.mark.asyncio
    async def test_sadd_with_serialization(self):
        """Test adding set members with serialization."""
        key = "test:set"
        members = ["string", 42, {"complex": "data"}]
        
        self.mock_redis_client.sadd = AsyncMock(return_value=3)
        
        result = await self.repository.sadd(key, *members, serialize=True)
        
        assert result == 3
        
        # Check call arguments
        call_args = self.mock_redis_client.sadd.call_args
        assert call_args[0][0] == key
        serialized_members = call_args[0][1:]
        
        # Check that complex data was serialized
        assert "string" in serialized_members
        assert 42 in serialized_members
        
        # Complex data should be JSON serialized
        import json
        complex_member = [m for m in serialized_members if isinstance(m, str) and m.startswith('{"')]
        assert len(complex_member) == 1
        assert json.loads(complex_member[0]) == {"complex": "data"}
    
    @pytest.mark.asyncio
    async def test_smembers_with_deserialization(self):
        """Test getting set members with deserialization."""
        key = "test:set"
        stored_members = {b"string", b"42", b'{"complex": "data"}'}
        
        self.mock_redis_client.smembers = AsyncMock(return_value=stored_members)
        
        result = await self.repository.smembers(key, deserialize=True)
        
        # Note: The set contains hashable representations of deserialized data
        # Convert set to list for checking since dictionary objects become tuples
        result_list = list(result)
        assert len(result_list) == 3
        
        # Check for simple string members
        assert "string" in result
        assert "42" in result
        
        # Check if complex data was deserialized and made hashable
        # The dictionary {"complex": "data"} becomes a tuple (('complex', 'data'),)
        complex_tuple = tuple(sorted({"complex": "data"}.items()))
        assert complex_tuple in result
    
    @pytest.mark.asyncio
    async def test_lpush_with_serialization(self):
        """Test pushing to list with serialization."""
        key = "test:list"
        values = ["item1", {"complex": "data"}, 42]
        
        self.mock_redis_client.lpush = AsyncMock(return_value=3)
        
        result = await self.repository.lpush(key, *values, serialize=True)
        
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_lrange_with_deserialization(self):
        """Test getting list range with deserialization."""
        key = "test:list"
        stored_values = [b"item1", b'{"complex": "data"}', b"42"]
        
        self.mock_redis_client.lrange = AsyncMock(return_value=stored_values)
        
        result = await self.repository.lrange(key, deserialize=True)
        
        assert result[0] == "item1"
        assert result[1] == {"complex": "data"}  # JSON deserialized
        assert result[2] == "42"  # Non-JSON string
    
    @pytest.mark.asyncio
    async def test_keys_pattern_matching(self):
        """Test getting keys with pattern matching."""
        pattern = "user:*"
        expected_keys = [b"user:1", b"user:2", b"user:3"]
        
        self.mock_redis_client.keys = AsyncMock(return_value=expected_keys)
        
        result = await self.repository.keys(pattern)
        
        expected_result = ["user:1", "user:2", "user:3"]
        assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_ping_connection(self):
        """Test Redis connection ping."""
        self.mock_redis_client.ping = AsyncMock(return_value=True)
        
        result = await self.repository.ping()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_ping_connection_failure(self):
        """Test Redis connection ping failure."""
        from redis.asyncio import RedisError
        
        self.mock_redis_client.ping = AsyncMock(side_effect=RedisError("Connection failed"))
        
        result = await self.repository.ping()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and conversion to RepositoryError."""
        from redis.asyncio import RedisError
        
        key = "test:key"
        self.mock_redis_client.get = AsyncMock(side_effect=RedisError("Redis error"))
        
        with pytest.raises(RepositoryError) as exc_info:
            await self.repository.get(key)
        
        assert "Failed to get Redis key" in str(exc_info.value)
        assert "test:key" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
