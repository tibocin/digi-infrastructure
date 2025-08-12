"""
Filepath: pcs/src/pcs/repositories/redis_repo.py
Purpose: Redis repository implementation for caching and context management operations
Related Components: Redis client, context caching, session management
Tags: redis, caching, context, async, ttl
"""

import json
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from uuid import UUID

import redis.asyncio as redis
from redis.asyncio import Redis
from structlog import get_logger

from .base import RepositoryError


class RedisRepository:
    """
    Redis repository for cache operations and data storage.
    
    Provides Redis operations with automatic JSON serialization/deserialization
    for complex data types while preserving simple strings and numbers.
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize Redis repository.
        
        Args:
            redis_client: Redis async client instance
        """
        self.redis = redis_client
        self.logger = get_logger(__name__)
    
    def _safe_deserialize(self, value_str: str) -> Any:
        """
        Safely deserialize a string value.
        
        Only deserializes proper JSON objects and arrays, keeps simple 
        strings and numbers as strings to preserve type expectations.
        
        Args:
            value_str: String value to potentially deserialize
            
        Returns:
            Deserialized object for JSON objects/arrays, original string otherwise
        """
        # Only attempt JSON deserialization for values that look like JSON objects or arrays
        stripped = value_str.strip()
        if stripped.startswith(('{', '[')):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                return value_str
        return value_str
    
    def _make_hashable(self, obj: Any) -> Any:
        """
        Convert an object to a hashable form for use in sets.
        
        Args:
            obj: Object to make hashable
            
        Returns:
            Hashable representation of the object
        """
        if isinstance(obj, dict):
            # Convert dict to a hashable tuple of sorted items
            return tuple(sorted(obj.items()))
        elif isinstance(obj, list):
            # Convert list to tuple
            return tuple(obj)
        return obj

    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[Union[int, timedelta]] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair with optional TTL.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live (seconds or timedelta)
            serialize: Whether to JSON serialize the value
            
        Returns:
            True if successful
            
        Raises:
            RepositoryError: If Redis operation fails
        """
        try:
            # Serialize value if needed
            if serialize and not isinstance(value, (str, bytes, int, float)):
                value = json.dumps(value, default=str)
            
            # Convert timedelta to seconds
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            
            result = await self.redis.set(key, value, ex=ttl)
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to set Redis key {key}: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise RepositoryError(f"Failed to serialize value for key {key}: {str(e)}") from e

    async def get(self, key: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: Redis key
            deserialize: Whether to JSON deserialize the value
            
        Returns:
            Value if found, None otherwise
        """
        try:
            value = await self.redis.get(key)
            if value is None:
                return None
            
            # Deserialize if needed
            if deserialize and isinstance(value, (str, bytes)):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Return as string if not valid JSON
                    return value.decode() if isinstance(value, bytes) else value
            
            return value.decode() if isinstance(value, bytes) else value
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis key {key}: {str(e)}") from e

    async def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if not keys:
                return 0
            return await self.redis.delete(*keys)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to delete Redis keys: {str(e)}") from e

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Redis key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            return bool(await self.redis.exists(key))
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to check existence of Redis key {key}: {str(e)}") from e

    async def expire(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """
        Set TTL on an existing key.
        
        Args:
            key: Redis key
            ttl: Time to live (seconds or timedelta)
            
        Returns:
            True if TTL was set, False if key doesn't exist
        """
        try:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            return bool(await self.redis.expire(key, ttl))
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to set TTL on Redis key {key}: {str(e)}") from e

    async def ttl(self, key: str) -> int:
        """
        Get TTL of a key.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            return await self.redis.ttl(key)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get TTL of Redis key {key}: {str(e)}") from e

    # Hash operations
    async def hset(self, key: str, mapping: Dict[str, Any], serialize: bool = True) -> int:
        """
        Set hash field values.
        
        Args:
            key: Redis hash key
            mapping: Dictionary of field-value pairs
            serialize: Whether to JSON serialize values
            
        Returns:
            Number of fields added
        """
        try:
            if serialize:
                serialized_mapping = {}
                for field, value in mapping.items():
                    if isinstance(value, (str, int, float)):
                        serialized_mapping[field] = value
                    else:
                        serialized_mapping[field] = json.dumps(value, default=str)
                mapping = serialized_mapping
            
            return await self.redis.hset(key, mapping=mapping)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to set Redis hash {key}: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise RepositoryError(f"Failed to serialize hash values for {key}: {str(e)}") from e

    async def hget(self, key: str, field: str, deserialize: bool = True) -> Optional[Any]:
        """
        Get hash field value.
        
        Args:
            key: Redis hash key
            field: Hash field name
            deserialize: Whether to JSON deserialize the value
            
        Returns:
            Field value if found, None otherwise
        """
        try:
            value = await self.redis.hget(key, field)
            if value is None:
                return None
            
            if deserialize and isinstance(value, (str, bytes)):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode() if isinstance(value, bytes) else value
            
            return value.decode() if isinstance(value, bytes) else value
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis hash field {key}.{field}: {str(e)}") from e

    async def hgetall(self, key: str, deserialize: bool = True) -> Dict[str, Any]:
        """
        Get all hash fields and values.
        
        Args:
            key: Redis hash key
            deserialize: Whether to JSON deserialize values
            
        Returns:
            Dictionary of field-value pairs
        """
        try:
            hash_data = await self.redis.hgetall(key)
            
            if not deserialize:
                return {k.decode() if isinstance(k, bytes) else k: 
                       v.decode() if isinstance(v, bytes) else v 
                       for k, v in hash_data.items()}
            
            result = {}
            for field, value in hash_data.items():
                field_str = field.decode() if isinstance(field, bytes) else field
                
                if isinstance(value, (str, bytes)):
                    value_str = value.decode() if isinstance(value, bytes) else value
                    result[field_str] = self._safe_deserialize(value_str)
                else:
                    result[field_str] = value
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get all Redis hash fields for {key}: {str(e)}") from e

    async def hdel(self, key: str, *fields: str) -> int:
        """
        Delete hash fields.
        
        Args:
            key: Redis hash key
            *fields: Field names to delete
            
        Returns:
            Number of fields deleted
        """
        try:
            if not fields:
                return 0
            return await self.redis.hdel(key, *fields)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to delete Redis hash fields from {key}: {str(e)}") from e

    # Set operations
    async def sadd(self, key: str, *members: Any, serialize: bool = True) -> int:
        """
        Add members to a set.
        
        Args:
            key: Redis set key
            *members: Members to add
            serialize: Whether to JSON serialize members
            
        Returns:
            Number of members added
        """
        try:
            if serialize:
                serialized_members = []
                for member in members:
                    if isinstance(member, (str, int, float)):
                        serialized_members.append(member)
                    else:
                        serialized_members.append(json.dumps(member, default=str))
                members = serialized_members
            
            return await self.redis.sadd(key, *members)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to add to Redis set {key}: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise RepositoryError(f"Failed to serialize set members for {key}: {str(e)}") from e

    async def smembers(self, key: str, deserialize: bool = True) -> set:
        """
        Get all set members.
        
        Args:
            key: Redis set key
            deserialize: Whether to JSON deserialize members
            
        Returns:
            Set of members
        """
        try:
            members = await self.redis.smembers(key)
            
            if not deserialize:
                return {m.decode() if isinstance(m, bytes) else m for m in members}
            
            result = set()
            for member in members:
                if isinstance(member, (str, bytes)):
                    member_str = member.decode() if isinstance(member, bytes) else member
                    deserialized = self._safe_deserialize(member_str)
                    # Make hashable for sets
                    hashable_value = self._make_hashable(deserialized)
                    result.add(hashable_value)
                else:
                    result.add(member)
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis set members for {key}: {str(e)}") from e

    async def srem(self, key: str, *members: Any, serialize: bool = True) -> int:
        """
        Remove members from a set.
        
        Args:
            key: Redis set key
            *members: Members to remove
            serialize: Whether to JSON serialize members
            
        Returns:
            Number of members removed
        """
        try:
            if serialize:
                serialized_members = []
                for member in members:
                    if isinstance(member, (str, int, float)):
                        serialized_members.append(member)
                    else:
                        serialized_members.append(json.dumps(member, default=str))
                members = serialized_members
            
            return await self.redis.srem(key, *members)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to remove from Redis set {key}: {str(e)}") from e

    # List operations
    async def lpush(self, key: str, *values: Any, serialize: bool = True) -> int:
        """
        Push values to the left (beginning) of a list.
        
        Args:
            key: Redis list key
            *values: Values to push
            serialize: Whether to JSON serialize values
            
        Returns:
            New list length
        """
        try:
            if serialize:
                serialized_values = []
                for value in values:
                    if isinstance(value, (str, int, float)):
                        serialized_values.append(value)
                    else:
                        serialized_values.append(json.dumps(value, default=str))
                values = serialized_values
            
            return await self.redis.lpush(key, *values)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to lpush to Redis list {key}: {str(e)}") from e

    async def rpush(self, key: str, *values: Any, serialize: bool = True) -> int:
        """
        Push values to the right (end) of a list.
        
        Args:
            key: Redis list key
            *values: Values to push
            serialize: Whether to JSON serialize values
            
        Returns:
            New list length
        """
        try:
            if serialize:
                serialized_values = []
                for value in values:
                    if isinstance(value, (str, int, float)):
                        serialized_values.append(value)
                    else:
                        serialized_values.append(json.dumps(value, default=str))
                values = serialized_values
            
            return await self.redis.rpush(key, *values)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to rpush to Redis list {key}: {str(e)}") from e

    async def lrange(self, key: str, start: int = 0, end: int = -1, deserialize: bool = True) -> List[Any]:
        """
        Get list elements by range.
        
        Args:
            key: Redis list key
            start: Start index
            end: End index (-1 for all)
            deserialize: Whether to JSON deserialize values
            
        Returns:
            List of values
        """
        try:
            values = await self.redis.lrange(key, start, end)
            
            if not deserialize:
                return [v.decode() if isinstance(v, bytes) else v for v in values]
            
            result = []
            for value in values:
                if isinstance(value, (str, bytes)):
                    value_str = value.decode() if isinstance(value, bytes) else value
                    result.append(self._safe_deserialize(value_str))
                else:
                    result.append(value)
            
            return result
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis list range for {key}: {str(e)}") from e

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """
        Trim list to specified range.
        
        Args:
            key: Redis list key
            start: Start index
            end: End index
            
        Returns:
            True if successful
        """
        try:
            result = await self.redis.ltrim(key, start, end)
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to trim Redis list {key}: {str(e)}") from e

    # Utility methods
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            List of matching keys
        """
        try:
            keys = await self.redis.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to get Redis keys with pattern {pattern}: {str(e)}") from e

    async def flushdb(self) -> bool:
        """
        Clear all keys in the current database.
        
        Returns:
            True if successful
        """
        try:
            result = await self.redis.flushdb()
            return bool(result)
        except redis.RedisError as e:
            raise RepositoryError(f"Failed to flush Redis database: {str(e)}") from e

    async def ping(self) -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connection is alive
        """
        try:
            result = await self.redis.ping()
            return bool(result)
        except redis.RedisError:
            return False
