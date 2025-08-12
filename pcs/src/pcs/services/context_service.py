"""
Filepath: pcs/src/pcs/services/context_service.py
Purpose: Context management service with Redis caching, TTL management, and intelligent context merging
Related Components: Redis repository, context models, template engine, rule engine
Tags: context-management, redis-caching, ttl-management, context-merging, performance
"""

import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from uuid import UUID, uuid4

from ..core.exceptions import PCSError
from ..repositories.redis_repo import RedisRepository
from ..repositories.base import RepositoryError


class ContextError(PCSError):
    """Custom exception for context-related errors."""
    pass


class ContextType(Enum):
    """Types of contexts in the system."""
    USER = "user"
    SESSION = "session" 
    APPLICATION = "application"
    CONVERSATION = "conversation"
    TEMPLATE = "template"
    GLOBAL = "global"
    TEMPORARY = "temporary"


class ContextScope(Enum):
    """Context scope levels."""
    PRIVATE = "private"      # User-specific
    SHARED = "shared"        # Shared across users
    PUBLIC = "public"        # Publicly accessible
    SYSTEM = "system"        # System-level context


class MergeStrategy(Enum):
    """Strategies for merging contexts."""
    REPLACE = "replace"      # Replace existing values
    MERGE_DEEP = "merge_deep"  # Deep merge objects
    MERGE_SHALLOW = "merge_shallow"  # Shallow merge
    APPEND = "append"        # Append to arrays
    PRESERVE = "preserve"    # Keep existing values


@dataclass
class ContextMetadata:
    """Metadata for context entries."""
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: Set[str] = None
    priority: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, set):
                data[key] = list(value)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextMetadata':
        """Create from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'updated_at', 'expires_at', 'last_accessed']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        
        if 'tags' in data:
            data['tags'] = set(data['tags'])
        
        return cls(**data)


@dataclass
class Context:
    """Represents a context with data and metadata."""
    context_id: str
    context_type: ContextType
    scope: ContextScope
    data: Dict[str, Any]
    metadata: ContextMetadata
    parent_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.context_type, str):
            self.context_type = ContextType(self.context_type)
        if isinstance(self.scope, str):
            self.scope = ContextScope(self.scope)
    
    def is_expired(self) -> bool:
        """Check if context has expired."""
        if self.metadata.expires_at:
            return datetime.now(datetime.UTC) > self.metadata.expires_at
        return False
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.metadata.access_count += 1
        self.metadata.last_accessed = datetime.now(datetime.UTC)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'context_id': self.context_id,
            'context_type': self.context_type.value,
            'scope': self.scope.value,
            'data': self.data,
            'metadata': self.metadata.to_dict(),
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Create from dictionary."""
        metadata_data = data.get('metadata', {})
        metadata = ContextMetadata.from_dict(metadata_data)
        
        return cls(
            context_id=data['context_id'],
            context_type=ContextType(data['context_type']),
            scope=ContextScope(data['scope']),
            data=data['data'],
            metadata=metadata,
            parent_id=data.get('parent_id')
        )


class ContextCache:
    """
    Context caching layer with intelligent caching strategies.
    
    Provides efficient caching with TTL management and memory optimization.
    """
    
    def __init__(self, redis_repo: RedisRepository, cache_prefix: str = "pcs:context"):
        """
        Initialize context cache.
        
        Args:
            redis_repo: Redis repository for caching
            cache_prefix: Prefix for cache keys
        """
        self.redis = redis_repo
        self.cache_prefix = cache_prefix
        
        # Cache key patterns
        self.context_key_pattern = f"{cache_prefix}:ctx"
        self.metadata_key_pattern = f"{cache_prefix}:meta"
        self.index_key_pattern = f"{cache_prefix}:idx"
    
    def _get_context_key(self, context_id: str) -> str:
        """Get Redis key for context data."""
        return f"{self.context_key_pattern}:{context_id}"
    
    def _get_metadata_key(self, context_id: str) -> str:
        """Get Redis key for context metadata."""
        return f"{self.metadata_key_pattern}:{context_id}"
    
    def _get_index_key(self, context_type: ContextType, scope: ContextScope) -> str:
        """Get Redis key for context index."""
        return f"{self.index_key_pattern}:{context_type.value}:{scope.value}"
    
    async def get_context(self, context_id: str) -> Optional[Context]:
        """
        Get context from cache.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Context object or None if not found
        """
        try:
            # Get context data and metadata in parallel
            context_data = await self.redis.get(self._get_context_key(context_id))
            metadata_data = await self.redis.get(self._get_metadata_key(context_id))
            
            if not context_data or not metadata_data:
                return None
            
            # Reconstruct context
            context_dict = {
                'context_id': context_id,
                'data': context_data,
                'metadata': metadata_data
            }
            
            context = Context.from_dict(context_dict)
            
            # Check if expired
            if context.is_expired():
                await self.delete_context(context_id)
                return None
            
            # Update access tracking
            context.update_access()
            await self._update_metadata(context)
            
            return context
            
        except Exception as e:
            raise ContextError(f"Failed to get context from cache: {str(e)}") from e
    
    async def set_context(self, context: Context, ttl: Optional[timedelta] = None) -> bool:
        """
        Set context in cache.
        
        Args:
            context: Context to cache
            ttl: Time to live
            
        Returns:
            True if successful
        """
        try:
            # Calculate TTL
            cache_ttl = None
            if ttl:
                cache_ttl = ttl
            elif context.metadata.expires_at:
                cache_ttl = context.metadata.expires_at - datetime.now(datetime.UTC)
                if cache_ttl.total_seconds() <= 0:
                    return False  # Already expired
            
            # Store context data and metadata
            context_key = self._get_context_key(context.context_id)
            metadata_key = self._get_metadata_key(context.context_id)
            
            await self.redis.set(context_key, context.data, ttl=cache_ttl)
            await self.redis.set(metadata_key, context.metadata.to_dict(), ttl=cache_ttl)
            
            # Update index
            index_key = self._get_index_key(context.context_type, context.scope)
            await self.redis.sadd(index_key, context.context_id)
            
            return True
            
        except Exception as e:
            raise ContextError(f"Failed to set context in cache: {str(e)}") from e
    
    async def delete_context(self, context_id: str) -> bool:
        """
        Delete context from cache.
        
        Args:
            context_id: Context identifier
            
        Returns:
            True if context was deleted
        """
        try:
            # Get context first to update indexes
            context = await self.get_context(context_id)
            
            # Delete context data and metadata
            context_key = self._get_context_key(context_id)
            metadata_key = self._get_metadata_key(context_id)
            
            deleted_count = await self.redis.delete(context_key, metadata_key)
            
            # Remove from index if context existed
            if context:
                index_key = self._get_index_key(context.context_type, context.scope)
                await self.redis.srem(index_key, context_id)
            
            return deleted_count > 0
            
        except Exception as e:
            raise ContextError(f"Failed to delete context from cache: {str(e)}") from e
    
    async def get_contexts_by_type(
        self, 
        context_type: ContextType, 
        scope: ContextScope
    ) -> List[Context]:
        """
        Get all contexts of a specific type and scope.
        
        Args:
            context_type: Type of contexts to retrieve
            scope: Scope of contexts to retrieve
            
        Returns:
            List of contexts
        """
        try:
            index_key = self._get_index_key(context_type, scope)
            context_ids = await self.redis.smembers(index_key)
            
            contexts = []
            for context_id in context_ids:
                context = await self.get_context(context_id)
                if context:  # Skip expired contexts
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            raise ContextError(f"Failed to get contexts by type: {str(e)}") from e
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired contexts.
        
        Returns:
            Number of contexts cleaned up
        """
        try:
            # Get all context keys
            pattern = f"{self.context_key_pattern}:*"
            context_keys = await self.redis.keys(pattern)
            
            cleaned_count = 0
            for key in context_keys:
                # Extract context ID from key
                context_id = key.split(':')[-1]
                
                # Check if context exists and is expired
                context = await self.get_context(context_id)
                if context and context.is_expired():
                    await self.delete_context(context_id)
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            raise ContextError(f"Failed to cleanup expired contexts: {str(e)}") from e
    
    async def _update_metadata(self, context: Context) -> None:
        """Update context metadata in cache."""
        try:
            metadata_key = self._get_metadata_key(context.context_id)
            await self.redis.set(metadata_key, context.metadata.to_dict())
        except Exception:
            # Non-critical operation, ignore errors
            pass


class ContextMerger:
    """
    Context merger for combining multiple contexts intelligently.
    
    Provides various merge strategies for different use cases.
    """
    
    def __init__(self):
        """Initialize context merger."""
        self.merge_handlers = {
            MergeStrategy.REPLACE: self._merge_replace,
            MergeStrategy.MERGE_DEEP: self._merge_deep,
            MergeStrategy.MERGE_SHALLOW: self._merge_shallow,
            MergeStrategy.APPEND: self._merge_append,
            MergeStrategy.PRESERVE: self._merge_preserve,
        }
    
    def merge_contexts(
        self, 
        base_context: Context, 
        update_context: Context,
        strategy: MergeStrategy = MergeStrategy.MERGE_DEEP
    ) -> Context:
        """
        Merge two contexts using the specified strategy.
        
        Args:
            base_context: Base context to merge into
            update_context: Context with updates
            strategy: Merge strategy to use
            
        Returns:
            Merged context
        """
        try:
            merge_func = self.merge_handlers.get(strategy)
            if not merge_func:
                raise ContextError(f"Unsupported merge strategy: {strategy}")
            
            # Create new merged context
            merged_data = merge_func(base_context.data, update_context.data)
            
            # Update metadata
            merged_metadata = ContextMetadata(
                created_at=base_context.metadata.created_at,
                updated_at=datetime.now(datetime.UTC),
                expires_at=update_context.metadata.expires_at or base_context.metadata.expires_at,
                access_count=base_context.metadata.access_count,
                last_accessed=base_context.metadata.last_accessed,
                tags=base_context.metadata.tags.union(update_context.metadata.tags),
                priority=max(base_context.metadata.priority, update_context.metadata.priority)
            )
            
            return Context(
                context_id=base_context.context_id,
                context_type=base_context.context_type,
                scope=base_context.scope,
                data=merged_data,
                metadata=merged_metadata,
                parent_id=base_context.parent_id
            )
            
        except Exception as e:
            raise ContextError(f"Failed to merge contexts: {str(e)}") from e
    
    def _merge_replace(self, base_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replace strategy: update data completely replaces base data."""
        return update_data.copy()
    
    def _merge_deep(self, base_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge strategy: recursively merge nested dictionaries."""
        result = base_data.copy()
        
        for key, value in update_data.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_deep(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _merge_shallow(self, base_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Shallow merge strategy: only merge top-level keys."""
        result = base_data.copy()
        result.update(update_data)
        return result
    
    def _merge_append(self, base_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Append strategy: append to lists, merge other types."""
        result = base_data.copy()
        
        for key, value in update_data.items():
            if key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        
        return result
    
    def _merge_preserve(self, base_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve strategy: only add new keys, preserve existing values."""
        result = base_data.copy()
        
        for key, value in update_data.items():
            if key not in result:
                result[key] = value
        
        return result


class ContextValidator:
    """
    Context validator for ensuring context integrity and security.
    
    Validates context data before storage and retrieval.
    """
    
    def __init__(self):
        """Initialize context validator."""
        self.max_data_size = 1024 * 1024  # 1MB
        self.max_context_depth = 10
        self.restricted_keys = {'__proto__', 'constructor', 'prototype'}
    
    def validate_context(self, context: Context) -> List[str]:
        """
        Validate context for security and integrity.
        
        Args:
            context: Context to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate basic fields
        if not context.context_id:
            errors.append("Context ID is required")
        
        if not context.data:
            errors.append("Context data cannot be empty")
        
        # Validate data size
        try:
            data_size = len(json.dumps(context.data))
            if data_size > self.max_data_size:
                errors.append(f"Context data too large: {data_size} bytes (max: {self.max_data_size})")
        except Exception:
            errors.append("Context data is not JSON serializable")
        
        # Validate data structure
        errors.extend(self._validate_data_structure(context.data))
        
        # Validate expiration
        if context.metadata.expires_at and context.metadata.expires_at <= datetime.now(datetime.UTC):
            errors.append("Context has already expired")
        
        return errors
    
    def _validate_data_structure(self, data: Any, depth: int = 0) -> List[str]:
        """Validate data structure recursively."""
        errors = []
        
        if depth > self.max_context_depth:
            errors.append(f"Context data too deeply nested (max depth: {self.max_context_depth})")
            return errors
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Check for restricted keys
                if key in self.restricted_keys:
                    errors.append(f"Restricted key not allowed: {key}")
                
                # Validate recursively
                errors.extend(self._validate_data_structure(value, depth + 1))
        
        elif isinstance(data, list):
            for item in data:
                errors.extend(self._validate_data_structure(item, depth + 1))
        
        return errors


class ContextManager:
    """
    Main context manager for the PCS system.
    
    Provides high-level context management operations with caching,
    merging, and validation.
    """
    
    def __init__(self, redis_repo: RedisRepository):
        """
        Initialize context manager.
        
        Args:
            redis_repo: Redis repository for caching
        """
        self.redis_repo = redis_repo
        self.cache = ContextCache(redis_repo)
        self.merger = ContextMerger()
        self.validator = ContextValidator()
    
    async def create_context(
        self,
        context_type: ContextType,
        scope: ContextScope,
        data: Dict[str, Any],
        ttl: Optional[timedelta] = None,
        tags: Optional[Set[str]] = None,
        parent_id: Optional[str] = None
    ) -> Context:
        """
        Create a new context.
        
        Args:
            context_type: Type of context
            scope: Context scope
            data: Context data
            ttl: Time to live
            tags: Context tags
            parent_id: Parent context ID
            
        Returns:
            Created context
            
        Raises:
            ContextError: If context creation fails
        """
        try:
            # Generate context ID
            context_id = str(uuid4())
            
            # Create metadata
            now = datetime.now(datetime.UTC)
            expires_at = now + ttl if ttl else None
            
            metadata = ContextMetadata(
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                tags=tags or set()
            )
            
            # Create context
            context = Context(
                context_id=context_id,
                context_type=context_type,
                scope=scope,
                data=data,
                metadata=metadata,
                parent_id=parent_id
            )
            
            # Validate context
            validation_errors = self.validator.validate_context(context)
            if validation_errors:
                raise ContextError(f"Context validation failed: {validation_errors}")
            
            # Store in cache
            await self.cache.set_context(context, ttl)
            
            return context
            
        except Exception as e:
            if isinstance(e, ContextError):
                raise
            raise ContextError(f"Failed to create context: {str(e)}") from e
    
    async def get_context(self, context_id: str) -> Optional[Context]:
        """
        Get context by ID.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Context or None if not found
        """
        try:
            return await self.cache.get_context(context_id)
        except Exception as e:
            raise ContextError(f"Failed to get context: {str(e)}") from e
    
    async def update_context(
        self,
        context_id: str,
        updates: Dict[str, Any],
        merge_strategy: MergeStrategy = MergeStrategy.MERGE_DEEP
    ) -> Optional[Context]:
        """
        Update an existing context.
        
        Args:
            context_id: Context identifier
            updates: Data updates
            merge_strategy: How to merge updates
            
        Returns:
            Updated context or None if not found
        """
        try:
            # Get existing context
            existing_context = await self.get_context(context_id)
            if not existing_context:
                return None
            
            # Create update context
            update_metadata = ContextMetadata(
                created_at=datetime.now(datetime.UTC),
                updated_at=datetime.now(datetime.UTC)
            )
            
            update_context = Context(
                context_id=context_id,
                context_type=existing_context.context_type,
                scope=existing_context.scope,
                data=updates,
                metadata=update_metadata
            )
            
            # Merge contexts
            merged_context = self.merger.merge_contexts(
                existing_context, 
                update_context, 
                merge_strategy
            )
            
            # Validate merged context
            validation_errors = self.validator.validate_context(merged_context)
            if validation_errors:
                raise ContextError(f"Context validation failed: {validation_errors}")
            
            # Store updated context
            await self.cache.set_context(merged_context)
            
            return merged_context
            
        except Exception as e:
            if isinstance(e, ContextError):
                raise
            raise ContextError(f"Failed to update context: {str(e)}") from e
    
    async def delete_context(self, context_id: str) -> bool:
        """
        Delete a context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            True if context was deleted
        """
        try:
            return await self.cache.delete_context(context_id)
        except Exception as e:
            raise ContextError(f"Failed to delete context: {str(e)}") from e
    
    async def merge_contexts(
        self,
        context_ids: List[str],
        merge_strategy: MergeStrategy = MergeStrategy.MERGE_DEEP
    ) -> Optional[Context]:
        """
        Merge multiple contexts into one.
        
        Args:
            context_ids: List of context IDs to merge
            merge_strategy: How to merge contexts
            
        Returns:
            Merged context or None if no contexts found
        """
        try:
            if not context_ids:
                return None
            
            # Get all contexts
            contexts = []
            for context_id in context_ids:
                context = await self.get_context(context_id)
                if context:
                    contexts.append(context)
            
            if not contexts:
                return None
            
            # Start with first context
            merged_context = contexts[0]
            
            # Merge with remaining contexts
            for context in contexts[1:]:
                merged_context = self.merger.merge_contexts(
                    merged_context, 
                    context, 
                    merge_strategy
                )
            
            return merged_context
            
        except Exception as e:
            raise ContextError(f"Failed to merge contexts: {str(e)}") from e
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired contexts.
        
        Returns:
            Number of contexts cleaned up
        """
        try:
            return await self.cache.cleanup_expired()
        except Exception as e:
            raise ContextError(f"Failed to cleanup expired contexts: {str(e)}") from e
    
    async def get_contexts_by_type(
        self,
        context_type: ContextType,
        scope: ContextScope = ContextScope.PRIVATE
    ) -> List[Context]:
        """
        Get all contexts of a specific type and scope.
        
        Args:
            context_type: Type of contexts to retrieve
            scope: Scope of contexts to retrieve
            
        Returns:
            List of contexts
        """
        try:
            return await self.cache.get_contexts_by_type(context_type, scope)
        except Exception as e:
            raise ContextError(f"Failed to get contexts by type: {str(e)}") from e
