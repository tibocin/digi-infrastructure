"""
Filepath: pcs/src/pcs/repositories/base.py
Purpose: Abstract repository pattern implementation with async support for data access layer
Related Components: SQLAlchemy models, database sessions, service layer
Tags: repository-pattern, async, sqlalchemy, data-access
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..models.base import BaseModel

# Generic type for model classes
ModelType = TypeVar("ModelType", bound=BaseModel)


class AbstractRepository(ABC, Generic[ModelType]):
    """
    Abstract repository interface defining standard CRUD operations.
    
    This abstract class ensures consistent interface across all repository implementations
    while allowing for database-specific optimizations in concrete classes.
    """

    @abstractmethod
    async def create(self, entity: ModelType) -> ModelType:
        """
        Create a new entity in the data store.
        
        Args:
            entity: The entity instance to create
            
        Returns:
            The created entity with any auto-generated fields populated
            
        Raises:
            IntegrityError: If entity violates unique constraints
            RepositoryError: For other database-related errors
        """
        pass

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """
        Retrieve an entity by its unique identifier.
        
        Args:
            id: The unique identifier of the entity
            
        Returns:
            The entity if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, id: UUID, updates: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update an existing entity with new values.
        
        Args:
            id: The unique identifier of the entity to update
            updates: Dictionary of field names and new values
            
        Returns:
            The updated entity if found, None otherwise
            
        Raises:
            IntegrityError: If updates violate constraints
            RepositoryError: For other database-related errors
        """
        pass

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """
        Delete an entity by its unique identifier.
        
        Args:
            id: The unique identifier of the entity to delete
            
        Returns:
            True if entity was deleted, False if not found
        """
        pass

    @abstractmethod
    async def find_by_criteria(self, **criteria) -> List[ModelType]:
        """
        Find entities matching the given criteria.
        
        Args:
            **criteria: Field names and values to match
            
        Returns:
            List of matching entities (may be empty)
        """
        pass

    @abstractmethod
    async def count(self, **criteria) -> int:
        """
        Count entities matching the given criteria.
        
        Args:
            **criteria: Field names and values to match
            
        Returns:
            Number of matching entities
        """
        pass

    @abstractmethod
    async def exists(self, id: UUID) -> bool:
        """
        Check if an entity exists by its unique identifier.
        
        Args:
            id: The unique identifier to check
            
        Returns:
            True if entity exists, False otherwise
        """
        pass


class BaseRepository(AbstractRepository[ModelType]):
    """
    Concrete base repository implementation for SQLAlchemy models.
    
    Provides standard CRUD operations using SQLAlchemy async patterns.
    Can be extended for specific models that need custom query logic.
    """

    def __init__(self, session: AsyncSession, model_class: Type[ModelType]):
        """
        Initialize repository with database session and model class.
        
        Args:
            session: AsyncSession for database operations
            model_class: The SQLAlchemy model class this repository manages
        """
        self.session = session
        self.model_class = model_class

    async def create(self, entity: ModelType) -> ModelType:
        """Create a new entity in the database."""
        try:
            self.session.add(entity)
            await self.session.commit()
            await self.session.refresh(entity)
            return entity
        except IntegrityError as e:
            await self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to create {self.model_class.__name__}: {str(e)}") from e

    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """Retrieve an entity by its ID."""
        try:
            result = await self.session.execute(
                select(self.model_class).where(self.model_class.id == id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to get {self.model_class.__name__} by ID: {str(e)}") from e

    async def update(self, id: UUID, updates: Dict[str, Any]) -> Optional[ModelType]:
        """Update an existing entity."""
        try:
            # First check if entity exists
            entity = await self.get_by_id(id)
            if not entity:
                return None

            # Apply updates
            stmt = (
                update(self.model_class)
                .where(self.model_class.id == id)
                .values(**updates)
            )
            await self.session.execute(stmt)
            await self.session.commit()
            
            # Return updated entity
            return await self.get_by_id(id)
        except IntegrityError as e:
            await self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to update {self.model_class.__name__}: {str(e)}") from e

    async def delete(self, id: UUID) -> bool:
        """Delete an entity by ID."""
        try:
            # Check if entity exists first
            entity = await self.get_by_id(id)
            if not entity:
                return False

            stmt = delete(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.rowcount > 0
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to delete {self.model_class.__name__}: {str(e)}") from e

    async def find_by_criteria(self, **criteria) -> List[ModelType]:
        """Find entities matching criteria."""
        try:
            stmt = select(self.model_class)
            
            # Apply filters based on criteria
            for field, value in criteria.items():
                if hasattr(self.model_class, field):
                    stmt = stmt.where(getattr(self.model_class, field) == value)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to find {self.model_class.__name__} by criteria: {str(e)}") from e

    async def count(self, **criteria) -> int:
        """Count entities matching criteria."""
        try:
            from sqlalchemy import func
            
            stmt = select(func.count(self.model_class.id))
            
            # Apply filters based on criteria
            for field, value in criteria.items():
                if hasattr(self.model_class, field):
                    stmt = stmt.where(getattr(self.model_class, field) == value)
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to count {self.model_class.__name__}: {str(e)}") from e

    async def exists(self, id: UUID) -> bool:
        """Check if entity exists by ID."""
        try:
            from sqlalchemy import func
            
            stmt = select(func.count(self.model_class.id)).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            count = result.scalar() or 0
            return count > 0
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to check existence of {self.model_class.__name__}: {str(e)}") from e


class RepositoryError(Exception):
    """Custom exception for repository-level errors."""
    pass
