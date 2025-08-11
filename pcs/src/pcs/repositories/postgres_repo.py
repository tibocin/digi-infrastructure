"""
Filepath: pcs/src/pcs/repositories/postgres_repo.py
Purpose: PostgreSQL-specific repository implementation with optimized queries and relationship handling
Related Components: SQLAlchemy models, database sessions, async PostgreSQL operations
Tags: postgresql, repository, async, relationships, optimization
"""

from typing import Any, Dict, List, Optional, Type
from uuid import UUID

from sqlalchemy import and_, or_, select, update, delete, func
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .base import BaseRepository, RepositoryError, ModelType
from ..models.base import BaseModel


class PostgreSQLRepository(BaseRepository[ModelType]):
    """
    PostgreSQL-specific repository implementation with optimizations.
    
    Extends the base repository with PostgreSQL-specific features like:
    - Relationship loading strategies
    - Complex query optimization
    - Bulk operations
    - JSON field queries
    """

    def __init__(self, session: AsyncSession, model_class: Type[ModelType]):
        """Initialize PostgreSQL repository with session and model class."""
        super().__init__(session, model_class)

    async def get_by_id_with_relationships(
        self, 
        id: UUID, 
        *relationships: str,
        load_strategy: str = "selectin"
    ) -> Optional[ModelType]:
        """
        Retrieve entity by ID with eagerly loaded relationships.
        
        Args:
            id: Entity identifier
            *relationships: Relationship attribute names to load
            load_strategy: Loading strategy ('selectin' or 'joined')
            
        Returns:
            Entity with loaded relationships or None
        """
        try:
            stmt = select(self.model_class).where(self.model_class.id == id)
            
            # Apply relationship loading based on strategy
            for relationship in relationships:
                if hasattr(self.model_class, relationship):
                    if load_strategy == "selectin":
                        stmt = stmt.options(selectinload(getattr(self.model_class, relationship)))
                    elif load_strategy == "joined":
                        stmt = stmt.options(joinedload(getattr(self.model_class, relationship)))
            
            result = await self.session.execute(stmt)
            return result.unique().scalar_one_or_none()
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to get {self.model_class.__name__} with relationships: {str(e)}") from e

    async def find_by_text_search(self, search_term: str, *text_fields: str) -> List[ModelType]:
        """
        Perform text search across specified fields using PostgreSQL text search.
        
        Args:
            search_term: Text to search for
            *text_fields: Field names to search in
            
        Returns:
            List of entities matching the search term
        """
        try:
            if not text_fields:
                raise ValueError("At least one text field must be specified")
            
            stmt = select(self.model_class)
            
            # Build OR conditions for text search
            search_conditions = []
            for field in text_fields:
                if hasattr(self.model_class, field):
                    field_attr = getattr(self.model_class, field)
                    # Use ILIKE for case-insensitive partial matching
                    search_conditions.append(field_attr.ilike(f"%{search_term}%"))
            
            if search_conditions:
                stmt = stmt.where(or_(*search_conditions))
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to perform text search on {self.model_class.__name__}: {str(e)}") from e

    async def find_by_json_field(self, json_field: str, json_path: str, value: Any) -> List[ModelType]:
        """
        Query entities by JSON field values using PostgreSQL JSON operators.
        
        Args:
            json_field: Name of the JSON field
            json_path: JSON path expression (e.g., "$.key" or "key")
            value: Value to match
            
        Returns:
            List of entities with matching JSON values
        """
        try:
            if not hasattr(self.model_class, json_field):
                raise ValueError(f"Field {json_field} does not exist on {self.model_class.__name__}")
            
            stmt = select(self.model_class)
            field_attr = getattr(self.model_class, json_field)
            
            # Use PostgreSQL JSON operators
            if json_path.startswith("$."):
                # JSONPath style
                stmt = stmt.where(field_attr.op("->")(json_path[2:]).astext == str(value))
            else:
                # Simple key access
                stmt = stmt.where(field_attr.op("->")(json_path).astext == str(value))
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to query JSON field on {self.model_class.__name__}: {str(e)}") from e

    async def bulk_create(self, entities: List[ModelType]) -> List[ModelType]:
        """
        Create multiple entities in a single transaction.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities with generated IDs
        """
        try:
            if not entities:
                return []
            
            self.session.add_all(entities)
            await self.session.commit()
            
            # Refresh all entities to get generated IDs
            for entity in entities:
                await self.session.refresh(entity)
            
            return entities
        except IntegrityError as e:
            await self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to bulk create {self.model_class.__name__}: {str(e)}") from e

    async def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple entities in a single operation.
        
        Args:
            updates: List of dictionaries with 'id' and update fields
            
        Returns:
            Number of entities updated
        """
        try:
            if not updates:
                return 0
            
            updated_count = 0
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                
                entity_id = update_data.pop('id')
                if update_data:  # Only update if there are fields to update
                    stmt = (
                        update(self.model_class)
                        .where(self.model_class.id == entity_id)
                        .values(**update_data)
                    )
                    result = await self.session.execute(stmt)
                    updated_count += result.rowcount
            
            await self.session.commit()
            return updated_count
        except IntegrityError as e:
            await self.session.rollback()
            raise e
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to bulk update {self.model_class.__name__}: {str(e)}") from e

    async def find_with_pagination(
        self, 
        page: int = 1, 
        page_size: int = 20, 
        order_by: Optional[str] = None,
        **criteria
    ) -> Dict[str, Any]:
        """
        Find entities with pagination support.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            order_by: Field name to order by (prefix with '-' for descending)
            **criteria: Filter criteria
            
        Returns:
            Dictionary with 'items', 'total', 'page', 'page_size', 'total_pages'
        """
        try:
            # Calculate offset
            offset = (page - 1) * page_size
            
            # Base query for items
            stmt = select(self.model_class)
            
            # Apply filters
            for field, value in criteria.items():
                if hasattr(self.model_class, field):
                    stmt = stmt.where(getattr(self.model_class, field) == value)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    # Descending order
                    field_name = order_by[1:]
                    if hasattr(self.model_class, field_name):
                        stmt = stmt.order_by(getattr(self.model_class, field_name).desc())
                else:
                    # Ascending order
                    if hasattr(self.model_class, order_by):
                        stmt = stmt.order_by(getattr(self.model_class, order_by))
            
            # Count query
            count_stmt = select(func.count(self.model_class.id))
            for field, value in criteria.items():
                if hasattr(self.model_class, field):
                    count_stmt = count_stmt.where(getattr(self.model_class, field) == value)
            
            # Execute both queries
            count_result = await self.session.execute(count_stmt)
            total = count_result.scalar() or 0
            
            # Add pagination to items query
            stmt = stmt.offset(offset).limit(page_size)
            items_result = await self.session.execute(stmt)
            items = list(items_result.scalars().all())
            
            # Calculate total pages
            total_pages = (total + page_size - 1) // page_size
            
            return {
                'items': items,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to paginate {self.model_class.__name__}: {str(e)}") from e

    async def find_by_date_range(
        self, 
        date_field: str, 
        start_date: Optional[Any] = None, 
        end_date: Optional[Any] = None
    ) -> List[ModelType]:
        """
        Find entities within a date range.
        
        Args:
            date_field: Name of the date/datetime field
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            List of entities within the date range
        """
        try:
            if not hasattr(self.model_class, date_field):
                raise ValueError(f"Date field {date_field} does not exist on {self.model_class.__name__}")
            
            stmt = select(self.model_class)
            field_attr = getattr(self.model_class, date_field)
            
            conditions = []
            if start_date is not None:
                conditions.append(field_attr >= start_date)
            if end_date is not None:
                conditions.append(field_attr <= end_date)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            raise RepositoryError(f"Failed to find {self.model_class.__name__} by date range: {str(e)}") from e

    async def soft_delete(self, id: UUID, deleted_field: str = "is_deleted") -> bool:
        """
        Perform soft delete by setting a boolean field.
        
        Args:
            id: Entity identifier
            deleted_field: Name of the boolean field to mark as deleted
            
        Returns:
            True if entity was soft deleted, False if not found
        """
        try:
            if not hasattr(self.model_class, deleted_field):
                raise ValueError(f"Deleted field {deleted_field} does not exist on {self.model_class.__name__}")
            
            stmt = (
                update(self.model_class)
                .where(self.model_class.id == id)
                .values(**{deleted_field: True})
            )
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return result.rowcount > 0
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise RepositoryError(f"Failed to soft delete {self.model_class.__name__}: {str(e)}") from e
