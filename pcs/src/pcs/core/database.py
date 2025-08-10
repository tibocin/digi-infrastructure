"""
Filepath: src/pcs/core/database.py
Purpose: Async database connection management with SQLAlchemy 2.0
Related Components: Models, Repositories, Health checks, Connection pooling
Tags: database, sqlalchemy, async, connection-pool, health-check
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional

from sqlalchemy import text, event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from .config import DatabaseSettings
from .exceptions import DatabaseError, ConnectionError


def get_base():
    """Lazy import of Base to avoid circular dependencies."""
    from ..models.base import Base
    return Base


class DatabaseManager:
    """
    Manages async database connections and sessions.
    """
    
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            self._engine = await self._create_async_engine()
            self._session_factory = sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            await self.health_check()
        except Exception as e:
            raise DatabaseError(f"Database initialization failed: {e}") from e
    
    async def _create_async_engine(self) -> AsyncEngine:
        """Create async SQLAlchemy engine with proper configuration."""
        database_url = self.settings.url
        
        # Configure pool settings based on database type
        pool_settings = {}
        
        if "sqlite" in database_url:
            # SQLite-specific configuration
            pool_settings = {
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False},
                "pool_pre_ping": self.settings.pool_pre_ping,
            }
        else:
            # PostgreSQL and other databases
            pool_settings = {
                "pool_size": self.settings.pool_size,
                "max_overflow": self.settings.max_overflow,
                "pool_timeout": self.settings.pool_timeout,
                "pool_pre_ping": self.settings.pool_pre_ping,
            }
        
        engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging in development
            future=True,
            **pool_settings
        )
        
        # Set up engine event listeners
        try:
            self._setup_engine_events(engine)
        except Exception as e:
            # Gracefully handle event setup failures (especially during testing)
            pass
        
        return engine
    
    def _setup_engine_events(self, engine: AsyncEngine) -> None:
        """Set up database engine event listeners."""
        
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance and reliability."""
            if "sqlite" in str(engine.url):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        
        @event.listens_for(engine.sync_engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Ping connection on checkout to ensure it's alive."""
            if connection_record.info.get('_ping_time', 0) < time.time() - 30:
                try:
                    # Ping the connection
                    cursor = dbapi_connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                    connection_record.info['_ping_time'] = time.time()
                except Exception:
                    # Connection is invalid, raise to get a new one
                    raise ConnectionError("Database connection ping failed")
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
        
        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check.
        
        Returns:
            Dictionary with health check results including connection status,
            pool statistics, and response time.
        """
        start_time = time.time()
        
        try:
            if not self._engine:
                raise DatabaseError("Database engine not initialized")
            
            # Test basic connectivity
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Get pool statistics
            pool = self._engine.pool
            pool_info = {
                "pool_type": pool.__class__.__name__,
            }
            
            # Only add pool statistics if the pool supports them
            # StaticPool (used by SQLite) doesn't have these methods
            if hasattr(pool, 'size'):
                pool_info.update({
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid(),
                })
            else:
                # For StaticPool or other pool types without statistics
                pool_info["note"] = "Pool statistics not available for this pool type"
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "database_url": f"{self.settings.host}:{self.settings.port}",
                "database_name": self.settings.name,
                "pool_info": pool_info
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            health_info = {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": round(response_time, 2),
                "database_url": f"{self.settings.host}:{self.settings.port}",
                "database_name": self.settings.name
            }
            
            raise DatabaseError(f"Database health check failed: {e}") from e
    
    async def create_all_tables(self) -> None:
        """Create all database tables."""
        if not self._engine:
            raise DatabaseError("Database engine not initialized")
        
        Base = get_base()
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_all_tables(self) -> None:
        """Drop all database tables."""
        if not self._engine:
            raise DatabaseError("Database engine not initialized")
        
        Base = get_base()
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self) -> None:
        """Close database engine and all connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    
    if _db_manager is None:
        from .config import get_settings
        settings = get_settings()
        _db_manager = DatabaseManager(settings.database)
        await _db_manager.initialize()
    
    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency function to get database session for FastAPI."""
    db_manager = await get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session


# Utility functions for testing
async def create_test_engine(database_url: str = "sqlite+aiosqlite:///:memory:") -> AsyncEngine:
    """Create a test database engine."""
    return create_async_engine(
        database_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )


async def setup_test_database(engine: AsyncEngine) -> None:
    """Set up test database with all tables."""
    Base = get_base()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
