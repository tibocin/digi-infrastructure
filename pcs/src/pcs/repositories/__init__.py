"""
Filepath: pcs/src/pcs/repositories/__init__.py
Purpose: Repository package initialization with exports for data access layer
Related Components: Core models, database connections, service layer
Tags: repository-pattern, data-access, database-abstraction
"""

from .base import AbstractRepository, BaseRepository, RepositoryError
from .postgres_repo import PostgreSQLRepository

# Optional imports for repositories that require external dependencies
try:
    from .redis_repo import RedisRepository
    _has_redis = True
except ImportError:
    RedisRepository = None
    _has_redis = False

try:
    from .neo4j_repo import Neo4jRepository
    _has_neo4j = True
except ImportError:
    Neo4jRepository = None
    _has_neo4j = False

try:
    from .qdrant_repo import EnhancedQdrantHTTPRepository as QdrantRepository
    _has_qdrant = True
except ImportError:
    QdrantRepository = None
    _has_qdrant = False

__all__ = [
    "AbstractRepository",
    "BaseRepository", 
    "RepositoryError",
    "PostgreSQLRepository",
]

# Add optional exports only if available
if _has_redis:
    __all__.append("RedisRepository")
if _has_neo4j:
    __all__.append("Neo4jRepository")
if _has_qdrant:
    __all__.append("QdrantRepository")
