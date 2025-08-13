"""
Global test configuration and fixtures.

Provides common test fixtures and environment setup for all tests.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

# Set comprehensive test environment variables before any imports
os.environ.update({
    "PCS_ENVIRONMENT": "testing",
    "PCS_DEBUG": "true",
    "PCS_SECURITY_SECRET_KEY": "test-secret-key-for-testing-only-must-be-at-least-16-chars",
    "PCS_SECURITY_JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-only-must-be-at-least-16-chars",
    "PCS_DB_HOST": "localhost",
    "PCS_DB_PORT": "5432",
    "PCS_DB_USER": "test_user",
    "PCS_DB_PASSWORD": "test_pass",
    "PCS_DB_NAME": "test_db",
    "PCS_DB_DIALECT": "sqlite+aiosqlite",  # Use SQLite for tests
    "PCS_REDIS_HOST": "localhost",
    "PCS_REDIS_PORT": "6379",
    "PCS_REDIS_DB": "15",
    "PCS_LOGGING_LEVEL": "DEBUG",
    "PCS_LOGGING_FILE_ENABLED": "false",
    
    # Legacy environment variables for backward compatibility
    "SECRET_KEY": "test-secret-key-for-testing-only-must-be-at-least-16-chars",
    "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-only-must-be-at-least-16-chars",
    "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
    "REDIS_URL": "redis://localhost:6379/15",
    "ENVIRONMENT": "testing",
    "DISABLE_EXTERNAL_SERVICES": "true"
})

# Import after setting environment variables
from pcs.core.config import get_test_settings


# Mock external services for testing
@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    return Mock()


@pytest.fixture
def mock_postgres_session():
    """Mock PostgreSQL session for testing."""
    session = AsyncMock(spec=AsyncSession)
    # Add common async methods
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = Mock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    return Mock()


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing."""
    return Mock()


@pytest.fixture
def test_settings():
    """Get test-specific settings."""
    return get_test_settings()


@pytest.fixture
def test_app(test_settings):
    """Create a test FastAPI application with test settings."""
    from pcs.main import create_app
    
    # Create app with test settings
    app = create_app(test_settings)
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client with the test app."""
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment for all tests."""
    # Ensure we're in test mode
    monkeypatch.setenv("PCS_ENVIRONMENT", "testing")
    monkeypatch.setenv("ENVIRONMENT", "testing")
    
    # Mock any external service calls
    monkeypatch.setenv("DISABLE_EXTERNAL_SERVICES", "true")
    
    # Prevent actual database connections
    monkeypatch.setenv("PCS_DB_DIALECT", "sqlite+aiosqlite")


@pytest.fixture
def mock_database_session():
    """Mock database session dependency."""
    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = Mock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_current_user():
    """Mock current user for authenticated endpoints."""
    return {
        "id": "test-user-id",
        "username": "test_user",
        "email": "test@example.com",
        "roles": ["user"]
    }

