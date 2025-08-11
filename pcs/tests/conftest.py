"""
Global test configuration and fixtures.

Provides common test fixtures and environment setup for all tests.
"""

import os
import pytest
from unittest.mock import Mock

# Set test environment variables to avoid config validation errors
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test_db")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ENVIRONMENT", "test")

# Mock external services for testing
@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    return Mock()

@pytest.fixture
def mock_postgres_session():
    """Mock PostgreSQL session for testing."""
    return Mock()

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    return Mock()

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for testing."""
    return Mock()

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment for all tests."""
    # Ensure we're in test mode
    monkeypatch.setenv("ENVIRONMENT", "test")
    
    # Mock any external service calls
    monkeypatch.setenv("DISABLE_EXTERNAL_SERVICES", "true")
