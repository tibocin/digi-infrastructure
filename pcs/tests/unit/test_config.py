"""
Filepath: tests/unit/test_config.py
Purpose: Unit tests for configuration management
Related Components: Core config, Settings validation, Environment variables
Tags: unit-tests, config, settings, validation
"""

import os
import pytest
from unittest.mock import patch

from pcs.core.config import (
    Settings,
    SecuritySettings,
    DatabaseSettings,
    RedisSettings,
    LoggingSettings,
    get_settings,
    get_test_settings
)
from pcs.core.exceptions import ValidationError


class TestSecuritySettings:
    """Test security settings validation."""
    
    def test_valid_security_settings(self):
        """Test creating security settings with valid data."""
        settings = SecuritySettings(
            secret_key="test_secret_key_with_enough_length",
            jwt_secret_key="test_jwt_secret_key_with_enough_length"
        )
        
        assert settings.secret_key == "test_secret_key_with_enough_length"
        assert settings.jwt_secret_key == "test_jwt_secret_key_with_enough_length"
        assert settings.jwt_algorithm == "HS256"
        assert settings.jwt_expire_minutes == 30
    
    def test_invalid_secret_key_too_short(self):
        """Test validation error for short secret key."""
        with pytest.raises(ValueError, match="SECRET_KEY must be provided"):
            SecuritySettings(
                secret_key="short",
                jwt_secret_key="test_jwt_secret_key_with_enough_length"
            )
    
    def test_invalid_jwt_secret_key_empty(self):
        """Test validation error for empty JWT secret key."""
        with pytest.raises(ValueError, match="SECRET_KEY must be provided"):
            SecuritySettings(
                secret_key="test_secret_key_with_enough_length",
                jwt_secret_key=""
            )


class TestDatabaseSettings:
    """Test database settings."""
    
    def test_postgresql_url_generation(self):
        """Test PostgreSQL URL generation."""
        settings = DatabaseSettings(
            dialect="postgresql+asyncpg",
            host="localhost",
            port=5432,
            user="test_user",
            password="test_password",
            name="test_db"
        )
        
        expected_url = "postgresql+asyncpg://test_user:test_password@localhost:5432/test_db"
        assert settings.url == expected_url
    
    def test_sqlite_url_generation(self):
        """Test SQLite URL generation."""
        settings = DatabaseSettings(
            dialect="sqlite+aiosqlite",
            name=":memory:"
        )
        
        expected_url = "sqlite+aiosqlite:///:memory:"
        assert settings.url == expected_url
    
    def test_database_url_without_password(self):
        """Test URL generation without password."""
        settings = DatabaseSettings(
            dialect="postgresql+asyncpg",
            host="localhost",
            port=5432,
            user="test_user",
            password="",
            name="test_db"
        )
        
        expected_url = "postgresql+asyncpg://test_user@localhost:5432/test_db"
        assert settings.url == expected_url
    
    def test_invalid_empty_host(self):
        """Test validation error for empty host."""
        with pytest.raises(ValueError, match="Database host must be provided"):
            DatabaseSettings(host="")


class TestRedisSettings:
    """Test Redis settings."""
    
    def test_redis_url_generation_basic(self):
        """Test basic Redis URL generation."""
        settings = RedisSettings(
            host="localhost",
            port=6379,
            db=0
        )
        
        expected_url = "redis://localhost:6379/0"
        assert settings.url == expected_url
    
    def test_redis_url_with_password(self):
        """Test Redis URL with password."""
        settings = RedisSettings(
            host="localhost",
            port=6379,
            db=0,
            password="secret"
        )
        
        expected_url = "redis://:secret@localhost:6379/0"
        assert settings.url == expected_url
    
    def test_redis_url_with_ssl(self):
        """Test Redis URL with SSL."""
        settings = RedisSettings(
            host="localhost",
            port=6380,
            db=0,
            ssl=True
        )
        
        expected_url = "rediss://localhost:6380/0"
        assert settings.url == expected_url


class TestMainSettings:
    """Test main application settings."""
    
    def test_development_environment_properties(self):
        """Test development environment properties."""
        settings = Settings(
            environment="development",
            security=SecuritySettings(
                secret_key="test_secret_key_with_enough_length",
                jwt_secret_key="test_jwt_secret_key_with_enough_length"
            ),
            database=DatabaseSettings(),
            redis=RedisSettings(),
            logging=LoggingSettings()
        )
        
        assert settings.is_development is True
        assert settings.is_testing is False
        assert settings.is_production is False
    
    def test_production_environment_properties(self):
        """Test production environment properties."""
        settings = Settings(
            environment="production",
            security=SecuritySettings(
                secret_key="test_secret_key_with_enough_length",
                jwt_secret_key="test_jwt_secret_key_with_enough_length"
            ),
            database=DatabaseSettings(),
            redis=RedisSettings(),
            logging=LoggingSettings()
        )
        
        assert settings.is_development is False
        assert settings.is_testing is False
        assert settings.is_production is True
    
    def test_invalid_environment(self):
        """Test validation error for invalid environment."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            Settings(
                environment="invalid",
                security=SecuritySettings(
                    secret_key="test_secret_key_with_enough_length",
                    jwt_secret_key="test_jwt_secret_key_with_enough_length"
                ),
                database=DatabaseSettings(),
                redis=RedisSettings(),
                logging=LoggingSettings()
            )


class TestSettingsFunctions:
    """Test settings utility functions."""
    
    @patch.dict(os.environ, {
        'PCS_SECURITY_SECRET_KEY': 'env_secret_key_with_enough_length',
        'PCS_SECURITY_JWT_SECRET_KEY': 'env_jwt_secret_key_with_enough_length',
        'PCS_ENVIRONMENT': 'testing',
        'PCS_DB_HOST': 'env_host',
        'PCS_DB_NAME': 'env_db'
    })
    def test_get_settings_from_environment(self):
        """Test loading settings from environment variables."""
        # Clear the cache first
        get_settings.cache_clear()
        
        settings = get_settings()
        
        assert settings.environment == "testing"
        assert settings.security.secret_key == "env_secret_key_with_enough_length"
        assert settings.security.jwt_secret_key == "env_jwt_secret_key_with_enough_length"
        assert settings.database.host == "env_host"
        assert settings.database.name == "env_db"
    
    def test_get_test_settings(self):
        """Test test-specific settings."""
        settings = get_test_settings()
        
        assert settings.environment == "testing"
        assert settings.debug is True
        assert settings.database.name == "pcs_test"
        assert settings.redis.db == 15
        assert settings.logging.file_enabled is False
