"""
Filepath: src/pcs/core/config.py
Purpose: Application configuration management using Pydantic settings
Related Components: Environment variables, Database settings, Security config
Tags: config, settings, environment, pydantic, validation
"""

import os
from functools import lru_cache
from typing import List, Optional
from pathlib import Path

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class SecuritySettings(BaseSettings):
    """Security-related configuration."""
    
    secret_key: str = Field(default="test-secret-key-for-development-only", description="Secret key for session management")
    jwt_secret_key: str = Field(default="test-jwt-secret-key-for-development-only", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, description="JWT expiration time in minutes")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    
    @field_validator('secret_key', 'jwt_secret_key')
    @classmethod
    def validate_secret_keys(cls, v: str) -> str:
        if not v or len(v) < 16:
            raise ValueError("SECRET_KEY must be provided and at least 16 characters long")
        return v
    
    model_config = ConfigDict(
        env_prefix="PCS_SECURITY_",
        env_file=".env",
        case_sensitive=False
    )


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    dialect: str = Field(default="postgresql+asyncpg", description="Database dialect")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    user: str = Field(default="pcs_user", description="Database user")
    password: str = Field(default="", description="Database password")
    name: str = Field(default="pcs_dev", description="Database name")
    
    # Pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_pre_ping: bool = Field(default=True, description="Enable pool pre-ping")
    
    @property
    def url(self) -> str:
        """Construct database URL from settings."""
        if "sqlite" in self.dialect:
            return f"{self.dialect}:///{self.name}"
        
        password_part = f":{self.password}" if self.password else ""
        return f"{self.dialect}://{self.user}{password_part}@{self.host}:{self.port}/{self.name}"
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v: str) -> str:
        if not v:
            raise ValueError("Database host must be provided")
        return v
    
    model_config = ConfigDict(
        env_prefix="PCS_DB_",
        env_file=".env",
        case_sensitive=False
    )


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL connection")
    socket_timeout: int = Field(default=5, description="Socket timeout")
    connection_timeout: int = Field(default=5, description="Connection timeout")
    max_connections: int = Field(default=20, description="Max connections in pool")
    
    @property
    def url(self) -> str:
        """Construct Redis URL from settings."""
        scheme = "rediss" if self.ssl else "redis"
        password_part = f":{self.password}@" if self.password else ""
        return f"{scheme}://{password_part}{self.host}:{self.port}/{self.db}"
    
    model_config = ConfigDict(
        env_prefix="PCS_REDIS_",
        env_file=".env",
        case_sensitive=False
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json/text)")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(default="logs/pcs.log", description="Log file path")
    
    model_config = ConfigDict(
        env_prefix="PCS_LOGGING_",
        env_file=".env",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(default="Prompt and Context Service (PCS)", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=8000, description="Port to bind to")
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    docs_url: Optional[str] = Field(default="/docs", description="OpenAPI docs URL")
    redoc_url: Optional[str] = Field(default="/redoc", description="ReDoc URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", description="OpenAPI JSON URL")
    
    # Nested settings
    security: SecuritySettings
    database: DatabaseSettings
    redis: RedisSettings
    logging: LoggingSettings
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    def __init__(self, **kwargs):
        # Initialize nested settings - they will handle their own environment loading
        kwargs.setdefault('security', SecuritySettings())
        kwargs.setdefault('database', DatabaseSettings())
        kwargs.setdefault('redis', RedisSettings())
        kwargs.setdefault('logging', LoggingSettings())
        
        super().__init__(**kwargs)
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "testing", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    model_config = ConfigDict(
        env_prefix="PCS_",
        env_file="/Users/stephensaunders/r/tibocin/digi-infrastructure/pcs/.env",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def get_test_settings() -> Settings:
    """Get test-specific settings with overrides."""
    test_overrides = {
        "environment": "testing",
        "debug": True,
        "database": DatabaseSettings(
            dialect="postgresql+asyncpg",
            host="localhost",
            port=5432,
            user="pcs_user",
            password="",
            name="pcs_test"
        ),
        "redis": RedisSettings(
            host="localhost",
            port=6379,
            db=15  # Use different DB for tests
        ),
        "security": SecuritySettings(
            secret_key="test_secret_key_that_is_long_enough_for_validation_purposes",
            jwt_secret_key="test_jwt_secret_key_that_is_long_enough_for_validation_purposes"
        ),
        "logging": LoggingSettings(
            level="DEBUG",
            file_enabled=False
        )
    }
    
    return Settings(**test_overrides)
