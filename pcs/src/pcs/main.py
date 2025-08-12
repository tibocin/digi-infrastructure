"""
Filepath: src/pcs/main.py
Purpose: Main FastAPI application factory and entry point
Related Components: FastAPI, Middleware, Routes, Lifespan management
Tags: fastapi, main, application-factory, middleware, startup
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .core.config import Settings, get_settings
from .core.database import get_database_manager
from .core.exceptions import PCSError, DatabaseError
from .api.dependencies import get_settings_dependency


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and timing."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    try:
        # Initialize database
        db_manager = await get_database_manager()
        print("Database manager initialized successfully")
        
        # Create tables if they don't exist
        await db_manager.create_all_tables()
        print("Database tables verified/created")
        
        yield
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        try:
            db_manager = await get_database_manager()
            await db_manager.close()
            print("Database connections closed")
        except Exception as e:
            print(f"Error during shutdown: {e}")


def create_app(settings: Settings = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings (optional, will use default if not provided)
        
    Returns:
        Configured FastAPI application instance
    """
    if settings is None:
        settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Autonomous coding agent with advanced prompt and context management",
        docs_url=settings.docs_url if settings.debug else None,
        redoc_url=settings.redoc_url if settings.debug else None,
        openapi_url=settings.openapi_url if settings.debug else None,
        lifespan=lifespan
    )
    
    # Set up middleware
    setup_middleware(app, settings)
    
    # Set up exception handlers
    setup_exception_handlers(app)
    
    # Set up routes
    setup_routes(app, settings)
    
    return app


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Configure application middleware."""
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
    )
    
    # GZip compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Trusted host middleware (for production)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", settings.host]
        )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure application exception handlers."""
    
    @app.exception_handler(PCSError)
    async def pcs_error_handler(request: Request, exc: PCSError) -> JSONResponse:
        """Handle custom PCS errors."""
        return JSONResponse(
            status_code=400,
            content=exc.to_dict()
        )
    
    @app.exception_handler(DatabaseError)
    async def database_error_handler(request: Request, exc: DatabaseError) -> JSONResponse:
        """Handle database errors."""
        return JSONResponse(
            status_code=503,
            content={
                "error": "DatabaseError",
                "message": "Database service temporarily unavailable",
                "details": {"internal_error": str(exc)}
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": {"validation_errors": exc.errors()}
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPError",
                "message": exc.detail,
                "details": {}
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": {"type": type(exc).__name__}
            }
        )


def setup_routes(app: FastAPI, settings: Settings) -> None:
    """Configure application routes."""
    
    # Root endpoint
    @app.get("/", response_model=Dict[str, Any])
    async def root(app_settings: Settings = Depends(get_settings_dependency)) -> Dict[str, Any]:
        """Root endpoint with basic API information."""
        return {
            "name": app_settings.app_name,
            "version": app_settings.version,
            "status": "operational",
            "environment": app_settings.environment,
            "docs_url": app_settings.docs_url,
            "api_version": "v1",
            "api_prefix": app_settings.api_prefix
        }
    
    # Include API routers
    from .api.v1.health import router as health_router
    from .api.v1.prompts import router as prompts_router
    from .api.v1.contexts import router as contexts_router
    from .api.v1.conversations import router as conversations_router
    from .api.v1.admin import router as admin_router
    
    app.include_router(health_router, prefix=f"{settings.api_prefix}/health", tags=["health"])
    app.include_router(prompts_router, prefix=settings.api_prefix, tags=["prompts"])
    app.include_router(contexts_router, prefix=settings.api_prefix, tags=["contexts"])
    app.include_router(conversations_router, prefix=settings.api_prefix, tags=["conversations"])
    app.include_router(admin_router, prefix=settings.api_prefix, tags=["admin"])


def main() -> None:
    """Main application entry point."""
    settings = get_settings()
    app = create_app(settings)
    
    print(f"Starting {settings.app_name} v{settings.version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    
    uvicorn.run(
        "pcs.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug and settings.is_development,
        log_level="info" if not settings.debug else "debug"
    )


def run_server() -> None:
    """Alternative entry point for running the server."""
    main()


# Create app instance for ASGI servers
app = create_app()


if __name__ == "__main__":
    main()
