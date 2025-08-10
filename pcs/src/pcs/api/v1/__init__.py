"""
Filepath: src/pcs/api/v1/__init__.py
Purpose: API v1 package initialization
Related Components: Health endpoints, Prompt endpoints, Context endpoints
Tags: api, v1, routers, endpoints
"""

from .health import router as health_router

__all__ = ["health_router"]
