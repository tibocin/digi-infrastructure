"""
Filepath: src/pcs/api/v1/__init__.py
Purpose: API v1 package initialization
Related Components: Health endpoints, Prompt endpoints, Context endpoints
Tags: api, v1, routers, endpoints
"""

from .health import router as health_router
from .prompts import router as prompts_router
from .contexts import router as contexts_router
from .conversations import router as conversations_router
from .admin import router as admin_router

__all__ = ["health_router", "prompts_router", "contexts_router", "conversations_router", "admin_router"]
