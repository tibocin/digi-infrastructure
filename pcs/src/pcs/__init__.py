"""
Filepath: src/pcs/__init__.py
Purpose: PCS (Prompt and Context Service) package initialization
Related Components: Main application, API, Core modules
Tags: package, init, pcs, main
"""

# Package metadata
__version__ = "1.0.0"
__description__ = "Prompt and Context Service - Autonomous coding agent system"

# Main entry points
from .main import create_app, main as run_main

__all__ = ["create_app", "run_main", "__version__", "__description__"]
