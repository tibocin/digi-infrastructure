"""
Filepath: src/pcs/sdk/__init__.py
Purpose: PCS SDK package initialization
Related Components: Python SDK, TypeScript SDK, Client libraries  
Tags: sdk, client, api, python, typescript
"""

from .python import PCSClient

__version__ = "1.0.0"
__all__ = ["PCSClient"]