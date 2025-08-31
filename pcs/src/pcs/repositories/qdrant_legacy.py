"""
Filepath: pcs/src/pcs/repositories/qdrant_legacy.py
Purpose: Backward compatibility methods for Qdrant repository
Related Components: Legacy API methods, compatibility layer
Tags: legacy, compatibility, qdrant
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .qdrant_types import VectorDocument, SimilarityResult

logger = logging.getLogger(__name__)


async def get_collection(
    repository,
    collection_name: str
) -> bool:
    """Legacy method: check if collection exists."""
    try:
        info = repository.client.get_collection_info(collection_name)
        return info is not None
    except Exception:
        return False


async def add_documents(
    repository,
    collection_name: str,
    documents: List[VectorDocument]
) -> Dict[str, Any]:
    """Legacy method: add documents to collection."""
    return repository.upsert_documents(collection_name, documents)


async def query_documents(
    repository,
    collection_name: str,
    query_embedding: List[float],
    n_results: int = 10,
    tenant_id: Optional[str] = None
) -> List[SimilarityResult]:
    """Legacy method: query documents by similarity."""
    return await repository.search_similar(
        collection_name=collection_name,
        query_embedding=query_embedding,
        limit=n_results,
        tenant_id=tenant_id
    )


async def get_documents(
    repository,
    collection_name: str,
    document_ids: List[str]
) -> List[VectorDocument]:
    """Legacy method: get documents by IDs (placeholder)."""
    # TODO: Implement actual document retrieval by IDs
    # For now, return empty list
    return []


async def similarity_search(
    repository,
    collection_name: str,
    query_embedding: List[float],
    n_results: int = 10,
    tenant_id: Optional[str] = None
) -> List[SimilarityResult]:
    """Legacy method: similarity search."""
    return await repository.search_similar(
        collection_name=collection_name,
        query_embedding=query_embedding,
        limit=n_results,
        tenant_id=tenant_id
    )


async def count_documents(
    repository,
    collection_name: str,
) -> int:
    """Legacy method: count documents in collection."""
    try:
        stats = repository.core.get_collection_stats(collection_name)
        return stats.points_count
    except Exception:
        return 0


def delete_documents(
    repository,
    collection_name: str,
    document_ids: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Enhanced delete_documents with legacy parameter support."""
    # Handle legacy parameter names
    if ids is not None:
        document_ids = ids
    elif document_ids is None:
        document_ids = []
    
    # TODO: Implement filtering by 'where' clause
    if where is not None:
        logger.warning("Filter-based deletion not yet implemented, using document IDs")
    
    return repository.client.delete_points(collection_name, document_ids)
