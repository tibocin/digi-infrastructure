"""
Filepath: pcs/src/pcs/repositories/qdrant_export.py
Purpose: Export and data retrieval methods for Qdrant collections
Related Components: Embedding export, data retrieval, tenant filtering
Tags: export, retrieval, embeddings, qdrant
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


async def export_embeddings(
    client,
    collection_name: str,
    format_type: str = "numpy",
    include_metadata: bool = True,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Export embeddings from collection in specified format."""
    try:
        # Get all points from collection
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Large limit to get all points
            with_payload=include_metadata,
            with_vectors=True
        )
        
        if not points:
            return {
                "collection_name": collection_name,
                "embeddings": [],
                "documents": [],
                "metadatas": [],
                "format": format_type,
                "document_count": 0,
                "tenant_id": tenant_id
            }
        
        # Extract embeddings and metadata
        embeddings = [point.vector for point in points if point.vector]
        metadata = [point.payload for point in points if point.payload] if include_metadata else []
        
        # Apply tenant filtering if specified
        if tenant_id:
            tenant_points = [p for p in points if p.payload and p.payload.get("tenant_id") == tenant_id]
            embeddings = [point.vector for point in tenant_points if point.vector]
            metadata = [point.payload for point in tenant_points if point.payload] if include_metadata else []
        
        result = {
            "collection_name": collection_name,
            "embeddings": embeddings,
            "documents": metadata,
            "metadatas": metadata,
            "format": format_type,
            "document_count": len(embeddings),
            "tenant_id": tenant_id
        }
        
        # Convert to numpy if requested
        if format_type == "numpy":
            try:
                import numpy as np
                result["embeddings"] = np.array(embeddings)
            except ImportError:
                logger.warning("NumPy not available, returning list format")
        elif format_type == "json":
            # Format for JSON export
            result["data"] = []
            for i, embedding in enumerate(embeddings):
                item = {"embedding": embedding}
                if metadata and i < len(metadata):
                    item.update(metadata[i])
                result["data"].append(item)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to export embeddings from collection {collection_name}: {e}")
        raise Exception(f"Failed to export embeddings: {e}")


async def get_collection_statistics(
    client,
    collection_name: str,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get comprehensive collection statistics."""
    try:
        # Get basic stats
        stats = client.get_collection_stats(collection_name)
        
        # For now, return basic stats with placeholder values
        # TODO: Implement actual statistics collection
        return {
            "vectors_count": stats.get("vectors_count", 0),
            "points_count": stats.get("points_count", 0),
            "segments_count": stats.get("segments_count", 0),
            "status": stats.get("status", "unknown"),
            "config": {
                "name": collection_name,
                "dimension": 384,  # Placeholder
                "document_count": stats.get("points_count", 0),
                "memory_usage_mb": 0.0  # Placeholder
            }
        }
    except Exception as e:
        logger.error(f"Failed to get collection statistics for {collection_name}: {e}")
        raise Exception(f"Failed to get collection statistics: {e}")
