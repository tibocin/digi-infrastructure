"""
Filepath: pcs/src/pcs/repositories/qdrant_types.py
Purpose: Type definitions and dataclasses for Qdrant repository operations
Related Components: Vector operations, collections, documents, similarity
Tags: types, dataclasses, enums, qdrant
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class QdrantDistance(Enum):
    """Qdrant distance metrics."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclid"
    DOT_PRODUCT = "Dot"
    MANHATTAN = "Manhattan"


class SimilarityAlgorithm(Enum):
    """Similarity algorithms for vector operations."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class VectorIndexType(Enum):
    """Vector index types for collections."""
    HNSW = "hnsw"
    IVFFLAT = "ivf_flat"
    SCALAR = "scalar"


@dataclass
class QdrantCollectionConfig:
    """Configuration for Qdrant collections."""
    name: str
    vector_size: int
    distance: Union[str, QdrantDistance] = QdrantDistance.COSINE
    on_disk_payload: bool = True
    hnsw_config: Optional[Dict[str, Any]] = None
    optimizers_config: Optional[Dict[str, Any]] = None
    replication_factor: int = 1
    write_consistency_factor: int = 1


@dataclass
class QdrantPoint:
    """Qdrant point representation."""
    id: str
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None
    version: Optional[int] = None


@dataclass
class QdrantSearchResult:
    """Qdrant search result."""
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    version: Optional[int] = None


@dataclass
class VectorDocument:
    """Vector document with metadata."""
    id: str
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    collection_name: Optional[str] = None
    tenant_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "collection_name": self.collection_name,
            "tenant_id": self.tenant_id
        }
    
    def to_qdrant_point(self) -> 'QdrantPoint':
        """Convert to Qdrant point format."""
        payload = {
            "content": self.content,
            "tenant_id": self.tenant_id
        }
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.collection_name:
            payload["collection_name"] = self.collection_name
        
        # Merge metadata directly into payload
        if self.metadata:
            payload.update(self.metadata)
        
        return QdrantPoint(
            id=self.id,
            vector=self.embedding,
            payload=payload
        )


@dataclass
class SimilarityResult:
    """Similarity search result."""
    id: Union[str, int]
    score: float
    similarity_score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    version: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    document: Optional[VectorDocument] = None  # Legacy field
    
    def __post_init__(self):
        """Handle legacy field mapping."""
        # For backward compatibility, create document if not provided
        if self.document is None and self.payload:
            # Try to reconstruct VectorDocument from payload
            try:
                self.document = VectorDocument(
                    id=str(self.id),
                    content=self.payload.get("content", ""),
                    embedding=self.vector or [],
                    metadata=self.payload.get("metadata", {}),
                    created_at=datetime.fromisoformat(self.payload.get("created_at", datetime.now().isoformat())) if self.payload.get("created_at") else None,
                    collection_name=self.payload.get("collection_name"),
                    tenant_id=self.payload.get("tenant_id")
                )
            except Exception:
                # If reconstruction fails, create minimal document
                self.document = VectorDocument(
                    id=str(self.id),
                    content="",
                    embedding=self.vector or [],
                    metadata=self.payload or {}
                )


@dataclass
class VectorSearchRequest:
    """Vector search request parameters."""
    collection_name: str
    query_vector: Optional[List[float]] = None
    query_embedding: Optional[List[float]] = None  # Legacy alias
    limit: int = 10
    n_results: int = 10  # Legacy alias
    score_threshold: Optional[float] = None
    similarity_threshold: float = 0.0  # Legacy alias
    tenant_id: Optional[str] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    metadata_filters: Optional[Dict[str, Any]] = None  # Legacy alias
    algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    include_vectors: bool = False
    include_payload: bool = True
    
    def __post_init__(self):
        """Handle legacy field aliases."""
        # Map legacy fields to new fields
        if self.query_embedding is not None and self.query_vector is None:
            self.query_vector = self.query_embedding
        if self.n_results != 10 and self.limit == 10:
            self.limit = self.n_results
        if self.similarity_threshold != 0.0 and self.score_threshold is None:
            self.score_threshold = self.similarity_threshold
        if self.metadata_filters is not None and self.filter_conditions is None:
            self.filter_conditions = self.metadata_filters


@dataclass
class BulkVectorOperation:
    """Container for bulk vector operations with multi-tenancy."""
    operation_type: str  # "upsert", "delete", "update"
    collection_name: str
    documents: Optional[List[Any]] = None  # List of VectorDocument or similar
    tenant_id: Optional[str] = None
    batch_size: int = 100
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VectorCollectionStats:
    """Collection statistics and metadata."""
    vectors_count: int
    points_count: int
    segments_count: int
    status: str
    config: Optional[Dict[str, Any]] = None
    payload_schema: Optional[Dict[str, Any]] = None
