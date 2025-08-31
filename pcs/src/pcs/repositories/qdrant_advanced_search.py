"""
Filepath: pcs/src/pcs/repositories/qdrant_advanced_search.py
Purpose: Advanced search functionality for Qdrant including semantic search, reranking, and advanced filtering
Related Components: Semantic search, result reranking, advanced filters, similarity algorithms
Tags: qdrant, advanced-search, semantic-search, reranking, filters
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .qdrant_types import SimilarityAlgorithm, QdrantSearchResult

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchRequest:
    """Request structure for advanced vector search operations."""
    collection_name: str
    query_vector: List[float]
    limit: int = 10
    score_threshold: Optional[float] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    similarity_algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    include_vectors: bool = False
    include_payload: bool = True


@dataclass
class SimilarityResult:
    """Enhanced search result with similarity metrics and metadata."""
    id: Union[str, int]
    score: float
    similarity_score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    version: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class QdrantAdvancedSearch:
    """
    Advanced search functionality for Qdrant vector database.
    
    This class provides sophisticated search capabilities including
    semantic search, result reranking, and advanced filtering.
    """
    
    def __init__(self, core_operations):
        """
        Initialize advanced search with core operations.
        
        Args:
            core_operations: QdrantCoreOperations instance for basic operations
        """
        self.core = core_operations
        self.logger = logging.getLogger(__name__)
    
    def _build_query_filter(self, request: VectorSearchRequest) -> Optional[Dict[str, Any]]:
        """Build advanced query filter from request conditions."""
        if not request.filter_conditions:
            return None
        
        # Convert filter conditions to Qdrant filter format
        # This is a simplified implementation - can be extended for complex filters
        return request.filter_conditions
    
    def _calculate_similarity(self, score: float, algorithm: SimilarityAlgorithm) -> float:
        """Calculate normalized similarity score based on algorithm."""
        try:
            if algorithm == SimilarityAlgorithm.COSINE:
                # Cosine similarity is already normalized [-1, 1]
                return max(0, score)  # Normalize to [0, 1]
            elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
                # Convert Euclidean distance to similarity (inverse relationship)
                # Assuming score is distance, convert to similarity
                return 1.0 / (1.0 + score) if score > 0 else 1.0
            elif algorithm == SimilarityAlgorithm.DOT_PRODUCT:
                # Dot product needs normalization - this is simplified
                return max(0, score)
            elif algorithm == SimilarityAlgorithm.MANHATTAN:
                # Convert Manhattan distance to similarity
                return 1.0 / (1.0 + score) if score > 0 else 1.0
            else:
                return score
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return score
    
    async def semantic_search_advanced(
        self, 
        request: VectorSearchRequest
    ) -> List[SimilarityResult]:
        """
        Perform advanced semantic search with enhanced filtering and scoring.
        
        Args:
            request: VectorSearchRequest containing search parameters
            
        Returns:
            List of SimilarityResult objects with enhanced metadata
        """
        try:
            # Build search parameters
            search_params = {
                "vector": request.query_vector,
                "limit": request.limit
            }
            
            if request.score_threshold is not None:
                search_params["score_threshold"] = request.score_threshold
            
            # Build advanced filter
            filter_conditions = self._build_query_filter(request)
            if filter_conditions:
                search_params["filter"] = filter_conditions
            
            # Perform basic search using core operations
            basic_results = self.core.search_points(
                collection_name=request.collection_name,
                query_vector=request.query_vector,
                limit=request.limit,
                score_threshold=request.score_threshold,
                filter_conditions=filter_conditions
            )
            
            # Convert to enhanced results
            enhanced_results = []
            for result in basic_results:
                                # Calculate normalized similarity score
                similarity_score = self._calculate_similarity(
                    result.score,
                    request.algorithm
                )
                
                # Build metadata
                metadata = {
                    "original_score": result.score,
                    "similarity_algorithm": request.algorithm.value,
                    "normalized_similarity": similarity_score
                }
                
                enhanced_result = SimilarityResult(
                    id=result.id,
                    score=result.score,
                    similarity_score=similarity_score,
                    payload=result.payload if request.include_payload else None,
                    vector=result.vector if request.include_vectors else None,
                    version=result.version,
                    metadata=metadata
                )
                
                enhanced_results.append(enhanced_result)
            
            # Sort by similarity score (descending)
            enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            self.logger.info(
                f"Advanced semantic search completed for {request.collection_name}, "
                f"found {len(enhanced_results)} results using {request.similarity_algorithm.value}"
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Advanced semantic search failed: {e}")
            raise Exception(f"Advanced semantic search failed: {e}")
    
    async def _rerank_results(
        self,
        results: List[SimilarityResult],
        reranking_strategy: str = "score_based"
    ) -> List[SimilarityResult]:
        """
        Rerank search results using various strategies.
        
        Args:
            results: List of SimilarityResult objects
            reranking_strategy: Strategy for reranking ("score_based", "diversity", "relevance")
            
        Returns:
            Reranked list of results
        """
        try:
            if reranking_strategy == "score_based":
                # Sort by similarity score (already done in semantic_search_advanced)
                return results
            
            elif reranking_strategy == "diversity":
                # Implement diversity-based reranking
                # This is a simplified implementation
                reranked = []
                used_vectors = set()
                
                for result in results:
                    if result.vector:
                        # Simple diversity check - avoid very similar vectors
                        vector_key = tuple(round(x, 3) for x in result.vector[:10])  # First 10 dimensions
                        if vector_key not in used_vectors:
                            reranked.append(result)
                            used_vectors.add(vector_key)
                        else:
                            # Add with lower priority
                            reranked.append(result)
                    else:
                        reranked.append(result)
                
                return reranked
            
            elif reranking_strategy == "relevance":
                # Implement relevance-based reranking
                # This could include payload-based scoring, user preferences, etc.
                return results
            
            else:
                self.logger.warning(f"Unknown reranking strategy: {reranking_strategy}")
                return results
                
        except Exception as e:
            self.logger.error(f"Result reranking failed: {e}")
            return results
    
    async def find_similar_documents(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE,
        rerank: bool = True
    ) -> List[SimilarityResult]:
        """
        Find similar documents with enhanced similarity scoring and optional reranking.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            algorithm: Similarity algorithm to use
            rerank: Whether to apply reranking
            
        Returns:
            List of SimilarityResult objects
        """
        try:
            # Build search request
            request = VectorSearchRequest(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=similarity_threshold,
                similarity_algorithm=algorithm,
                include_vectors=rerank,  # Need vectors for reranking
                include_payload=True
            )
            
            # Perform advanced search
            results = await self.semantic_search_advanced(request)
            
            # Apply reranking if requested
            if rerank and results:
                results = await self._rerank_results(results, reranking_strategy="diversity")
            
            self.logger.info(
                f"Similarity search completed for {collection_name}, "
                f"found {len(results)} results above threshold {similarity_threshold}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            raise Exception(f"Similarity search failed: {e}")
    
    def build_metadata_filter(
        self,
        metadata_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build Qdrant-compatible filter from metadata conditions.
        
        Args:
            metadata_conditions: Dictionary of metadata field conditions
            
        Returns:
            Qdrant filter structure
        """
        try:
            if not metadata_conditions:
                return {}
            
            # Convert metadata conditions to Qdrant filter format
            # This is a simplified implementation
            filter_conditions = {}
            
            for field, condition in metadata_conditions.items():
                if isinstance(condition, dict):
                    # Handle range queries, etc.
                    filter_conditions[field] = condition
                else:
                    # Simple equality
                    filter_conditions[field] = {"$eq": condition}
            
            return filter_conditions
            
        except Exception as e:
            self.logger.error(f"Failed to build metadata filter: {e}")
            return {}
    
    async def search_with_metadata(
        self,
        collection_name: str,
        query_vector: List[float],
        metadata_filter: Dict[str, Any],
        limit: int = 10,
        algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE
    ) -> List[SimilarityResult]:
        """
        Search for similar documents with metadata filtering.
        
        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            metadata_filter: Metadata conditions to filter by
            limit: Maximum number of results to return
            algorithm: Similarity algorithm to use
            
        Returns:
            List of SimilarityResult objects matching both vector similarity and metadata
        """
        try:
            # Build Qdrant filter from metadata
            qdrant_filter = self.build_metadata_filter(metadata_filter)
            
            # Build search request with metadata filter
            request = VectorSearchRequest(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                filter_conditions=qdrant_filter,
                similarity_algorithm=algorithm,
                include_payload=True
            )
            
            # Perform advanced search
            results = await self.semantic_search_advanced(request)
            
            self.logger.info(
                f"Metadata-filtered search completed for {collection_name}, "
                f"found {len(results)} results matching metadata criteria"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Metadata-filtered search failed: {e}")
            raise Exception(f"Metadata-filtered search failed: {e}")
