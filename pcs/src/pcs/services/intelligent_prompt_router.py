"""
Filepath: pcs/src/pcs/services/intelligent_prompt_router.py
Purpose: Intelligent prompt routing service that combines vector search with reasoning for optimal prompt selection
Related Components: Qdrant repository, Neo4j service, prompt service, context manager
Tags: intelligent-routing, vector-search, reasoning, prompt-selection, optimization
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from ..core.exceptions import PCSError
from ..repositories.qdrant_repo import EnhancedQdrantRepository, SimilarityResult
from .prompt_service import PromptGenerator


class PromptRoutingError(PCSError):
    """Custom exception for prompt routing errors."""
    pass


class RoutingStrategy(Enum):
    """Different routing strategies for prompt selection."""
    VECTOR_ONLY = "vector_only"           # Use only vector similarity
    REASONING_ONLY = "reasoning_only"     # Use only reasoning logic
    HYBRID = "hybrid"                     # Combine both approaches
    CONFIDENCE_BASED = "confidence_based" # Route based on confidence scores


@dataclass
class PromptRoutingRequest:
    """Request for intelligent prompt routing."""
    query_text: str
    app_id: str
    user_context: Dict[str, Any]
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID
    max_candidates: int = 5
    similarity_threshold: float = 0.7
    include_cross_app: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PromptRoutingResult:
    """Result of intelligent prompt routing."""
    selected_prompt_id: str
    confidence_score: float
    routing_strategy: RoutingStrategy
    vector_score: Optional[float] = None
    reasoning_score: Optional[float] = None
    cross_app_source: Optional[str] = None
    reasoning_explanation: Optional[str] = None
    alternatives: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []


class IntelligentPromptRouter:
    """
    Intelligent prompt router that combines vector search with reasoning.
    
    This service provides sophisticated prompt selection by:
    1. Using vector similarity to find candidate prompts
    2. Applying reasoning logic to select the optimal prompt
    3. Supporting cross-application prompt sharing (when enabled)
    4. Learning from usage patterns and feedback
    """
    
    def __init__(
        self,
        qdrant_repo: EnhancedQdrantRepository,
        prompt_generator: PromptGenerator,
        neo4j_service: Optional[Any] = None,  # Will be implemented in Phase 3
        enable_cross_app: bool = False
    ):
        """
        Initialize the intelligent prompt router.
        
        Args:
            qdrant_repo: Qdrant repository for vector search
            prompt_generator: Prompt generator service
            neo4j_service: Neo4j service for reasoning (optional)
            enable_cross_app: Whether to enable cross-application prompts
        """
        self.qdrant_repo = qdrant_repo
        self.prompt_generator = prompt_generator
        self.neo4j_service = neo4j_service
        self.enable_cross_app = enable_cross_app
        
        # Performance tracking
        self.total_routing_requests = 0
        self.successful_routings = 0
        self.cross_app_usage_count = 0
    
    async def route_to_optimal_prompt(
        self, 
        request: PromptRoutingRequest
    ) -> PromptRoutingResult:
        """
        Route to the optimal prompt using the specified strategy.
        
        Args:
            request: Routing request with query and context
            
        Returns:
            Routing result with selected prompt and confidence
        """
        self.total_routing_requests += 1
        
        try:
            # Step 1: Find candidate prompts using vector search
            candidates = await self._find_candidate_prompts(request)
            
            if not candidates:
                raise PromptRoutingError(f"No candidate prompts found for query: {request.query_text}")
            
            # Step 2: Apply routing strategy
            if request.routing_strategy == RoutingStrategy.VECTOR_ONLY:
                result = await self._route_vector_only(request, candidates)
            elif request.routing_strategy == RoutingStrategy.REASONING_ONLY:
                result = await self._route_reasoning_only(request, candidates)
            elif request.routing_strategy == RoutingStrategy.HYBRID:
                result = await self._route_hybrid(request, candidates)
            elif request.routing_strategy == RoutingStrategy.CONFIDENCE_BASED:
                result = await self._route_confidence_based(request, candidates)
            else:
                raise PromptRoutingError(f"Unknown routing strategy: {request.routing_strategy}")
            
            # Step 3: Update usage statistics
            self.successful_routings += 1
            if result.cross_app_source:
                self.cross_app_usage_count += 1
            
            return result
            
        except Exception as e:
            raise PromptRoutingError(f"Failed to route to optimal prompt: {str(e)}") from e
    
    async def _find_candidate_prompts(
        self, 
        request: PromptRoutingRequest
    ) -> List[SimilarityResult]:
        """Find candidate prompts using vector search."""
        try:
            # Search in app-specific collection
            app_candidates = await self.prompt_generator.search_similar_prompts(
                query_text=request.query_text,
                app_id=request.app_id,
                limit=request.max_candidates,
                similarity_threshold=request.similarity_threshold
            )
            
            candidates = list(app_candidates)
            
            # Add cross-app candidates if enabled
            if request.include_cross_app and self.enable_cross_app:
                cross_app_candidates = await self._find_cross_app_candidates(request)
                candidates.extend(cross_app_candidates)
            
            # Sort by similarity score
            candidates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return candidates[:request.max_candidates]
            
        except Exception as e:
            raise PromptRoutingError(f"Failed to find candidate prompts: {str(e)}") from e
    
    async def _find_cross_app_candidates(
        self, 
        request: PromptRoutingRequest
    ) -> List[SimilarityResult]:
        """Find candidate prompts from other applications."""
        if not self.enable_cross_app:
            return []
        
        try:
            # For now, we'll search in a unified collection
            # In Phase 3, this will use Neo4j for cross-app reasoning
            unified_candidates = await self.prompt_generator.search_similar_prompts(
                query_text=request.query_text,
                app_id="unified",  # Special collection for cross-app prompts
                limit=request.max_candidates // 2,  # Limit cross-app candidates
                similarity_threshold=request.similarity_threshold
            )
            
            # Mark these as cross-app candidates
            for candidate in unified_candidates:
                candidate.payload = candidate.payload or {}
                candidate.payload["cross_app"] = True
                candidate.payload["source_app"] = candidate.payload.get("app_id", "unknown")
            
            return unified_candidates
            
        except Exception as e:
            # Cross-app search failure shouldn't break the main flow
            # Log the error and continue with app-specific candidates
            print(f"Warning: Cross-app search failed: {str(e)}")
            return []
    
    async def _route_vector_only(
        self, 
        request: PromptRoutingRequest, 
        candidates: List[SimilarityResult]
    ) -> PromptRoutingResult:
        """Route using only vector similarity scores."""
        if not candidates:
            raise PromptRoutingError("No candidates available for vector-only routing")
        
        # Select the highest scoring candidate
        best_candidate = candidates[0]
        
        return PromptRoutingResult(
            selected_prompt_id=best_candidate.document.id,
            confidence_score=best_candidate.similarity_score,
            routing_strategy=RoutingStrategy.VECTOR_ONLY,
            vector_score=best_candidate.similarity_score,
            alternatives=[
                {
                    "prompt_id": c.document.id,
                    "score": c.similarity_score,
                    "name": c.document.payload.get("name", "Unknown"),
                    "category": c.document.payload.get("category", "Unknown")
                }
                for c in candidates[1:3]  # Top 3 alternatives
            ]
        )
    
    async def _route_reasoning_only(
        self, 
        request: PromptRoutingRequest, 
        candidates: List[SimilarityResult]
    ) -> PromptRoutingResult:
        """Route using only reasoning logic (placeholder for Phase 3)."""
        if not candidates:
            raise PromptRoutingError("No candidates available for reasoning-only routing")
        
        # For now, fall back to vector-only routing
        # In Phase 3, this will use Neo4j reasoning
        return await self._route_vector_only(request, candidates)
    
    async def _route_hybrid(
        self, 
        request: PromptRoutingRequest, 
        candidates: List[SimilarityResult]
    ) -> PromptRoutingResult:
        """Route using a combination of vector similarity and reasoning."""
        if not candidates:
            raise PromptRoutingError("No candidates available for hybrid routing")
        
        try:
            # Step 1: Calculate vector scores
            vector_scores = {}
            for candidate in candidates:
                vector_scores[candidate.document.id] = candidate.similarity_score
            
            # Step 2: Apply reasoning logic (placeholder for Phase 3)
            reasoning_scores = await self._calculate_reasoning_scores(request, candidates)
            
            # Step 3: Combine scores (simple weighted average for now)
            combined_scores = {}
            for prompt_id in vector_scores:
                vector_score = vector_scores[prompt_id]
                reasoning_score = reasoning_scores.get(prompt_id, 0.5)  # Default to neutral
                
                # Weight: 70% vector, 30% reasoning
                combined_score = (0.7 * vector_score) + (0.3 * reasoning_score)
                combined_scores[prompt_id] = combined_score
            
            # Step 4: Select best candidate
            best_prompt_id = max(combined_scores, key=combined_scores.get)
            best_candidate = next(c for c in candidates if c.document.id == best_prompt_id)
            
            return PromptRoutingResult(
                selected_prompt_id=best_prompt_id,
                confidence_score=combined_scores[best_prompt_id],
                routing_strategy=RoutingStrategy.HYBRID,
                vector_score=vector_scores[best_prompt_id],
                reasoning_score=reasoning_scores.get(best_prompt_id, 0.5),
                alternatives=[
                    {
                        "prompt_id": c.document.id,
                        "combined_score": combined_scores.get(c.document.id, 0),
                        "vector_score": vector_scores.get(c.document.id, 0),
                        "reasoning_score": reasoning_scores.get(c.document.id, 0.5),
                        "name": c.document.payload.get("name", "Unknown") if c.document.payload else "Unknown"
                    }
                    for c in candidates[:3] if c.document.id != best_prompt_id
                ]
            )
            
        except Exception as e:
            # Fall back to vector-only routing if hybrid fails
            print(f"Warning: Hybrid routing failed, falling back to vector-only: {str(e)}")
            return await self._route_vector_only(request, candidates)
    
    async def _route_confidence_based(
        self, 
        request: PromptRoutingRequest, 
        candidates: List[SimilarityResult]
    ) -> PromptRoutingResult:
        """Route based on confidence scores and fallback strategies."""
        if not candidates:
            raise PromptRoutingError("No candidates available for confidence-based routing")
        
        try:
            # Calculate confidence scores for all candidates
            confidence_scores = {}
            for candidate in candidates:
                # Base confidence from vector similarity
                base_confidence = candidate.similarity_score
                
                # Boost confidence based on metadata
                metadata_boost = self._calculate_metadata_boost(candidate, request)
                
                # Final confidence score
                confidence_scores[candidate.document.id] = base_confidence + metadata_boost
            
            # Select candidate with highest confidence
            best_prompt_id = max(confidence_scores, key=confidence_scores.get)
            best_candidate = next(c for c in candidates if c.document.id == best_prompt_id)
            
            return PromptRoutingResult(
                selected_prompt_id=best_prompt_id,
                confidence_score=confidence_scores[best_prompt_id],
                routing_strategy=RoutingStrategy.CONFIDENCE_BASED,
                vector_score=best_candidate.similarity_score,
                alternatives=[
                    {
                        "prompt_id": c.document.id,
                        "confidence_score": confidence_scores.get(c.document.id, 0),
                        "name": c.document.payload.get("name", "Unknown") if c.document.payload else "Unknown"
                    }
                    for c in candidates[:3] if c.document.id != best_prompt_id
                ]
            )
            
        except Exception as e:
            # Fall back to vector-only routing if confidence-based fails
            print(f"Warning: Confidence-based routing failed, falling back to vector-only: {str(e)}")
            return await self._route_vector_only(request, candidates)
    
    async def _calculate_reasoning_scores(
        self, 
        request: PromptRoutingRequest, 
        candidates: List[SimilarityResult]
    ) -> Dict[str, float]:
        """Calculate reasoning scores for candidates (placeholder for Phase 3)."""
        # For now, return neutral scores
        # In Phase 3, this will use Neo4j for sophisticated reasoning
        reasoning_scores = {}
        for candidate in candidates:
            prompt_id = candidate.document.id
            payload = candidate.document.payload or {}
            
            # Simple heuristics for now
            score = 0.5  # Neutral base score
            
            # Boost for system prompts
            if payload.get("is_system", False):
                score += 0.1
            
            # Boost for matching category
            if payload.get("category") and request.user_context.get("topic"):
                if payload["category"].lower() in request.user_context["topic"].lower():
                    score += 0.1
            
            # Boost for matching tags
            if payload.get("tags") and request.user_context.get("interests"):
                common_tags = set(payload["tags"]) & set(request.user_context.get("interests", []))
                score += 0.05 * len(common_tags)
            
            reasoning_scores[prompt_id] = min(1.0, score)
        
        return reasoning_scores
    
    def _calculate_metadata_boost(
        self, 
        candidate: SimilarityResult, 
        request: PromptRoutingRequest
    ) -> float:
        """Calculate confidence boost based on metadata."""
        payload = candidate.document.payload or {}
        boost = 0.0
        
        # Boost for system prompts
        if payload.get("is_system", False):
            boost += 0.1
        
        # Boost for recent updates
        if payload.get("updated_at"):
            # This would need proper date parsing in production
            boost += 0.05
        
        # Boost for cross-app prompts (if enabled)
        if payload.get("cross_app") and request.include_cross_app:
            boost += 0.05
        
        return min(0.2, boost)  # Cap boost at 0.2
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics."""
        success_rate = (self.successful_routings / self.total_routing_requests) if self.total_routing_requests > 0 else 0
        
        return {
            "total_requests": self.total_routing_requests,
            "successful_routings": self.successful_routings,
            "success_rate": success_rate,
            "cross_app_usage_count": self.cross_app_usage_count,
            "cross_app_usage_rate": (self.cross_app_usage_count / self.successful_routings) if self.successful_routings > 0 else 0
        }
