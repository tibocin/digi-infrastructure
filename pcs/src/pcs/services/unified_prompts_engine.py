"""
Filepath: pcs/src/pcs/services/unified_prompts_engine.py
Purpose: Unified prompts engine that combines vector search, reasoning, and cross-application intelligence
Related Components: Intelligent prompt router, Neo4j service, Qdrant repository, prompt generator
Tags: unified-engine, cross-app, intelligence, vector-search, reasoning, prompt-optimization
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import asyncio
import logging
from uuid import uuid4

from ..core.exceptions import PCSError
from .intelligent_prompt_router import IntelligentPromptRouter, PromptRoutingRequest, PromptRoutingResult, RoutingStrategy
from .neo4j_service import PCSNeo4jService, ReasoningType
from .prompt_service import PromptGenerator


class UnifiedPromptsError(PCSError):
    """Custom exception for unified prompts engine errors."""
    pass


class CrossAppMode(Enum):
    """Different modes for cross-application prompt sharing."""
    DISABLED = "disabled"           # No cross-app prompts
    READ_ONLY = "read_only"         # Can read from other apps but not share
    SHARED = "shared"               # Full cross-app sharing
    INTELLIGENT = "intelligent"     # AI-driven cross-app selection


@dataclass
class UnifiedPromptRequest:
    """Request for unified prompt generation."""
    query_text: str
    app_id: str
    user_context: Dict[str, Any]
    routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID
    cross_app_mode: CrossAppMode = CrossAppMode.INTELLIGENT
    max_candidates: int = 5
    similarity_threshold: float = 0.7
    include_reasoning: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UnifiedPromptResult:
    """Result of unified prompt generation."""
    prompt_id: str
    prompt_content: str
    confidence_score: float
    routing_strategy: RoutingStrategy
    reasoning_chain_id: Optional[str] = None
    cross_app_source: Optional[str] = None
    vector_score: Optional[float] = None
    reasoning_score: Optional[float] = None
    alternatives: List[Dict[str, Any]] = None
    optimization_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []


class UnifiedPromptsEngine:
    """
    Unified prompts engine that provides intelligent, cross-application prompt selection.
    
    This engine combines:
    1. Vector-based prompt discovery
    2. Reasoning-based prompt selection
    3. Cross-application intelligence
    4. Continuous learning and optimization
    """
    
    def __init__(
        self,
        prompt_generator: PromptGenerator,
        intelligent_router: IntelligentPromptRouter,
        neo4j_service: PCSNeo4jService,
        enable_cross_app: bool = True,
        cross_app_mode: CrossAppMode = CrossAppMode.INTELLIGENT
    ):
        """
        Initialize the unified prompts engine.
        
        Args:
            prompt_generator: Core prompt generator service
            intelligent_router: Intelligent prompt router
            neo4j_service: Neo4j service for reasoning
            enable_cross_app: Whether to enable cross-application features
            cross_app_mode: Mode for cross-application sharing
        """
        self.prompt_generator = prompt_generator
        self.intelligent_router = intelligent_router
        self.neo4j_service = neo4j_service
        self.enable_cross_app = enable_cross_app
        self.cross_app_mode = cross_app_mode
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.cross_app_usage_count = 0
        self.reasoning_chain_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    async def generate_unified_prompt(
        self, 
        request: UnifiedPromptRequest
    ) -> UnifiedPromptResult:
        """
        Generate a prompt using the unified engine.
        
        Args:
            request: Unified prompt request
            
        Returns:
            Unified prompt result with reasoning and optimization
        """
        self.total_requests += 1
        
        try:
            # Step 1: Route to optimal prompt
            routing_result = await self._route_to_optimal_prompt(request)
            
            # Step 2: Generate the actual prompt content
            prompt_content = await self._generate_prompt_content(
                routing_result.selected_prompt_id,
                request.user_context
            )
            
            # Step 3: Track reasoning chain if enabled
            reasoning_chain_id = None
            if request.include_reasoning:
                reasoning_chain_id = await self._track_reasoning_chain(
                    request, routing_result, prompt_content
                )
            
            # Step 4: Generate optimization suggestions
            optimization_suggestions = await self._generate_optimization_suggestions(
                request, routing_result
            )
            
            # Step 5: Create unified result
            result = UnifiedPromptResult(
                prompt_id=routing_result.selected_prompt_id,
                prompt_content=prompt_content,
                confidence_score=routing_result.confidence_score,
                routing_strategy=routing_result.routing_strategy,
                reasoning_chain_id=reasoning_chain_id,
                cross_app_source=routing_result.cross_app_source,
                vector_score=routing_result.vector_score,
                reasoning_score=routing_result.reasoning_score,
                alternatives=routing_result.alternatives,
                optimization_suggestions=optimization_suggestions
            )
            
            # Update statistics
            self.successful_requests += 1
            if result.cross_app_source:
                self.cross_app_usage_count += 1
            if reasoning_chain_id:
                self.reasoning_chain_count += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate unified prompt: {str(e)}")
            raise UnifiedPromptsError(f"Unified prompt generation failed: {str(e)}") from e
    
    async def _route_to_optimal_prompt(
        self, 
        request: UnifiedPromptRequest
    ) -> PromptRoutingResult:
        """Route to the optimal prompt using the intelligent router."""
        try:
            # Prepare routing request
            routing_request = PromptRoutingRequest(
                query_text=request.query_text,
                app_id=request.app_id,
                user_context=request.user_context,
                routing_strategy=request.routing_strategy,
                max_candidates=request.max_candidates,
                similarity_threshold=request.similarity_threshold,
                include_cross_app=(request.cross_app_mode != CrossAppMode.DISABLED),
                metadata=request.metadata
            )
            
            # Route to optimal prompt
            routing_result = await self.intelligent_router.route_to_optimal_prompt(routing_request)
            
            return routing_result
            
        except Exception as e:
            raise UnifiedPromptsError(f"Prompt routing failed: {str(e)}") from e
    
    async def _generate_prompt_content(
        self, 
        prompt_id: str, 
        user_context: Dict[str, Any]
    ) -> str:
        """Generate the actual prompt content from the selected template."""
        try:
            # For now, we'll use a placeholder approach
            # In production, this would use the actual prompt template system
            
            # Create a simple prompt request
            from .prompt_service import PromptRequest, OptimizationLevel
            
            prompt_request = PromptRequest(
                request_id=str(uuid4()),
                template_name=prompt_id,  # Use prompt_id as template name for now
                context_data=user_context,
                optimization_level=OptimizationLevel.BASIC
            )
            
            # Generate prompt using the core generator
            response = await self.prompt_generator.generate_prompt(prompt_request)
            
            if response.status.value == "completed":
                return response.generated_prompt
            else:
                # Fallback to a simple template
                return f"Based on your request and context: {user_context.get('query', 'Please provide more details')}"
                
        except Exception as e:
            self.logger.warning(f"Failed to generate prompt content, using fallback: {str(e)}")
            # Fallback to simple template
            return f"Please help with: {user_context.get('query', 'your request')}"
    
    async def _track_reasoning_chain(
        self,
        request: UnifiedPromptRequest,
        routing_result: PromptRoutingResult,
        prompt_content: str
    ) -> Optional[str]:
        """Track the reasoning chain in Neo4j."""
        try:
            # Determine reasoning approach based on routing strategy
            reasoning_approach = self._determine_reasoning_approach(routing_result.routing_strategy)
            
            # Track the reasoning chain
            reasoning_chain_id = await self.neo4j_service.track_reasoning_chain(
                query_text=request.query_text,
                reasoning_approach=reasoning_approach,
                prompt_id=routing_result.selected_prompt_id,
                app_id=request.app_id,
                user_context=request.user_context,
                confidence=routing_result.confidence_score
            )
            
            return reasoning_chain_id
            
        except Exception as e:
            self.logger.warning(f"Failed to track reasoning chain: {str(e)}")
            return None
    
    def _determine_reasoning_approach(self, routing_strategy: RoutingStrategy) -> str:
        """Determine the reasoning approach based on routing strategy."""
        if routing_strategy == RoutingStrategy.VECTOR_ONLY:
            return "vector_similarity"
        elif routing_strategy == RoutingStrategy.REASONING_ONLY:
            return "pure_reasoning"
        elif routing_strategy == RoutingStrategy.HYBRID:
            return "hybrid_vector_reasoning"
        elif routing_strategy == RoutingStrategy.CONFIDENCE_BASED:
            return "confidence_weighted"
        else:
            return "unknown"
    
    async def _generate_optimization_suggestions(
        self,
        request: UnifiedPromptRequest,
        routing_result: PromptRoutingResult
    ) -> List[str]:
        """Generate optimization suggestions based on the routing result."""
        suggestions = []
        
        try:
            # Analyze success patterns for the app
            success_patterns = await self.neo4j_service.analyze_success_patterns(
                app_id=request.app_id,
                time_window_days=30
            )
            
            # Generate suggestions based on patterns
            if success_patterns:
                top_pattern = success_patterns[0]
                if top_pattern["usage_count"] > 10:  # Significant usage
                    suggestions.append(
                        f"Consider using '{top_pattern['reasoning_approach']}' approach more frequently "
                        f"(success rate: {top_pattern['success_indicator']} uses)"
                    )
            
            # Cross-app optimization suggestions
            if request.cross_app_mode in [CrossAppMode.SHARED, CrossAppMode.INTELLIGENT]:
                cross_app_insights = await self.neo4j_service.find_cross_app_insights(
                    query_intent=request.query_text[:100],  # First 100 chars
                    current_app_id=request.app_id,
                    limit=3
                )
                
                for insight in cross_app_insights:
                    if insight["success_count"] > 5:  # Significant success
                        suggestions.append(
                            f"App '{insight['source_app']}' has success with '{insight['reasoning_approach']}' "
                            f"approach ({insight['success_count']} successful uses)"
                        )
            
            # Confidence-based suggestions
            if routing_result.confidence_score < 0.7:
                suggestions.append(
                    "Low confidence in prompt selection. Consider adding more specific context "
                    "or enabling cross-application prompts for better matches."
                )
            
            # Vector score suggestions
            if routing_result.vector_score and routing_result.vector_score < 0.8:
                suggestions.append(
                    "Vector similarity is below optimal threshold. Consider expanding prompt "
                    "templates or improving template descriptions for better matching."
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate optimization suggestions: {str(e)}")
            suggestions.append("Unable to generate optimization suggestions due to system error.")
        
        return suggestions
    
    async def enable_cross_app_sharing(
        self, 
        mode: CrossAppMode,
        target_apps: Optional[List[str]] = None
    ) -> bool:
        """
        Enable cross-application prompt sharing.
        
        Args:
            mode: Cross-app sharing mode
            target_apps: Specific apps to share with (optional)
            
        Returns:
            True if enabled successfully
        """
        try:
            self.cross_app_mode = mode
            self.enable_cross_app = (mode != CrossAppMode.DISABLED)
            
            # Update the intelligent router
            self.intelligent_router.enable_cross_app = self.enable_cross_app
            
            self.logger.info(f"Cross-app sharing enabled with mode: {mode.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable cross-app sharing: {str(e)}")
            return False
    
    async def get_cross_app_insights(
        self,
        query_intent: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get insights from other applications."""
        try:
            insights = await self.neo4j_service.find_cross_app_insights(
                query_intent=query_intent,
                current_app_id="pcs",  # Current app
                limit=limit
            )
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get cross-app insights: {str(e)}")
            return []
    
    async def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        try:
            # Get routing statistics
            routing_stats = await self.intelligent_router.get_routing_statistics()
            
            # Get Neo4j statistics
            neo4j_stats = await self.neo4j_service.get_reasoning_statistics()
            
            # Combine statistics
            stats = {
                "engine": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "success_rate": (self.successful_requests / self.total_requests) if self.total_requests > 0 else 0,
                    "cross_app_usage_count": self.cross_app_usage_count,
                    "reasoning_chain_count": self.reasoning_chain_count
                },
                "routing": routing_stats,
                "neo4j": neo4j_stats,
                "cross_app_mode": self.cross_app_mode.value,
                "cross_app_enabled": self.enable_cross_app
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get engine statistics: {str(e)}")
            return {
                "error": str(e),
                "engine": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests
                }
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the unified engine."""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Check Neo4j service
            neo4j_stats = await self.neo4j_service.get_reasoning_statistics()
            health["components"]["neo4j"] = {
                "status": "healthy" if "error" not in neo4j_stats else "unhealthy",
                "details": neo4j_stats
            }
            
            # Check intelligent router
            router_stats = await self.intelligent_router.get_routing_statistics()
            health["components"]["intelligent_router"] = {
                "status": "healthy" if router_stats.get("success_rate", 0) > 0 else "degraded",
                "details": router_stats
            }
            
            # Check prompt generator
            prompt_health = await self.prompt_generator.health_check()
            health["components"]["prompt_generator"] = prompt_health
            
        except Exception as e:
            health["components"]["error"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health status
        component_statuses = [comp.get("status", "unknown") for comp in health["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health["status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
        
        return health
