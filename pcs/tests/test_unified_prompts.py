"""
Filepath: pcs/tests/test_unified_prompts.py
Purpose: Test the unified prompts engine functionality
Related Components: Unified prompts engine, intelligent router, Neo4j service
Tags: testing, unified-engine, prompts, vector-search, reasoning
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from pcs.services.unified_prompts_engine import (
    UnifiedPromptsEngine,
    UnifiedPromptRequest,
    CrossAppMode
)
from pcs.services.intelligent_prompt_router import (
    IntelligentPromptRouter,
    PromptRoutingResult,
    RoutingStrategy
)
from pcs.services.neo4j_service import PCSNeo4jService


@pytest.fixture
def mock_prompt_generator():
    """Mock prompt generator."""
    generator = Mock()
    generator.generate_prompt = AsyncMock()
    generator.health_check = AsyncMock(return_value={"status": "healthy"})
    return generator


@pytest.fixture
def mock_intelligent_router():
    """Mock intelligent router."""
    router = Mock()
    router.route_to_optimal_prompt = AsyncMock()
    router.get_routing_statistics = AsyncMock(return_value={
        "total_requests": 10,
        "successful_routings": 9,
        "success_rate": 0.9
    })
    return router


@pytest.fixture
def mock_neo4j_service():
    """Mock Neo4j service."""
    service = Mock()
    service.track_reasoning_chain = AsyncMock(return_value="reasoning_123")
    service.analyze_success_patterns = AsyncMock(return_value=[
        {
            "reasoning_approach": "hybrid_vector_reasoning",
            "prompt_id": "prompt_456",
            "usage_count": 15,
            "success_indicator": 15
        }
    ])
    service.find_cross_app_insights = AsyncMock(return_value=[])
    service.get_reasoning_statistics = AsyncMock(return_value={
        "node_counts": {"Query": 100, "Reasoning": 50, "Prompt": 25},
        "relationship_counts": {"REQUIRES_REASONING": 50, "GENERATES_PROMPT": 25}
    })
    return service


@pytest.fixture
def unified_engine(
    mock_prompt_generator,
    mock_intelligent_router,
    mock_neo4j_service
):
    """Create unified prompts engine with mocked dependencies."""
    return UnifiedPromptsEngine(
        prompt_generator=mock_prompt_generator,
        intelligent_router=mock_intelligent_router,
        neo4j_service=mock_neo4j_service,
        enable_cross_app=True,
        cross_app_mode=CrossAppMode.INTELLIGENT
    )


class TestUnifiedPromptsEngine:
    """Test the unified prompts engine."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, unified_engine):
        """Test engine initialization."""
        assert unified_engine.enable_cross_app is True
        assert unified_engine.cross_app_mode == CrossAppMode.INTELLIGENT
        assert unified_engine.total_requests == 0
        assert unified_engine.successful_requests == 0
    
    @pytest.mark.asyncio
    async def test_generate_unified_prompt_success(
        self,
        unified_engine,
        mock_intelligent_router,
        mock_prompt_generator
    ):
        """Test successful prompt generation."""
        # Setup mocks
        routing_result = PromptRoutingResult(
            selected_prompt_id="prompt_123",
            confidence_score=0.85,
            routing_strategy=RoutingStrategy.HYBRID,
            vector_score=0.8,
            reasoning_score=0.9
        )
        
        mock_intelligent_router.route_to_optimal_prompt.return_value = routing_result
        
        # Mock prompt generation
        mock_prompt_generator.generate_prompt.return_value = Mock(
            status=Mock(value="completed"),
            generated_prompt="Generated prompt content"
        )
        
        # Create request
        request = UnifiedPromptRequest(
            query_text="How do I implement authentication?",
            app_id="test_app",
            user_context={"topic": "security", "level": "beginner"}
        )
        
        # Generate prompt
        result = await unified_engine.generate_unified_prompt(request)
        
        # Verify result
        assert result.prompt_id == "prompt_123"
        assert result.prompt_content == "Generated prompt content"
        assert result.confidence_score == 0.85
        assert result.routing_strategy == RoutingStrategy.HYBRID
        assert result.reasoning_chain_id == "reasoning_123"
        assert result.vector_score == 0.8
        assert result.reasoning_score == 0.9
        
        # Verify statistics
        assert unified_engine.total_requests == 1
        assert unified_engine.successful_requests == 1
        assert unified_engine.reasoning_chain_count == 1
    
    @pytest.mark.asyncio
    async def test_cross_app_sharing_enabled(self, unified_engine):
        """Test enabling cross-app sharing."""
        success = await unified_engine.enable_cross_app_sharing(CrossAppMode.SHARED)
        assert success is True
        assert unified_engine.cross_app_mode == CrossAppMode.SHARED
        assert unified_engine.enable_cross_app is True
    
    @pytest.mark.asyncio
    async def test_cross_app_sharing_disabled(self, unified_engine):
        """Test disabling cross-app sharing."""
        success = await unified_engine.enable_cross_app_sharing(CrossAppMode.DISABLED)
        assert success is True
        assert unified_engine.cross_app_mode == CrossAppMode.DISABLED
        assert unified_engine.enable_cross_app is False
    
    @pytest.mark.asyncio
    async def test_get_engine_statistics(self, unified_engine):
        """Test getting engine statistics."""
        stats = await unified_engine.get_engine_statistics()
        
        assert "engine" in stats
        assert "routing" in stats
        assert "neo4j" in stats
        
        engine_stats = stats["engine"]
        assert engine_stats["total_requests"] == 0
        assert engine_stats["successful_requests"] == 0
        assert engine_stats["cross_app_usage_count"] == 0
        assert engine_stats["reasoning_chain_count"] == 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, unified_engine):
        """Test health check functionality."""
        health = await unified_engine.health_check()
        
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health
        assert "timestamp" in health
        
        # Verify components are checked
        assert "neo4j" in health["components"]
        assert "intelligent_router" in health["components"]
        assert "prompt_generator" in health["components"]
    
    @pytest.mark.asyncio
    async def test_get_cross_app_insights(self, unified_engine, mock_neo4j_service):
        """Test getting cross-app insights."""
        insights = await unified_engine.get_cross_app_insights("authentication")
        assert isinstance(insights, list)
        
        # Verify Neo4j service was called
        mock_neo4j_service.find_cross_app_insights.assert_called_once_with(
            query_intent="authentication",
            current_app_id="pcs",
            limit=5
        )


class TestUnifiedPromptRequest:
    """Test the unified prompt request."""
    
    def test_request_initialization(self):
        """Test request initialization."""
        request = UnifiedPromptRequest(
            query_text="Test query",
            app_id="test_app",
            user_context={"key": "value"}
        )
        
        assert request.query_text == "Test query"
        assert request.app_id == "test_app"
        assert request.user_context == {"key": "value"}
        assert request.routing_strategy == RoutingStrategy.HYBRID
        assert request.cross_app_mode == CrossAppMode.INTELLIGENT
        assert request.max_candidates == 5
        assert request.similarity_threshold == 0.7
        assert request.include_reasoning is True
        assert request.metadata == {}


class TestCrossAppMode:
    """Test cross-app mode enum."""
    
    def test_modes(self):
        """Test all cross-app modes."""
        modes = list(CrossAppMode)
        assert len(modes) == 4
        
        assert CrossAppMode.DISABLED in modes
        assert CrossAppMode.READ_ONLY in modes
        assert CrossAppMode.SHARED in modes
        assert CrossAppMode.INTELLIGENT in modes
    
    def test_mode_values(self):
        """Test mode string values."""
        assert CrossAppMode.DISABLED.value == "disabled"
        assert CrossAppMode.READ_ONLY.value == "read_only"
        assert CrossAppMode.SHARED.value == "shared"
        assert CrossAppMode.INTELLIGENT.value == "intelligent"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
