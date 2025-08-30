#!/usr/bin/env python3
"""
Filepath: pcs/demo_unified_engine.py
Purpose: Demo script showing the unified prompts engine in action
Related Components: Unified prompts engine, intelligent router, Neo4j service
Tags: demo, unified-engine, prompts, vector-search, reasoning
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pcs.services.unified_prompts_engine import (
    UnifiedPromptsEngine,
    UnifiedPromptRequest,
    CrossAppMode
)
from pcs.services.intelligent_prompt_router import (
    IntelligentPromptRouter,
    RoutingStrategy
)
from pcs.services.neo4j_service import PCSNeo4jService
from pcs.services.prompt_service import PromptGenerator


async def demo_unified_engine():
    """Demonstrate the unified prompts engine."""
    print("🚀 PCS Unified Prompts Engine Demo")
    print("=" * 50)
    
    try:
        # Initialize services (with mocked dependencies for demo)
        print("\n1. Initializing services...")
        
        # Mock prompt generator
        prompt_generator = MockPromptGenerator()
        
        # Mock intelligent router
        intelligent_router = MockIntelligentRouter()
        
        # Mock Neo4j service
        neo4j_service = MockNeo4jService()
        
        # Create unified engine
        unified_engine = UnifiedPromptsEngine(
            prompt_generator=prompt_generator,
            intelligent_router=intelligent_router,
            neo4j_service=neo4j_service,
            enable_cross_app=True,
            cross_app_mode=CrossAppMode.INTELLIGENT
        )
        
        print("✅ Services initialized successfully!")
        
        # Demo 1: Basic prompt generation
        print("\n2. Demo 1: Basic Prompt Generation")
        print("-" * 40)
        
        request = UnifiedPromptRequest(
            query_text="How do I implement user authentication?",
            app_id="beep_boop",
            user_context={
                "topic": "security",
                "level": "beginner",
                "framework": "python",
                "interests": ["web development", "security"]
            }
        )
        
        result = await unified_engine.generate_unified_prompt(request)
        
        print(f"Query: {request.query_text}")
        print(f"Selected Prompt ID: {result.prompt_id}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Routing Strategy: {result.routing_strategy.value}")
        print(f"Reasoning Chain ID: {result.reasoning_chain_id}")
        print(f"Cross-App Source: {result.cross_app_source or 'None'}")
        print(f"Vector Score: {result.vector_score:.2f}")
        print(f"Reasoning Score: {result.reasoning_score:.2f}")
        
        if result.alternatives:
            print("\nAlternative Prompts:")
            for alt in result.alternatives:
                print(f"  - {alt.get('name', 'Unknown')}: {alt.get('score', 0):.2f}")
        
        if result.optimization_suggestions:
            print("\nOptimization Suggestions:")
            for suggestion in result.optimization_suggestions:
                print(f"  - {suggestion}")
        
        # Demo 2: Cross-app insights
        print("\n3. Demo 2: Cross-Application Insights")
        print("-" * 40)
        
        insights = await unified_engine.get_cross_app_insights("authentication")
        if insights:
            print("Cross-app insights found:")
            for insight in insights:
                print(f"  - App: {insight['source_app']}")
                print(f"    Approach: {insight['reasoning_approach']}")
                print(f"    Success Count: {insight['success_count']}")
        else:
            print("No cross-app insights available yet.")
        
        # Demo 3: Engine statistics
        print("\n4. Demo 3: Engine Statistics")
        print("-" * 40)
        
        stats = await unified_engine.get_engine_statistics()
        print("Engine Statistics:")
        print(f"  Total Requests: {stats['engine']['total_requests']}")
        print(f"  Successful Requests: {stats['engine']['successful_requests']}")
        print(f"  Success Rate: {stats['engine']['success_rate']:.2%}")
        print(f"  Cross-App Usage: {stats['engine']['cross_app_usage_count']}")
        print(f"  Reasoning Chains: {stats['engine']['reasoning_chain_count']}")
        
        # Demo 4: Health check
        print("\n5. Demo 4: Health Check")
        print("-" * 40)
        
        health = await unified_engine.health_check()
        print(f"Overall Status: {health['status']}")
        print("Component Status:")
        for component, status_info in health['components'].items():
            print(f"  {component}: {status_info['status']}")
        
        # Demo 5: Cross-app sharing modes
        print("\n6. Demo 5: Cross-App Sharing Modes")
        print("-" * 40)
        
        modes = [
            CrossAppMode.DISABLED,
            CrossAppMode.READ_ONLY,
            CrossAppMode.SHARED,
            CrossAppMode.INTELLIGENT
        ]
        
        for mode in modes:
            success = await unified_engine.enable_cross_app_sharing(mode)
            print(f"  {mode.value}: {'✅' if success else '❌'}")
        
        # Reset to intelligent mode
        await unified_engine.enable_cross_app_sharing(CrossAppMode.INTELLIGENT)
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Vector-based prompt discovery")
        print("  ✅ Intelligent routing strategies")
        print("  ✅ Reasoning chain tracking")
        print("  ✅ Cross-application intelligence")
        print("  ✅ Optimization suggestions")
        print("  ✅ Health monitoring")
        print("  ✅ Configurable cross-app sharing")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


class MockPromptGenerator:
    """Mock prompt generator for demo purposes."""
    
    async def generate_prompt(self, request):
        """Mock prompt generation."""
        from pcs.services.prompt_service import PromptResponse, PromptStatus
        
        return PromptResponse(
            request_id=request.request_id,
            status=PromptStatus.COMPLETED,
            generated_prompt=f"Generated prompt for: {request.template_name}",
            processing_time_ms=150.0
        )
    
    async def health_check(self):
        """Mock health check."""
        return {"status": "healthy"}


class MockIntelligentRouter:
    """Mock intelligent router for demo purposes."""
    
    def __init__(self):
        self.enable_cross_app = True
        self.total_routing_requests = 0
        self.successful_routings = 0
        self.cross_app_usage_count = 0
    
    async def route_to_optimal_prompt(self, request):
        """Mock prompt routing."""
        from pcs.services.intelligent_prompt_router import PromptRoutingResult
        
        self.total_routing_requests += 1
        self.successful_routings += 1
        
        return PromptRoutingResult(
            selected_prompt_id="prompt_123",
            confidence_score=0.85,
            routing_strategy=RoutingStrategy.HYBRID,
            vector_score=0.8,
            reasoning_score=0.9,
            alternatives=[
                {"prompt_id": "prompt_456", "score": 0.75, "name": "Alternative 1"},
                {"prompt_id": "prompt_789", "score": 0.65, "name": "Alternative 2"}
            ]
        )
    
    async def get_routing_statistics(self):
        """Mock routing statistics."""
        return {
            "total_requests": self.total_routing_requests,
            "successful_routings": self.successful_routings,
            "success_rate": (self.successful_routings / self.total_routing_requests) if self.total_routing_requests > 0 else 0,
            "cross_app_usage_count": self.cross_app_usage_count
        }


class MockNeo4jService:
    """Mock Neo4j service for demo purposes."""
    
    def __init__(self):
        self.total_operations = 0
        self.successful_operations = 0
    
    async def track_reasoning_chain(self, **kwargs):
        """Mock reasoning chain tracking."""
        self.total_operations += 1
        self.successful_operations += 1
        return f"reasoning_{self.successful_operations}"
    
    async def analyze_success_patterns(self, app_id, time_window_days=30):
        """Mock success pattern analysis."""
        return [
            {
                "reasoning_approach": "hybrid_vector_reasoning",
                "prompt_id": "prompt_456",
                "usage_count": 15,
                "success_indicator": 15
            }
        ]
    
    async def find_cross_app_insights(self, query_intent, current_app_id, limit=5):
        """Mock cross-app insights."""
        return [
            {
                "source_app": "lernmi",
                "reasoning_approach": "success_pattern_replication",
                "prompt_id": "prompt_789",
                "success_count": 12
            }
        ]
    
    async def get_reasoning_statistics(self):
        """Mock reasoning statistics."""
        return {
            "node_counts": {"Query": 100, "Reasoning": 50, "Prompt": 25},
            "relationship_counts": {"REQUIRES_REASONING": 50, "GENERATES_PROMPT": 25},
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "success_rate": (self.successful_operations / self.total_operations) if self.total_operations > 0 else 0
        }


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_unified_engine())
