"""
Integration tests for Prompt Service.

Tests the complete prompt generation pipeline including template engine,
rule engine, context manager, and caching working together.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from uuid import uuid4

import sys
from pathlib import Path

# Add src to path for direct imports without triggering main app initialization
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pcs.services.prompt_service import (
    PromptGenerator,
    PromptCache,
    PromptOptimizer,
    PromptRequest,
    PromptResponse,
    PromptStatus,
    OptimizationLevel,
    PromptError
)
from pcs.services.template_service import TemplateEngine
from pcs.services.rule_engine import RuleEngine
from pcs.services.context_service import (
    ContextManager, Context, ContextType, ContextScope, ContextMetadata
)
from pcs.repositories.redis_repo import RedisRepository


class TestPromptRequest:
    """Test PromptRequest functionality."""
    
    def test_create_prompt_request(self):
        """Test creating a prompt request."""
        request = PromptRequest(
            request_id="req-123",
            template_name="greeting",
            context_data={"name": "Alice"},
            variables={"score": 95},
            optimization_level=OptimizationLevel.BASIC
        )
        
        assert request.request_id == "req-123"
        assert request.template_name == "greeting"
        assert request.context_data == {"name": "Alice"}
        assert request.variables == {"score": 95}
        assert request.optimization_level == OptimizationLevel.BASIC
    
    def test_prompt_request_defaults(self):
        """Test prompt request with default values."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test"
        )
        
        assert request.context_data == {}
        assert request.context_ids == []
        assert request.rule_names == []
        assert request.variables == {}
        assert request.metadata == {}
        assert request.optimization_level == OptimizationLevel.BASIC
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        request1 = PromptRequest(
            request_id="req-123",
            template_name="greeting",
            context_data={"name": "Alice"},
            variables={"score": 95}
        )
        
        request2 = PromptRequest(
            request_id="req-456",  # Different request ID
            template_name="greeting",
            context_data={"name": "Alice"},
            variables={"score": 95}
        )
        
        # Cache key should be the same for requests with same content
        # but different request IDs
        key1 = request1.get_cache_key()
        key2 = request2.get_cache_key()
        assert key1 == key2
        
        # Different content should produce different cache keys
        request3 = PromptRequest(
            request_id="req-789",
            template_name="greeting",
            context_data={"name": "Bob"},  # Different data
            variables={"score": 95}
        )
        
        key3 = request3.get_cache_key()
        assert key1 != key3


class TestPromptResponse:
    """Test PromptResponse functionality."""
    
    def test_create_prompt_response(self):
        """Test creating a prompt response."""
        response = PromptResponse(
            request_id="req-123",
            status=PromptStatus.COMPLETED,
            generated_prompt="Hello Alice!",
            processing_time_ms=150.5,
            cache_hit=False
        )
        
        assert response.request_id == "req-123"
        assert response.status == PromptStatus.COMPLETED
        assert response.generated_prompt == "Hello Alice!"
        assert response.processing_time_ms == 150.5
        assert response.cache_hit is False
    
    def test_prompt_response_to_dict(self):
        """Test converting response to dictionary."""
        response = PromptResponse(
            request_id="req-123",
            status=PromptStatus.COMPLETED,
            generated_prompt="Hello Alice!",
            template_variables={"name", "greeting"}
        )
        
        data = response.to_dict()
        
        assert data["request_id"] == "req-123"
        assert data["status"] == "completed"
        assert data["generated_prompt"] == "Hello Alice!"
        assert set(data["template_variables"]) == {"name", "greeting"}


class TestPromptCache:
    """Test PromptCache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = Mock(spec=RedisRepository)
        self.cache = PromptCache(self.mock_redis)
    
    @pytest.mark.asyncio
    async def test_get_cached_prompt_hit(self):
        """Test cache hit scenario."""
        request = PromptRequest(request_id="req-123", template_name="test")
        cached_data = {
            "request_id": "req-123",
            "status": "completed",
            "generated_prompt": "Cached result",
            "cache_hit": False
        }
        
        self.mock_redis.get = AsyncMock(return_value=cached_data)
        
        result = await self.cache.get_cached_prompt(request)
        
        assert result is not None
        assert result.request_id == "req-123"
        assert result.generated_prompt == "Cached result"
        assert result.cache_hit is True  # Should be set to True when retrieved from cache
        assert self.cache.cache_hits == 1
        assert self.cache.cache_misses == 0
    
    @pytest.mark.asyncio
    async def test_get_cached_prompt_miss(self):
        """Test cache miss scenario."""
        request = PromptRequest(request_id="req-123", template_name="test")
        
        self.mock_redis.get = AsyncMock(return_value=None)
        
        result = await self.cache.get_cached_prompt(request)
        
        assert result is None
        assert self.cache.cache_hits == 0
        assert self.cache.cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_prompt(self):
        """Test caching a prompt response."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            cache_ttl=timedelta(hours=1)
        )
        
        response = PromptResponse(
            request_id="req-123",
            status=PromptStatus.COMPLETED,
            generated_prompt="Test result"
        )
        
        self.mock_redis.set = AsyncMock(return_value=True)
        
        result = await self.cache.cache_prompt(request, response)
        
        assert result is True
        self.mock_redis.set.assert_called_once()
        
        # Check the call arguments
        call_args = self.mock_redis.set.call_args
        assert call_args[1]['ttl'] == timedelta(hours=1)
    
    def test_get_cache_stats(self):
        """Test cache statistics."""
        # Simulate some cache operations
        self.cache.cache_hits = 7
        self.cache.cache_misses = 3
        
        stats = self.cache.get_cache_stats()
        
        assert stats['cache_hits'] == 7
        assert stats['cache_misses'] == 3
        assert stats['total_requests'] == 10
        assert stats['hit_rate'] == 0.7


class TestPromptOptimizer:
    """Test PromptOptimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PromptOptimizer()
        self.mock_context_manager = Mock(spec=ContextManager)
    
    @pytest.mark.asyncio
    async def test_optimize_none_level(self):
        """Test no optimization."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            optimization_level=OptimizationLevel.NONE
        )
        
        result = await self.optimizer.optimize_request(request, self.mock_context_manager)
        
        # Should return the same request unchanged
        assert result is request
    
    @pytest.mark.asyncio
    async def test_optimize_basic_level(self):
        """Test basic optimization with context merging."""
        # Create mock merged context
        now = datetime.utcnow()
        metadata = ContextMetadata(created_at=now, updated_at=now)
        merged_context = Context(
            context_id="merged",
            context_type=ContextType.USER,
            scope=ContextScope.PRIVATE,
            data={"merged": "data", "user": "Alice"},
            metadata=metadata
        )
        
        self.mock_context_manager.merge_contexts = AsyncMock(return_value=merged_context)
        
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            context_ids=["ctx1", "ctx2"],
            context_data={"existing": "data"},
            optimization_level=OptimizationLevel.BASIC
        )
        
        result = await self.optimizer.optimize_request(request, self.mock_context_manager)
        
        # Should merge contexts and update context_data
        assert result.context_data["existing"] == "data"  # Preserved
        assert result.context_data["merged"] == "data"  # From merged context
        assert result.context_data["user"] == "Alice"  # From merged context
        assert result.context_ids == []  # Should be cleared
    
    @pytest.mark.asyncio
    async def test_optimize_aggressive_level(self):
        """Test aggressive optimization."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        result = await self.optimizer.optimize_request(request, self.mock_context_manager)
        
        # Should set longer cache TTL
        assert result.cache_ttl == timedelta(hours=24)


class TestPromptGenerator:
    """Test complete PromptGenerator integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create real instances with minimal configuration
        self.template_engine = TemplateEngine(enable_cache=False)
        self.rule_engine = RuleEngine()
        
        # Mock components that require external dependencies
        self.mock_redis_repo = Mock(spec=RedisRepository)
        self.mock_context_manager = Mock(spec=ContextManager)
        
        # Create prompt generator
        self.generator = PromptGenerator(
            template_engine=self.template_engine,
            rule_engine=self.rule_engine,
            context_manager=self.mock_context_manager,
            redis_repo=self.mock_redis_repo
        )
        
        # Mock cache operations
        self.generator.cache.get_cached_prompt = AsyncMock(return_value=None)
        self.generator.cache.cache_prompt = AsyncMock(return_value=True)
    
    @pytest.mark.asyncio
    async def test_generate_simple_prompt(self):
        """Test generating a simple prompt without rules or contexts."""
        request = PromptRequest(
            request_id="req-123",
            template_name="greeting",
            template_content="Hello {{ name }}! Welcome to {{ app }}.",
            variables={"name": "Alice", "app": "PCS"},
            optimization_level=OptimizationLevel.NONE  # No caching
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.status == PromptStatus.COMPLETED
        assert response.generated_prompt == "Hello Alice! Welcome to PCS."
        assert response.request_id == "req-123"
        assert response.processing_time_ms is not None
        assert response.cache_hit is False
    
    @pytest.mark.asyncio
    async def test_generate_prompt_with_context(self):
        """Test generating prompt with context data."""
        # Mock context manager to return some context
        self.mock_context_manager.get_context = AsyncMock(return_value=None)
        
        request = PromptRequest(
            request_id="req-123",
            template_name="notification",
            template_content="{{ user.name }}, you have {{ count }} new {{ item_type }}(s).",
            context_data={"user": {"name": "Bob"}},
            variables={"count": 3, "item_type": "message"},
            optimization_level=OptimizationLevel.NONE
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.status == PromptStatus.COMPLETED
        assert response.generated_prompt == "Bob, you have 3 new message(s)."
    
    @pytest.mark.asyncio
    async def test_generate_prompt_with_rules(self):
        """Test generating prompt with rule evaluation."""
        # Add a rule to the engine
        rule_def = {
            "name": "high_score_rule",
            "description": "Set grade for high scores",
            "condition": {
                "field": "score",
                "operator": "gt",
                "value": 90
            },
            "actions": [
                {
                    "type": "set_variable",
                    "parameters": {"name": "grade", "value": "A"}
                }
            ]
        }
        
        self.rule_engine.add_rule(rule_def)
        
        request = PromptRequest(
            request_id="req-123",
            template_name="result",
            template_content="Score: {{ score }}, Grade: {{ grade | default('N/A') }}",
            context_data={"score": 95},
            rule_names=["high_score_rule"],
            optimization_level=OptimizationLevel.NONE
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.status == PromptStatus.COMPLETED
        assert response.generated_prompt == "Score: 95, Grade: A"
        assert len(response.rules_applied) == 1
        assert response.rules_applied[0].rule_name == "high_score_rule"
        assert response.rules_applied[0].matched is True
    
    @pytest.mark.asyncio
    async def test_generate_prompt_with_cache_hit(self):
        """Test generating prompt with cache hit."""
        cached_response = PromptResponse(
            request_id="req-123",
            status=PromptStatus.COMPLETED,
            generated_prompt="Cached result",
            cache_hit=True
        )
        
        self.generator.cache.get_cached_prompt = AsyncMock(return_value=cached_response)
        
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            optimization_level=OptimizationLevel.BASIC
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.generated_prompt == "Cached result"
        assert response.cache_hit is True
        # Processing time should be updated even for cached responses
        assert response.processing_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_generate_prompt_template_error(self):
        """Test handling template rendering errors."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            template_content="Hello {{ undefined_variable }}!",  # Will cause error
            optimization_level=OptimizationLevel.NONE
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.status == PromptStatus.FAILED
        assert response.error_message is not None
        assert "undefined variable" in response.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_generate_prompt_with_template_variables(self):
        """Test extracting template variables."""
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            template_content="Hello {{ name }}! Your score is {{ score }} and level is {{ level }}.",
            variables={"name": "Alice", "score": 95, "level": 5},
            optimization_level=OptimizationLevel.NONE
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert response.status == PromptStatus.COMPLETED
        expected_vars = {"name", "score", "level"}
        assert response.template_variables == expected_vars
    
    @pytest.mark.asyncio
    async def test_generate_prompt_performance_tracking(self):
        """Test performance tracking functionality."""
        initial_requests = self.generator.total_requests
        initial_time = self.generator.total_processing_time
        
        request = PromptRequest(
            request_id="req-123",
            template_name="test",
            template_content="Hello {{ name }}!",
            variables={"name": "Alice"},
            optimization_level=OptimizationLevel.NONE
        )
        
        response = await self.generator.generate_prompt(request)
        
        assert self.generator.total_requests == initial_requests + 1
        assert self.generator.total_processing_time > initial_time
        assert response.processing_time_ms is not None
        assert response.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        # Mock context manager operations for health check
        now = datetime.utcnow()
        metadata = ContextMetadata(created_at=now, updated_at=now)
        test_context = Context(
            context_id="health-test",
            context_type=ContextType.TEMPORARY,
            scope=ContextScope.PRIVATE,
            data={"test": "data"},
            metadata=metadata
        )
        
        self.mock_context_manager.create_context = AsyncMock(return_value=test_context)
        self.mock_context_manager.get_context = AsyncMock(return_value=test_context)
        self.mock_context_manager.delete_context = AsyncMock(return_value=True)
        
        health = await self.generator.health_check()
        
        assert health['status'] == 'healthy'
        assert 'template_engine' in health['components']
        assert 'rule_engine' in health['components']
        assert 'context_manager' in health['components']
        
        # All components should be healthy
        for component in health['components'].values():
            assert component['status'] == 'healthy'
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Simulate some requests
        self.generator.total_requests = 10
        self.generator.total_processing_time = 1500.0  # 1.5 seconds
        
        stats = self.generator.get_performance_stats()
        
        assert stats['total_requests'] == 10
        assert stats['total_processing_time_ms'] == 1500.0
        assert stats['average_processing_time_ms'] == 150.0
        assert 'cache_stats' in stats
    
    @pytest.mark.asyncio
    async def test_invalidate_cache(self):
        """Test cache invalidation."""
        self.generator.cache.invalidate_cache = AsyncMock(return_value=5)
        
        result = await self.generator.invalidate_cache("pattern*")
        
        assert result == 5
        self.generator.cache.invalidate_cache.assert_called_once_with("pattern*")


class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""
    
    def setup_method(self):
        """Set up complete integration test environment."""
        # Use real components where possible
        self.template_engine = TemplateEngine(enable_cache=False)
        self.rule_engine = RuleEngine()
        
        # Mock external dependencies
        self.mock_redis_repo = Mock(spec=RedisRepository)
        self.mock_context_manager = Mock(spec=ContextManager)
        
        self.generator = PromptGenerator(
            template_engine=self.template_engine,
            rule_engine=self.rule_engine,
            context_manager=self.mock_context_manager,
            redis_repo=self.mock_redis_repo
        )
        
        # Disable caching for predictable tests
        self.generator.cache.get_cached_prompt = AsyncMock(return_value=None)
        self.generator.cache.cache_prompt = AsyncMock(return_value=True)
    
    @pytest.mark.asyncio
    async def test_complete_prompt_generation_workflow(self):
        """Test a complete prompt generation workflow with all components."""
        # Add rules to the engine
        welcome_rule = {
            "name": "vip_welcome",
            "description": "Special welcome for VIP users",
            "condition": {
                "field": "user.tier",
                "operator": "eq",
                "value": "VIP"
            },
            "actions": [
                {
                    "type": "set_variable",
                    "parameters": {"name": "greeting_style", "value": "premium"}
                }
            ]
        }
        
        score_rule = {
            "name": "achievement_unlock",
            "description": "Unlock achievement for high scores",
            "condition": {
                "field": "score",
                "operator": "gte",
                "value": 100
            },
            "actions": [
                {
                    "type": "add_context",
                    "parameters": {
                        "data": {
                            "achievement": "Century Club",
                            "badge": "🏆"
                        }
                    }
                }
            ]
        }
        
        self.rule_engine.add_rule(welcome_rule)
        self.rule_engine.add_rule(score_rule)
        
        # Create prompt request
        request = PromptRequest(
            request_id="integration-test",
            template_name="user_dashboard",
            template_content="""
{%- if greeting_style == 'premium' -%}
🌟 Welcome back, {{ user.name }}! 
{%- else -%}
Hello {{ user.name }}!
{%- endif %}

Your current score: {{ score }}
{%- if achievement %}
🎉 New Achievement: {{ achievement }} {{ badge }}
{%- endif %}

{% if notifications %}
You have {{ notifications | length }} new notification(s):
{%- for notif in notifications %}
• {{ notif.message }}
{%- endfor %}
{%- endif %}
            """.strip(),
            context_data={
                "user": {"name": "Alice", "tier": "VIP"},
                "score": 150,
                "notifications": [
                    {"message": "Welcome to the platform!"},
                    {"message": "Your profile is 90% complete"}
                ]
            },
            rule_names=["vip_welcome", "achievement_unlock"],
            optimization_level=OptimizationLevel.NONE
        )
        
        # Generate prompt
        response = await self.generator.generate_prompt(request)
        
        # Verify response
        assert response.status == PromptStatus.COMPLETED
        assert response.request_id == "integration-test"
        
        expected_content = """🌟 Welcome back, Alice!

Your current score: 150
🎉 New Achievement: Century Club 🏆

You have 2 new notification(s):
• Welcome to the platform!
• Your profile is 90% complete"""
        
        # Normalize whitespace for comparison (removes extra blank lines)
        actual_normalized = '\n'.join(line.rstrip() for line in response.generated_prompt.split('\n') if line.strip() or line == '')
        expected_normalized = '\n'.join(line.rstrip() for line in expected_content.split('\n') if line.strip() or line == '')
        actual_normalized = '\n'.join(line for line in actual_normalized.split('\n') if line.strip())
        expected_normalized = '\n'.join(line for line in expected_normalized.split('\n') if line.strip())
        
        assert actual_normalized == expected_normalized
        
        # Verify rules were applied
        assert len(response.rules_applied) == 2
        rule_names = [rule.rule_name for rule in response.rules_applied]
        assert "vip_welcome" in rule_names
        assert "achievement_unlock" in rule_names
        
        # Verify context updates from rules
        assert "greeting_style" in response.context_used
        assert response.context_used["greeting_style"] == "premium"
        assert "achievement" in response.context_used
        assert response.context_used["achievement"] == "Century Club"
        
        # Verify template variables were extracted
        expected_vars = {"greeting_style", "user", "score", "achievement", "badge", "notifications"}
        assert response.template_variables == expected_vars


if __name__ == "__main__":
    pytest.main([__file__])
