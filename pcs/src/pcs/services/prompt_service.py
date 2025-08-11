"""
Filepath: pcs/src/pcs/services/prompt_service.py
Purpose: Core prompt generation service integrating template engine, rule engine, and context manager
Related Components: Template engine, rule engine, context manager, repositories, caching
Tags: prompt-generation, template-processing, context-injection, rule-evaluation, caching
"""

import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

from ..core.exceptions import PCSError
from ..repositories.redis_repo import RedisRepository
from .template_service import TemplateEngine, TemplateError
from .rule_engine import RuleEngine, RuleEvaluationResult, RuleError
from .context_service import (
    ContextManager, Context, ContextType, ContextScope, 
    MergeStrategy, ContextError
)


class PromptError(PCSError):
    """Custom exception for prompt-related errors."""
    pass


class PromptStatus(Enum):
    """Status of prompt generation."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class OptimizationLevel(Enum):
    """Optimization levels for prompt generation."""
    NONE = "none"          # No optimization
    BASIC = "basic"        # Basic caching
    AGGRESSIVE = "aggressive"  # Aggressive caching and pre-processing


@dataclass
class PromptRequest:
    """
    Request for prompt generation.
    
    Contains all information needed to generate a prompt including
    template, context, rules, and generation options.
    """
    request_id: str
    template_name: str
    template_content: Optional[str] = None  # Override template content
    context_data: Dict[str, Any] = None
    context_ids: List[str] = None  # Context IDs to merge
    rule_names: List[str] = None  # Rules to apply
    variables: Dict[str, Any] = None  # Additional variables
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    cache_ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}
        if self.context_ids is None:
            self.context_ids = []
        if self.rule_names is None:
            self.rule_names = []
        if self.variables is None:
            self.variables = {}
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.optimization_level, str):
            self.optimization_level = OptimizationLevel(self.optimization_level)
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request."""
        # Create deterministic hash from request components
        cache_data = {
            'template_name': self.template_name,
            'template_content': self.template_content,
            'context_data': self.context_data,
            'context_ids': sorted(self.context_ids),
            'rule_names': sorted(self.rule_names),
            'variables': self.variables
        }
        
        # Convert to JSON string and hash
        import json
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.sha256(cache_string.encode()).hexdigest()


@dataclass
class PromptResponse:
    """
    Response from prompt generation.
    
    Contains the generated prompt along with metadata about
    the generation process.
    """
    request_id: str
    status: PromptStatus
    generated_prompt: Optional[str] = None
    processing_time_ms: Optional[float] = None
    context_used: Dict[str, Any] = None
    rules_applied: List[RuleEvaluationResult] = None
    template_variables: Set[str] = None
    cache_hit: bool = False
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context_used is None:
            self.context_used = {}
        if self.rules_applied is None:
            self.rules_applied = []
        if self.template_variables is None:
            self.template_variables = set()
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.status, str):
            self.status = PromptStatus(self.status)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        if 'template_variables' in data:
            data['template_variables'] = list(data['template_variables'])
        # Convert enum to string
        if 'status' in data:
            data['status'] = data['status'].value if hasattr(data['status'], 'value') else str(data['status'])
        return data


class PromptCache:
    """
    Caching layer for prompt generation.
    
    Provides intelligent caching with TTL management and
    cache invalidation strategies.
    """
    
    def __init__(self, redis_repo: RedisRepository, cache_prefix: str = "pcs:prompt"):
        """
        Initialize prompt cache.
        
        Args:
            redis_repo: Redis repository for caching
            cache_prefix: Prefix for cache keys
        """
        self.redis = redis_repo
        self.cache_prefix = cache_prefix
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, request: PromptRequest) -> str:
        """Get cache key for prompt request."""
        return f"{self.cache_prefix}:prompt:{request.get_cache_key()}"
    
    def _get_metadata_key(self, request: PromptRequest) -> str:
        """Get cache key for prompt metadata."""
        return f"{self.cache_prefix}:meta:{request.get_cache_key()}"
    
    async def get_cached_prompt(self, request: PromptRequest) -> Optional[PromptResponse]:
        """
        Get cached prompt response.
        
        Args:
            request: Prompt request
            
        Returns:
            Cached response or None if not found
        """
        try:
            cache_key = self._get_cache_key(request)
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                self.cache_hits += 1
                
                # Reconstruct response
                response = PromptResponse(**cached_data)
                response.cache_hit = True
                return response
            else:
                self.cache_misses += 1
                return None
                
        except Exception as e:
            # Cache errors shouldn't break prompt generation
            return None
    
    async def cache_prompt(self, request: PromptRequest, response: PromptResponse) -> bool:
        """
        Cache prompt response.
        
        Args:
            request: Prompt request
            response: Generated response
            
        Returns:
            True if successfully cached
        """
        try:
            cache_key = self._get_cache_key(request)
            cache_ttl = request.cache_ttl or timedelta(hours=1)  # Default 1 hour
            
            # Cache the response
            await self.redis.set(cache_key, response.to_dict(), ttl=cache_ttl)
            
            return True
            
        except Exception:
            # Cache errors shouldn't break prompt generation
            return False
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cached prompts.
        
        Args:
            pattern: Cache key pattern to match (optional)
            
        Returns:
            Number of keys invalidated
        """
        try:
            if pattern:
                cache_pattern = f"{self.cache_prefix}:prompt:{pattern}*"
            else:
                cache_pattern = f"{self.cache_prefix}:prompt:*"
            
            keys_to_delete = await self.redis.keys(cache_pattern)
            
            if keys_to_delete:
                return await self.redis.delete(*keys_to_delete)
            
            return 0
            
        except Exception:
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class PromptOptimizer:
    """
    Prompt optimizer for improving generation performance.
    
    Provides optimization strategies based on usage patterns
    and performance requirements.
    """
    
    def __init__(self):
        """Initialize prompt optimizer."""
        self.optimization_strategies = {
            OptimizationLevel.NONE: self._optimize_none,
            OptimizationLevel.BASIC: self._optimize_basic,
            OptimizationLevel.AGGRESSIVE: self._optimize_aggressive,
        }
    
    async def optimize_request(
        self, 
        request: PromptRequest,
        context_manager: ContextManager
    ) -> PromptRequest:
        """
        Optimize prompt request based on optimization level.
        
        Args:
            request: Original prompt request
            context_manager: Context manager for optimization
            
        Returns:
            Optimized prompt request
        """
        try:
            optimizer = self.optimization_strategies.get(request.optimization_level)
            if optimizer:
                return await optimizer(request, context_manager)
            
            return request
            
        except Exception:
            # If optimization fails, return original request
            return request
    
    async def _optimize_none(
        self, 
        request: PromptRequest, 
        context_manager: ContextManager
    ) -> PromptRequest:
        """No optimization - return request as-is."""
        return request
    
    async def _optimize_basic(
        self, 
        request: PromptRequest, 
        context_manager: ContextManager
    ) -> PromptRequest:
        """Basic optimization - merge contexts efficiently."""
        # Pre-merge contexts if multiple context IDs provided
        if len(request.context_ids) > 1:
            merged_context = await context_manager.merge_contexts(
                request.context_ids,
                MergeStrategy.MERGE_DEEP
            )
            
            if merged_context:
                # Replace context IDs with merged context data
                request.context_data.update(merged_context.data)
                request.context_ids = []
        
        return request
    
    async def _optimize_aggressive(
        self, 
        request: PromptRequest, 
        context_manager: ContextManager
    ) -> PromptRequest:
        """Aggressive optimization - pre-process everything possible."""
        # Run basic optimization first
        request = await self._optimize_basic(request, context_manager)
        
        # Set longer cache TTL for aggressive optimization
        if not request.cache_ttl:
            request.cache_ttl = timedelta(hours=24)
        
        return request


class PromptGenerator:
    """
    Core prompt generator that orchestrates the generation process.
    
    Integrates template engine, rule engine, context manager, and optimization
    to provide end-to-end prompt generation.
    """
    
    def __init__(
        self,
        template_engine: TemplateEngine,
        rule_engine: RuleEngine,
        context_manager: ContextManager,
        redis_repo: RedisRepository
    ):
        """
        Initialize prompt generator.
        
        Args:
            template_engine: Template engine for template processing
            rule_engine: Rule engine for conditional logic
            context_manager: Context manager for context operations
            redis_repo: Redis repository for caching
        """
        self.template_engine = template_engine
        self.rule_engine = rule_engine
        self.context_manager = context_manager
        
        # Initialize supporting components
        self.cache = PromptCache(redis_repo)
        self.optimizer = PromptOptimizer()
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
    
    async def generate_prompt(self, request: PromptRequest) -> PromptResponse:
        """
        Generate a prompt from the given request.
        
        Args:
            request: Prompt generation request
            
        Returns:
            Generated prompt response
        """
        start_time = time.time()
        
        try:
            self.total_requests += 1
            
            # Create initial response
            response = PromptResponse(
                request_id=request.request_id,
                status=PromptStatus.PROCESSING
            )
            
            # Check cache first (unless optimization level is NONE)
            if request.optimization_level != OptimizationLevel.NONE:
                cached_response = await self.cache.get_cached_prompt(request)
                if cached_response:
                    # Update processing time for cached response
                    processing_time = (time.time() - start_time) * 1000
                    cached_response.processing_time_ms = processing_time
                    return cached_response
            
            # Optimize request
            optimized_request = await self.optimizer.optimize_request(request, self.context_manager)
            
            # Build context
            context = await self._build_context(optimized_request)
            
            # Apply rules
            rules_applied = await self._apply_rules(optimized_request, context)
            
            # Update context with rule results
            context = self._update_context_with_rules(context, rules_applied)
            
            # Get template content
            template_content = await self._get_template_content(optimized_request)
            
            # Extract template variables
            template_variables = self.template_engine.extract_variables(template_content)
            
            # Prepare final variables for template
            final_variables = self._prepare_template_variables(optimized_request, context)
            
            # Render template
            generated_prompt = await self.template_engine.render_template(
                template_content,
                final_variables,
                validate=True,
                required_vars=template_variables
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            
            # Build successful response
            response = PromptResponse(
                request_id=request.request_id,
                status=PromptStatus.COMPLETED,
                generated_prompt=generated_prompt,
                processing_time_ms=processing_time,
                context_used=context,
                rules_applied=rules_applied,
                template_variables=template_variables,
                cache_hit=False,
                metadata=request.metadata.copy()
            )
            
            # Cache the response
            if request.optimization_level != OptimizationLevel.NONE:
                await self.cache.cache_prompt(optimized_request, response)
            
            return response
            
        except Exception as e:
            # Calculate processing time even for errors
            processing_time = (time.time() - start_time) * 1000
            
            # Create error response
            return PromptResponse(
                request_id=request.request_id,
                status=PromptStatus.FAILED,
                processing_time_ms=processing_time,
                error_message=str(e),
                metadata=request.metadata.copy()
            )
    
    async def _build_context(self, request: PromptRequest) -> Dict[str, Any]:
        """
        Build context from request data and context IDs.
        
        Args:
            request: Prompt request
            
        Returns:
            Combined context dictionary
        """
        try:
            context = request.context_data.copy()
            
            # Add variables
            context.update(request.variables)
            
            # Merge contexts from context IDs
            if request.context_ids:
                for context_id in request.context_ids:
                    ctx = await self.context_manager.get_context(context_id)
                    if ctx:
                        context.update(ctx.data)
            
            # Add system context
            context.update({
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request.request_id,
                'template_name': request.template_name
            })
            
            return context
            
        except Exception as e:
            raise PromptError(f"Failed to build context: {str(e)}") from e
    
    async def _apply_rules(
        self, 
        request: PromptRequest, 
        context: Dict[str, Any]
    ) -> List[RuleEvaluationResult]:
        """
        Apply rules to modify context and behavior.
        
        Args:
            request: Prompt request
            context: Current context
            
        Returns:
            List of rule evaluation results
        """
        try:
            if not request.rule_names:
                return []
            
            # Apply specified rules
            return await self.rule_engine.evaluate_rules(context, request.rule_names)
            
        except Exception as e:
            # Rule errors shouldn't break prompt generation
            # Return empty list and continue
            return []
    
    def _update_context_with_rules(
        self, 
        context: Dict[str, Any], 
        rules_applied: List[RuleEvaluationResult]
    ) -> Dict[str, Any]:
        """
        Update context with results from rule evaluation.
        
        Args:
            context: Original context
            rules_applied: Rule evaluation results
            
        Returns:
            Updated context
        """
        updated_context = context.copy()
        
        for rule_result in rules_applied:
            if rule_result.matched and rule_result.context_updates:
                updated_context.update(rule_result.context_updates)
        
        return updated_context
    
    async def _get_template_content(self, request: PromptRequest) -> str:
        """
        Get template content from request or template store.
        
        Args:
            request: Prompt request
            
        Returns:
            Template content string
        """
        # If template content is provided in request, use it
        if request.template_content:
            return request.template_content
        
        # Otherwise, look up template by name
        # For now, this is a placeholder - in a real implementation,
        # you would load from a template repository
        template_templates = {
            'greeting': 'Hello {{ name }}! Welcome to {{ application }}.',
            'summary': 'Summary for {{ title }}: {{ content | truncate_words(50) }}',
            'notification': '{{ user.name }}, you have {{ count }} new {{ item_type }}(s).',
            'default': 'Template: {{ template_name }} with variables: {{ variables | json }}'
        }
        
        return template_templates.get(request.template_name, template_templates['default'])
    
    def _prepare_template_variables(
        self, 
        request: PromptRequest, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare final variables for template rendering.
        
        Args:
            request: Prompt request
            context: Current context
            
        Returns:
            Variables dictionary for template
        """
        # Start with context
        variables = context.copy()
        
        # Add any additional metadata
        variables.update({
            'template_name': request.template_name,
            'request_metadata': request.metadata
        })
        
        return variables
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_processing_time = (
            self.total_processing_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            'total_requests': self.total_requests,
            'total_processing_time_ms': self.total_processing_time,
            'average_processing_time_ms': avg_processing_time,
            'cache_stats': self.cache.get_cache_stats()
        }
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cached prompts."""
        return await self.cache.invalidate_cache(pattern)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on prompt generation components.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Test template engine
            test_template = "Hello {{ name }}!"
            test_result = await self.template_engine.render_template(
                test_template, 
                {'name': 'test'}
            )
            health['components']['template_engine'] = {
                'status': 'healthy' if test_result == 'Hello test!' else 'unhealthy',
                'test_result': test_result
            }
        except Exception as e:
            health['components']['template_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        try:
            # Test rule engine
            test_rule = {
                'name': 'test_rule',
                'description': 'Test rule',
                'condition': {
                    'field': 'test',
                    'operator': 'eq',
                    'value': 'value'
                },
                'actions': []
            }
            self.rule_engine.add_rule(test_rule)
            test_results = await self.rule_engine.evaluate_rules({'test': 'value'}, ['test_rule'])
            
            health['components']['rule_engine'] = {
                'status': 'healthy' if test_results[0].matched else 'unhealthy',
                'test_result': test_results[0].matched
            }
            
            # Clean up test rule
            self.rule_engine.remove_rule('test_rule')
            
        except Exception as e:
            health['components']['rule_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        try:
            # Test context manager
            test_context = await self.context_manager.create_context(
                ContextType.TEMPORARY,
                ContextScope.PRIVATE,
                {'test': 'data'},
                ttl=timedelta(seconds=1)
            )
            
            retrieved_context = await self.context_manager.get_context(test_context.context_id)
            
            health['components']['context_manager'] = {
                'status': 'healthy' if retrieved_context else 'unhealthy',
                'test_result': retrieved_context is not None
            }
            
            # Clean up test context
            await self.context_manager.delete_context(test_context.context_id)
            
        except Exception as e:
            health['components']['context_manager'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Overall health status
        component_statuses = [comp['status'] for comp in health['components'].values()]
        if all(status == 'healthy' for status in component_statuses):
            health['status'] = 'healthy'
        elif any(status == 'healthy' for status in component_statuses):
            health['status'] = 'degraded'
        else:
            health['status'] = 'unhealthy'
        
        return health
