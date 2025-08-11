"""
Filepath: pcs/src/pcs/services/__init__.py
Purpose: Services package initialization with exports for business logic layer
Related Components: Template engine, rule engine, context manager, prompt generator
Tags: services, business-logic, template-engine, rule-engine, context-management, prompt-generation
"""

# Template Engine exports
from .template_service import (
    TemplateEngine,
    TemplateValidator,
    TemplateValidationResult,
    VariableInjector,
    TemplateError
)

# Rule Engine exports
from .rule_engine import (
    RuleEngine,
    ConditionEvaluator,
    ActionExecutor,
    RuleCompiler,
    Rule,
    Condition,
    LogicalCondition,
    Action,
    RuleEvaluationResult,
    ComparisonOperator,
    LogicalOperator,
    ActionType,
    RuleError
)

# Context Management exports
from .context_service import (
    ContextManager,
    ContextCache,
    ContextMerger,
    ContextValidator,
    Context,
    ContextMetadata,
    ContextType,
    ContextScope,
    MergeStrategy,
    ContextError
)

# Prompt Service exports
from .prompt_service import (
    PromptGenerator,
    PromptCache,
    PromptOptimizer,
    PromptRequest,
    PromptResponse,
    PromptStatus,
    OptimizationLevel,
    PromptError
)

__all__ = [
    # Template Engine
    "TemplateEngine",
    "TemplateValidator", 
    "TemplateValidationResult",
    "VariableInjector",
    "TemplateError",
    
    # Rule Engine
    "RuleEngine",
    "ConditionEvaluator",
    "ActionExecutor",
    "RuleCompiler",
    "Rule",
    "Condition",
    "LogicalCondition",
    "Action",
    "RuleEvaluationResult",
    "ComparisonOperator",
    "LogicalOperator",
    "ActionType",
    "RuleError",
    
    # Context Management
    "ContextManager",
    "ContextCache",
    "ContextMerger",
    "ContextValidator",
    "Context",
    "ContextMetadata",
    "ContextType",
    "ContextScope",
    "MergeStrategy",
    "ContextError",
    
    # Prompt Service
    "PromptGenerator",
    "PromptCache",
    "PromptOptimizer",
    "PromptRequest",
    "PromptResponse",
    "PromptStatus",
    "OptimizationLevel",
    "PromptError",
]
