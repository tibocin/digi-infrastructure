"""
Filepath: pcs/src/pcs/services/__init__.py
Purpose: Services package initialization with exports for business logic layer
Related Components: Template engine, rule engine, context manager, prompt generator, background tasks, webhooks, rate limiting, monitoring, performance optimization
Tags: services, business-logic, template-engine, rule-engine, context-management, prompt-generation, background-tasks, webhooks, rate-limiting, monitoring, prometheus, alerting, performance-optimization, benchmarking
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

# Background Task Service exports (optional)
try:
    from .background_task_service import (
        BackgroundTaskService,
        TaskRegistry,
        TaskDefinition,
        TaskStatus,
        TaskPriority,
        TaskResult,
        BackgroundTaskError,
        get_background_task_service,
        background_task_lifespan,
    )
    _has_background_tasks = True
except ImportError:
    # Provide None or mock implementations when background tasks are not available
    BackgroundTaskService = None
    TaskRegistry = None
    TaskDefinition = None
    TaskStatus = None
    TaskPriority = None
    TaskResult = None
    BackgroundTaskError = None
    get_background_task_service = None
    background_task_lifespan = None
    enqueue_fastapi_task = None
    _has_background_tasks = False

# Webhook Service exports (optional)
try:
    from .webhook_service import (
        WebhookService,
        WebhookDeliveryEngine,
        WebhookSecurityValidator,
        WebhookEndpoint,
        WebhookPayload,
        WebhookEvent,
        WebhookStatus,
        WebhookError,
        get_webhook_service,
        send_task_completion_webhook,
    )
    _has_webhooks = True
except ImportError:
    # Provide None for webhook services when not available
    WebhookService = None
    WebhookDeliveryEngine = None
    WebhookSecurityValidator = None
    WebhookEndpoint = None
    WebhookPayload = None
    WebhookEvent = None
    WebhookStatus = None
    WebhookError = None
    get_webhook_service = None
    send_task_completion_webhook = None
    send_task_failure_webhook = None
    send_custom_webhook = None
    _has_webhooks = False

# Rate Limiting Service exports (optional)
try:
    from .rate_limiting_service import (
        RateLimitingService,
        RateLimitConfig,
        RateLimitResult,
        RateLimitAlgorithm,
        RateLimitScope,
        RateLimitError,
        TokenBucketAlgorithm,
        SlidingWindowAlgorithm,
        FixedWindowAlgorithm,
        LeakyBucketAlgorithm,
        RateLimitMiddleware,
        get_rate_limiting_service,
        check_ip_rate_limit,
        check_user_rate_limit
    )
    _has_rate_limiting = True
except ImportError:
    # Provide None for rate limiting services when not available
    RateLimitingService = None
    RateLimitConfig = None
    RateLimitResult = None
    RateLimitAlgorithm = None
    RateLimitScope = None
    RateLimitError = None
    TokenBucketAlgorithm = None
    SlidingWindowAlgorithm = None
    FixedWindowAlgorithm = None
    LeakyBucketAlgorithm = None
    RateLimitMiddleware = None
    get_rate_limiting_service = None
    check_ip_rate_limit = None
    check_user_rate_limit = None
    _has_rate_limiting = False

# Monitoring Service exports (optional)
try:
    from .monitoring_service import (
        MonitoringService,
        PrometheusExporter,
        AlertManager,
        HealthChecker,
        AlertRule,
        Alert,
        AlertSeverity,
        AlertStatus,
        HealthCheck,
        HealthCheckResult,
        MetricType,
        HealthCheckError,
        MonitoringError,
        get_monitoring_service,
        monitoring_lifespan,
        monitor_http_request,
        monitor_db_query,
        monitor_background_task,
        monitor_webhook_delivery,
        monitor_rate_limit_check
    )
    _has_monitoring = True
except ImportError:
    # Provide None for monitoring services when not available
    MonitoringService = None
    PrometheusExporter = None
    AlertManager = None
    HealthChecker = None
    AlertRule = None
    Alert = None
    AlertSeverity = None
    AlertStatus = None
    HealthCheck = None
    HealthCheckResult = None
    MetricType = None
    HealthCheckError = None
    MonitoringError = None
    get_monitoring_service = None
    monitoring_lifespan = None
    monitor_http_request = None
    monitor_db_query = None
    monitor_background_task = None
    monitor_webhook_delivery = None
    monitor_rate_limit_check = None
    _has_monitoring = False

# Performance Optimization Service exports
from .performance_optimization_service import (
    PerformanceOptimizationService,
    BottleneckAnalyzer,
    PerformanceOptimizer,
    PerformanceBenchmark,
    Bottleneck,
    OptimizationResult,
    BenchmarkResult,
    PerformanceMetric,
    OptimizationType,
    BottleneckSeverity,
    OptimizationStatus,
    PerformanceError,
    get_performance_optimization_service,
    analyze_performance_bottlenecks,
    run_performance_optimization,
    benchmark_system_performance,
    performance_profiler
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
    
    # Background Task Service
    "BackgroundTaskService",
    "TaskRegistry",
    "TaskDefinition",
    "TaskStatus",
    "TaskPriority",
    "TaskResult",
    "BackgroundTaskError",
    "get_background_task_service",
    "background_task_lifespan",
    "enqueue_fastapi_task",
    
    # Webhook Service
    "WebhookService",
    "WebhookDeliveryEngine",
    "WebhookSecurityValidator",
    "WebhookEndpoint",
    "WebhookPayload",
    "WebhookEvent",
    "WebhookStatus",
    "WebhookError",
    "get_webhook_service",
    "send_task_completion_webhook",
    "send_task_failure_webhook",
    "send_custom_webhook",
    
    # Rate Limiting Service
    "RateLimitingService",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitAlgorithm",
    "RateLimitScope",
    "RateLimitError",
    "TokenBucketAlgorithm",
    "SlidingWindowAlgorithm",
    "FixedWindowAlgorithm",
    "LeakyBucketAlgorithm",
    "RateLimitMiddleware",
    "get_rate_limiting_service",
    "check_ip_rate_limit",
    "check_user_rate_limit",
    
    # Monitoring Service
    "MonitoringService",
    "PrometheusExporter",
    "AlertManager",
    "HealthChecker",
    "AlertRule",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "HealthCheck",
    "HealthCheckResult",
    "MetricType",
    "HealthCheckError",
    "MonitoringError",
    "get_monitoring_service",
    "monitoring_lifespan",
    "monitor_http_request",
    "monitor_db_query",
    "monitor_background_task",
    "monitor_webhook_delivery",
    "monitor_rate_limit_check",
    
    # Performance Optimization Service
    "PerformanceOptimizationService",
    "BottleneckAnalyzer",
    "PerformanceOptimizer",
    "PerformanceBenchmark",
    "Bottleneck",
    "OptimizationResult",
    "BenchmarkResult",
    "PerformanceMetric",
    "OptimizationType",
    "BottleneckSeverity",
    "OptimizationStatus",
    "PerformanceError",
    "get_performance_optimization_service",
    "analyze_performance_bottlenecks",
    "run_performance_optimization",
    "benchmark_system_performance",
    "performance_profiler",
]
