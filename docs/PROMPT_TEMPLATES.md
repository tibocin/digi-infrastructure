# Prompt Templates System

**Filepath:** `docs/PROMPT_TEMPLATES.md`  
**Purpose:** Comprehensive guide to the PCS prompt template system, variable management, and rule engine  
**Related Components:** PCS Core, Template Engine, Rule Engine, Variable System, Versioning Service  
**Tags:** prompt-templates, rule-engine, variable-system, template-versioning, pcs

## Overview

The Prompt Templates System is the core of the PCS, providing a flexible, rule-based approach to dynamic prompt generation. This system enables developers to create sophisticated, context-aware prompts that adapt to user interactions, preferences, and application states through a combination of templates, variables, and conditional logic.

## Template System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Prompt Template System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Template      │  │   Variable      │  │   Rule          │              │
│  │   Engine        │  │   System        │  │   Engine        │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│           │                     │                     │                     │
│  ┌───────┴────────┐  ┌─────────┴─────────┐  ┌───────┴────────┐              │
│  │   Template     │  │   Variable        │  │   Rule         │              │
│  │   Store        │  │   Validator       │  │   Evaluator    │              │
│  └─────────────────┘  └───────────────────┘  └───────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Template Lifecycle

1. **Template Creation** → Template authoring and validation
2. **Template Storage** → Versioned storage in database
3. **Template Retrieval** → Context-aware template selection
4. **Variable Resolution** → Dynamic content injection
5. **Rule Evaluation** → Conditional logic application
6. **Prompt Generation** → Final prompt assembly
7. **Template Evolution** → A/B testing and optimization

## Template Structure

### Basic Template Format

```yaml
# Basic Prompt Template
name: "user_onboarding_welcome"
version: "1.0.0"
category: "onboarding"
description: "Welcome message for new users based on their profile"

template: |
  Welcome to {{app_name}}, {{user_name}}! 🎉

  We're excited to help you {{learning_goal}} with {{topic}}.
  
  Based on your {{expertise_level}} level, we've prepared a personalized learning path.
  
  {{#if is_first_time}}
    Let's start with the basics and build your foundation.
  {{else}}
    Welcome back! Let's continue where you left off.
  {{/if}}

variables:
  - name: "app_name"
    type: "string"
    required: true
    description: "Name of the application"
    
  - name: "user_name"
    type: "string"
    required: true
    description: "User's display name"
    
  - name: "learning_goal"
    type: "string"
    required: true
    description: "User's primary learning objective"
    
  - name: "topic"
    type: "string"
    required: true
    description: "Main subject area"
    
  - name: "expertise_level"
    type: "enum"
    values: ["beginner", "intermediate", "advanced"]
    required: true
    description: "User's skill level"
    
  - name: "is_first_time"
    type: "boolean"
    required: false
    default: true
    description: "Whether this is the user's first visit"

rules:
  - name: "expertise_based_greeting"
    condition: "expertise_level == 'beginner'"
    action: "use_beginner_greeting"
    fallback: "use_generic_greeting"
    
  - name: "time_based_welcome"
    condition: "hour_of_day >= 6 && hour_of_day <= 12"
    action: "use_morning_greeting"
    fallback: "use_standard_greeting"

metadata:
  author: "content_team"
  tags: ["welcome", "onboarding", "personalization"]
  target_audience: "new_users"
  language: "en"
  created_at: "2024-01-15T10:00:00Z"
  last_updated: "2024-01-15T10:00:00Z"
```

### Advanced Template Features

```yaml
# Advanced Template with Complex Logic
name: "adaptive_learning_prompt"
version: "2.1.0"
category: "education"
description: "Adaptive learning prompt with multi-modal support"

template: |
  {{#with user_profile}}
    {{#if learning_style == 'visual'}}
      🎨 **Visual Learning Path for {{name}}**
      
      I'll create visual diagrams and charts to help you understand {{current_topic}}.
      {{#if preferred_language != 'en'}}
        I'll also provide explanations in {{preferred_language}}.
      {{/if}}
    {{else if learning_style == 'auditory'}}
      🔊 **Audio-Focused Learning for {{name}}**
      
      I'll provide detailed verbal explanations and audio summaries for {{current_topic}}.
      {{#if has_audio_preference}}
        I'll use {{audio_voice}} voice for better engagement.
      {{/if}}
    {{else}}
      ✋ **Interactive Learning for {{name}}**
      
      Let's work through {{current_topic}} with hands-on exercises and real-world examples.
    {{/if}}
  {{/with}}

  **Current Progress:** {{progress_percentage}}% complete
  
  {{#if progress_percentage < 30}}
    Let's start with the fundamentals and build a strong foundation.
  {{else if progress_percentage < 70}}
    Great progress! Let's dive deeper into advanced concepts.
  {{else}}
    Excellent work! Let's tackle some challenging problems to master this topic.
  {{/if}}

  {{#if has_learning_disability}}
    I'll adapt my teaching style to accommodate your {{learning_disability}} needs.
  {{/if}}

variables:
  - name: "user_profile"
    type: "object"
    required: true
    schema:
      name: "string"
      learning_style: "enum[visual, auditory, kinesthetic]"
      preferred_language: "string"
      audio_voice: "string"
      has_audio_preference: "boolean"
      has_learning_disability: "string?"
      
  - name: "current_topic"
    type: "string"
    required: true
    
  - name: "progress_percentage"
    type: "number"
    required: true
    min: 0
    max: 100

rules:
  - name: "learning_style_adaptation"
    condition: "user_profile.learning_style == 'visual'"
    action: "include_visual_elements"
    priority: "high"
    
  - name: "accessibility_adaptation"
    condition: "user_profile.has_learning_disability"
    action: "apply_accessibility_modifications"
    priority: "critical"
    
  - name: "progress_based_difficulty"
    condition: "progress_percentage < 30"
    action: "use_beginner_content"
    fallback: "use_adaptive_content"

hooks:
  pre_generation:
    - "validate_user_profile"
    - "check_learning_preferences"
    
  post_generation:
    - "log_template_usage"
    - "update_user_progress"
    - "trigger_analytics"

validation:
  required_variables: ["user_profile", "current_topic", "progress_percentage"]
  variable_constraints:
    progress_percentage:
      min: 0
      max: 100
      type: "number"
```

## Variable System

### Variable Types and Validation

```typescript
interface VariableDefinition {
  name: string;
  type: VariableType;
  required: boolean;
  description?: string;
  default?: any;
  validation?: VariableValidation;
  transform?: VariableTransform;
}

enum VariableType {
  STRING = 'string',
  NUMBER = 'number',
  BOOLEAN = 'boolean',
  ARRAY = 'array',
  OBJECT = 'object',
  ENUM = 'enum',
  DATE = 'date',
  EMAIL = 'email',
  URL = 'url',
  UUID = 'uuid'
}

interface VariableValidation {
  min?: number;
  max?: number;
  pattern?: string;
  enum?: string[];
  required?: boolean;
  custom?: (value: any) => boolean;
}

interface VariableTransform {
  type: 'uppercase' | 'lowercase' | 'capitalize' | 'trim' | 'custom';
  function?: (value: any) => any;
}
```

### Variable Resolution Engine

```typescript
class VariableResolver {
  /**
   * Resolves all variables in a template with validation and transformation
   * @param template - Template containing variables
   * @param context - Context data for variable resolution
   * @param options - Resolution options
   */
  async resolveVariables(
    template: string,
    context: Context,
    options: VariableResolutionOptions = {}
  ): Promise<ResolvedTemplate> {
    const { strictMode = true, allowMissing = false } = options;
    
    // Extract all variables from template
    const variables = this.extractVariables(template);
    
    // Validate required variables
    if (strictMode) {
      await this.validateRequiredVariables(variables, context);
    }
    
    // Resolve each variable
    const resolvedVariables = new Map<string, any>();
    
    for (const variable of variables) {
      try {
        const value = await this.resolveVariable(variable, context);
        resolvedVariables.set(variable.name, value);
      } catch (error) {
        if (allowMissing) {
          resolvedVariables.set(variable.name, variable.default);
        } else {
          throw new VariableResolutionError(
            `Failed to resolve variable: ${variable.name}`,
            error
          );
        }
      }
    }
    
    // Apply transformations
    const transformedVariables = await this.applyTransformations(
      resolvedVariables,
      variables
    );
    
    // Substitute variables in template
    const resolvedTemplate = this.substituteVariables(
      template,
      transformedVariables
    );
    
    return {
      template: resolvedTemplate,
      variables: transformedVariables,
      metadata: {
        resolutionTime: Date.now(),
        variablesResolved: variables.length,
        transformationsApplied: this.countTransformations(variables)
      }
    };
  }
  
  private async resolveVariable(
    variable: VariableDefinition,
    context: Context
  ): Promise<any> {
    // Check context first
    if (context.hasOwnProperty(variable.name)) {
      return context[variable.name];
    }
    
    // Check environment variables
    if (variable.name.startsWith('env.')) {
      const envVar = variable.name.substring(4);
      return process.env[envVar];
    }
    
    // Check computed values
    if (variable.name.startsWith('computed.')) {
      return this.computeValue(variable.name, context);
    }
    
    // Return default value if available
    if (variable.default !== undefined) {
      return variable.default;
    }
    
    throw new VariableNotFoundError(`Variable not found: ${variable.name}`);
  }
  
  private async computeValue(computedName: string, context: Context): Promise<any> {
    const computation = computedName.substring(9);
    
    switch (computation) {
      case 'current_time':
        return new Date();
        
      case 'user_age':
        if (context.birthDate) {
          const birthDate = new Date(context.birthDate);
          const today = new Date();
          return today.getFullYear() - birthDate.getFullYear();
        }
        throw new Error('Birth date not available for age computation');
        
      case 'progress_percentage':
        if (context.completedLessons && context.totalLessons) {
          return Math.round((context.completedLessons / context.totalLessons) * 100);
        }
        return 0;
        
      default:
        throw new Error(`Unknown computation: ${computation}`);
    }
  }
}
```

### Variable Substitution

```typescript
class VariableSubstitutor {
  /**
   * Substitutes resolved variables into the template
   * @param template - Template with variable placeholders
   * @param variables - Resolved variable values
   */
  substituteVariables(template: string, variables: Map<string, any>): string {
    let result = template;
    
    // Handle different variable syntax patterns
    const patterns = [
      /\{\{(\w+)\}\}/g,           // {{variable}}
      /\{\{(\w+\.\w+)\}\}/g,      // {{object.property}}
      /\{\{#if\s+(\w+)\}\}(.*?)\{\{\/if\}\}/gs,  // {{#if condition}}...{{/if}}
      /\{\{#with\s+(\w+)\}\}(.*?)\{\{\/with\}\}/gs,  // {{#with object}}...{{/with}}
      /\{\{#each\s+(\w+)\}\}(.*?)\{\{\/each\}\}/gs   // {{#each array}}...{{/each}}
    ];
    
    // Substitute simple variables
    result = result.replace(/\{\{(\w+)\}\}/g, (match, varName) => {
      return variables.get(varName) || match;
    });
    
    // Substitute object properties
    result = result.replace(/\{\{(\w+\.\w+)\}\}/g, (match, path) => {
      const [objName, propName] = path.split('.');
      const obj = variables.get(objName);
      return obj && obj[propName] !== undefined ? obj[propName] : match;
    });
    
    // Handle conditional blocks
    result = this.processConditionals(result, variables);
    
    // Handle with blocks
    result = this.processWithBlocks(result, variables);
    
    // Handle each loops
    result = this.processEachLoops(result, variables);
    
    return result;
  }
  
  private processConditionals(
    template: string,
    variables: Map<string, any>
  ): string {
    return template.replace(
      /\{\{#if\s+(\w+)\}\}(.*?)\{\{\/if\}\}/gs,
      (match, condition, content) => {
        const value = variables.get(condition);
        return this.evaluateCondition(value) ? content : '';
      }
    );
  }
  
  private evaluateCondition(value: any): boolean {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') return value.length > 0;
    if (typeof value === 'number') return value !== 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'object') return value !== null;
    return false;
  }
}
```

## Rule Engine

### Rule Definition and Structure

```typescript
interface Rule {
  name: string;
  condition: string | RuleCondition;
  action: string | RuleAction;
  fallback?: string | RuleAction;
  priority: 'low' | 'medium' | 'high' | 'critical';
  metadata?: RuleMetadata;
}

interface RuleCondition {
  type: 'expression' | 'function' | 'template';
  value: string | Function;
  parameters?: any[];
}

interface RuleAction {
  type: 'template_switch' | 'variable_modification' | 'custom_function';
  value: string | Function;
  parameters?: any[];
}

interface RuleMetadata {
  author: string;
  created_at: Date;
  last_updated: Date;
  tags: string[];
  description?: string;
  test_cases?: RuleTestCase[];
}
```

### Rule Evaluation Engine

```typescript
class RuleEngine {
  /**
   * Evaluates rules against context and applies actions
   * @param rules - Rules to evaluate
   * @param context - Current context
   * @param template - Current template
   */
  async evaluateRules(
    rules: Rule[],
    context: Context,
    template: Template
  ): Promise<RuleEvaluationResult> {
    // Sort rules by priority
    const sortedRules = this.sortRulesByPriority(rules);
    
    const results: RuleResult[] = [];
    let modifiedTemplate = template;
    let modifiedContext = { ...context };
    
    for (const rule of sortedRules) {
      try {
        const evaluation = await this.evaluateRule(rule, modifiedContext, modifiedTemplate);
        
        if (evaluation.matched) {
          const actionResult = await this.executeAction(
            rule.action,
            evaluation,
            modifiedContext,
            modifiedTemplate
          );
          
          // Update template and context based on action
          if (actionResult.template) {
            modifiedTemplate = actionResult.template;
          }
          if (actionResult.context) {
            modifiedContext = { ...modifiedContext, ...actionResult.context };
          }
          
          results.push({
            rule: rule.name,
            matched: true,
            action: actionResult,
            executionTime: Date.now()
          });
          
          // Check if we should stop processing (critical rules)
          if (rule.priority === 'critical') {
            break;
          }
        } else if (rule.fallback) {
          // Execute fallback action
          const fallbackResult = await this.executeAction(
            rule.fallback,
            evaluation,
            modifiedContext,
            modifiedTemplate
          );
          
          results.push({
            rule: rule.name,
            matched: false,
            fallback: fallbackResult,
            executionTime: Date.now()
          });
        }
      } catch (error) {
        results.push({
          rule: rule.name,
          matched: false,
          error: error.message,
          executionTime: Date.now()
        });
      }
    }
    
    return {
      results,
      finalTemplate: modifiedTemplate,
      finalContext: modifiedContext,
      executionSummary: this.generateExecutionSummary(results)
    };
  }
  
  private async evaluateRule(
    rule: Rule,
    context: Context,
    template: Template
  ): Promise<RuleEvaluation> {
    if (typeof rule.condition === 'string') {
      return this.evaluateExpression(rule.condition, context);
    } else {
      return this.evaluateFunction(rule.condition, context);
    }
  }
  
  private async evaluateExpression(
    expression: string,
    context: Context
  ): Promise<RuleEvaluation> {
    try {
      // Safe expression evaluation with context
      const safeContext = this.createSafeContext(context);
      const result = this.evaluateSafeExpression(expression, safeContext);
      
      return {
        matched: Boolean(result),
        value: result,
        context: safeContext
      };
    } catch (error) {
      throw new RuleEvaluationError(`Failed to evaluate expression: ${expression}`, error);
    }
  }
  
  private createSafeContext(context: Context): any {
    // Create a safe context object with only allowed properties
    const safeContext: any = {};
    
    for (const [key, value] of Object.entries(context)) {
      if (this.isSafeValue(value)) {
        safeContext[key] = value;
      }
    }
    
    return safeContext;
  }
  
  private isSafeValue(value: any): boolean {
    // Only allow primitive types and simple objects
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      return true;
    }
    
    if (Array.isArray(value)) {
      return value.every(item => this.isSafeValue(item));
    }
    
    if (typeof value === 'object' && value !== null) {
      return Object.values(value).every(val => this.isSafeValue(val));
    }
    
    return false;
  }
}
```

### Rule Actions and Execution

```typescript
class RuleActionExecutor {
  /**
   * Executes rule actions and returns results
   * @param action - Action to execute
   * @param evaluation - Rule evaluation result
   * @param context - Current context
   * @param template - Current template
   */
  async executeAction(
    action: RuleAction,
    evaluation: RuleEvaluation,
    context: Context,
    template: Template
  ): Promise<ActionResult> {
    switch (action.type) {
      case 'template_switch':
        return this.executeTemplateSwitch(action, evaluation, context);
        
      case 'variable_modification':
        return this.executeVariableModification(action, evaluation, context);
        
      case 'custom_function':
        return this.executeCustomFunction(action, evaluation, context, template);
        
      default:
        throw new Error(`Unknown action type: ${action.type}`);
    }
  }
  
  private async executeTemplateSwitch(
    action: RuleAction,
    evaluation: RuleEvaluation,
    context: Context
  ): Promise<ActionResult> {
    const templateName = action.value as string;
    
    // Load the new template
    const newTemplate = await this.templateStore.getTemplate(templateName);
    
    if (!newTemplate) {
      throw new Error(`Template not found: ${templateName}`);
    }
    
    return {
      template: newTemplate,
      context: context,
      actionType: 'template_switch',
      metadata: {
        fromTemplate: context.currentTemplate,
        toTemplate: templateName,
        reason: evaluation.value
      }
    };
  }
  
  private async executeVariableModification(
    action: RuleAction,
    evaluation: RuleEvaluation,
    context: Context
  ): Promise<ActionResult> {
    const modifications = action.parameters || {};
    const modifiedContext = { ...context };
    
    // Apply variable modifications
    for (const [key, value] of Object.entries(modifications)) {
      if (typeof value === 'function') {
        modifiedContext[key] = value(context, evaluation);
      } else {
        modifiedContext[key] = value;
      }
    }
    
    return {
      context: modifiedContext,
      actionType: 'variable_modification',
      metadata: {
        modifications: Object.keys(modifications),
        reason: evaluation.value
      }
    };
  }
}
```

## Template Versioning and A/B Testing

### Version Management System

```typescript
interface TemplateVersion {
  id: string;
  templateId: string;
  version: string;
  content: Template;
  status: 'draft' | 'active' | 'deprecated' | 'archived';
  created_at: Date;
  created_by: string;
  metadata: VersionMetadata;
}

interface VersionMetadata {
  change_log: string;
  performance_metrics?: PerformanceMetrics;
  a_b_test_config?: ABTestConfig;
  rollout_percentage?: number;
  target_audience?: AudienceTarget;
}

interface ABTestConfig {
  test_id: string;
  variant_name: string;
  traffic_percentage: number;
  start_date: Date;
  end_date?: Date;
  success_metrics: string[];
  minimum_sample_size: number;
}
```

### A/B Testing Engine

```typescript
class ABTestingEngine {
  /**
   * Manages A/B testing for template variants
   * @param templateId - Template to test
   * @param context - User context for variant selection
   */
  async selectTemplateVariant(
    templateId: string,
    context: Context
  ): Promise<TemplateVariant> {
    const activeTests = await this.getActiveTests(templateId);
    
    if (activeTests.length === 0) {
      // Return production template
      return await this.getProductionTemplate(templateId);
    }
    
    // Select test variant based on user context
    const selectedTest = this.selectTestForUser(activeTests, context);
    const variant = await this.selectVariant(selectedTest, context);
    
    // Track variant selection
    await this.trackVariantSelection(selectedTest, variant, context);
    
    return {
      template: variant.template,
      testId: selectedTest.test_id,
      variantName: variant.variant_name,
      isTestVariant: true
    };
  }
  
  private selectTestForUser(
    tests: ABTestConfig[],
    context: Context
  ): ABTestConfig {
    // Use consistent hashing for user assignment
    const userHash = this.hashUserContext(context);
    
    for (const test of tests) {
      if (this.isUserInTest(userHash, test)) {
        return test;
      }
    }
    
    // User not in any test, return null
    return null;
  }
  
  private isUserInTest(userHash: string, test: ABTestConfig): boolean {
    const hashValue = parseInt(userHash.substring(0, 8), 16);
    const normalizedValue = (hashValue % 100) + 1;
    
    return normalizedValue <= test.traffic_percentage;
  }
  
  private async selectVariant(
    test: ABTestConfig,
    context: Context
  ): Promise<ABTestVariant> {
    const variants = await this.getTestVariants(test.test_id);
    
    if (variants.length === 1) {
      return variants[0];
    }
    
    // Use contextual bandits for variant selection
    return this.selectContextualVariant(variants, context, test);
  }
  
  private async selectContextualVariant(
    variants: ABTestVariant[],
    context: Context,
    test: ABTestConfig
  ): Promise<ABTestVariant> {
    // Get historical performance for each variant
    const performanceData = await Promise.all(
      variants.map(variant => this.getVariantPerformance(variant, context))
    );
    
    // Use UCB1 algorithm for exploration vs exploitation
    const totalTrials = performanceData.reduce((sum, data) => sum + data.trials, 0);
    
    let bestVariant = variants[0];
    let bestScore = -Infinity;
    
    for (let i = 0; i < variants.length; i++) {
      const data = performanceData[i];
      const exploitation = data.successRate;
      const exploration = Math.sqrt((2 * Math.log(totalTrials)) / data.trials);
      const score = exploitation + exploration;
      
      if (score > bestScore) {
        bestScore = score;
        bestVariant = variants[i];
      }
    }
    
    return bestVariant;
  }
}
```

### Performance Tracking and Optimization

```typescript
class TemplatePerformanceTracker {
  /**
   * Tracks template performance metrics for optimization
   * @param templateId - Template identifier
   * @param variantId - Variant identifier (if A/B testing)
   * @param metrics - Performance metrics
   */
  async trackPerformance(
    templateId: string,
    variantId: string,
    metrics: TemplatePerformanceMetrics
  ): Promise<void> {
    const performanceRecord = {
      template_id: templateId,
      variant_id: variantId,
      timestamp: new Date(),
      metrics: metrics,
      context_snapshot: metrics.context
    };
    
    // Store performance data
    await this.performanceStore.recordPerformance(performanceRecord);
    
    // Update real-time metrics
    await this.updateRealTimeMetrics(templateId, variantId, metrics);
    
    // Check for performance anomalies
    await this.checkPerformanceAnomalies(templateId, variantId, metrics);
    
    // Trigger optimization if needed
    if (this.shouldTriggerOptimization(templateId, variantId, metrics)) {
      await this.triggerOptimization(templateId, variantId);
    }
  }
  
  private async checkPerformanceAnomalies(
    templateId: string,
    variantId: string,
    metrics: TemplatePerformanceMetrics
  ): Promise<void> {
    const baseline = await this.getPerformanceBaseline(templateId, variantId);
    
    if (baseline) {
      const deviation = this.calculateDeviation(metrics, baseline);
      
      if (deviation > this.anomalyThreshold) {
        await this.alertPerformanceAnomaly(templateId, variantId, metrics, deviation);
      }
    }
  }
  
  private calculateDeviation(
    current: TemplatePerformanceMetrics,
    baseline: TemplatePerformanceMetrics
  ): number {
    const metrics = ['responseTime', 'successRate', 'userSatisfaction'];
    let totalDeviation = 0;
    
    for (const metric of metrics) {
      if (baseline[metric] && current[metric]) {
        const deviation = Math.abs(
          (current[metric] - baseline[metric]) / baseline[metric]
        );
        totalDeviation += deviation;
      }
    }
    
    return totalDeviation / metrics.length;
  }
}
```

## Template Management and Administration

### Template CRUD Operations

```typescript
class TemplateManager {
  /**
   * Creates a new template with validation
   * @param templateData - Template data
   * @param options - Creation options
   */
  async createTemplate(
    templateData: TemplateCreationData,
    options: TemplateCreationOptions = {}
  ): Promise<Template> {
    // Validate template structure
    await this.validateTemplate(templateData);
    
    // Check for naming conflicts
    await this.checkNamingConflicts(templateData.name);
    
    // Create initial version
    const template = await this.templateStore.createTemplate({
      ...templateData,
      version: '1.0.0',
      status: 'draft',
      created_at: new Date(),
      created_by: options.author || 'system'
    });
    
    // Set up monitoring
    await this.setupTemplateMonitoring(template);
    
    // Trigger webhooks
    await this.notifyTemplateChange(template, 'create');
    
    return template;
  }
  
  /**
   * Updates an existing template with versioning
   * @param templateId - Template to update
   * @param updates - Update data
   * @param options - Update options
   */
  async updateTemplate(
    templateId: string,
    updates: Partial<Template>,
    options: TemplateUpdateOptions = {}
  ): Promise<Template> {
    const currentTemplate = await this.getTemplate(templateId);
    
    // Validate updates
    await this.validateTemplateUpdates(currentTemplate, updates);
    
    // Create new version
    const newVersion = await this.createTemplateVersion(
      templateId,
      updates,
      options
    );
    
    // Update template status
    if (options.activate) {
      await this.activateTemplateVersion(templateId, newVersion.version);
    }
    
    // Trigger webhooks
    await this.notifyTemplateChange(newVersion, 'update');
    
    return newVersion;
  }
  
  /**
   * Deletes a template with cleanup
   * @param templateId - Template to delete
   * @param options - Deletion options
   */
  async deleteTemplate(
    templateId: string,
    options: TemplateDeletionOptions = {}
  ): Promise<void> {
    const template = await this.getTemplate(templateId);
    
    // Check dependencies
    const dependencies = await this.checkTemplateDependencies(templateId);
    
    if (dependencies.length > 0 && !options.force) {
      throw new TemplateDependencyError(
        `Template has dependencies: ${dependencies.join(', ')}`
      );
    }
    
    // Archive template
    await this.archiveTemplate(templateId);
    
    // Clean up related data
    await this.cleanupTemplateData(templateId);
    
    // Trigger webhooks
    await this.notifyTemplateChange(template, 'delete');
  }
}
```

## Integration Examples

### SDK Integration

```typescript
// PCS SDK template usage
const pcs = new PCSSDK({
  baseUrl: process.env.PCS_BASE_URL,
  apiKey: process.env.PCS_API_KEY,
  appId: process.env.PCS_APP_ID,
});

// Generate prompt using template
const prompt = await pcs.templates.generatePrompt({
  templateName: 'adaptive_learning_prompt',
  context: {
    user_profile: {
      name: 'John',
      learning_style: 'visual',
      preferred_language: 'en'
    },
    current_topic: 'machine_learning',
    progress_percentage: 45
  },
  options: {
    enableABTesting: true,
    trackPerformance: true,
    fallbackStrategy: 'default'
  }
});

// Create custom template
const newTemplate = await pcs.templates.createTemplate({
  name: 'custom_welcome',
  template: 'Welcome {{user_name}} to {{app_name}}!',
  variables: [
    { name: 'user_name', type: 'string', required: true },
    { name: 'app_name', type: 'string', required: true }
  ],
  category: 'custom',
  description: 'Custom welcome message'
});
```

### Webhook Integration

```typescript
// Webhook handler for template changes
app.post('/webhooks/template-updated', async (req, res) => {
  const { templateId, changeType, templateData } = req.body;
  
  try {
    switch (changeType) {
      case 'create':
        await handleTemplateCreated(templateId, templateData);
        break;
        
      case 'update':
        await handleTemplateUpdated(templateId, templateData);
        break;
        
      case 'delete':
        await handleTemplateDeleted(templateId);
        break;
        
      default:
        console.warn(`Unknown change type: ${changeType}`);
    }
    
    res.status(200).json({ success: true });
  } catch (error) {
    console.error('Template webhook error:', error);
    res.status(500).json({ error: error.message });
  }
});

async function handleTemplateUpdated(templateId: string, templateData: any) {
  // Update local cache
  await localTemplateCache.update(templateId, templateData);
  
  // Notify dependent services
  await eventBus.emit('template:updated', {
    templateId,
    version: templateData.version,
    timestamp: new Date()
  });
  
  // Update analytics tracking
  await analyticsService.trackTemplateUpdate(templateId, templateData);
}
```

## Monitoring and Analytics

### Template Performance Dashboard

```typescript
class TemplateAnalytics {
  /**
   * Generates comprehensive template performance report
   * @param timeRange - Time range for analysis
   * @param filters - Additional filters
   */
  async generatePerformanceReport(
    timeRange: TimeRange,
    filters: TemplateFilters = {}
  ): Promise<TemplatePerformanceReport> {
    const templates = await this.getTemplatesInRange(timeRange, filters);
    
    const report = {
      summary: await this.generateSummary(templates, timeRange),
      topPerformers: await this.identifyTopPerformers(templates, timeRange),
      underperformers: await this.identifyUnderperformers(templates, timeRange),
      trends: await this.analyzeTrends(templates, timeRange),
      recommendations: await this.generateRecommendations(templates, timeRange)
    };
    
    return report;
  }
  
  private async generateSummary(
    templates: Template[],
    timeRange: TimeRange
  ): Promise<TemplateSummary> {
    const totalTemplates = templates.length;
    const activeTemplates = templates.filter(t => t.status === 'active').length;
    
    const performanceData = await Promise.all(
      templates.map(t => this.getTemplatePerformance(t.id, timeRange))
    );
    
    const avgResponseTime = this.calculateAverage(
      performanceData.map(d => d.responseTime)
    );
    
    const avgSuccessRate = this.calculateAverage(
      performanceData.map(d => d.successRate)
    );
    
    return {
      totalTemplates,
      activeTemplates,
      averageResponseTime: avgResponseTime,
      averageSuccessRate: avgSuccessRate,
      totalUsage: performanceData.reduce((sum, d) => sum + d.usage, 0)
    };
  }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Template Not Found**
   - Check template name and version
   - Verify template status (active/draft)
   - Check access permissions

2. **Variable Resolution Errors**
   - Validate required variables are provided
   - Check variable type constraints
   - Verify context data structure

3. **Rule Evaluation Failures**
   - Check rule syntax and conditions
   - Verify context data availability
   - Review rule priorities and dependencies

4. **Performance Issues**
   - Monitor template complexity
   - Check variable resolution performance
   - Review caching strategies

5. **A/B Testing Problems**
   - Verify test configuration
   - Check traffic allocation
   - Monitor variant performance

## Related Documentation

- [Dynamic Prompting Architecture](DYNAMIC_PROMPTING_ARCHITECTURE.md)
- [Context Management](CONTEXT_MANAGEMENT.md)
- [PCS SDK Reference](PCS_SDK_REFERENCE.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
- [Schema Management](SCHEMA_MANAGEMENT.md)
