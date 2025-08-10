# Context Management System

**Filepath:** `docs/CONTEXT_MANAGEMENT.md`  
**Purpose:** Comprehensive guide to the Prompt and Context Service (PCS) context management system  
**Related Components:** PCS Core, Context Store, Context Engine, PCS SDK  
**Tags:** context-management, pcs, state-management, user-experience

## Overview

The Context Management System is the backbone of the PCS, responsible for maintaining, retrieving, and updating contextual information that enables personalized and adaptive AI interactions. This system ensures that every prompt generation is informed by relevant user history, preferences, and current session state.

## Context Types

### 1. Session Context

**Purpose:** Maintains current conversation state and immediate user interactions

```typescript
interface SessionContext {
  sessionId: string;
  userId?: string;
  appId: string;
  currentTopic: string;
  conversationHistory: Message[];
  activePrompts: string[];
  lastActivity: Date;
  metadata: Record<string, any>;
}
```

**Storage Strategy:** Redis with TTL (Time To Live) for session management
**Retention:** Session duration + 24 hours for analysis

### 2. User Profile Context

**Purpose:** Long-term user preferences, learning patterns, and behavioral data

```typescript
interface UserProfileContext {
  userId: string;
  learningStyle: "visual" | "auditory" | "kinesthetic";
  preferredLanguage: string;
  expertiseLevel: "beginner" | "intermediate" | "advanced";
  interests: string[];
  learningGoals: LearningGoal[];
  historicalPerformance: PerformanceMetric[];
  lastUpdated: Date;
}
```

**Storage Strategy:** PostgreSQL with JSONB for flexible schema
**Retention:** Indefinite (with GDPR compliance controls)

### 3. Application Context

**Purpose:** App-specific settings, configurations, and feature flags

```typescript
interface AppContext {
  appId: string;
  appVersion: string;
  enabledFeatures: string[];
  configuration: AppConfig;
  rateLimits: RateLimitConfig;
  webhookEndpoints: WebhookConfig[];
  lastSync: Date;
}
```

**Storage Strategy:** PostgreSQL with versioning
**Retention:** Application lifecycle + 90 days

### 4. Temporal Context

**Purpose:** Time-based relevance, seasonal adjustments, and temporal patterns

```typescript
interface TemporalContext {
  timestamp: Date;
  timezone: string;
  season: "spring" | "summer" | "autumn" | "winter";
  dayOfWeek: number;
  hourOfDay: number;
  isBusinessHours: boolean;
  holidays: Holiday[];
  specialEvents: SpecialEvent[];
}
```

**Storage Strategy:** Computed on-demand with caching
**Retention:** 30 days for pattern analysis

## Context Storage Architecture

### Primary Storage (PostgreSQL)

```sql
-- Enhanced context storage with versioning
CREATE TABLE pcs_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_type VARCHAR(50) NOT NULL,
    context_key VARCHAR(255) NOT NULL,
    context_data JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Composite unique constraint
    UNIQUE(context_type, context_key, version)
);

-- Context relationships for complex context graphs
CREATE TABLE pcs_context_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_context_id UUID REFERENCES pcs_contexts(id),
    target_context_id UUID REFERENCES pcs_contexts(id),
    relationship_type VARCHAR(50) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(source_context_id, target_context_id, relationship_type)
);

-- Context access patterns for optimization
CREATE TABLE pcs_context_access_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_id UUID REFERENCES pcs_contexts(id),
    access_type VARCHAR(20) NOT NULL, -- 'read', 'write', 'update'
    app_id VARCHAR(255),
    user_id VARCHAR(255),
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Cache Layer (Redis)

```yaml
# Redis configuration for context caching
redis:
  context_cache:
    # Session contexts: fast access, short TTL
    session_ttl: 3600 # 1 hour

    # User profiles: longer TTL, frequent access
    profile_ttl: 86400 # 24 hours

    # App contexts: medium TTL, moderate access
    app_ttl: 7200 # 2 hours

    # Temporal contexts: computed on-demand
    temporal_ttl: 300 # 5 minutes
```

## Context Management Operations

### 1. Context Retrieval

```typescript
class ContextManager {
  /**
   * Retrieves context with intelligent fallback and caching
   * @param contextKey - Unique identifier for the context
   * @param contextType - Type of context to retrieve
   * @param options - Retrieval options including fallback strategy
   */
  async getContext(
    contextKey: string,
    contextType: ContextType,
    options: ContextRetrievalOptions = {}
  ): Promise<Context> {
    const { useCache = true, fallbackStrategy = "default" } = options;

    // Try cache first if enabled
    if (useCache) {
      const cached = await this.cache.get(`${contextType}:${contextKey}`);
      if (cached) return cached;
    }

    // Retrieve from database
    const context = await this.database.getContext(contextKey, contextType);

    // Apply fallback strategy if context not found
    if (!context && fallbackStrategy !== "none") {
      return this.applyFallbackStrategy(
        contextKey,
        contextType,
        fallbackStrategy
      );
    }

    // Cache the result
    if (useCache && context) {
      await this.cache.set(`${contextType}:${contextKey}`, context);
    }

    return context;
  }
}
```

### 2. Context Updates

```typescript
class ContextManager {
  /**
   * Updates context with conflict resolution and versioning
   * @param contextKey - Context identifier
   * @param updates - Partial context updates
   * @param options - Update options including conflict resolution
   */
  async updateContext(
    contextKey: string,
    updates: Partial<Context>,
    options: ContextUpdateOptions = {}
  ): Promise<Context> {
    const { mergeStrategy = "smart", conflictResolution = "latest" } = options;

    // Get current context
    const currentContext = await this.getContext(contextKey, updates.type);

    // Apply merge strategy
    const mergedContext = this.mergeContexts(
      currentContext,
      updates,
      mergeStrategy
    );

    // Handle conflicts if they exist
    if (this.hasConflicts(currentContext, updates)) {
      const resolvedContext = await this.resolveConflicts(
        currentContext,
        updates,
        conflictResolution
      );
      return resolvedContext;
    }

    // Create new version
    const newVersion = await this.database.createContextVersion(
      contextKey,
      mergedContext
    );

    // Invalidate cache
    await this.cache.invalidate(`${updates.type}:${contextKey}`);

    // Trigger webhooks for context changes
    await this.notifyContextChange(contextKey, "update", newVersion);

    return newVersion;
  }
}
```

### 3. Context Merging Strategies

```typescript
enum MergeStrategy {
  REPLACE = "replace", // Completely replace context
  MERGE = "merge", // Deep merge all fields
  SMART = "smart", // Intelligent merge based on field type
  APPEND = "append", // Append to arrays, merge objects
  SELECTIVE = "selective", // Merge only specified fields
}

class ContextMerger {
  /**
   * Merges contexts based on the specified strategy
   * @param existing - Current context data
   * @param updates - New context updates
   * @param strategy - Merge strategy to apply
   */
  mergeContexts(existing: any, updates: any, strategy: MergeStrategy): any {
    switch (strategy) {
      case MergeStrategy.REPLACE:
        return { ...updates };

      case MergeStrategy.MERGE:
        return this.deepMerge(existing, updates);

      case MergeStrategy.SMART:
        return this.smartMerge(existing, updates);

      case MergeStrategy.APPEND:
        return this.appendMerge(existing, updates);

      case MergeStrategy.SELECTIVE:
        return this.selectiveMerge(existing, updates);

      default:
        return this.deepMerge(existing, updates);
    }
  }

  private smartMerge(existing: any, updates: any): any {
    const merged = { ...existing };

    for (const [key, value] of Object.entries(updates)) {
      if (this.isArray(value) && this.isArray(existing[key])) {
        // Merge arrays intelligently
        merged[key] = this.mergeArrays(existing[key], value);
      } else if (this.isObject(value) && this.isObject(existing[key])) {
        // Recursively merge objects
        merged[key] = this.smartMerge(existing[key], value);
      } else {
        // Replace primitive values
        merged[key] = value;
      }
    }

    return merged;
  }
}
```

## Context Lifecycle Management

### 1. Context Creation

```typescript
class ContextLifecycleManager {
  /**
   * Creates new context with validation and initialization
   * @param contextData - Initial context data
   * @param options - Creation options
   */
  async createContext(
    contextData: ContextCreationData,
    options: ContextCreationOptions = {}
  ): Promise<Context> {
    // Validate context data
    await this.validateContext(contextData);

    // Initialize default values
    const initializedContext = this.initializeContext(contextData);

    // Create in database
    const context = await this.database.createContext(initializedContext);

    // Set up monitoring and alerts
    await this.setupContextMonitoring(context);

    // Trigger creation webhooks
    await this.notifyContextChange(context.id, "create", context);

    return context;
  }
}
```

### 2. Context Expiration

```typescript
class ContextLifecycleManager {
  /**
   * Manages context expiration and cleanup
   * @param contextId - Context to manage
   */
  async manageContextExpiration(contextId: string): Promise<void> {
    const context = await this.getContextById(contextId);

    if (this.isExpired(context)) {
      // Archive expired context
      await this.archiveContext(context);

      // Clean up related data
      await this.cleanupRelatedContexts(context);

      // Notify stakeholders
      await this.notifyContextExpiration(context);
    }
  }

  private isExpired(context: Context): boolean {
    if (!context.expiresAt) return false;
    return new Date() > context.expiresAt;
  }
}
```

## Context Analytics and Insights

### 1. Access Pattern Analysis

```typescript
class ContextAnalytics {
  /**
   * Analyzes context access patterns for optimization
   * @param timeRange - Time range for analysis
   */
  async analyzeAccessPatterns(
    timeRange: TimeRange
  ): Promise<AccessPatternReport> {
    const accessLogs = await this.database.getAccessLogs(timeRange);

    return {
      mostAccessedContexts: this.identifyHotContexts(accessLogs),
      accessFrequency: this.calculateAccessFrequency(accessLogs),
      performanceMetrics: this.analyzePerformance(accessLogs),
      optimizationRecommendations: this.generateRecommendations(accessLogs),
    };
  }
}
```

### 2. Context Effectiveness Metrics

```typescript
class ContextAnalytics {
  /**
   * Measures context effectiveness in prompt generation
   * @param contextId - Context to analyze
   */
  async measureContextEffectiveness(
    contextId: string
  ): Promise<EffectivenessMetrics> {
    const context = await this.getContextById(contextId);
    const relatedPrompts = await this.getRelatedPrompts(contextId);

    return {
      contextUtilization: this.calculateUtilization(context, relatedPrompts),
      promptSuccessRate: this.measurePromptSuccess(relatedPrompts),
      userSatisfaction: this.measureUserSatisfaction(relatedPrompts),
      performanceImpact: this.measurePerformanceImpact(context),
    };
  }
}
```

## Security and Privacy

### 1. Context Encryption

```typescript
class ContextSecurity {
  /**
   * Encrypts sensitive context data before storage
   * @param contextData - Context data to encrypt
   * @param encryptionKey - Encryption key
   */
  async encryptContext(
    contextData: any,
    encryptionKey: string
  ): Promise<EncryptedContext> {
    const sensitiveFields = this.identifySensitiveFields(contextData);

    for (const field of sensitiveFields) {
      contextData[field] = await this.encryptValue(
        contextData[field],
        encryptionKey
      );
    }

    return {
      data: contextData,
      encryptionMetadata: {
        algorithm: "AES-256-GCM",
        timestamp: new Date(),
        keyId: this.getKeyId(encryptionKey),
      },
    };
  }
}
```

### 2. Access Control

```typescript
class ContextAccessControl {
  /**
   * Validates access permissions for context operations
   * @param userId - User requesting access
   * @param contextId - Context being accessed
   * @param operation - Operation being performed
   */
  async validateAccess(
    userId: string,
    contextId: string,
    operation: ContextOperation
  ): Promise<AccessValidationResult> {
    const context = await this.getContextById(contextId);
    const userPermissions = await this.getUserPermissions(userId);

    // Check ownership
    if (context.userId && context.userId !== userId) {
      return { allowed: false, reason: "Not owner of context" };
    }

    // Check app permissions
    if (!this.hasAppPermission(userPermissions, context.appId, operation)) {
      return { allowed: false, reason: "Insufficient app permissions" };
    }

    // Check rate limits
    if (await this.isRateLimited(userId, operation)) {
      return { allowed: false, reason: "Rate limit exceeded" };
    }

    return { allowed: true };
  }
}
```

## Performance Optimization

### 1. Context Caching Strategies

```typescript
class ContextCacheManager {
  /**
   * Implements intelligent caching for context data
   * @param contextKey - Context identifier
   * @param contextType - Type of context
   */
  async getCachedContext(
    contextKey: string,
    contextType: ContextType
  ): Promise<Context | null> {
    const cacheKey = `${contextType}:${contextKey}`;

    // Try L1 cache (in-memory)
    let context = this.l1Cache.get(cacheKey);
    if (context) return context;

    // Try L2 cache (Redis)
    context = await this.l2Cache.get(cacheKey);
    if (context) {
      // Populate L1 cache
      this.l1Cache.set(cacheKey, context);
      return context;
    }

    return null;
  }
}
```

### 2. Context Preloading

```typescript
class ContextPreloader {
  /**
   * Preloads likely-needed contexts based on user behavior
   * @param userId - User to preload contexts for
   */
  async preloadUserContexts(userId: string): Promise<void> {
    const userBehavior = await this.analyzeUserBehavior(userId);
    const likelyContexts = this.predictLikelyContexts(userBehavior);

    // Preload in background
    this.preloadContexts(likelyContexts).catch((error) => {
      this.logger.warn("Context preloading failed", { userId, error });
    });
  }

  private async preloadContexts(contextKeys: string[]): Promise<void> {
    const contexts = await Promise.all(
      contextKeys.map((key) => this.getContext(key, "user_profile"))
    );

    // Cache preloaded contexts
    for (const context of contexts) {
      if (context) {
        await this.cache.set(`user_profile:${context.id}`, context);
      }
    }
  }
}
```

## Integration Examples

### 1. SDK Integration

```typescript
// PCS SDK context management
const pcs = new PCSSDK({
  baseUrl: process.env.PCS_BASE_URL,
  apiKey: process.env.PCS_API_KEY,
  appId: process.env.PCS_APP_ID,
});

// Get user context for personalized experience
const userContext = await pcs.context.getUserContext(userId, {
  includeHistory: true,
  includePreferences: true,
  includePerformance: true,
});

// Update context based on user interaction
await pcs.context.updateUserContext(userId, {
  lastActivity: new Date(),
  currentTopic: "machine_learning",
  interactionCount: userContext.interactionCount + 1,
});
```

### 2. Webhook Integration

```typescript
// Webhook handler for context changes
app.post("/webhooks/context-updated", async (req, res) => {
  const { contextId, changeType, contextData } = req.body;

  // Update local cache
  await localContextCache.update(contextId, contextData);

  // Trigger dependent processes
  await eventBus.emit("context:updated", {
    contextId,
    changeType,
    timestamp: new Date(),
  });

  res.status(200).json({ success: true });
});
```

## Monitoring and Alerting

### 1. Context Health Checks

```typescript
class ContextHealthMonitor {
  /**
   * Monitors context system health and performance
   */
  async performHealthCheck(): Promise<HealthCheckResult> {
    const checks = await Promise.all([
      this.checkDatabaseConnectivity(),
      this.checkCachePerformance(),
      this.checkContextIntegrity(),
      this.checkAccessPatterns(),
    ]);

    const overallHealth = this.aggregateHealthResults(checks);

    if (overallHealth.status === "unhealthy") {
      await this.triggerAlerts(overallHealth);
    }

    return overallHealth;
  }
}
```

### 2. Performance Metrics

```typescript
class ContextMetrics {
  /**
   * Collects and reports context performance metrics
   */
  async collectMetrics(): Promise<ContextMetricsReport> {
    return {
      responseTime: await this.measureResponseTime(),
      cacheHitRate: await this.calculateCacheHitRate(),
      contextSize: await this.measureContextSize(),
      accessFrequency: await this.measureAccessFrequency(),
      errorRate: await this.calculateErrorRate(),
    };
  }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Context Not Found**

   - Check if context has expired
   - Verify access permissions
   - Check cache invalidation

2. **Performance Degradation**

   - Monitor cache hit rates
   - Check database query performance
   - Review context size and complexity

3. **Data Inconsistency**
   - Verify merge strategies
   - Check conflict resolution
   - Review transaction handling

## Related Documentation

- [Dynamic Prompting Architecture](DYNAMIC_PROMPTING_ARCHITECTURE.md)
- [PCS SDK Reference](PCS_SDK_REFERENCE.md)
- [Prompt Templates](PROMPT_TEMPLATES.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
- [Schema Management](SCHEMA_MANAGEMENT.md)
