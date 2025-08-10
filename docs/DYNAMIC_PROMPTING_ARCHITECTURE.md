# Dynamic Prompting Architecture & Prompt and Context Service (PCS)

## Overview

The Dynamic Prompting Architecture is the core intelligence layer of the Digi ecosystem, providing adaptive, context-aware prompting capabilities through the Prompt and Context Service (PCS). This service acts as the foundation for all applications, enabling sophisticated AI interactions with dynamic context management.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Dynamic Prompting Layer                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Prompt and    │  │   Context       │  │   Dynamic       │              │
│  │   Context       │  │   Management    │  │   Orchestration │              │
│  │   Service (PCS) │  │   Engine        │  │   Engine        │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│           │                     │                     │                     │
│  ┌───────┴────────┐  ┌─────────┴─────────┐  ┌───────┴────────┐              │
│  │   PCS SDK      │  │   Context Store   │  │   Prompt       │              │
│  │   (Client      │  │   (PostgreSQL +   │  │   Templates    │              │
│  │   Libraries)   │  │   Redis Cache)    │  │   & Rules      │              │
│  └─────────────────┘  └───────────────────┘  └───────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prompt and Context Service (PCS)

### Core Responsibilities

The PCS serves as the central intelligence hub for:

1. **Dynamic Prompt Generation**: Creates context-aware prompts based on user interactions
2. **Context Management**: Maintains conversation state and user context across sessions
3. **Prompt Orchestration**: Coordinates multiple prompt strategies and fallbacks
4. **Performance Optimization**: Caches and optimizes prompt responses
5. **Analytics & Insights**: Tracks prompt effectiveness and user engagement

### Service Architecture

```yaml
# PCS Service Definition
services:
  pcs-core:
    image: digi/pcs-core:latest
    container_name: digi-pcs-core
    environment:
      - PCS_DATABASE_URL=${PCS_DATABASE_URL}
      - PCS_REDIS_URL=${PCS_REDIS_URL}
      - PCS_OPENAI_API_KEY=${OPENAI_API_KEY}
      - PCS_LOG_LEVEL=${PCS_LOG_LEVEL:-info}
    ports:
      - "8000:8000"
    volumes:
      - ./pcs/config:/app/config
      - ./pcs/templates:/app/templates
    networks:
      - digi-net
    depends_on:
      - postgres
      - redis

  pcs-worker:
    image: digi/pcs-worker:latest
    container_name: digi-pcs-worker
    environment:
      - PCS_DATABASE_URL=${PCS_DATABASE_URL}
      - PCS_REDIS_URL=${PCS_REDIS_URL}
      - PCS_QUEUE_NAME=pcs_tasks
    volumes:
      - ./pcs/jobs:/app/jobs
    networks:
      - digi-net
    depends_on:
      - postgres
      - redis
```

### Database Schema

```sql
-- Core PCS Tables
CREATE TABLE pcs_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    template TEXT NOT NULL,
    variables JSONB,
    rules JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE pcs_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    context_data JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE pcs_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    prompt_id UUID REFERENCES pcs_prompts(id),
    input_text TEXT,
    response_text TEXT,
    context_snapshot JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE pcs_apps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    app_name VARCHAR(255) NOT NULL UNIQUE,
    app_key VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB,
    rate_limits JSONB,
    webhook_urls JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## PCS SDK

### SDK Architecture

The PCS SDK provides client libraries for multiple languages, enabling seamless integration:

```typescript
// TypeScript/JavaScript SDK Example
import { PCSSDK } from "@digi/pcs-sdk";

const pcs = new PCSSDK({
  baseUrl: "http://localhost:8000",
  apiKey: "your-app-api-key",
  appId: "your-app-id",
});

// Dynamic prompt generation
const response = await pcs.generatePrompt({
  promptName: "user_onboarding",
  context: {
    userLevel: "beginner",
    preferredLanguage: "en",
    useCase: "learning",
  },
  variables: {
    userName: "John",
    topic: "machine learning",
  },
});
```

```python
# Python SDK Example
from digi_pcs import PCSSDK

pcs = PCSSDK(
    base_url="http://localhost:8000",
    api_key="your-app-api-key",
    app_id="your-app-id"
)

# Context-aware conversation
response = pcs.conversation.create(
    prompt_name="adaptive_learning",
    context={
        "user_progress": 0.7,
        "learning_style": "visual",
        "difficulty_level": "intermediate"
    }
)
```

### SDK Features

1. **Automatic Context Management**: Maintains conversation state automatically
2. **Prompt Caching**: Intelligent caching of frequently used prompts
3. **Fallback Strategies**: Automatic fallback to alternative prompts
4. **Performance Monitoring**: Built-in metrics and analytics
5. **Rate Limiting**: Respects app-specific rate limits
6. **Webhook Support**: Real-time notifications for prompt updates

## Dynamic Prompting Engine

### Prompt Template System

```yaml
# Prompt Template Example
name: "adaptive_learning_prompt"
template: |
  You are an expert tutor helping {{user_name}} learn {{topic}}.

  Based on their learning style ({{learning_style}}) and current progress ({{progress_percentage}}%), 
  provide a {{difficulty_level}} explanation that builds on their previous knowledge.

  Context from previous sessions: {{session_context}}

  Focus on: {{focus_areas}}

  Avoid: {{avoid_topics}}

variables:
  - user_name: string
  - topic: string
  - learning_style: enum[visual, auditory, kinesthetic]
  - progress_percentage: number
  - difficulty_level: enum[beginner, intermediate, advanced]
  - session_context: string
  - focus_areas: array
  - avoid_topics: array

rules:
  - condition: "progress_percentage < 0.3"
    action: "use_beginner_explanation"
    fallback: "basic_concept_intro"

  - condition: "learning_style == 'visual'"
    action: "include_visual_elements"
    fallback: "text_based_explanation"

metadata:
  category: "education"
  tags: ["learning", "adaptive", "personalized"]
  version: "1.0.0"
```

### Context Management

The context management system maintains:

1. **Session Context**: Current conversation state and user preferences
2. **Historical Context**: Previous interactions and learning patterns
3. **User Profile**: Learning style, preferences, and progress
4. **App Context**: Application-specific settings and configurations
5. **Temporal Context**: Time-based relevance and seasonal adjustments

### Dynamic Orchestration

```typescript
// Orchestration Logic Example
class PromptOrchestrator {
  async orchestratePrompt(request: PromptRequest): Promise<PromptResponse> {
    // 1. Analyze context and user state
    const context = await this.analyzeContext(request);

    // 2. Select optimal prompt strategy
    const strategy = await this.selectStrategy(context);

    // 3. Generate personalized prompt
    const prompt = await this.generatePrompt(strategy, context);

    // 4. Apply fallback rules if needed
    const finalPrompt = await this.applyFallbacks(prompt, context);

    // 5. Track performance metrics
    await this.trackMetrics(finalPrompt, context);

    return finalPrompt;
  }
}
```

## App Onboarding & Initialization

### Onboarding Process

1. **App Registration**

   ```bash
   # Register new app with PCS
   curl -X POST http://localhost:8000/api/v1/apps \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "app_name": "my_new_app",
       "description": "AI-powered learning platform",
       "permissions": ["prompt_read", "context_write"],
       "rate_limits": {"requests_per_minute": 100}
     }'
   ```

2. **SDK Integration**

   ```typescript
   // Initialize SDK with app credentials
   const pcs = new PCSSDK({
     baseUrl: process.env.PCS_BASE_URL,
     apiKey: process.env.PCS_API_KEY,
     appId: process.env.PCS_APP_ID,
   });

   // Test connection
   await pcs.health.check();
   ```

3. **Prompt Configuration**
   ```yaml
   # App-specific prompt configuration
   app_prompts:
     - name: "app_welcome"
       template: "Welcome to {{app_name}}! {{user_greeting}}"
       variables:
         - app_name: string
         - user_greeting: string
   ```

### Initialization Checklist

- [ ] App registered with PCS
- [ ] SDK integrated and tested
- [ ] Initial prompts configured
- [ ] Context schema defined
- [ ] Rate limits configured
- [ ] Webhook endpoints set up
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery tested

## Monitoring & Analytics

### Key Metrics

1. **Prompt Performance**

   - Response time
   - Success rate
   - User satisfaction scores
   - Fallback usage frequency

2. **Context Management**

   - Context hit rate
   - Context size and complexity
   - Context update frequency
   - Memory usage

3. **App Usage**
   - API call volume
   - Error rates
   - Rate limit violations
   - Feature adoption

### Grafana Dashboards

The PCS provides pre-configured Grafana dashboards for:

- Real-time prompt performance
- Context management efficiency
- App usage analytics
- System health monitoring
- Error tracking and alerting

## Security & Access Control

### Authentication

- API key-based authentication for apps
- JWT tokens for user sessions
- Rate limiting per app and user
- IP whitelisting for production access

### Authorization

- Role-based access control (RBAC)
- App-specific permissions
- Prompt access restrictions
- Context data privacy controls

## Deployment & Scaling

### Horizontal Scaling

```yaml
# Production PCS deployment
services:
  pcs-core:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
    environment:
      - PCS_ENVIRONMENT=production
      - PCS_LOG_LEVEL=warn
```

### Load Balancing

- Round-robin load balancing across PCS instances
- Health checks and automatic failover
- Session affinity for context consistency
- Global load balancing for multi-region deployment

## Integration Patterns

### Microservices Integration

```typescript
// Service-to-service communication
class LearningService {
  async createLesson(userId: string, topic: string) {
    // Get personalized prompt from PCS
    const prompt = await this.pcs.generatePrompt({
      promptName: "lesson_creation",
      context: { userId, topic },
      variables: { topic },
    });

    // Use prompt to generate lesson content
    const lesson = await this.aiService.generateContent(prompt);

    // Update user context
    await this.pcs.updateContext(userId, {
      lastLesson: topic,
      lessonCount: (await this.getLessonCount(userId)) + 1,
    });

    return lesson;
  }
}
```

### Event-Driven Architecture

```typescript
// Event handling with PCS
class PCSEventHandler {
  async handleUserProgressUpdate(event: UserProgressEvent) {
    // Update context based on progress
    await this.pcs.updateContext(event.userId, {
      progress: event.newProgress,
      lastActivity: new Date(),
      achievements: event.achievements,
    });

    // Trigger adaptive prompt generation
    await this.pcs.triggerPrompt("progress_celebration", {
      userId: event.userId,
      progress: event.newProgress,
    });
  }
}
```

## Future Enhancements

1. **Multi-Modal Prompts**: Support for image, audio, and video context
2. **Federated Learning**: Collaborative prompt improvement across apps
3. **Real-time Collaboration**: Shared context for collaborative sessions
4. **Advanced Analytics**: ML-powered prompt optimization
5. **Edge Computing**: Local prompt processing for low-latency applications

## Related Documentation

- [App Onboarding Guide](APP_ONBOARDING.md)
- [PCS SDK Reference](PCS_SDK_REFERENCE.md)
- [Context Management](CONTEXT_MANAGEMENT.md)
- [Prompt Templates](PROMPT_TEMPLATES.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
