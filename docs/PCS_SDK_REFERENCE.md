# PCS SDK Reference

## Overview

The Prompt and Context Service (PCS) SDK provides client libraries for multiple programming languages, enabling seamless integration with the Digi ecosystem's dynamic prompting capabilities. This reference covers all available SDKs, their APIs, and usage patterns.

## Supported Languages

- **TypeScript/JavaScript** - `@digi/pcs-sdk`
- **Python** - `digi-pcs-sdk`
- **Go** - `github.com/digi/pcs-sdk`

## TypeScript/JavaScript SDK

### Installation

```bash
npm install @digi/pcs-sdk
# or
yarn add @digi/pcs-sdk
```

### Basic Setup

```typescript
import { PCSSDK } from "@digi/pcs-sdk";

const pcs = new PCSSDK({
  baseUrl: "http://localhost:8000",
  apiKey: "your-app-api-key",
  appId: "your-app-id",
  timeout: 30000,
  retries: 3,
});
```

### Configuration Options

```typescript
interface PCSSDKConfig {
  baseUrl: string; // PCS service URL
  apiKey: string; // App authentication key
  appId: string; // App identifier
  timeout?: number; // Request timeout in ms (default: 30000)
  retries?: number; // Retry attempts (default: 3)
  cacheEnabled?: boolean; // Enable response caching (default: true)
  cacheTTL?: number; // Cache TTL in seconds (default: 300)
  logLevel?: "debug" | "info" | "warn" | "error";
}
```

### Core API Methods

#### Prompt Management

```typescript
// Generate a prompt with context
const response = await pcs.prompts.generate({
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

// Create a new prompt template
const prompt = await pcs.prompts.create({
  name: "custom_prompt",
  template: "Hello {{user_name}}, welcome to {{app_name}}!",
  variables: ["user_name", "app_name"],
  rules: [
    {
      condition: "user_name.length > 0",
      action: "use_personalized_greeting",
    },
  ],
  metadata: {
    category: "greeting",
    tags: ["welcome", "personalized"],
  },
});

// Update existing prompt
await pcs.prompts.update("custom_prompt", {
  template: "Hi {{user_name}}, welcome to {{app_name}}!",
  variables: ["user_name", "app_name"],
});

// Delete prompt
await pcs.prompts.delete("custom_prompt");

// List all prompts
const prompts = await pcs.prompts.list({
  category: "greeting",
  tags: ["welcome"],
});
```

#### Context Management

```typescript
// Create or update user context
await pcs.context.update("user_123", {
  learningProgress: 0.7,
  preferredTopics: ["AI", "ML", "Data Science"],
  lastActivity: new Date(),
  sessionCount: 15,
});

// Get user context
const context = await pcs.context.get("user_123");

// Get context with specific fields
const progress = await pcs.context.get("user_123", [
  "learningProgress",
  "preferredTopics",
]);

// Delete user context
await pcs.context.delete("user_123");

// Search contexts
const contexts = await pcs.context.search({
  query: "learningProgress > 0.5",
  limit: 10,
});
```

#### Conversation Management

```typescript
// Start a new conversation
const conversation = await pcs.conversations.create({
  sessionId: "session_456",
  userId: "user_123",
  initialContext: {
    topic: "machine learning",
    difficulty: "intermediate",
  },
});

// Add message to conversation
await pcs.conversations.addMessage("session_456", {
  type: "user",
  content: "How do neural networks work?",
  timestamp: new Date(),
});

// Get conversation history
const history = await pcs.conversations.getHistory("session_456", {
  limit: 50,
  includeContext: true,
});

// Update conversation context
await pcs.conversations.updateContext("session_456", {
  currentTopic: "neural networks",
  userUnderstanding: "beginner",
});
```

#### Health & Monitoring

```typescript
// Check service health
const health = await pcs.health.check();

// Get service metrics
const metrics = await pcs.health.metrics();

// Get app usage statistics
const stats = await pcs.health.appStats();
```

### Advanced Features

#### Batch Operations

```typescript
// Batch prompt generation
const responses = await pcs.prompts.generateBatch([
  {
    promptName: "greeting",
    context: { userLevel: "beginner" },
    variables: { userName: "Alice" },
  },
  {
    promptName: "greeting",
    context: { userLevel: "advanced" },
    variables: { userName: "Bob" },
  },
]);

// Batch context updates
await pcs.context.updateBatch([
  { userId: "user_1", context: { progress: 0.5 } },
  { userId: "user_2", context: { progress: 0.8 } },
]);
```

#### Streaming Responses

```typescript
// Stream prompt generation
const stream = await pcs.prompts.generateStream({
  promptName: "long_explanation",
  context: { userLevel: "beginner" },
  variables: { topic: "machine learning" },
});

for await (const chunk of stream) {
  console.log("Received chunk:", chunk.content);
}
```

#### Event Handling

```typescript
// Subscribe to prompt updates
const unsubscribe = pcs.events.subscribe("prompt.updated", (event) => {
  console.log("Prompt updated:", event.promptName);
  // Refresh local prompt cache
});

// Subscribe to context changes
pcs.events.subscribe("context.changed", (event) => {
  console.log("Context changed for user:", event.userId);
  // Update local state
});

// Unsubscribe
unsubscribe();
```

### Error Handling

```typescript
try {
  const response = await pcs.prompts.generate({
    promptName: "nonexistent_prompt",
    context: {},
    variables: {},
  });
} catch (error) {
  if (error instanceof PCSError) {
    switch (error.code) {
      case "PROMPT_NOT_FOUND":
        console.error("Prompt does not exist");
        break;
      case "RATE_LIMIT_EXCEEDED":
        console.error("Rate limit exceeded, retry later");
        break;
      case "INVALID_CONTEXT":
        console.error("Invalid context provided");
        break;
      default:
        console.error("Unknown error:", error.message);
    }
  }
}
```

## Python SDK

### Installation

```bash
pip install digi-pcs-sdk
```

### Basic Setup

```python
from digi_pcs import PCSSDK

pcs = PCSSDK(
    base_url="http://localhost:8000",
    api_key="your-app-api-key",
    app_id="your-app-id",
    timeout=30,
    retries=3
)
```

### Core API Methods

```python
# Generate prompt
response = pcs.prompts.generate(
    prompt_name="user_onboarding",
    context={
        "user_level": "beginner",
        "preferred_language": "en"
    },
    variables={
        "user_name": "John",
        "topic": "machine learning"
    }
)

# Update context
await pcs.context.update(
    user_id="user_123",
    context_data={
        "learning_progress": 0.7,
        "preferred_topics": ["AI", "ML"]
    }
)

# Create conversation
conversation = await pcs.conversations.create(
    session_id="session_456",
    user_id="user_123",
    initial_context={"topic": "machine learning"}
)
```

### Async Support

```python
import asyncio
from digi_pcs import AsyncPCSSDK

async def main():
    pcs = AsyncPCSSDK(
        base_url="http://localhost:8000",
        api_key="your-app-api-key",
        app_id="your-app-id"
    )

    response = await pcs.prompts.generate(
        prompt_name="greeting",
        context={"user_level": "beginner"},
        variables={"user_name": "Alice"}
    )

    print(response.content)

# Run async function
asyncio.run(main())
```

## Go SDK

### Installation

```bash
go get github.com/digi/pcs-sdk
```

### Basic Setup

```go
package main

import (
    "log"
    "github.com/digi/pcs-sdk"
)

func main() {
    client := pcs.NewClient(&pcs.Config{
        BaseURL: "http://localhost:8000",
        APIKey:  "your-app-api-key",
        AppID:   "your-app-id",
        Timeout: 30 * time.Second,
    })

    // Generate prompt
    response, err := client.Prompts.Generate(&pcs.GenerateRequest{
        PromptName: "user_onboarding",
        Context: map[string]interface{}{
            "userLevel": "beginner",
        },
        Variables: map[string]interface{}{
            "userName": "John",
        },
    })

    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Generated prompt: %s", response.Content)
}
```

### Core API Methods

```go
// Context management
err = client.Context.Update("user_123", map[string]interface{}{
    "learningProgress": 0.7,
    "preferredTopics": []string{"AI", "ML"},
})

// Conversation management
conversation, err := client.Conversations.Create(&pcs.CreateConversationRequest{
    SessionID: "session_456",
    UserID:    "user_123",
    Context: map[string]interface{}{
        "topic": "machine learning",
    },
})
```

## Testing & Mocking

### Mock PCS for Testing

```typescript
// Create mock PCS for unit tests
class MockPCSSDK {
  private responses = new Map<string, any>();

  setResponse(method: string, params: any, response: any) {
    const key = `${method}:${JSON.stringify(params)}`;
    this.responses.set(key, response);
  }

  prompts = {
    generate: async (request: any) => {
      const key = `generate:${JSON.stringify(request)}`;
      return (
        this.responses.get(key) || {
          content: "Mock response",
          metadata: { mocked: true },
        }
      );
    },
  };
}

// Usage in tests
const mockPCS = new MockPCSSDK();
mockPCS.setResponse(
  "generate",
  {
    promptName: "test",
    context: { userId: "123" },
  },
  {
    content: "Test response",
    metadata: { test: true },
  }
);
```

## SDK Updates & Migration

### Version Compatibility

```typescript
// Check SDK version
import { version } from "@digi/pcs-sdk/package.json";
console.log("PCS SDK Version:", version);

// Feature detection
if (pcs.features.supports("streaming")) {
  // Use streaming features
} else {
  // Fallback to non-streaming
}
```

### Migration Guide

```typescript
// v1.x to v2.x migration
// Old API
const response = await pcs.generatePrompt("prompt_name", context, variables);

// New API
const response = await pcs.prompts.generate({
  promptName: "prompt_name",
  context,
  variables,
});
```

## Support & Resources

### Documentation

- **API Reference**: [PCS API Documentation](https://docs.digi.com/pcs/api)
- **Examples**: [GitHub Examples Repository](https://github.com/digi/pcs-examples)
- **Tutorials**: [Getting Started Guide](https://docs.digi.com/pcs/getting-started)

### Community

- **Discord**: [Digi Ecosystem Discord](https://discord.gg/digi)
- **GitHub Issues**: [Report bugs and request features](https://github.com/digi/pcs-sdk/issues)
- **Discussions**: [Community discussions](https://github.com/digi/pcs-sdk/discussions)

### Support

- **Email**: support@digi.com
- **Slack**: #pcs-support channel
- **Documentation**: [Troubleshooting Guide](https://docs.digi.com/pcs/troubleshooting)

## Related Documentation

- [Dynamic Prompting Architecture](DYNAMIC_PROMPTING_ARCHITECTURE.md)
- [App Onboarding Guide](APP_ONBOARDING.md)
- [Context Management](CONTEXT_MANAGEMENT.md)
- [Prompt Templates](PROMPT_TEMPLATES.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
