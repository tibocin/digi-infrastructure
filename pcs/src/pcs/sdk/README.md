# PCS SDK - Prompt and Context Service SDK

**Version**: 1.0.0  
**Languages**: TypeScript, Python, Go  
**Documentation**: Complete implementation recipes for AI-powered applications

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Patterns](#architecture-patterns)
- [Implementation Recipes](#implementation-recipes)
- [Best Practices](#best-practices)
- [Contributing](#contributing)

## Overview

The PCS SDK provides comprehensive tools for building AI-powered applications with dynamic prompting, context management, and conversation handling. It integrates seamlessly with the Digi Infrastructure ecosystem to provide shared databases, monitoring, and AI services.

### Key Features

- **Dynamic Prompting**: Context-aware prompt generation and management
- **Multi-Modal Support**: Handle text, code, images, and structured data
- **Conversation Management**: Persistent conversation history and context
- **RAG Integration**: Vector databases and semantic search
- **Agent Orchestration**: Support for multi-agent systems
- **Real-time Updates**: WebSocket support for live interactions
- **Monitoring & Analytics**: Built-in metrics and performance tracking

### Supported Languages

| Language | Package | Documentation |
|----------|---------|---------------|
| TypeScript/JavaScript | `@pcs/typescript-sdk` | [TypeScript Guide](typescript/) |
| Python | `digi-pcs-sdk` | [Python Guide](python/) |
| Go | `github.com/digi/pcs-sdk` | [Go Guide](go/) |

## Installation

### Prerequisites

1. **Digi Infrastructure** running (PostgreSQL, Neo4j, Redis, ChromaDB)
2. **PCS Service** deployed and accessible
3. **Ollama** (for local LLM support)
4. **Node.js 18+** (for TypeScript) or **Python 3.9+** (for Python)

### TypeScript/JavaScript

```bash
npm install @pcs/typescript-sdk
# or
yarn add @pcs/typescript-sdk
```

### Python

```bash
pip install digi-pcs-sdk
# or
poetry add digi-pcs-sdk
```

### Go

```bash
go get github.com/digi/pcs-sdk
```

## Quick Start

### 1. Initialize PCS Client

```typescript
// TypeScript
import { PCSClient } from '@pcs/typescript-sdk'

const pcs = new PCSClient({
  base_url: 'http://localhost:8000',
  api_key: process.env.PCS_API_KEY,
  app_id: process.env.PCS_APP_ID
})
```

```python
# Python
from digi_pcs import PCSClient

pcs = PCSClient(
    base_url="http://localhost:8000",
    api_key=os.getenv("PCS_API_KEY"),
    app_id=os.getenv("PCS_APP_ID")
)
```

### 2. Create Your First Prompt

```typescript
const prompt = await pcs.createPrompt({
  name: 'user_greeting',
  content: 'Hello {{user_name}}! I\'m {{agent_name}}, how can I help you with {{domain}}?',
  variables: {
    user_name: 'string',
    agent_name: 'string', 
    domain: 'string'
  },
  category: 'greetings'
})
```

### 3. Generate Dynamic Content

```typescript
const response = await pcs.generatePrompt('user_greeting', {
  context: {
    user_name: 'Alice',
    agent_name: 'Beep-Boop',
    domain: 'personal assistance'
  }
})

console.log(response.generated_prompt)
// Output: "Hello Alice! I'm Beep-Boop, how can I help you with personal assistance?"
```

## Architecture Patterns

The PCS SDK supports several architectural patterns for building AI applications:

### 1. Single Agent Pattern
- One AI agent with dynamic prompting
- Context-aware responses
- Conversation history

### 2. Multi-Agent Orchestration  
- Multiple specialized agents
- Shared context and knowledge base
- Agent-to-agent communication

### 3. RAG-Enhanced Pattern
- Vector database integration
- Semantic search and retrieval
- Context injection into prompts

### 4. Reinforcement Learning Pattern
- Training feedback loops
- Performance optimization
- Continuous improvement

## Implementation Recipes

Complete, production-ready implementations for real-world AI applications:

### 🤖 [Recipe 1: Beep-Boop - Multi-Modal Digital Twin](../../../docs/recipes/beep-boop.md)
A conversational AI digital twin that learns about its user through interactions, maintains a personal RAG database, and provides personalized multi-modal responses.

**Key Features:**
- Personal memory management with ChromaDB
- Multi-modal content processing (text, images, audio, code)
- Relationship tracking with Neo4j
- Continuous learning and personality adaptation

### 🛠️ [Recipe 2: DEVAO - Virtual Development Shop](../../../docs/recipes/devao.md)
A multi-agent development team using LangGraph to coordinate 10 specialized Ollama agents that autonomously build, test, monitor, and maintain software projects.

**Key Features:**
- 10 specialized development agents (Architect, Frontend, Backend, QA, etc.)
- LangGraph workflow orchestration
- Automated code generation, testing, and deployment
- Continuous integration and monitoring

### 🧠 [Recipe 3: LERNMI - Reinforcement Learning Agent](../../../docs/recipes/lernmi.md)
An Ollama-powered reinforcement AI agent that interacts with Beep-Boop for continuous training and improvement through feedback loops.

**Key Features:**
- Reinforcement learning with reward signals
- Training episode management
- Performance optimization through interaction
- Feedback analysis and pattern recognition

### ₿ [Recipe 4: Bitscrow - Bitcoin Smart Contract AI](../../../docs/recipes/bitscrow.md)
A conversational AI specialized in facilitating Bitcoin/Lightning Network smart contracts and oracle-powered agreements between parties.

**Key Features:**
- Smart contract negotiation and generation
- Bitcoin/Lightning Network integration
- Oracle data validation
- Dispute resolution mechanisms

## Best Practices

### 1. Prompt Design
- **Use specific, contextual prompts** for each agent's specialization
- **Include clear instructions** about response format and expectations
- **Provide relevant context** from conversation history and user profile
- **Test prompts extensively** with various scenarios and edge cases

### 2. Memory Management
- **Store structured memories** with confidence scores and relevance tags
- **Implement memory decay** to prioritize recent and frequently accessed information
- **Use vector embeddings** for semantic memory retrieval
- **Regular memory cleanup** to prevent storage bloat

### 3. Multi-Agent Coordination
- **Clear role definitions** for each agent's responsibilities
- **Shared context management** through PCS
- **Event-driven communication** between agents
- **Conflict resolution** mechanisms for disagreements

### 4. Performance Optimization
- **Cache frequently used prompts** and responses
- **Implement connection pooling** for database connections
- **Use async/await** for non-blocking operations
- **Monitor and optimize** query performance

### 5. Error Handling
- **Graceful degradation** when services are unavailable
- **Comprehensive logging** for debugging and monitoring
- **Retry mechanisms** with exponential backoff
- **User-friendly error messages** that maintain agent personality

## Troubleshooting

### Common Issues

1. **PCS Connection Failures**
   - Check network connectivity and API credentials
   - Verify PCS service is running and accessible
   - Review authentication token expiration

2. **Database Performance Issues**
   - Monitor query execution times
   - Check index usage and optimize queries
   - Review connection pool settings

3. **Memory Collection Errors**
   - Verify ChromaDB service status
   - Check embedding model availability
   - Review memory storage limits

4. **Agent Response Quality**
   - Review prompt templates for clarity
   - Check context data quality and relevance
   - Monitor feedback loops and learning metrics

## Contributing

We welcome contributions to improve these recipes and add new ones!

### Guidelines
- Follow the established pattern for new recipes
- Include comprehensive database schemas
- Provide complete code examples
- Add thorough documentation
- Test with real scenarios

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/digi-infrastructure.git
cd digi-infrastructure

# Start infrastructure
make up

# Install SDK dependencies
cd pcs/src/pcs/sdk
npm install  # for TypeScript
pip install -e .  # for Python
```

### Adding New Recipes

1. Create a new file in `docs/recipes/your-recipe.md`
2. Follow the established recipe structure
3. Include complete implementation code
4. Add comprehensive examples and usage
5. Update this README with a link to your recipe

## Related Documentation

- [Dynamic Prompting Architecture](../../../docs/DYNAMIC_PROMPTING_ARCHITECTURE.md)
- [PCS API Reference](../../../docs/PCS_SDK_REFERENCE.md)
- [Infrastructure Overview](../../../docs/INFRASTRUCTURE_SUMMARY.md)
- [App Onboarding Guide](../../../docs/APP_ONBOARDING.md)

---

*For detailed implementation guides, see the individual recipe files in [docs/recipes/](../../../docs/recipes/).*

