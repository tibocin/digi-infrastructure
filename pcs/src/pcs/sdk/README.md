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
  - [Recipe 1: Beep-Boop - Multi-Modal Digital Twin](#recipe-1-beep-boop---multi-modal-digital-twin)
  - [Recipe 2: DEVAO - Virtual Development Shop](#recipe-2-devao---virtual-development-shop)
  - [Recipe 3: LERNMI - Reinforcement Learning Agent](#recipe-3-lernmi---reinforcement-learning-agent)
  - [Recipe 4: Bitscrow - Bitcoin Smart Contract AI](#recipe-4-bitscrow---bitcoin-smart-contract-ai)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
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

---

# Implementation Recipes

The following recipes provide complete, end-to-end implementations for real-world AI applications using the PCS SDK.
