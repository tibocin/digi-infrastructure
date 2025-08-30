# PCS Unified Prompts Engine

**Filepath:** `pcs/README_UNIFIED_PROMPTS.md`  
**Purpose:** Comprehensive documentation for the unified prompts engine implementation  
**Related Components:** Vector search, Neo4j reasoning, cross-application intelligence  
**Tags:** unified-engine, documentation, vector-search, reasoning, cross-app

## 🚀 Overview

The **PCS Unified Prompts Engine** is a sophisticated, AI-driven prompt selection and optimization system that combines vector search, reasoning chains, and cross-application intelligence. This engine transforms PCS from a simple template engine into an intelligent, learning system that continuously improves prompt effectiveness.

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Unified Prompts Engine                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   Vector-Based  │  │   Intelligent   │  │   Neo4j         │              │
│  │   Prompt        │  │   Router        │  │   Reasoning     │              │
│  │   Discovery     │  │                 │  │   Engine        │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│           │                     │                     │                     │
│  ┌───────┴────────┐  ┌─────────┴─────────┐  ┌───────┴────────┐              │
│  │   Qdrant       │  │   Routing         │  │   Reasoning     │              │
│  │   Vector DB    │  │   Strategies      │  │   Chains        │              │
│  └─────────────────┘  └───────────────────┘  └───────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query Input** → User query with context
2. **Vector Search** → Find similar prompts using embeddings
3. **Intelligent Routing** → Apply reasoning strategies for optimal selection
4. **Reasoning Tracking** → Store reasoning chains in Neo4j
5. **Cross-App Intelligence** → Learn from other applications
6. **Optimization** → Generate improvement suggestions

## 🔧 Implementation Phases

### Phase 1: Vector-Based Prompt Indexing ✅

**Status:** COMPLETED  
**Components:** Enhanced prompt service with Qdrant integration

#### Features
- **Prompt Vectorization**: Convert prompt templates to vector embeddings
- **Semantic Search**: Find similar prompts using vector similarity
- **Multi-tenant Support**: App-specific collections with isolation
- **Metadata Filtering**: Rich filtering capabilities

#### Code Example
```python
# Index a prompt template for vector search
await prompt_generator.index_prompt_template(
    prompt_template=template,
    app_id="beep_boop",
    collection_name="beep_boop_prompts"
)

# Search for similar prompts
similar_prompts = await prompt_generator.search_similar_prompts(
    query_text="How do I implement authentication?",
    app_id="beep_boop",
    limit=5,
    similarity_threshold=0.7
)
```

### Phase 2: Intelligent Prompt Router ✅

**Status:** COMPLETED  
**Components:** Sophisticated routing with multiple strategies

#### Routing Strategies
- **Vector Only**: Pure similarity-based selection
- **Reasoning Only**: Logic-based selection (placeholder for Phase 3)
- **Hybrid**: Combine vector and reasoning scores
- **Confidence Based**: Weighted scoring with metadata boosts

#### Code Example
```python
# Create routing request
request = PromptRoutingRequest(
    query_text="Authentication implementation help",
    app_id="beep_boop",
    user_context={"topic": "security", "level": "beginner"},
    routing_strategy=RoutingStrategy.HYBRID,
    max_candidates=5,
    similarity_threshold=0.7
)

# Route to optimal prompt
result = await intelligent_router.route_to_optimal_prompt(request)
print(f"Selected: {result.selected_prompt_id}")
print(f"Confidence: {result.confidence_score:.2f}")
```

### Phase 3: Neo4j Reasoning Engine ✅

**Status:** COMPLETED  
**Components:** Graph-based reasoning and cross-app intelligence

#### Features
- **Reasoning Chains**: Track Query → Reasoning → Prompt relationships
- **Success Pattern Analysis**: Learn from successful prompt selections
- **Cross-App Insights**: Discover patterns across applications
- **Context Optimization**: Track and analyze context improvements

#### Code Example
```python
# Track reasoning chain
reasoning_id = await neo4j_service.track_reasoning_chain(
    query_text="How to implement OAuth2?",
    reasoning_approach="hybrid_vector_reasoning",
    prompt_id="prompt_123",
    app_id="beep_boop",
    user_context={"topic": "security"},
    confidence=0.85
)

# Analyze success patterns
patterns = await neo4j_service.analyze_success_patterns(
    app_id="beep_boop",
    time_window_days=30
)
```

## 🌐 Cross-Application Intelligence

### Sharing Modes

| Mode | Description | Cross-App Prompts | Cross-App Learning |
|------|-------------|-------------------|-------------------|
| `DISABLED` | No cross-app features | ❌ | ❌ |
| `READ_ONLY` | Can read from other apps | ✅ | ✅ |
| `SHARED` | Full cross-app sharing | ✅ | ✅ |
| `INTELLIGENT` | AI-driven cross-app selection | ✅ | ✅ |

### Benefits

1. **Shared Knowledge**: Learn from successful prompts across apps
2. **Pattern Recognition**: Identify common success patterns
3. **Resource Optimization**: Avoid duplicating effective prompts
4. **Continuous Improvement**: Cross-pollination of best practices

## 📊 Analytics & Insights

### What Gets Tracked

- **Prompt Performance**: Success rates, usage patterns
- **Reasoning Effectiveness**: Which approaches work best
- **Cross-App Usage**: How often cross-app prompts are selected
- **Context Optimization**: Performance improvements from context changes

### Sample Analytics Output

```json
{
  "engine": {
    "total_requests": 150,
    "successful_requests": 142,
    "success_rate": 0.947,
    "cross_app_usage_count": 23,
    "reasoning_chain_count": 142
  },
  "routing": {
    "total_requests": 150,
    "successful_routings": 142,
    "success_rate": 0.947,
    "cross_app_usage_rate": 0.162
  },
  "neo4j": {
    "node_counts": {
      "Query": 150,
      "Reasoning": 142,
      "Prompt": 89
    },
    "relationship_counts": {
      "REQUIRES_REASONING": 142,
      "GENERATES_PROMPT": 142
    }
  }
}
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 2. Configure Services

```python
from pcs.services.unified_prompts_engine import UnifiedPromptsEngine
from pcs.services.intelligent_prompt_router import IntelligentPromptRouter
from pcs.services.neo4j_service import PCSNeo4jService

# Initialize services
neo4j_service = PCSNeo4jService(
    neo4j_uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)

# Create unified engine
unified_engine = UnifiedPromptsEngine(
    prompt_generator=your_prompt_generator,
    intelligent_router=your_intelligent_router,
    neo4j_service=neo4j_service,
    enable_cross_app=True,
    cross_app_mode=CrossAppMode.INTELLIGENT
)
```

### 3. Generate Intelligent Prompts

```python
# Create request
request = UnifiedPromptRequest(
    query_text="How do I implement user authentication?",
    app_id="your_app",
    user_context={
        "topic": "security",
        "level": "beginner",
        "framework": "python"
    },
    routing_strategy=RoutingStrategy.HYBRID,
    cross_app_mode=CrossAppMode.INTELLIGENT
)

# Generate prompt
result = await unified_engine.generate_unified_prompt(request)

print(f"Selected Prompt: {result.prompt_id}")
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Reasoning Chain: {result.reasoning_chain_id}")
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_unified_prompts.py -v

# Run with coverage
uv run pytest --cov=pcs
```

### Demo Script

```bash
# Run the interactive demo
uv run python demo_unified_engine.py
```

## 🔍 Monitoring & Health Checks

### Health Check Endpoint

```python
# Check overall system health
health = await unified_engine.health_check()

print(f"Status: {health['status']}")
for component, status in health['components'].items():
    print(f"{component}: {status['status']}")
```

### Performance Metrics

```python
# Get comprehensive statistics
stats = await unified_engine.get_engine_statistics()

print(f"Success Rate: {stats['engine']['success_rate']:.2%}")
print(f"Cross-App Usage: {stats['engine']['cross_app_usage_count']}")
print(f"Reasoning Chains: {stats['engine']['reasoning_chain_count']}")
```

## 🎯 Use Cases

### 1. **Beep-Boop Learning Assistant**
- **Query**: "How do I learn Python decorators?"
- **Vector Search**: Find similar learning prompts
- **Cross-App**: Learn from LernMI's successful teaching approaches
- **Result**: Optimized prompt with proven success patterns

### 2. **LernMI Content Creation**
- **Query**: "Create a lesson plan for machine learning basics"
- **Vector Search**: Find similar educational content prompts
- **Cross-App**: Learn from Beep-Boop's user engagement patterns
- **Result**: Content creation prompt optimized for engagement

### 3. **Cross-Application Pattern Recognition**
- **Pattern**: "Visual examples work better for beginners"
- **Source**: Beep-Boop user feedback
- **Application**: LernMI lesson planning
- **Result**: Automatic preference for visual prompts

## 🔮 Future Enhancements

### Phase 4: Advanced Reasoning
- **Multi-hop Reasoning**: Complex reasoning chains across multiple steps
- **Causal Inference**: Understand cause-effect relationships
- **Temporal Patterns**: Learn from time-based success patterns

### Phase 5: Predictive Optimization
- **Predictive Prompt Selection**: Anticipate user needs
- **Automatic A/B Testing**: Continuous optimization
- **Adaptive Learning**: Real-time strategy adjustment

### Phase 6: Federated Learning
- **Privacy-Preserving Sharing**: Share insights without sharing data
- **Distributed Reasoning**: Collaborative intelligence across apps
- **Edge Optimization**: Local optimization with global insights

## 🛠️ Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check Neo4j service status
   - Verify connection credentials
   - Ensure database exists

2. **Vector Search Not Working**
   - Check Qdrant service status
   - Verify collection exists
   - Check embedding generation

3. **Cross-App Features Disabled**
   - Verify `enable_cross_app` is True
   - Check cross-app mode setting
   - Ensure proper app isolation

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging for troubleshooting
unified_engine.logger.setLevel(logging.DEBUG)
```

## 📚 API Reference

### Core Classes

- **`UnifiedPromptsEngine`**: Main engine orchestrating all components
- **`IntelligentPromptRouter`**: Routes queries to optimal prompts
- **`PCSNeo4jService`**: Manages reasoning chains and cross-app intelligence
- **`PromptGenerator`**: Enhanced with vector indexing capabilities

### Key Methods

- **`generate_unified_prompt()`**: Main entry point for prompt generation
- **`route_to_optimal_prompt()`**: Intelligent prompt routing
- **`track_reasoning_chain()`**: Store reasoning for analysis
- **`get_cross_app_insights()`**: Discover cross-app patterns

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/unified-prompts`
3. **Make changes and test**: `uv run pytest`
4. **Submit pull request**

### Code Standards

- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings for all methods
- **Testing**: 90%+ test coverage required
- **Formatting**: Use `black` and `isort`

## 📄 License

This project is part of the Digi Infrastructure ecosystem and follows the same licensing terms.

---

**🎉 Congratulations!** You now have a production-ready, intelligent prompts engine that combines the power of vector search, reasoning, and cross-application intelligence. This system will continuously learn and improve, making your AI applications smarter with every interaction.
