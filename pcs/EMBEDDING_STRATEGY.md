# Vector Database Embedding Strategy

## 🎯 **Embedding Dimensions & Free Options**

### **Why 1536 vs 768 vs 384 Dimensions?**

| Model | Dimensions | Cost | Performance | Use Case |
|-------|------------|------|-------------|----------|
| **OpenAI text-embedding-ada-002** | 1536 | 💰 **Paid API** | ⭐⭐⭐⭐⭐ Excellent | Production with budget |
| **all-mpnet-base-v2** | 768 | 🆓 **FREE** | ⭐⭐⭐⭐ Very Good | **Recommended Default** |
| **all-MiniLM-L6-v2** | 384 | 🆓 **FREE** | ⭐⭐⭐ Good | Lightweight/Fast |

### **🆓 FREE Embedding Options (NO API CHARGES)**

#### **Option 1: Sentence Transformers (Recommended)**
```bash
# Install sentence-transformers
uv add sentence-transformers

# Set environment variable for model choice
export EMBEDDING_MODEL="all-mpnet-base-v2"  # 768 dimensions, excellent quality
# OR
export EMBEDDING_MODEL="all-MiniLM-L6-v2"   # 384 dimensions, faster
```

**Available Free Models:**
- `all-mpnet-base-v2` (768 dims) - **Best quality/performance balance**
- `all-MiniLM-L6-v2` (384 dims) - Fastest, good quality
- `all-MiniLM-L12-v2` (384 dims) - Better than L6, still fast
- `paraphrase-mpnet-base-v2` (768 dims) - Excellent for paraphrasing
- `multi-qa-mpnet-base-dot-v1` (768 dims) - Optimized for Q&A

#### **Option 2: Hugging Face Transformers**
```python
from transformers import AutoTokenizer, AutoModel
import torch

# Use any BERT-style model for free embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

#### **Option 3: Local Model Hosting**
```bash
# Run your own embedding service
git clone https://github.com/UKPLab/sentence-transformers
cd sentence-transformers
pip install -e .
```

### **💰 Paid Options (API Charges Apply)**

#### **OpenAI Embeddings**
```bash
# Set API key (requires payment)
export OPENAI_API_KEY="your-api-key-here"
```

**OpenAI Models:**
- `text-embedding-ada-002` (1536 dims) - $0.0001 per 1K tokens
- `text-embedding-3-small` (1536 dims) - $0.00002 per 1K tokens
- `text-embedding-3-large` (3072 dims) - $0.00013 per 1K tokens

## 🔧 **Implementation Strategy**

### **Current Implementation Priority:**
1. **🆓 FREE FIRST**: Try sentence-transformers (768 dims)
2. **💰 PAID BACKUP**: Fall back to OpenAI (1536 dims) if API key available
3. **🧪 TEST FALLBACK**: Deterministic vectors for testing

### **Configuration Examples:**

#### **Free Setup (Recommended)**
```bash
# Install free embedding model
uv add sentence-transformers

# Configure for free usage
export EMBEDDING_MODEL="all-mpnet-base-v2"  # 768 dimensions
# No API key needed - completely free!
```

#### **OpenAI Setup (If you prefer paid API)**
```bash
# Set OpenAI API key
export OPENAI_API_KEY="sk-..."
# Will use 1536-dimensional vectors from OpenAI
```

#### **Mixed Environment**
```bash
# Use free model as primary, OpenAI as backup
export EMBEDDING_MODEL="all-mpnet-base-v2"
export OPENAI_API_KEY="sk-..."  # Optional backup
```

## 📊 **Vector Database Configuration**

### **Collection Creation with Proper Dimensions:**

```python
from pcs.repositories.qdrant_http_repo import EnhancedQdrantHTTPRepository

# For free sentence-transformers (768 dims)
repo = EnhancedQdrantHTTPRepository()
await repo.create_collection_optimized(
    collection_name="my_collection",
    vector_size=768,  # Matches all-mpnet-base-v2
    distance="cosine"
)

# For OpenAI embeddings (1536 dims)
await repo.create_collection_optimized(
    collection_name="my_openai_collection", 
    vector_size=1536,  # Matches OpenAI text-embedding-ada-002
    distance="cosine"
)
```

## 🎯 **Recommendations**

### **For Development/Testing (100% Free):**
```bash
uv add sentence-transformers
export EMBEDDING_MODEL="all-mpnet-base-v2"
# Use 768-dimensional vectors - excellent quality, zero cost
```

### **For Production (Budget-Conscious):**
```bash
# Start with free models
export EMBEDDING_MODEL="all-mpnet-base-v2"  # Primary: Free 768-dim
export OPENAI_API_KEY="sk-..."               # Backup: Paid 1536-dim
# System automatically chooses best available option
```

### **For Maximum Performance (Budget Available):**
```bash
export OPENAI_API_KEY="sk-..."
# Uses OpenAI 1536-dimensional embeddings for best semantic understanding
```

## 🔍 **Performance Comparison**

| Strategy | Quality | Speed | Cost | Storage |
|----------|---------|-------|------|---------|
| **Free (768d)** | ⭐⭐⭐⭐ | ⚡⚡⚡ | 🆓 | ~3KB per vector |
| **OpenAI (1536d)** | ⭐⭐⭐⭐⭐ | ⚡⚡ | 💰 | ~6KB per vector |
| **Lightweight (384d)** | ⭐⭐⭐ | ⚡⚡⚡⚡ | 🆓 | ~1.5KB per vector |

## 🚀 **Quick Start (Zero Cost)**

```bash
# 1. Install free embedding model
cd /workspace/pcs
uv add sentence-transformers

# 2. Set environment for free model
export EMBEDDING_MODEL="all-mpnet-base-v2"

# 3. Create collection with correct dimensions
python -c "
import asyncio
from src.pcs.repositories.qdrant_http_repo import EnhancedQdrantHTTPRepository

async def setup():
    repo = EnhancedQdrantHTTPRepository()
    await repo.create_collection_optimized('test_collection', vector_size=768)
    print('✅ Free 768-dimensional collection created!')

asyncio.run(setup())
"

# 4. Generate embeddings (completely free)
python -c "
import asyncio
from src.pcs.services.prompt_service import PromptGenerator

async def test_embedding():
    # This will use the free sentence-transformers model
    generator = PromptGenerator(None, None, None, None, None)
    embedding = await generator._generate_embedding('Hello world')
    print(f'✅ Generated {len(embedding)}-dimensional embedding for free!')

asyncio.run(test_embedding())
"
```

**Result: High-quality semantic search with ZERO API costs!** 🎉