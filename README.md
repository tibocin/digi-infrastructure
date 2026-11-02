# Digi Infrastructure

Shared infrastructure foundation for the Digi ecosystem, including database services, dynamic prompting capabilities, and the Prompt and Context Service (PCS).

## Overview

This repository serves as the foundation for the entire Digi ecosystem, providing:

- **Shared Database Infrastructure**: PostgreSQL, Neo4j, Qdrant, Redis
- **Dynamic Prompting Architecture**: Prompt and Context Service (PCS) with intelligent context management
- **Advanced Vector Operations**: Enhanced Qdrant repository with clustering, bulk operations, and performance optimization
- **Multi-Tenant Support**: Isolated data management for multiple applications
- **Monitoring & Observability**: Prometheus, Grafana, and comprehensive health checks
- **App Onboarding & Integration**: Complete SDK and onboarding processes for new applications
- **Multi-App Support**: Infrastructure designed to serve multiple applications simultaneously

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Infrastructure                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  PostgreSQL │  │    Neo4j    │  │    Qdrant   │          │
│  │   Container │  │  Container  │  │  Container  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │ digi_core   │  │ digi_core   │  │ digi_core   │          │
│  │ lernmi      │  │ lernmi      │  │ lernmi      │          │
│  │ beep_boop   │  │ beep_boop   │  │ beep_boop   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │     PCS     │  │     PCS     │  │     PCS     │          │
│  │  Enhanced   │  │  Enhanced   │  │  Enhanced   │          │
│  │  Qdrant     │  │  Qdrant     │  │  Qdrant     │          │
│  │ Repository  │  │ Repository  │  │ Repository  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 🆕 **Latest Features: Enhanced PCS with Advanced Qdrant Repository**

The **Prompt and Context Service (PCS)** now features a **production-ready, modular Qdrant repository** with:

### 🧠 **Advanced Vector Operations**
- **Modular Architecture**: Specialized modules for core operations, advanced search, bulk operations, performance monitoring, and clustering
- **Multi-Tenant Support**: Isolated collections for digi-core, beep-boop, lernmi, and other applications
- **Performance Optimization**: Automatic collection tuning, HNSW configuration, and real-time metrics
- **Clustering Capabilities**: K-means and DBSCAN document clustering algorithms
- **Comprehensive Testing**: 100% test coverage with modern testing practices

### 🔧 **Integration Ready**
- **SDK Support**: TypeScript/JavaScript, Python, and Go client libraries
- **Multi-App Architecture**: Designed for seamless integration with existing applications
- **Backward Compatibility**: Legacy API methods for existing integrations
- **Performance Monitoring**: Real-time metrics and optimization recommendations

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/digi-infrastructure.git
   cd digi-infrastructure
   ```

2. **Configure environment**

   ```bash
   cp config/env.example .env
   # Edit .env with your values
   ```

3. **Start infrastructure**

   **Public / private host (ports unpublished — preferred for showcase):**

   ```bash
   make private-up
   make private-health
   make assert-private-ports
   ```

   See [docs/PRIVATE_STACK.md](docs/PRIVATE_STACK.md).

   **Local digi (publishes ports via override):**

   ```bash
   make up
   make health
   ```

## Services

| Service    | Container port | Purpose                      | Host publish                          |
| ---------- | -------------- | ---------------------------- | ------------------------------------- |
| PostgreSQL | 5432           | Relational database          | private: none / local override: 5432  |
| Neo4j      | 7474 / 7687    | Graph database               | private: none / local: 7474, 7687     |
| Qdrant     | 6333 / 6334    | Vector database              | private: none / local: 6333, 6334     |
| Redis      | 6379           | Cache & sessions             | private: none / local: 6379           |
| **PCS**    | **8000**       | **Prompt & Context Service** | private: none / local: 8000           |
| Prometheus | 9090           | Metrics collection           | local override only                   |
| Grafana    | 3000           | Monitoring dashboards        | local override only                   |

On the private stack, peers reach services by compose DNS on `digi-net`
(e.g. `http://pcs:8000`), not via host localhost.

## Multi-App Database Setup

This infrastructure supports multiple applications, each with its own database:

- **digi_core**: Main RAG application database
- **lernmi**: Learning and evaluation database
- **beep_boop**: Bot application database
- **pcs**: Prompt and Context Service database
- **stackr**: Database of bitcoin price, dca purchase schedule, etc
- **bitscrow**: Bitcoin escrow terms, contracts, ifps, psbt database
- **satsflow**: Lightening nodes, routing, fees, etc database
- **devao**: Agentric Devshop agents, prompts, output, etc
- **cvpunk**: Company, Job and Hiring Manager database

Each app connects using its own credentials and database name.

## PCS Integration Examples

### Using PCS in Your Applications

```python
# In digi-core or beep-boop
from pcs.repositories.qdrant_repo import EnhancedQdrantRepository

# Initialize repository
repo = EnhancedQdrantRepository(
    host="localhost",
    port=6333,
    use_async=True
)

# Create app-specific collection
await repo.create_collection_optimized(
    collection_name="digi_core_documents",
    vector_size=384,
    distance="cosine"
)

# Add documents with tenant isolation
await repo.bulk_upsert_documents(
    collection_name="digi_core_documents",
    documents=my_documents,
    tenant_id="digi_core"
)

# Search with tenant filtering
results = await repo.search_similar(
    collection_name="digi_core_documents",
    query_embedding=query_vector,
    tenant_id="digi_core"
)
```

### Multi-Tenant Setup

```python
# Each app gets isolated collections
digi_core_collection = "digi_core_documents"
beep_boop_collection = "beep_boop_documents"

# Tenant-specific operations
await repo.upsert_documents(digi_core_collection, documents, tenant_id="digi_core")
await repo.search_similar(beep_boop_collection, query_vector, tenant_id="beep_boop")
```

## Development

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM available
- 50GB+ disk space
- Python 3.11+ with UV package manager (for PCS development)

### Common Commands

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Health check
make health

# Show status
make status

# Create backup
make backup

# Restore from backup
make restore

# PCS Development
cd pcs
uv sync --extra dev
uv run pytest  # Run tests
uv run pcs-server  # Start development server
```

## Connecting Applications

Applications connect to this infrastructure using external links:

```yaml
# In your app's docker-compose.yml
services:
  my-app:
    external_links:
      - digi-infrastructure_postgres_1:postgres
      - digi-infrastructure_neo4j_1:neo4j
      - digi-infrastructure_qdrant_1:qdrant
      - digi-infrastructure_redis_1:redis
    networks:
      - digi-net
```

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474
- **Qdrant**: http://localhost:6333
- **PCS API**: http://localhost:8000

## Documentation

- [Central Documentation Index](./DOCS_INDEX.md) - Single entry point for all Digi Infrastructure docs, with quick links and summaries.

- [PCS Setup Guide](./pcs/SETUP.md) - **UPDATED** - Complete setup guide with enhanced Qdrant repository
- [PCS README](./pcs/README.md) - **UPDATED** - Comprehensive overview of PCS capabilities
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Schema Management](docs/SCHEMA_MANAGEMENT.md)
- [Infrastructure Repository Design](docs/INFRASTRUCTURE_REPOSITORY.md)
- [Multi-App Deployment Strategy](docs/MULTI_APP_DEPLOYMENT.md)
- [App Onboarding Guide](docs/APP_ONBOARDING.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make health` and `uv run pytest` (in PCS directory)
5. Submit a pull request

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
