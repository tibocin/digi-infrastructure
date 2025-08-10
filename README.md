# Digi Infrastructure

Shared infrastructure foundation for the Digi ecosystem, including database services, dynamic prompting capabilities, and the Prompt and Context Service (PCS).

## Overview

This repository serves as the foundation for the entire Digi ecosystem, providing:

- **Shared Database Infrastructure**: PostgreSQL, Neo4j, ChromaDB, Redis
- **Dynamic Prompting Architecture**: Prompt and Context Service (PCS) with intelligent context management
- **Monitoring & Observability**: Prometheus, Grafana, and comprehensive health checks
- **App Onboarding & Integration**: Complete SDK and onboarding processes for new applications
- **Multi-App Support**: Infrastructure designed to serve multiple applications simultaneously

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared Infrastructure                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  PostgreSQL │  │    Neo4j    │  │   ChromaDB  │          │
│  │   Container │  │  Container  │  │  Container  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                 │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐          │
│  │ digi_core   │  │ digi_core   │  │ digi_core   │          │
│  │ lernmi      │  │ lernmi      │  │ lernmi      │          │
│  │ beep_boop   │  │ beep_boop   │  │ beep_boop   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

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

   ```bash
   make up
   ```

4. **Verify health**
   ```bash
   make health
   ```

## Services

| Service    | Port | Purpose               | Access                                     |
| ---------- | ---- | --------------------- | ------------------------------------------ |
| PostgreSQL | 5432 | Relational database   | `postgresql://user:pass@localhost:5432/db` |
| Neo4j      | 7474 | Graph database        | `http://localhost:7474`                    |
| ChromaDB   | 8001 | Vector database       | `http://localhost:8001`                    |
| Redis      | 6379 | Cache & sessions      | `redis://localhost:6379`                   |
| Prometheus | 9090 | Metrics collection    | `http://localhost:9090`                    |
| Grafana    | 3000 | Monitoring dashboards | `http://localhost:3000`                    |

## Multi-App Database Setup

This infrastructure supports multiple applications, each with its own database:

- **digi_core**: Main RAG application database
- **lernmi**: Learning and evaluation database
- **beep_boop**: Bot application database

Each app connects using its own credentials and database name.

## Development

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM available
- 50GB+ disk space

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
      - digi-infrastructure_chroma_1:chroma
      - digi-infrastructure_redis_1:redis
    networks:
      - digi-net
```

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474
- **ChromaDB**: http://localhost:8001

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Schema Management](docs/SCHEMA_MANAGEMENT.md)
- [Infrastructure Repository Design](docs/INFRASTRUCTURE_REPOSITORY.md)
- [Multi-App Deployment Strategy](docs/MULTI_APP_DEPLOYMENT.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `make health`
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
