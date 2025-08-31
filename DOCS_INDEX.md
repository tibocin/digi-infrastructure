# Digi Infrastructure Documentation Index

Welcome to the central entry point for all Digi Infrastructure documentation. This index provides quick links and brief descriptions to help you navigate the project docs efficiently.

---

## Project Overview

- [Root README](./README.md) - Overview of the Digi ecosystem shared infrastructure, core components, architecture, and quick start.

## Setup and Getting Started

- [PCS Setup Guide](./pcs/SETUP.md) - **UPDATED** - Step-wise guide for setting up the Prompt and Context Service (PCS) with enhanced Qdrant repository and 100% test coverage.

## API Reference

- [PCS API Documentation](./pcs/docs/API_DOCUMENTATION.md) - Detailed REST API endpoints for the PCS service including authentication, health checks, prompts, contexts, and admin.

## Application Onboarding

- [App Onboarding Guide](./docs/APP_ONBOARDING.md) - Comprehensive guide to onboarding new applications into the Digi ecosystem including planning, architecture review, and resource planning.

## Schema and Database Management

- [Schema Management](./docs/SCHEMA_MANAGEMENT.md) - Explanation of schema and migration management per app in the multi-app architecture.

## Infrastructure

- [Infrastructure Repository](./docs/INFRASTRUCTURE_REPOSITORY.md) - Details of the shared infrastructure repository, components, and docker-compose configuration with Qdrant connection examples.
- [Qdrant Infrastructure](./docs/QDRANT_INFRASTRUCTURE.md) - Comprehensive Qdrant vector database configuration, performance optimization, and multi-tenant deployment.
- [Tenant Qdrant Integration](./docs/TENANT_QDRANT_INTEGRATION.md) - Multi-tenant Qdrant integration patterns, collection management, and isolation strategies.

## Implementation Plans

- [Background Agent Implementation Plan](./docs/BACKGROUND_AGENT_IMPLEMENTATION_PLAN.md) - Micro-step implementation strategy for the PCS autonomous agent system including tech stack and dev setup.

## Core Architecture

- [Dynamic Prompting Architecture](./docs/DYNAMIC_PROMPTING_ARCHITECTURE.md) - Overview of the dynamic prompting layer, PCS components, and service architecture.

## Downstream App Integration

- [App Onboarding Guide](./docs/APP_ONBOARDING.md) - Complete guide for apps to connect to digi-infrastructure including Qdrant integration examples.
- [Multi-App Deployment](./docs/MULTI_APP_DEPLOYMENT.md) - Multi-tenant deployment patterns with Qdrant collection management and isolation strategies.

---

## PCS Core Code Documentation

### Core Modules

- [Configuration Management](./pcs/src/pcs/core/config.py) - Application configuration management using Pydantic. Includes security, database, and Redis settings with env validations.
- [Database Management](./pcs/src/pcs/core/database.py) - Async database connection management using SQLAlchemy 2.0. Handles engine/session creation, pooling, events, and health checks.

### Repositories

- [**ENHANCED** Qdrant Repository](./pcs/src/pcs/repositories/qdrant_repo.py) - **NEW** - Advanced modular Qdrant repository orchestrating specialized modules for vector database operations, clustering, bulk operations, and performance monitoring.
- [Qdrant Core Operations](./pcs/src/pcs/repositories/qdrant_core.py) - **NEW** - Basic CRUD operations, collection management, and point operations.
- [Qdrant Advanced Search](./pcs/src/pcs/repositories/qdrant_advanced_search.py) - **NEW** - Semantic search with multiple algorithms, filtering, and similarity scoring.
- [Qdrant Bulk Operations](./pcs/src/pcs/repositories/qdrant_bulk.py) - **NEW** - Batch processing with error handling, progress tracking, and retry mechanisms.
- [Qdrant Performance Monitor](./pcs/src/pcs/repositories/qdrant_performance.py) - **NEW** - Real-time performance metrics, optimization recommendations, and collection tuning.
- [Qdrant Clustering](./pcs/src/pcs/repositories/qdrant_clustering.py) - **NEW** - Document clustering algorithms (K-means, DBSCAN) for vector analysis.
- [Qdrant Export](./pcs/src/pcs/repositories/qdrant_export.py) - **NEW** - Data export functionality in multiple formats (numpy, JSON, list) with tenant filtering.
- [Qdrant Legacy](./pcs/src/pcs/repositories/qdrant_legacy.py) - **NEW** - Backward compatibility methods for existing integrations.
- [Qdrant HTTP Repository](./pcs/src/pcs/repositories/qdrant_http_repo.py) - HTTP-based Qdrant operations for applications that prefer REST API over gRPC.
- [Qdrant HTTP Client](./pcs/src/pcs/repositories/qdrant_http_client.py) - HTTP client wrapper for Qdrant with connection pooling and error handling.

### Models

- [Prompt Models](./pcs/src/pcs/models/prompts.py) - Data models for prompt templates, versions, rules with SQLAlchemy ORM and enums for status and priority.

### API Endpoints

- [Health API](./pcs/src/pcs/api/v1/health.py) - Health check API endpoints for monitoring service status including basic and detailed checks.

---

### Additional PCS Core Modules and Components

- [Connection Pool Manager](./pcs/src/pcs/core/connection_pool_manager.py) - Unified connection pool management for Postgres, Redis, Neo4j, and Qdrant with health monitoring and performance metrics.
- [Optimized PostgreSQL Repository](./pcs/src/pcs/repositories/postgres_repo.py) - PostgreSQL-specific repository with batch processing, caching, and query optimizations.
- [Contexts Models](./pcs/src/pcs/models/contexts.py) - Data models for context types, relationships, and scopes, including validation schemas.
- [Prompts API](./pcs/src/pcs/api/v1/prompts.py) - REST API endpoints for managing prompt templates, versions, and rules.

---

## PCS Testing and Quality Assurance

### Test Coverage Status

✅ **100% Test Coverage Achieved for Core Qdrant Functionality:**

- [Export Tests](./pcs/tests/unit/test_qdrant_export.py) - **16/16 PASSING** - NumPy array handling, format conversion, tenant filtering
- [Async Tests](./pcs/tests/unit/test_qdrant_async.py) - **22/22 PASSING** - Mock configurations, method signatures, async/await patterns  
- [Legacy Tests](./pcs/tests/unit/test_qdrant_legacy.py) - **20/20 PASSING** - Backward compatibility, parameter handling, method delegation

### Test Infrastructure

- [Test Configuration](./pcs/tests/unit/conftest.py) - **UPDATED** - Shared test fixtures and mocks for Qdrant tests with proper async support.
- [Test Results](./pcs/TEST_RESULTS.md) - **UPDATED** - Comprehensive test results and coverage reports.

---

## PCS Integration Examples

### Downstream Application Integration

- [Integration Example](./pcs/examples/downstream_app_integration.py) - **NEW** - Complete example showing how digi-core, beep-boop, and other applications integrate with PCS Qdrant system.

### Multi-Tenant Setup

- [Multi-App Support](./docs/MULTI_APP_DEPLOYMENT.md) - **UPDATED** - Patterns for multiple applications sharing infrastructure with isolated data.

---

## Digi Infrastructure Core Code Documentation

- [Port Validator](./port-management/digi-infrastructure/port-validator.py) - Validates port allocations and detects conflicts to ensure no port clashes across infrastructure services, with YAML-based configuration.

---

## Latest Updates and Features

### 🆕 **Enhanced Qdrant Repository (Latest)**

The PCS now features a **production-ready, modular Qdrant repository** with:

- **Specialized Modules**: Core operations, advanced search, bulk operations, performance monitoring, clustering
- **Multi-Tenant Support**: Isolated collections for digi-core, beep-boop, and other applications
- **Performance Optimization**: Automatic collection tuning and HNSW configuration
- **Clustering Capabilities**: K-means and DBSCAN document clustering algorithms
- **Comprehensive Testing**: 100% test coverage with modern testing practices

### 🔧 **Integration Ready**

- **SDK Support**: TypeScript/JavaScript, Python, and Go client libraries
- **Multi-App Architecture**: Designed for digi-core, beep-boop, lernmi, and other applications
- **Backward Compatibility**: Legacy API methods for existing integrations
- **Performance Monitoring**: Real-time metrics and optimization recommendations

### 📚 **Documentation Status**

- **Core README**: Updated with latest functionality and examples
- **Setup Guide**: Comprehensive setup instructions with enhanced repository features
- **API Reference**: Complete endpoint documentation
- **Integration Examples**: Ready-to-use code examples for downstream applications

---

For more detailed or specific docs, please explore the documented directories or relevant markdown files within the project.

If you want, the next step can be to scan more code files and enhance this documentation index further.

---

_Generated by Goose AI Documentation Assistant - Updated with Latest PCS Functionality_
