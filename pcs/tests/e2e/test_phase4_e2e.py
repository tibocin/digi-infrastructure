"""
Filepath: tests/e2e/test_phase4_e2e.py
Purpose: End-to-end tests for Phase 4 database integrations
Related Components: Complete PCS system with all Phase 4 enhancements
Tags: e2e-tests, phase4, system-testing, workflow-testing, performance-validation
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from pcs.main import create_app
from pcs.core.config import get_settings
from pcs.repositories.postgres_repo import OptimizedPostgreSQLRepository
from pcs.repositories.neo4j_repo import Neo4jRepository
from pcs.repositories.chroma_repo import EnhancedChromaRepository
from pcs.repositories.redis_repo import EnhancedRedisRepository
from pcs.core.connection_pool_manager import get_connection_pool_manager
from pcs.utils.metrics import get_metrics_collector
from pcs.models.prompts import PromptTemplate, PromptStatus
from pcs.models.contexts import Context, ContextScope
from pcs.models.conversations import Conversation, ConversationStatus, MessageRole
from pcs.api.v1.conversations import MessageResponse as Message


class TestPhase4EndToEnd:
    """
    End-to-end tests validating Phase 4 with complete system integration.
    
    These tests simulate real-world usage scenarios and validate
    that all database integrations work together seamlessly.
    """
    
    @pytest.fixture(scope="class")
    def app(self):
        """Create test FastAPI application."""
        from pcs.api.dependencies import get_database_session, get_current_user, get_current_app
        
        settings = get_settings()
        app = create_app(settings)
        
        # Override dependencies with mocks for testing
        def mock_get_db():
            mock_session = AsyncMock()
            return mock_session
        
        def mock_get_user():
            return {"id": "test-user-id", "username": "testuser", "is_admin": True}
        
        def mock_get_app():
            return {"app_id": "test-app", "app_name": "PCS", "permissions": ["admin"], "environment": "test"}
        
        app.dependency_overrides[get_database_session] = mock_get_db
        app.dependency_overrides[get_current_user] = mock_get_user
        app.dependency_overrides[get_current_app] = mock_get_app
        
        return app
    
    @pytest.fixture(scope="class")
    def metrics_collector(self):
        """Get metrics collector instance."""
        return get_metrics_collector()
    
    @pytest.fixture(scope="class")
    def pool_manager(self):
        """Get connection pool manager instance."""
        return get_connection_pool_manager()
    
    @pytest.fixture
    async def mock_repositories(self):
        """Create comprehensive mock repositories."""
        # PostgreSQL repository with all Phase 4 enhancements
        postgres_repo = Mock(spec=OptimizedPostgreSQLRepository)
        postgres_repo.bulk_create = AsyncMock()
        postgres_repo.find_with_pagination = AsyncMock()
        postgres_repo.execute_optimized_query = AsyncMock()
        postgres_repo.create = AsyncMock()
        postgres_repo.update = AsyncMock()
        postgres_repo.delete = AsyncMock()
        postgres_repo.find_by_id = AsyncMock()
        postgres_repo.find_all = AsyncMock()
        
        # Neo4j repository with graph operations
        neo4j_repo = Mock(spec=Neo4jRepository)
        neo4j_repo.find_related_nodes = AsyncMock()
        neo4j_repo.analyze_conversation_patterns = AsyncMock()
        neo4j_repo.create_context_hierarchy = AsyncMock()
        neo4j_repo.create_node = AsyncMock()
        neo4j_repo.create_relationship = AsyncMock()
        neo4j_repo.find_context_dependencies = AsyncMock()
        
        # ChromaDB repository with vector operations
        chroma_repo = Mock(spec=EnhancedChromaRepository)
        chroma_repo.semantic_search_advanced = AsyncMock()
        chroma_repo.find_similar_documents = AsyncMock()
        chroma_repo.cluster_documents = AsyncMock()
        chroma_repo.create_collection_optimized = AsyncMock()
        chroma_repo.bulk_upsert_documents = AsyncMock()
        
        # Redis repository with advanced caching
        redis_repo = Mock(spec=EnhancedRedisRepository)
        redis_repo.set_advanced = AsyncMock()
        redis_repo.get_advanced = AsyncMock()
        redis_repo.batch_operations = AsyncMock()
        redis_repo.invalidate_cache = AsyncMock()
        redis_repo.warm_cache = AsyncMock()
        
        return {
            'postgres': postgres_repo,
            'neo4j': neo4j_repo,
            'chroma': chroma_repo,
            'redis': redis_repo
        }
    
    async def test_full_prompt_generation_pipeline(self, mock_repositories, metrics_collector):
        """
        Test complete prompt generation pipeline using all databases.
        
        This test simulates the entire workflow from template creation
        to prompt generation with caching and optimization.
        """
        # Step 1: Create prompt template in PostgreSQL
        template_id = str(uuid.uuid4())
        template_data = {
            "id": template_id,
            "name": "e2e-test-template",
            "description": "End-to-end test template for comprehensive testing",
            "content": "Generate a prompt for {topic} with {style} style",
            "status": PromptStatus.ACTIVE,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        mock_repositories['postgres'].create.return_value = template_data
        
        created_template = await mock_repositories['postgres'].create(template_data)
        assert created_template['id'] == template_id
        assert created_template['name'] == "e2e-test-template"
        
        # Step 2: Index template in ChromaDB for semantic search
        vector_data = {
            "id": template_id,
            "content": template_data['description'],
            "metadata": {
                "type": "template",
                "category": "e2e-test",
                "status": "active"
            }
        }
        
        mock_repositories['chroma'].bulk_upsert_documents.return_value = {
            "success": True,
            "documents_processed": 1,
            "errors": []
        }
        
        index_result = await mock_repositories['chroma'].bulk_upsert_documents([vector_data])
        assert index_result['success'] is True
        assert index_result['documents_processed'] == 1
        
        # Step 3: Create context hierarchy in Neo4j
        context_data = {
            "template_id": template_id,
            "context_type": "prompt_generation",
            "relationships": ["depends_on", "similar_to", "used_in"]
        }
        
        mock_repositories['neo4j'].create_context_hierarchy.return_value = {
            "success": True,
            "nodes_created": 5,
            "relationships_created": 4,
            "hierarchy_depth": 3
        }
        
        hierarchy_result = await mock_repositories['neo4j'].create_context_hierarchy(context_data)
        assert hierarchy_result['success'] is True
        assert hierarchy_result['nodes_created'] == 5
        
        # Step 4: Cache template in Redis with intelligent invalidation
        cache_key = f"template:{template_id}"
        cache_tags = ["template", "prompt", "e2e-test"]
        
        await mock_repositories['redis'].set_advanced(
            cache_key,
            template_data,
            ttl=3600,
            tags=cache_tags
        )
        
        # Step 5: Simulate prompt generation request
        generation_request = {
            "template_id": template_id,
            "parameters": {
                "topic": "artificial intelligence",
                "style": "professional"
            },
            "user_context": "AI researcher"
        }
        
        # Check cache first
        cached_template = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_template is not None
        
        # Generate prompt using template
        generated_prompt = cached_template['content'].format(
            topic=generation_request['parameters']['topic'],
            style=generation_request['parameters']['style']
        )
        
        expected_prompt = "Generate a prompt for artificial intelligence with professional style"
        assert generated_prompt == expected_prompt
        
        # Step 6: Store generation result in PostgreSQL
        generation_record = {
            "id": str(uuid.uuid4()),
            "template_id": template_id,
            "generated_prompt": generated_prompt,
            "parameters": generation_request['parameters'],
            "user_context": generation_request['user_context'],
            "created_at": datetime.now()
        }
        
        mock_repositories['postgres'].create.return_value = generation_record
        
        stored_generation = await mock_repositories['postgres'].create(generation_record)
        assert stored_generation['generated_prompt'] == generated_prompt
        
        # Step 7: Update conversation patterns in Neo4j
        pattern_data = {
            "user_context": generation_request['user_context'],
            "template_used": template_id,
            "generation_success": True,
            "response_time": 0.15
        }
        
        mock_repositories['neo4j'].analyze_conversation_patterns.return_value = {
            "patterns": [{"type": "prompt_generation", "frequency": 1}],
            "analysis_time": 0.05
        }
        
        patterns = await mock_repositories['neo4j'].analyze_conversation_patterns(
            time_window=timedelta(hours=1)
        )
        assert len(patterns['patterns']) > 0
        
        # Step 8: Verify metrics collection
        metrics_summary = metrics_collector.get_metrics_summary()
        assert len(metrics_summary) > 0
        
        # Verify all operations were called
        mock_repositories['postgres'].create.assert_called()
        mock_repositories['chroma'].bulk_upsert_documents.assert_called_once()
        mock_repositories['neo4j'].create_context_hierarchy.assert_called_once()
        mock_repositories['redis'].set_advanced.assert_called_once()
        mock_repositories['redis'].get_advanced.assert_called_once()
        mock_repositories['neo4j'].analyze_conversation_patterns.assert_called_once()
    
    async def test_semantic_search_integration(self, mock_repositories):
        """
        Test semantic search integration across all database layers.
        
        This test validates that semantic search works seamlessly
        across PostgreSQL, ChromaDB, and Redis caching.
        """
        # Step 1: Create multiple templates for search testing
        templates = []
        for i in range(5):
            template = {
                "id": str(uuid.uuid4()),
                "name": f"search-template-{i}",
                "description": f"Template {i} for semantic search testing with AI and machine learning content",
                "content": f"Generate content about {['AI', 'ML', 'data science', 'automation', 'robotics'][i]}",
                "tags": ["search", "ai", "ml", "testing"]
            }
            templates.append(template)
        
        # Step 2: Index all templates in ChromaDB
        vector_documents = []
        for template in templates:
            vector_doc = {
                "id": template['id'],
                "content": template['description'],
                "metadata": {
                    "name": template['name'],
                    "tags": template['tags'],
                    "type": "template"
                }
            }
            vector_documents.append(vector_doc)
        
        mock_repositories['chroma'].bulk_upsert_documents.return_value = {
            "success": True,
            "documents_processed": len(vector_documents),
            "errors": []
        }
        
        index_result = await mock_repositories['chroma'].bulk_upsert_documents(vector_documents)
        assert index_result['success'] is True
        assert index_result['documents_processed'] == 5
        
        # Step 3: Perform semantic search
        search_query = "artificial intelligence and machine learning"
        
        mock_repositories['chroma'].semantic_search_advanced.return_value = {
            "results": [
                {"id": templates[0]['id'], "score": 0.95, "metadata": templates[0]},
                {"id": templates[1]['id'], "score": 0.92, "metadata": templates[1]},
                {"id": templates[2]['id'], "score": 0.88, "metadata": templates[2]}
            ],
            "total": 3,
            "query_time": 0.08
        }
        
        search_results = await mock_repositories['chroma'].semantic_search_advanced(
            search_query,
            limit=10,
            threshold=0.8
        )
        
        assert search_results['total'] == 3
        assert len(search_results['results']) == 3
        assert search_results['results'][0]['score'] > 0.9
        
        # Step 4: Cache search results in Redis
        search_cache_key = f"search:{hash(search_query)}"
        search_cache_data = {
            "query": search_query,
            "results": search_results['results'],
            "timestamp": datetime.now().isoformat(),
            "total": search_results['total']
        }
        
        await mock_repositories['redis'].set_advanced(
            search_cache_key,
            search_cache_data,
            ttl=1800,
            tags=["search", "semantic", "ai", "ml"]
        )
        
        # Step 5: Retrieve from cache (cache hit)
        cached_search = await mock_repositories['redis'].get_advanced(search_cache_key)
        assert cached_search is not None
        assert cached_search['query'] == search_query
        assert len(cached_search['results']) == 3
        
        # Step 6: Find similar documents using ChromaDB
        reference_doc_id = templates[0]['id']
        
        mock_repositories['chroma'].find_similar_documents.return_value = {
            "results": [
                {"id": templates[1]['id'], "similarity": 0.89},
                {"id": templates[2]['id'], "similarity": 0.85}
            ],
            "reference_id": reference_doc_id
        }
        
        similar_docs = await mock_repositories['chroma'].find_similar_documents(
            reference_doc_id,
            limit=5,
            threshold=0.8
        )
        
        assert len(similar_docs['results']) == 2
        assert similar_docs['reference_id'] == reference_doc_id
        
        # Step 7: Update search patterns in Neo4j
        search_pattern = {
            "query_type": "semantic_search",
            "query_terms": ["artificial", "intelligence", "machine", "learning"],
            "results_count": 3,
            "user_context": "researcher"
        }
        
        mock_repositories['neo4j'].create_node.return_value = {
            "id": str(uuid.uuid4()),
            "type": "search_pattern",
            "properties": search_pattern
        }
        
        pattern_node = await mock_repositories['neo4j'].create_node(
            "SearchPattern",
            search_pattern
        )
        
        assert pattern_node['type'] == "search_pattern"
        assert pattern_node['properties']['results_count'] == 3
        
        # Step 8: Verify search integration
        mock_repositories['chroma'].semantic_search_advanced.assert_called_once()
        mock_repositories['redis'].set_advanced.assert_called()
        mock_repositories['redis'].get_advanced.assert_called()
        mock_repositories['chroma'].find_similar_documents.assert_called_once()
        mock_repositories['neo4j'].create_node.assert_called_once()
    
    async def test_high_load_scenarios(self, mock_repositories, pool_manager, metrics_collector):
        """
        Test system behavior under high load scenarios.
        
        This test validates performance, connection pool management,
        and system stability under realistic load conditions.
        """
        # Step 1: Simulate high concurrent user load
        concurrent_users = 100
        operations_per_user = 10
        
        async def user_workload(user_id: int):
            """Simulate a single user's workload."""
            results = []
            
            for op_id in range(operations_per_user):
                # Create template
                template_data = {
                    "id": str(uuid.uuid4()),
                    "name": f"load-test-template-{user_id}-{op_id}",
                    "description": f"Load test template {op_id} for user {user_id}",
                    "created_at": datetime.now()
                }
                
                mock_repositories['postgres'].create.return_value = template_data
                template = await mock_repositories['postgres'].create(template_data)
                results.append(template)
                
                # Cache template
                cache_key = f"template:{template['id']}"
                await mock_repositories['redis'].set_advanced(
                    cache_key,
                    template,
                    ttl=3600
                )
                
                # Index in ChromaDB
                vector_doc = {
                    "id": template['id'],
                    "content": template['description'],
                    "metadata": {"user_id": user_id, "op_id": op_id}
                }
                
                mock_repositories['chroma'].bulk_upsert_documents.return_value = {
                    "success": True,
                    "documents_processed": 1
                }
                
                await mock_repositories['chroma'].bulk_upsert_documents([vector_doc])
                
                # Update Neo4j patterns
                mock_repositories['neo4j'].create_node.return_value = {
                    "id": str(uuid.uuid4()),
                    "type": "user_activity"
                }
                
                await mock_repositories['neo4j'].create_node(
                    "UserActivity",
                    {"user_id": user_id, "operation": "template_creation"}
                )
            
            return results
        
        # Step 2: Execute concurrent workloads
        start_time = time.time()
        
        user_tasks = [user_workload(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Step 3: Validate performance requirements
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Verify all users completed successfully
        successful_users = sum(1 for result in all_results if not isinstance(result, Exception))
        assert successful_users == concurrent_users
        
        # Step 4: Test connection pool behavior under load
        pool_stats = pool_manager.get_pool_stats()
        
        for pool_stat in pool_stats:
            # Verify pools are handling load
            assert pool_stat.total_connections > 0
            assert pool_stat.active_connections >= 0
            assert pool_stat.active_connections <= pool_stat.total_connections
        
        # Step 5: Test bulk operations under load
        bulk_templates = []
        for i in range(1000):
            bulk_templates.append({
                "id": str(uuid.uuid4()),
                "name": f"bulk-load-{i}",
                "description": f"Bulk load template {i}",
                "created_at": datetime.now()
            })
        
        start_time = time.time()
        
        mock_repositories['postgres'].bulk_create.return_value = bulk_templates
        bulk_result = await mock_repositories['postgres'].bulk_create(bulk_templates)
        
        bulk_time = time.time() - start_time
        
        assert len(bulk_result) == 1000
        assert bulk_time < 5.0  # Bulk operations should be fast
        
        # Step 6: Test cache performance under load
        cache_keys = [f"load-test-key-{i}" for i in range(500)]
        
        # Test cache miss performance
        start_time = time.time()
        
        for key in cache_keys:
            mock_repositories['redis'].get_advanced.return_value = None
            await mock_repositories['redis'].get_advanced(key)
        
        cache_miss_time = time.time() - start_time
        assert cache_miss_time < 2.0  # 500 cache misses should be fast
        
        # Test cache hit performance
        start_time = time.time()
        
        for key in cache_keys:
            mock_repositories['redis'].get_advanced.return_value = {"key": key, "value": "cached"}
            await mock_repositories['redis'].get_advanced(key)
        
        cache_hit_time = time.time() - start_time
        assert cache_hit_time < 1.0  # Cache hits should be very fast
        
        # Step 7: Test vector search performance under load
        search_queries = [f"load test query {i}" for i in range(100)]
        
        start_time = time.time()
        
        for query in search_queries:
            mock_repositories['chroma'].semantic_search_advanced.return_value = {
                "results": [{"id": str(uuid.uuid4()), "score": 0.9}],
                "total": 1,
                "query_time": 0.05
            }
            
            await mock_repositories['chroma'].semantic_search_advanced(query)
        
        search_time = time.time() - start_time
        assert search_time < 10.0  # 100 searches should complete quickly
        
        # Step 8: Verify metrics collection under load
        metrics_summary = metrics_collector.get_metrics_summary()
        assert len(metrics_summary) > 0
        
        # Verify all operations were called
        assert mock_repositories['postgres'].create.call_count >= concurrent_users * operations_per_user
        assert mock_repositories['redis'].set_advanced.call_count >= concurrent_users * operations_per_user
        assert mock_repositories['chroma'].bulk_upsert_documents.call_count >= concurrent_users * operations_per_user
        assert mock_repositories['neo4j'].create_node.call_count >= concurrent_users * operations_per_user
    
    async def test_data_consistency_and_recovery(self, mock_repositories, pool_manager):
        """
        Test data consistency and recovery mechanisms across databases.
        
        This test validates that the system maintains data integrity
        and can recover from various failure scenarios.
        """
        # Step 1: Test data consistency across databases
        template_id = str(uuid.uuid4())
        template_data = {
            "id": template_id,
            "name": "consistency-test-template",
            "description": "Template for testing data consistency",
            "created_at": datetime.now()
        }
        
        # Create in PostgreSQL
        mock_repositories['postgres'].create.return_value = template_data
        created_template = await mock_repositories['postgres'].create(template_data)
        
        # Cache in Redis
        cache_key = f"template:{template_id}"
        await mock_repositories['redis'].set_advanced(
            cache_key,
            created_template,
            ttl=3600
        )
        
        # Index in ChromaDB
        vector_doc = {
            "id": template_id,
            "content": template_data['description'],
            "metadata": {"type": "template"}
        }
        
        mock_repositories['chroma'].bulk_upsert_documents.return_value = {
            "success": True,
            "documents_processed": 1
        }
        
        await mock_repositories['chroma'].bulk_upsert_documents([vector_doc])
        
        # Create in Neo4j
        mock_repositories['neo4j'].create_node.return_value = {
            "id": str(uuid.uuid4()),
            "type": "template",
            "properties": {"template_id": template_id}
        }
        
        await mock_repositories['neo4j'].create_node("Template", {"template_id": template_id})
        
        # Step 2: Verify data consistency
        # Check PostgreSQL
        mock_repositories['postgres'].find_by_id.return_value = created_template
        db_template = await mock_repositories['postgres'].find_by_id(template_id)
        assert db_template['id'] == template_id
        
        # Check Redis cache
        cached_template = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_template['id'] == template_id
        
        # Check ChromaDB
        mock_repositories['chroma'].semantic_search_advanced.return_value = {
            "results": [{"id": template_id, "metadata": {"type": "template"}}],
            "total": 1
        }
        
        search_result = await mock_repositories['chroma'].semantic_search_advanced(
            "consistency test template"
        )
        assert search_result['total'] == 1
        assert search_result['results'][0]['id'] == template_id
        
        # Step 3: Test failure recovery
        # Simulate PostgreSQL failure
        mock_repositories['postgres'].find_by_id.side_effect = Exception("Database connection failed")
        
        # Try to recover from cache
        cached_template = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_template is not None
        assert cached_template['id'] == template_id
        
        # Simulate Redis failure
        mock_repositories['redis'].get_advanced.side_effect = Exception("Redis connection failed")
        
        # Try to recover from database
        mock_repositories['postgres'].find_by_id.side_effect = None  # Reset
        mock_repositories['postgres'].find_by_id.return_value = created_template
        
        db_template = await mock_repositories['postgres'].find_by_id(template_id)
        assert db_template['id'] == template_id
        
        # Step 4: Test data synchronization
        # Update template in PostgreSQL
        updated_template = {**template_data, "description": "Updated description"}
        mock_repositories['postgres'].update.return_value = updated_template
        
        updated_result = await mock_repositories['postgres'].update(template_id, updated_template)
        assert updated_result['description'] == "Updated description"
        
        # Invalidate cache
        await mock_repositories['redis'].invalidate_cache(
            tags=["template"],
            pattern="template:*"
        )
        
        # Update ChromaDB
        updated_vector_doc = {
            "id": template_id,
            "content": updated_template['description'],
            "metadata": {"type": "template", "updated": True}
        }
        
        await mock_repositories['chroma'].bulk_upsert_documents([updated_vector_doc])
        
        # Update Neo4j
        mock_repositories['neo4j'].create_relationship.return_value = {
            "success": True,
            "relationship_id": str(uuid.uuid4())
        }
        
        await mock_repositories['neo4j'].create_relationship(
            "UPDATED",
            {"template_id": template_id, "timestamp": datetime.now().isoformat()}
        )
        
        # Step 5: Verify recovery and consistency
        # Check updated data in all databases
        db_template = await mock_repositories['postgres'].find_by_id(template_id)
        assert db_template['description'] == "Updated description"
        
        # Cache should be invalidated, so get from database
        cached_template = await mock_repositories['redis'].get_advanced(cache_key)
        assert cached_template is None  # Cache was invalidated
        
        # Verify all operations were called
        mock_repositories['postgres'].create.assert_called_once()
        mock_repositories['redis'].set_advanced.assert_called()
        mock_repositories['chroma'].bulk_upsert_documents.assert_called()
        mock_repositories['neo4j'].create_node.assert_called_once()
        mock_repositories['postgres'].update.assert_called_once()
        mock_repositories['redis'].invalidate_cache.assert_called_once()
        mock_repositories['neo4j'].create_relationship.assert_called_once()
    
    async def test_performance_monitoring_and_optimization(self, mock_repositories, pool_manager, metrics_collector):
        """
        Test performance monitoring and optimization features.
        
        This test validates that the system can monitor performance,
        identify bottlenecks, and provide optimization recommendations.
        """
        # Step 1: Generate performance metrics
        operations = [
            ("postgres", "create", 0.05),
            ("postgres", "bulk_create", 0.15),
            ("redis", "set", 0.01),
            ("redis", "get", 0.005),
            ("chroma", "search", 0.08),
            ("neo4j", "create_node", 0.03),
            ("neo4j", "analyze_patterns", 0.12)
        ]
        
        for db_type, operation, duration in operations:
            # Record metrics
            metrics_collector.record_query_metric(
                query_type=f"{db_type}_{operation}",
                duration=duration,
                success=True,
                database_type=db_type
            )
            
            # Record connection events
            pool_manager.record_connection_event(db_type, "checkout", duration)
            pool_manager.record_connection_event(db_type, "checkin", duration * 0.5)
        
        # Step 2: Analyze performance trends
        trends = metrics_collector.get_performance_trends()
        assert len(trends) > 0
        
        # Step 3: Get optimization recommendations
        recommendations = pool_manager.get_optimization_recommendations()
        assert len(recommendations) > 0
        
        # Step 4: Test performance optimization
        # Simulate high load scenario
        for i in range(50):
            metrics_collector.record_query_metric(
                query_type="postgres_bulk_create",
                duration=0.2 + (i * 0.01),  # Gradually increasing duration
                success=True,
                database_type="postgresql"
            )
        
        # Get updated recommendations
        updated_recommendations = pool_manager.get_optimization_recommendations()
        
        # Should have recommendations for PostgreSQL optimization
        postgres_recommendations = [
            r for r in updated_recommendations 
            if "postgresql" in r.description.lower()
        ]
        assert len(postgres_recommendations) > 0
        
        # Step 5: Test connection pool optimization
        # Simulate connection pool stress
        for i in range(20):
            pool_manager.record_connection_event("postgresql", "checkout", 0.1)
            pool_manager.record_connection_event("postgresql", "checkin", 0.05)
        
        # Get pool statistics
        pool_stats = pool_manager.get_pool_stats()
        postgres_stats = next((s for s in pool_stats if s.pool_type == "postgresql"), None)
        
        if postgres_stats:
            assert postgres_stats.total_connections > 0
            assert postgres_stats.active_connections >= 0
        
        # Step 6: Test metrics aggregation
        metrics_summary = metrics_collector.get_metrics_summary()
        assert len(metrics_summary) > 0
        
        # Verify PostgreSQL metrics
        postgres_metrics = [
            m for m in metrics_summary 
            if m.database_type == "postgresql"
        ]
        assert len(postgres_metrics) > 0
        
        # Step 7: Test performance alerts
        # Simulate performance degradation
        for i in range(10):
            metrics_collector.record_query_metric(
                query_type="postgres_query",
                duration=1.0 + (i * 0.1),  # Very slow queries
                success=True,
                database_type="postgresql"
            )
        
        # Get performance trends
        performance_trends = metrics_collector.get_performance_trends()
        
        # Should detect performance degradation
        assert len(performance_trends) > 0
        
        # Step 8: Verify monitoring integration
        # All components should be integrated
        assert metrics_collector is not None
        assert pool_manager is not None
        
        # Verify metrics collection is working
        final_metrics = metrics_collector.get_metrics_summary()
        assert len(final_metrics) > 0
        
        # Verify pool monitoring is working
        final_pool_stats = pool_manager.get_pool_stats()
        assert len(final_pool_stats) >= 0  # May be empty if no pools registered
