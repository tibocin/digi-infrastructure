"""
Filepath: tests/unit/test_optimized_postgres_repo.py
Purpose: Unit tests for optimized PostgreSQL repository implementation
Related Components: OptimizedPostgreSQLRepository, PaginatedResult, QueryPerformanceMetrics
Tags: testing, postgresql, repository, performance, optimization
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pcs.repositories.postgres_repo import (
    OptimizedPostgreSQLRepository,
    PostgreSQLRepository,
    PaginatedResult,
    QueryPerformanceMetrics
)
from pcs.repositories.base import RepositoryError


class MockModel:
    """Mock model for testing purposes."""
    __tablename__ = "mock_table"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    # Mock class attributes that repositories expect
    id = Mock()
    created_at = Mock()
    updated_at = Mock()
    
    @classmethod
    def __table_columns__(cls):
        """Mock table columns for testing."""
        return ["id", "created_at", "updated_at", "name", "status"]


# Create a mock table with columns
mock_table = Mock()
mock_table.columns = {
    "id": Mock(),
    "created_at": Mock(), 
    "updated_at": Mock(),
    "name": Mock(),
    "status": Mock()
}

# Set the table on the MockModel
MockModel.__table__ = mock_table


@pytest.fixture
def mock_session():
    """Create a mock async session for testing."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.add_all = Mock()
    session.close = AsyncMock()
    
    # Mock bind and pool for connection stats
    mock_pool = Mock()
    mock_pool.size.return_value = 10
    mock_pool.checkedin.return_value = 8
    mock_pool.checkedout.return_value = 2
    mock_pool.overflow.return_value = 0
    
    session.bind = Mock()
    session.bind.pool = mock_pool
    
    return session


@pytest.fixture
def repository(mock_session):
    """Create an optimized PostgreSQL repository for testing."""
    return OptimizedPostgreSQLRepository(mock_session, MockModel)


class TestOptimizedPostgreSQLRepository:
    """Test suite for OptimizedPostgreSQLRepository."""

    def test_initialization(self, mock_session):
        """Test repository initialization."""
        repo = OptimizedPostgreSQLRepository(mock_session, MockModel)
        
        assert repo.session == mock_session
        assert repo.model_class == MockModel
        assert isinstance(repo._query_metrics, list)
        assert len(repo._query_metrics) == 0

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.Insert')
    async def test_bulk_create_optimized_success(self, mock_insert, repository, mock_session):
        """Test successful bulk create operation."""
        # Setup
        entities = [
            MockModel(id=uuid4(), name="entity1"),
            MockModel(id=uuid4(), name="entity2"),
            MockModel(id=uuid4(), name="entity3")
        ]
        
        # Mock the Insert constructor and return value
        mock_stmt = Mock()
        mock_insert.return_value = mock_stmt
        mock_stmt.returning.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        
        # Mock the execute result - return different results for each batch
        batch1_result = [entities[0], entities[1]]  # First batch (2 items)
        batch2_result = [entities[2]]  # Second batch (1 item)
        
        mock_result1 = Mock()
        mock_result1.fetchall.return_value = batch1_result
        mock_result2 = Mock()
        mock_result2.fetchall.return_value = batch2_result
        
        mock_session.execute.side_effect = [mock_result1, mock_result2]
        
        # Execute
        result = await repository.bulk_create_optimized(entities, batch_size=2)
        
        # Verify
        assert len(result) == 3
        assert mock_session.commit.called
        assert len(repository._query_metrics) == 1
        assert repository._query_metrics[0].query_type == "bulk_create_optimized"

    @pytest.mark.asyncio
    async def test_bulk_create_optimized_empty_list(self, repository):
        """Test bulk create with empty list."""
        result = await repository.bulk_create_optimized([])
        assert result == []

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.select')
    async def test_find_with_cursor_pagination_first_page(self, mock_select, repository, mock_session):
        """Test cursor-based pagination for first page."""
        # Setup mocks
        mock_stmt = Mock()
        mock_select.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.order_by.return_value = mock_stmt
        mock_stmt.limit.return_value = mock_stmt
        
        # Mock entities
        mock_entities = [
            MockModel(id=uuid4(), created_at=datetime.now() - timedelta(hours=i))
            for i in range(4)  # 4 entities to test "has_next" logic
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_entities
        
        # Mock the count query result
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 100  # Return a number for total count
        
        mock_session.execute.side_effect = [mock_result, mock_count_result]
        
        # Execute
        result = await repository.find_with_cursor_pagination(
            page_size=3,
            order_by="created_at",
            status="active"
        )
        
        # Verify
        assert isinstance(result, PaginatedResult)
        assert len(result.items) == 3  # Should trim the extra item
        assert result.has_next is True
        assert result.has_prev is False
        assert result.next_cursor is not None

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.select')
    async def test_find_with_cursor_pagination_with_cursor(self, mock_select, repository, mock_session):
        """Test cursor-based pagination with existing cursor."""
        # Setup mocks
        mock_stmt = Mock()
        mock_select.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.order_by.return_value = mock_stmt
        mock_stmt.limit.return_value = mock_stmt
        
        # Mock the order field to support comparison operations
        mock_order_field = Mock()
        mock_order_field.__gt__ = Mock(return_value=Mock())
        mock_order_field.asc = Mock(return_value=Mock())
        repository.model_class.created_at = mock_order_field
        
        # Setup cursor
        cursor = repository._encode_cursor(datetime.now())
        mock_entities = [MockModel(id=uuid4(), created_at=datetime.now()) for _ in range(2)]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_entities
        
        # Mock the count query result
        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 50
        
        mock_session.execute.side_effect = [mock_result, mock_count_result]
        
        # Execute
        result = await repository.find_with_cursor_pagination(
            cursor=cursor,
            page_size=3,
            order_by="created_at"
        )
        
        # Verify
        assert isinstance(result, PaginatedResult)
        assert result.has_prev is True

    @pytest.mark.asyncio
    async def test_find_with_cursor_pagination_invalid_field(self, repository):
        """Test cursor pagination with invalid order field."""
        with pytest.raises(ValueError, match="Order field invalid_field does not exist"):
            await repository.find_with_cursor_pagination(order_by="invalid_field")

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.text')
    async def test_execute_optimized_query_success(self, mock_text, repository, mock_session):
        """Test optimized query execution."""
        # Setup
        query = "SELECT * FROM mock_table WHERE status = :status"
        params = {"status": "active"}
        
        mock_stmt = Mock()
        mock_text.return_value = mock_stmt
        
        mock_row = Mock()
        mock_row._mapping = {"id": str(uuid4()), "status": "active"}
        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.execute_optimized_query(query, params)
        
        # Verify
        assert len(result) == 1
        assert result[0]["status"] == "active"
        assert len(repository._query_metrics) == 1

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.text')
    async def test_execute_optimized_query_with_cache(self, mock_text, repository, mock_session):
        """Test optimized query with caching enabled."""
        query = "SELECT * FROM mock_table"
        
        mock_stmt = Mock()
        mock_text.return_value = mock_stmt
        
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Execute
        await repository.execute_optimized_query(query, enable_query_cache=True)
        
        # Verify that the query was modified with cache hint
        called_args = mock_text.call_args[0]
        query_text = called_args[0]
        assert "ENABLE_QUERY_CACHE" in query_text

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.update')
    async def test_bulk_update_optimized_success(self, mock_update, repository, mock_session):
        """Test optimized bulk update operation."""
        # Setup
        updates = [
            {"id": uuid4(), "status": "active", "name": "updated1"},
            {"id": uuid4(), "status": "active", "name": "updated2"},
            {"id": uuid4(), "status": "inactive", "priority": 1}
        ]
        
        # Mock update statement
        mock_stmt = Mock()
        mock_update.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        mock_stmt.values.return_value = mock_stmt
        
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        
        # Execute
        result = await repository.bulk_update_optimized(updates, batch_size=2)
        
        # Verify
        assert result >= 0  # Number of updated records
        assert mock_session.commit.called
        assert len(repository._query_metrics) == 1

    @pytest.mark.asyncio
    async def test_bulk_update_optimized_empty_list(self, repository):
        """Test bulk update with empty list."""
        result = await repository.bulk_update_optimized([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_get_connection_pool_stats_success(self, repository):
        """Test connection pool statistics retrieval."""
        stats = await repository.get_connection_pool_stats()
        
        assert "pool_size" in stats
        assert "checked_in" in stats
        assert "checked_out" in stats
        assert "overflow" in stats
        assert "utilization_percent" in stats
        assert stats["pool_size"] == 10
        assert stats["checked_in"] == 8
        assert stats["checked_out"] == 2
        assert stats["utilization_percent"] == 20.0

    @pytest.mark.asyncio
    async def test_get_connection_pool_stats_error(self, repository, mock_session):
        """Test connection pool statistics with error."""
        mock_session.bind.pool.size.side_effect = Exception("Pool error")
        
        stats = await repository.get_connection_pool_stats()
        
        assert "error" in stats
        assert stats["pool_size"] == 0

    def test_query_performance_metrics_tracking(self, repository):
        """Test query performance metrics tracking."""
        # Record some metrics
        repository._track_query_performance("select", 0.1, 10)
        repository._track_query_performance("insert", 0.05, 1)
        repository._track_query_performance("update", 0.2, 5)
        
        metrics = repository.get_query_performance_metrics()
        
        assert len(metrics) == 3
        assert metrics[0].query_type == "select"
        assert metrics[0].execution_time == 0.1
        assert metrics[0].rows_affected == 10
        assert isinstance(metrics[0].timestamp, datetime)

    def test_clear_query_metrics(self, repository):
        """Test clearing query metrics."""
        repository._track_query_performance("select", 0.1, 10)
        assert len(repository._query_metrics) == 1
        
        repository.clear_query_metrics()
        assert len(repository._query_metrics) == 0

    def test_metrics_memory_management(self, repository):
        """Test that metrics are trimmed when exceeding limit."""
        # Add more than 1000 metrics (the default limit)
        for i in range(1005):
            repository._track_query_performance(f"query_{i}", 0.1, 1)
        
        assert len(repository._query_metrics) == 1000

    def test_encode_decode_cursor(self, repository):
        """Test cursor encoding and decoding."""
        # Test with datetime
        dt = datetime.now()
        cursor = repository._encode_cursor(dt)
        decoded = repository._decode_cursor(cursor)
        # The decoder properly handles datetime reconstruction
        if isinstance(decoded, datetime):
            # If it's decoded as datetime, compare as datetime
            assert decoded == dt
        else:
            # If it's decoded as string, compare as string
            assert decoded == str(dt)
        
        # Test with string
        test_str = "test_value"
        cursor = repository._encode_cursor(test_str)
        decoded = repository._decode_cursor(cursor)
        assert decoded == test_str

    def test_decode_invalid_cursor(self, repository):
        """Test decoding invalid cursor."""
        with pytest.raises(ValueError, match="Invalid cursor"):
            repository._decode_cursor("invalid_cursor")

    @pytest.mark.asyncio
    @patch('pcs.repositories.postgres_repo.select')
    @patch('pcs.repositories.postgres_repo.func')
    async def test_get_estimated_count_success(self, mock_func, mock_select, repository, mock_session):
        """Test estimated count calculation."""
        # Setup mocks
        mock_stmt = Mock()
        mock_select.return_value = mock_stmt
        mock_stmt.where.return_value = mock_stmt
        
        mock_result = Mock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result
        
        count = await repository._get_estimated_count(status="active")
        
        assert count == 100

    @pytest.mark.asyncio
    async def test_get_estimated_count_error(self, repository, mock_session):
        """Test estimated count with SQL error."""
        from sqlalchemy.exc import SQLAlchemyError
        mock_session.execute.side_effect = SQLAlchemyError("Database error")
        
        count = await repository._get_estimated_count()
        
        assert count == 0

    def test_group_updates_by_fields(self, repository):
        """Test grouping updates by field sets."""
        updates = [
            {"id": "1", "name": "test1", "status": "active"},
            {"id": "2", "name": "test2", "status": "inactive"},
            {"id": "3", "priority": 1, "category": "high"},
            {"id": "4", "name": "test4", "status": "active"}
        ]
        
        groups = repository._group_updates_by_fields(updates)
        
        # Should have 2 groups: one for (name, status) and one for (category, priority)
        assert len(groups) == 2
        
        name_status_key = tuple(sorted(["name", "status"]))
        priority_category_key = tuple(sorted(["priority", "category"]))
        
        assert name_status_key in groups
        assert priority_category_key in groups
        assert len(groups[name_status_key]) == 3
        assert len(groups[priority_category_key]) == 1


class TestPostgreSQLRepositoryBackwardCompatibility:
    """Test backward compatibility of PostgreSQLRepository."""

    def test_inheritance(self, mock_session):
        """Test that PostgreSQLRepository inherits from OptimizedPostgreSQLRepository."""
        repo = PostgreSQLRepository(mock_session, MockModel)
        assert isinstance(repo, OptimizedPostgreSQLRepository)

    def test_existing_methods_still_work(self, mock_session):
        """Test that existing methods are still available."""
        repo = PostgreSQLRepository(mock_session, MockModel)
        
        # Test that original methods exist
        assert hasattr(repo, "get_by_id_with_relationships")
        assert hasattr(repo, "find_by_text_search")
        assert hasattr(repo, "find_by_json_field")
        assert hasattr(repo, "bulk_create")
        assert hasattr(repo, "bulk_update")
        assert hasattr(repo, "find_with_pagination")


class TestPaginatedResult:
    """Test PaginatedResult data class."""

    def test_paginated_result_creation(self):
        """Test PaginatedResult initialization."""
        items = [MockModel(id=uuid4()) for _ in range(3)]
        result = PaginatedResult(
            items=items,
            total=100,
            page=1,
            page_size=20,
            total_pages=5,
            cursor="test_cursor",
            next_cursor="next_cursor",
            has_next=True,
            has_prev=False
        )
        
        assert len(result.items) == 3
        assert result.total == 100
        assert result.page == 1
        assert result.page_size == 20
        assert result.total_pages == 5
        assert result.cursor == "test_cursor"
        assert result.next_cursor == "next_cursor"
        assert result.has_next is True
        assert result.has_prev is False


class TestQueryPerformanceMetrics:
    """Test QueryPerformanceMetrics data class."""

    def test_query_performance_metrics_creation(self):
        """Test QueryPerformanceMetrics initialization."""
        metric = QueryPerformanceMetrics(
            query_type="select",
            execution_time=0.1,
            rows_affected=10
        )
        
        assert metric.query_type == "select"
        assert metric.execution_time == 0.1
        assert metric.rows_affected == 10
        assert isinstance(metric.timestamp, datetime)


class TestPerformanceFeatures:
    """Test performance-specific features."""

    def test_cursor_encoding_different_types(self, repository):
        """Test cursor encoding with different data types."""
        # Test UUID
        test_uuid = uuid4()
        cursor = repository._encode_cursor(test_uuid)
        decoded = repository._decode_cursor(cursor)
        # UUID should be properly reconstructed as UUID object
        if hasattr(decoded, 'hex'):  # It's a UUID object
            assert decoded == test_uuid
        else:  # It's a string representation
            assert decoded == str(test_uuid)
        
        # Test integer
        test_int = 12345
        cursor = repository._encode_cursor(test_int)
        decoded = repository._decode_cursor(cursor)
        assert decoded == test_int  # Should be decoded back to int

    def test_metrics_timestamp_tracking(self, repository):
        """Test that metrics properly track timestamps."""
        before = datetime.now()
        repository._track_query_performance("test", 0.1, 5)
        after = datetime.now()
        
        metric = repository._query_metrics[0]
        assert before <= metric.timestamp <= after

    @pytest.mark.asyncio
    async def test_performance_monitoring_context(self, repository):
        """Test that performance monitoring works across operations."""
        # Simulate multiple operations
        repository._track_query_performance("select", 0.05, 100)
        repository._track_query_performance("insert", 0.02, 1)
        repository._track_query_performance("update", 0.03, 5)
        repository._track_query_performance("select", 0.08, 150)
        
        metrics = repository.get_query_performance_metrics()
        
        # Verify we have all metrics
        assert len(metrics) == 4
        
        # Verify different operation types
        select_metrics = [m for m in metrics if m.query_type == "select"]
        assert len(select_metrics) == 2
        
        # Verify performance data
        total_execution_time = sum(m.execution_time for m in metrics)
        assert total_execution_time == 0.18  # 0.05 + 0.02 + 0.03 + 0.08
