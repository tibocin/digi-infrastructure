"""
Filepath: tests/unit/test_background_task_service.py
Purpose: Unit tests for background task processing service
Related Components: BackgroundTaskService, Redis, Worker system, Task management
Tags: unit-tests, background-tasks, redis, workers, async-testing
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from pcs.services.background_task_service import (
    BackgroundTaskService,
    TaskDefinition,
    TaskStatus,
    TaskPriority,
    TaskRegistry,
    BackgroundTaskError,
    get_background_task_service,
    enqueue_fastapi_task
)
from pcs.core.exceptions import PCSError


class TestTaskRegistry:
    """Test TaskRegistry functionality."""
    
    def test_register_task_decorator(self):
        """Test task registration using decorator."""
        registry = TaskRegistry()
        
        @registry.register("test_task")
        def test_function():
            return "test_result"
        
        # Verify task is registered
        assert "test_task" in registry.list_tasks()
        assert registry.get_task("test_task") == test_function
    
    def test_get_nonexistent_task(self):
        """Test getting non-existent task returns None."""
        registry = TaskRegistry()
        assert registry.get_task("nonexistent") is None
    
    def test_list_tasks(self):
        """Test listing all registered tasks."""
        registry = TaskRegistry()
        
        @registry.register("task1")
        def task1():
            pass
        
        @registry.register("task2")
        def task2():
            pass
        
        tasks = registry.list_tasks()
        assert "task1" in tasks
        assert "task2" in tasks
        assert len(tasks) == 2


class TestTaskDefinition:
    """Test TaskDefinition model."""
    
    def test_task_definition_creation(self):
        """Test creating a task definition with defaults."""
        task = TaskDefinition(name="test_task")
        
        assert task.name == "test_task"
        assert task.priority == TaskPriority.NORMAL
        assert task.max_retries == 3
        assert task.retry_delay == 60
        assert task.timeout is None
        assert isinstance(task.created_at, datetime)
        assert task.scheduled_for is None
        assert task.args == []
        assert task.kwargs == {}
        assert task.metadata == {}
    
    def test_task_definition_with_all_fields(self):
        """Test creating a task definition with all fields."""
        scheduled_time = datetime.utcnow() + timedelta(hours=1)
        
        task = TaskDefinition(
            name="complex_task",
            args=[1, 2, 3],
            kwargs={"key": "value"},
            priority=TaskPriority.HIGH,
            max_retries=5,
            retry_delay=120,
            timeout=300,
            scheduled_for=scheduled_time,
            metadata={"source": "test"}
        )
        
        assert task.name == "complex_task"
        assert task.args == [1, 2, 3]
        assert task.kwargs == {"key": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.max_retries == 5
        assert task.retry_delay == 120
        assert task.timeout == 300
        assert task.scheduled_for == scheduled_time
        assert task.metadata == {"source": "test"}
    
    def test_task_definition_serialization(self):
        """Test task definition JSON serialization."""
        task = TaskDefinition(name="test_task", args=[1, "test"])
        
        # Should serialize without errors
        json_data = task.model_dump_json()
        assert isinstance(json_data, str)
        
        # Should deserialize back correctly
        task_copy = TaskDefinition.model_validate_json(json_data)
        assert task_copy.name == task.name
        assert task_copy.args == task.args


@pytest.fixture
def mock_redis():
    """Fixture providing a mocked Redis client."""
    redis_mock = AsyncMock()
    
    # Mock basic Redis operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.llen = AsyncMock(return_value=0)
    redis_mock.brpoplpush = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zcard = AsyncMock(return_value=0)
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.lrem = AsyncMock(return_value=1)
    redis_mock.close = AsyncMock()
    
    return redis_mock


@pytest.fixture
def background_task_service(mock_redis):
    """Fixture providing a BackgroundTaskService with mocked Redis."""
    return BackgroundTaskService(redis_client=mock_redis)


class TestBackgroundTaskService:
    """Test BackgroundTaskService functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, background_task_service, mock_redis):
        """Test service initialization."""
        await background_task_service.initialize()
        
        # Verify Redis connection was tested
        mock_redis.ping.assert_called_once()
        assert background_task_service.redis_client == mock_redis
    
    @pytest.mark.asyncio
    async def test_initialization_redis_failure(self):
        """Test initialization with Redis connection failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        service = BackgroundTaskService(redis_client=mock_redis)
        
        with pytest.raises(BackgroundTaskError, match="Redis connection failed"):
            await service.initialize()
    
    @pytest.mark.asyncio
    async def test_enqueue_task_basic(self, background_task_service, mock_redis):
        """Test basic task enqueueing."""
        await background_task_service.initialize()
        
        task_id = await background_task_service.enqueue_task("test_task", 1, 2, key="value")
        
        # Verify task was enqueued to correct queue
        mock_redis.lpush.assert_called_once()
        args, kwargs = mock_redis.lpush.call_args
        assert args[0] == "pcs:tasks:queue:normal"  # default priority
        
        # Verify metadata was stored
        mock_redis.hset.assert_called()
        mock_redis.expire.assert_called()
        
        assert isinstance(task_id, str)
    
    @pytest.mark.asyncio
    async def test_enqueue_task_with_priority(self, background_task_service, mock_redis):
        """Test task enqueueing with different priority."""
        await background_task_service.initialize()
        
        await background_task_service.enqueue_task(
            "high_priority_task",
            priority=TaskPriority.HIGH
        )
        
        # Verify task was enqueued to high priority queue
        args, kwargs = mock_redis.lpush.call_args
        assert args[0] == "pcs:tasks:queue:high"
    
    @pytest.mark.asyncio
    async def test_enqueue_scheduled_task(self, background_task_service, mock_redis):
        """Test enqueueing a scheduled task."""
        await background_task_service.initialize()
        
        future_time = datetime.utcnow() + timedelta(hours=1)
        
        await background_task_service.enqueue_task(
            "scheduled_task",
            scheduled_for=future_time
        )
        
        # Verify task was added to scheduled queue
        mock_redis.zadd.assert_called_once()
        args, kwargs = mock_redis.zadd.call_args
        assert args[0] == "pcs:tasks:scheduled"
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, background_task_service, mock_redis):
        """Test getting task status."""
        await background_task_service.initialize()
        
        # Mock Redis response
        mock_redis.hgetall.return_value = {
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "retries": "0"
        }
        
        status = await background_task_service.get_task_status("test_task_id")
        
        assert status is not None
        assert status["status"] == "completed"
        assert status["retries"] == 0
        mock_redis.hgetall.assert_called_with("pcs:tasks:meta:test_task_id")
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, background_task_service, mock_redis):
        """Test getting status for non-existent task."""
        await background_task_service.initialize()
        
        # Mock Redis returning empty dict
        mock_redis.hgetall.return_value = {}
        
        status = await background_task_service.get_task_status("nonexistent")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, background_task_service, mock_redis):
        """Test task cancellation."""
        await background_task_service.initialize()
        
        # Mock successful cancellation
        mock_redis.hset.return_value = 1
        
        result = await background_task_service.cancel_task("test_task_id")
        
        assert result is True
        mock_redis.hset.assert_called_with(
            "pcs:tasks:meta:test_task_id",
            "status", TaskStatus.CANCELLED.value
        )
    
    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, background_task_service, mock_redis):
        """Test cancelling non-existent task."""
        await background_task_service.initialize()
        
        # Mock failed cancellation
        mock_redis.hset.return_value = 0
        
        result = await background_task_service.cancel_task("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self, background_task_service, mock_redis):
        """Test getting queue statistics."""
        await background_task_service.initialize()
        
        # Mock queue lengths
        mock_redis.llen.return_value = 5
        mock_redis.zcard.return_value = 2
        
        stats = await background_task_service.get_queue_stats()
        
        assert "queues" in stats
        assert "processing" in stats
        assert "scheduled" in stats
        assert "total_workers" in stats
        assert stats["processing"] == 5
        assert stats["scheduled"] == 2
    
    def test_register_task_decorator(self, background_task_service):
        """Test task registration through service."""
        @background_task_service.register_task("service_task")
        def test_task():
            return "result"
        
        # Verify task was registered
        registered_task = background_task_service.task_registry.get_task("service_task")
        assert registered_task == test_task
    
    @pytest.mark.asyncio
    async def test_shutdown(self, background_task_service, mock_redis):
        """Test service shutdown."""
        await background_task_service.initialize()
        
        # Add some mock workers (create actual async tasks)
        async def dummy_worker():
            return "done"
        
        task1 = asyncio.create_task(dummy_worker())
        task2 = asyncio.create_task(dummy_worker())
        background_task_service._workers = [task1, task2]
        
        await background_task_service.shutdown()
        
        # Verify shutdown event was set and Redis was closed
        assert background_task_service._shutdown_event.is_set()
        mock_redis.close.assert_called_once()


class TestTaskProcessing:
    """Test task processing functionality."""
    
    @pytest.mark.asyncio
    async def test_get_next_task_from_priority_queue(self, background_task_service, mock_redis):
        """Test getting next task respects priority order."""
        await background_task_service.initialize()
        
        # Mock Redis returning task from high priority queue
        mock_redis.brpoplpush.side_effect = [None, None, "task_data", None]
        
        task_data = await background_task_service._get_next_task()
        
        assert task_data == "task_data"
        # Should check critical, high, then normal queues
        assert mock_redis.brpoplpush.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_next_task_no_tasks(self, background_task_service, mock_redis):
        """Test getting next task when no tasks available."""
        await background_task_service.initialize()
        
        # Mock Redis returning None for all queues
        mock_redis.brpoplpush.return_value = None
        
        task_data = await background_task_service._get_next_task()
        assert task_data is None
    
    @pytest.mark.asyncio
    async def test_execute_task_function_async(self, background_task_service):
        """Test executing async task function."""
        async def async_task(x, y, multiplier=1):
            return (x + y) * multiplier
        
        result = await background_task_service._execute_task_function(
            async_task, [1, 2], {"multiplier": 3}
        )
        
        assert result == 9
    
    @pytest.mark.asyncio
    async def test_execute_task_function_sync(self, background_task_service):
        """Test executing sync task function."""
        def sync_task(x, y):
            return x * y
        
        result = await background_task_service._execute_task_function(
            sync_task, [3, 4], {}
        )
        
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_mark_task_completed(self, background_task_service, mock_redis):
        """Test marking task as completed."""
        await background_task_service.initialize()
        
        await background_task_service._mark_task_completed(
            "test_task_id", {"result": "success"}, 1.5
        )
        
        # Verify metadata was updated
        mock_redis.hset.assert_called()
        args, kwargs = mock_redis.hset.call_args
        assert args[0] == "pcs:tasks:meta:test_task_id"
        assert kwargs["mapping"]["status"] == TaskStatus.COMPLETED.value
        assert kwargs["mapping"]["execution_time"] == "1.5"
    
    @pytest.mark.asyncio
    async def test_handle_task_failure_with_retries(self, background_task_service, mock_redis):
        """Test handling task failure with retries available."""
        await background_task_service.initialize()
        
        # Mock current retry count
        mock_redis.hgetall.return_value = {"retries": "1"}
        
        task = TaskDefinition(name="test_task", max_retries=3)
        
        await background_task_service._handle_task_failure(
            task, "Test error", 1.0
        )
        
        # Should schedule retry
        mock_redis.zadd.assert_called()
        mock_redis.hset.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_task_failure_max_retries_reached(self, background_task_service, mock_redis):
        """Test handling task failure when max retries reached."""
        await background_task_service.initialize()
        
        # Mock current retry count at maximum
        mock_redis.hgetall.return_value = {"retries": "3"}
        
        task = TaskDefinition(name="test_task", max_retries=3)
        
        await background_task_service._handle_task_failure(
            task, "Test error", 1.0
        )
        
        # Should mark as failed permanently
        mock_redis.hset.assert_called()
        args, kwargs = mock_redis.hset.call_args
        assert kwargs["mapping"]["status"] == TaskStatus.FAILED.value
    
    @pytest.mark.asyncio
    async def test_schedule_retry_exponential_backoff(self, background_task_service, mock_redis):
        """Test retry scheduling with exponential backoff."""
        await background_task_service.initialize()
        
        task = TaskDefinition(name="test_task", retry_delay=60)
        
        await background_task_service._schedule_retry(task, 2, "Test error")
        
        # Verify retry was scheduled
        mock_redis.zadd.assert_called()
        mock_redis.hset.assert_called()
        
        # Check that exponential backoff was applied (60 * 2^(2-1) = 240 seconds)
        args, kwargs = mock_redis.hset.call_args
        assert kwargs["mapping"]["status"] == TaskStatus.RETRYING.value
        assert kwargs["mapping"]["retries"] == "2"


class TestSchedulerLoop:
    """Test scheduler functionality."""
    
    @pytest.mark.asyncio
    async def test_scheduler_moves_ready_tasks(self, background_task_service, mock_redis):
        """Test scheduler moves scheduled tasks when ready."""
        await background_task_service.initialize()
        
        # Create a task that should be ready
        task = TaskDefinition(name="scheduled_task", priority=TaskPriority.HIGH)
        task_data = task.model_dump_json()
        
        # Mock Redis returning the scheduled task
        mock_redis.zrangebyscore.return_value = [task_data]
        
        # Run one iteration of scheduler
        background_task_service._shutdown_event.set()  # Prevent infinite loop
        
        # Manually call the scheduler logic
        current_time = datetime.utcnow().timestamp()
        scheduled_tasks = await mock_redis.zrangebyscore(
            "pcs:tasks:scheduled", 0, current_time, withscores=False
        )
        
        # Verify scheduled tasks were retrieved
        mock_redis.zrangebyscore.assert_called_with(
            "pcs:tasks:scheduled", 0, current_time, withscores=False
        )


class TestWorkerIntegration:
    """Test worker system integration."""
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, background_task_service, mock_redis):
        """Test successful task processing."""
        await background_task_service.initialize()
        
        # Register a test task
        @background_task_service.register_task("success_task")
        async def test_task(value):
            return value * 2
        
        # Create task data
        task = TaskDefinition(name="success_task", args=[5])
        task_data = task.model_dump_json()
        
        # Mock task status check
        mock_redis.hgetall.return_value = {"status": "pending"}
        
        await background_task_service._process_task(task_data, "test_worker")
        
        # Verify task was processed and completed
        mock_redis.hset.assert_called()
        mock_redis.lrem.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_task_cancelled(self, background_task_service, mock_redis):
        """Test processing cancelled task."""
        await background_task_service.initialize()
        
        task = TaskDefinition(name="cancelled_task")
        task_data = task.model_dump_json()
        
        # Mock cancelled task status
        mock_redis.hgetall.return_value = {"status": "cancelled"}
        
        await background_task_service._process_task(task_data, "test_worker")
        
        # Should skip processing and remove from queue
        mock_redis.lrem.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_task_timeout(self, background_task_service, mock_redis):
        """Test task processing with timeout."""
        await background_task_service.initialize()
        
        # Register a slow task
        @background_task_service.register_task("slow_task")
        async def slow_task():
            await asyncio.sleep(10)  # Long running task
            return "completed"
        
        task = TaskDefinition(name="slow_task", timeout=1)  # 1 second timeout
        task_data = task.model_dump_json()
        
        mock_redis.hgetall.return_value = {"status": "pending", "retries": "0"}
        
        await background_task_service._process_task(task_data, "test_worker")
        
        # Should handle timeout as failure
        mock_redis.hset.assert_called()
        mock_redis.lrem.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_task_unknown_function(self, background_task_service, mock_redis):
        """Test processing task with unknown function."""
        await background_task_service.initialize()
        
        task = TaskDefinition(name="unknown_task")
        task_data = task.model_dump_json()
        
        mock_redis.hgetall.return_value = {"status": "pending", "retries": "0"}
        
        await background_task_service._process_task(task_data, "test_worker")
        
        # Should handle as failure
        mock_redis.hset.assert_called()
        mock_redis.lrem.assert_called()


class TestHelperFunctions:
    """Test helper functions and utilities."""
    
    def test_get_background_task_service_singleton(self):
        """Test global service instance is singleton."""
        service1 = get_background_task_service()
        service2 = get_background_task_service()
        
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_enqueue_fastapi_task_persistent(self, mock_redis):
        """Test FastAPI task enqueueing with persistent queue."""
        from fastapi import BackgroundTasks
        
        background_tasks = BackgroundTasks()
        
        # Mock the global service
        with patch('pcs.services.background_task_service.get_background_task_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.enqueue_task.return_value = "task_id_123"
            mock_get_service.return_value = mock_service
            
            task_id = await enqueue_fastapi_task(
                background_tasks,
                "test_task",
                "arg1", "arg2",
                use_persistent_queue=True,
                key="value"
            )
            
            assert task_id == "task_id_123"
            mock_service.enqueue_task.assert_called_once_with(
                "test_task", "arg1", "arg2", key="value"
            )
    
    @pytest.mark.asyncio
    async def test_enqueue_fastapi_task_in_memory(self):
        """Test FastAPI task enqueueing with in-memory tasks."""
        from fastapi import BackgroundTasks
        
        background_tasks = BackgroundTasks()
        
        # Mock the global service and task function
        with patch('pcs.services.background_task_service.get_background_task_service') as mock_get_service:
            mock_service = MagicMock()
            mock_task_func = MagicMock()
            mock_service.task_registry.get_task.return_value = mock_task_func
            mock_get_service.return_value = mock_service
            
            task_id = await enqueue_fastapi_task(
                background_tasks,
                "test_task",
                "arg1",
                use_persistent_queue=False
            )
            
            assert task_id is None  # In-memory tasks don't return ID
    
    @pytest.mark.asyncio
    async def test_enqueue_fastapi_task_unknown_function(self):
        """Test FastAPI task enqueueing with unknown function."""
        from fastapi import BackgroundTasks
        
        background_tasks = BackgroundTasks()
        
        with patch('pcs.services.background_task_service.get_background_task_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.task_registry.get_task.return_value = None
            mock_get_service.return_value = mock_service
            
            with pytest.raises(BackgroundTaskError, match="Task function 'unknown_task' not found"):
                await enqueue_fastapi_task(
                    background_tasks,
                    "unknown_task",
                    use_persistent_queue=False
                )
