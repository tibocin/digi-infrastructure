"""
Filepath: src/pcs/services/background_task_service.py
Purpose: Background task processing system with Redis queues, worker management, and monitoring
Related Components: Redis, FastAPI BackgroundTasks, Metrics, Logging
Tags: background-tasks, redis, queues, workers, async, monitoring
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import BackgroundTasks as FastAPIBackgroundTasks
from pydantic import BaseModel, Field

from pcs.core.config import get_settings
from pcs.core.exceptions import PCSError, ServiceError
from pcs.utils.logger import get_logger
from pcs.utils.metrics import get_metrics_collector, record_manual_metric

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskResult:
    """Container for task execution result."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    completed_at: Optional[datetime] = None


class TaskDefinition(BaseModel):
    """Task definition with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Task name/type")
    args: List[Any] = Field(default_factory=list, description="Task arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Task keyword arguments")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=60, description="Retry delay in seconds")
    timeout: Optional[int] = Field(default=None, description="Task timeout in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_for: Optional[datetime] = Field(default=None, description="Scheduled execution time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BackgroundTaskError(PCSError):
    """Background task related errors."""
    pass


class TaskRegistry:
    """Registry for task functions."""
    
    def __init__(self):
        self._tasks: Dict[str, Callable] = {}
    
    def register(self, name: str) -> Callable:
        """
        Decorator to register a task function.
        
        Args:
            name: Task name to register
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            self._tasks[name] = func
            logger.info(f"Registered task: {name}")
            return func
        return decorator
    
    def get_task(self, name: str) -> Optional[Callable]:
        """Get task function by name."""
        return self._tasks.get(name)
    
    def list_tasks(self) -> List[str]:
        """List all registered task names."""
        return list(self._tasks.keys())


class BackgroundTaskService:
    """
    Background task processing service with Redis queues.
    
    Features:
    - Persistent task queues using Redis
    - Retry logic with exponential backoff
    - Task priority handling
    - Worker management
    - Monitoring and metrics
    - Task scheduling
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        """
        Initialize background task service.
        
        Args:
            redis_client: Redis client instance (optional)
        """
        self.settings = get_settings()
        self.redis_client = redis_client
        self.task_registry = TaskRegistry()
        self.metrics_collector = get_metrics_collector()
        
        # Queue names
        self.task_queue = "pcs:tasks:queue"
        self.processing_queue = "pcs:tasks:processing"
        self.completed_queue = "pcs:tasks:completed"
        self.failed_queue = "pcs:tasks:failed"
        
        # Worker control
        self._workers: List[asyncio.Task] = []
        self._worker_count = 3
        self._shutdown_event = asyncio.Event()
        
        logger.info("Background task service initialized")
    
    async def initialize(self) -> None:
        """Initialize Redis connection and setup queues."""
        if not self.redis_client:
            self.redis_client = aioredis.from_url(
                self.settings.redis.url,
                decode_responses=True,
                max_connections=self.settings.redis.max_connections,
                socket_timeout=self.settings.redis.socket_timeout,
                socket_connect_timeout=self.settings.redis.connection_timeout
            )
        
        # Test connection
        try:
            await self.redis_client.ping()
            logger.info("Redis connection established for background tasks")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise BackgroundTaskError("Redis connection failed")
    
    async def shutdown(self) -> None:
        """Shutdown the background task service."""
        logger.info("Shutting down background task service...")
        
        # Signal workers to stop
        self._shutdown_event.set()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Background task service shutdown complete")
    
    async def enqueue_task(
        self,
        task_name: str,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: int = 60,
        timeout: Optional[int] = None,
        scheduled_for: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Enqueue a background task.
        
        Args:
            task_name: Name of the task to execute
            *args: Task arguments
            priority: Task priority
            max_retries: Maximum retry attempts
            retry_delay: Retry delay in seconds
            timeout: Task timeout in seconds
            scheduled_for: Scheduled execution time
            metadata: Additional metadata
            **kwargs: Task keyword arguments
            
        Returns:
            Task ID
        """
        if not self.redis_client:
            await self.initialize()
        
        task = TaskDefinition(
            name=task_name,
            args=list(args),
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            scheduled_for=scheduled_for,
            metadata=metadata or {}
        )
        
        # Serialize task
        task_data = task.model_dump_json()
        
        # Add to appropriate queue based on scheduling
        if scheduled_for and scheduled_for > datetime.utcnow():
            # Add to delayed queue with score as timestamp
            score = scheduled_for.timestamp()
            await self.redis_client.zadd("pcs:tasks:scheduled", {task_data: score})
            logger.info(f"Scheduled task {task.id} for {scheduled_for}")
        else:
            # Add to immediate queue based on priority
            queue_name = f"{self.task_queue}:{priority.value}"
            await self.redis_client.lpush(queue_name, task_data)
            logger.info(f"Enqueued task {task.id} with priority {priority.value}")
        
        # Store task metadata
        await self.redis_client.hset(
            f"pcs:tasks:meta:{task.id}",
            mapping={
                "status": TaskStatus.PENDING.value,
                "created_at": task.created_at.isoformat(),
                "retries": "0"
            }
        )
        
        # Set expiration for metadata (30 days)
        await self.redis_client.expire(f"pcs:tasks:meta:{task.id}", 30 * 24 * 60 * 60)
        
        # Record metrics
        record_manual_metric(
            query_type="task_enqueue",
            execution_time=0.0,
            rows_affected=1,
            table_name="background_tasks"
        )
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status and metadata.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        if not self.redis_client:
            await self.initialize()
        
        metadata = await self.redis_client.hgetall(f"pcs:tasks:meta:{task_id}")
        if not metadata:
            return None
        
        return {
            "task_id": task_id,
            "status": metadata.get("status"),
            "created_at": metadata.get("created_at"),
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at"),
            "retries": int(metadata.get("retries", 0)),
            "error": metadata.get("error"),
            "result": metadata.get("result")
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or scheduled task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False if not found or already processed
        """
        if not self.redis_client:
            await self.initialize()
        
        # Update status to cancelled
        result = await self.redis_client.hset(
            f"pcs:tasks:meta:{task_id}",
            "status", TaskStatus.CANCELLED.value
        )
        
        if result:
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    async def start_workers(self, worker_count: Optional[int] = None) -> None:
        """
        Start background task workers.
        
        Args:
            worker_count: Number of workers to start (default: 3)
        """
        if worker_count:
            self._worker_count = worker_count
        
        if not self.redis_client:
            await self.initialize()
        
        logger.info(f"Starting {self._worker_count} background task workers")
        
        # Start workers
        for i in range(self._worker_count):
            worker_task = asyncio.create_task(
                self._worker_loop(f"worker-{i}")
            )
            self._workers.append(worker_task)
        
        # Start scheduler for delayed tasks
        scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._workers.append(scheduler_task)
    
    async def _worker_loop(self, worker_name: str) -> None:
        """
        Main worker loop to process tasks.
        
        Args:
            worker_name: Name of the worker
        """
        logger.info(f"Worker {worker_name} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Try to get task from priority queues
                task_data = await self._get_next_task()
                
                if task_data:
                    await self._process_task(task_data, worker_name)
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _scheduler_loop(self) -> None:
        """Scheduler loop to move scheduled tasks to ready queue."""
        logger.info("Task scheduler started")
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow().timestamp()
                
                # Get tasks scheduled for now or earlier
                scheduled_tasks = await self.redis_client.zrangebyscore(
                    "pcs:tasks:scheduled",
                    0,
                    current_time,
                    withscores=False
                )
                
                for task_data in scheduled_tasks:
                    try:
                        task = TaskDefinition.model_validate_json(task_data)
                        
                        # Move to appropriate priority queue
                        queue_name = f"{self.task_queue}:{task.priority.value}"
                        await self.redis_client.lpush(queue_name, task_data)
                        
                        # Remove from scheduled queue
                        await self.redis_client.zrem("pcs:tasks:scheduled", task_data)
                        
                        logger.info(f"Moved scheduled task {task.id} to ready queue")
                        
                    except Exception as e:
                        logger.error(f"Error processing scheduled task: {e}")
                        # Remove malformed task
                        await self.redis_client.zrem("pcs:tasks:scheduled", task_data)
                
                # Sleep for a minute before checking again
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Task scheduler stopped")
    
    async def _get_next_task(self) -> Optional[str]:
        """Get next task from priority queues."""
        # Check queues in priority order
        priority_queues = [
            f"{self.task_queue}:{TaskPriority.CRITICAL.value}",
            f"{self.task_queue}:{TaskPriority.HIGH.value}",
            f"{self.task_queue}:{TaskPriority.NORMAL.value}",
            f"{self.task_queue}:{TaskPriority.LOW.value}"
        ]
        
        for queue in priority_queues:
            task_data = await self.redis_client.brpoplpush(
                queue,
                self.processing_queue,
                timeout=1
            )
            if task_data:
                return task_data
        
        return None
    
    async def _process_task(self, task_data: str, worker_name: str) -> None:
        """
        Process a single task.
        
        Args:
            task_data: Serialized task data
            worker_name: Name of the worker processing the task
        """
        start_time = time.time()
        task = None
        
        try:
            # Parse task
            task = TaskDefinition.model_validate_json(task_data)
            
            # Check if task is cancelled
            status_info = await self.get_task_status(task.id)
            if status_info and status_info.get("status") == TaskStatus.CANCELLED.value:
                logger.info(f"Skipping cancelled task {task.id}")
                await self.redis_client.lrem(self.processing_queue, 1, task_data)
                return
            
            # Update status to running
            await self.redis_client.hset(
                f"pcs:tasks:meta:{task.id}",
                mapping={
                    "status": TaskStatus.RUNNING.value,
                    "started_at": datetime.utcnow().isoformat(),
                    "worker": worker_name
                }
            )
            
            logger.info(f"Worker {worker_name} processing task {task.id} ({task.name})")
            
            # Get task function
            task_func = self.task_registry.get_task(task.name)
            if not task_func:
                raise BackgroundTaskError(f"Task function '{task.name}' not found")
            
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_task_function(task_func, task.args, task.kwargs),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_task_function(task_func, task.args, task.kwargs)
            
            # Task completed successfully
            execution_time = time.time() - start_time
            
            await self._mark_task_completed(task.id, result, execution_time)
            
            logger.info(f"Task {task.id} completed successfully in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Task timed out after {task.timeout}s"
            await self._handle_task_failure(task, error_msg, execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            await self._handle_task_failure(task, error_msg, execution_time)
            
        finally:
            # Remove from processing queue
            await self.redis_client.lrem(self.processing_queue, 1, task_data)
    
    async def _execute_task_function(
        self,
        task_func: Callable,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        """Execute task function with proper async handling."""
        if asyncio.iscoroutinefunction(task_func):
            return await task_func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))
    
    async def _mark_task_completed(
        self,
        task_id: str,
        result: Any,
        execution_time: float
    ) -> None:
        """Mark task as completed with result."""
        completed_at = datetime.utcnow()
        
        # Serialize result
        try:
            result_str = json.dumps(result) if result is not None else None
        except (TypeError, ValueError):
            result_str = str(result)
        
        await self.redis_client.hset(
            f"pcs:tasks:meta:{task_id}",
            mapping={
                "status": TaskStatus.COMPLETED.value,
                "completed_at": completed_at.isoformat(),
                "execution_time": str(execution_time),
                "result": result_str
            }
        )
        
        # Record metrics
        record_manual_metric(
            query_type="task_completed",
            execution_time=execution_time,
            rows_affected=1,
            table_name="background_tasks"
        )
    
    async def _handle_task_failure(
        self,
        task: Optional[TaskDefinition],
        error_msg: str,
        execution_time: float
    ) -> None:
        """Handle task failure with retry logic."""
        if not task:
            logger.error(f"Task failure with no task data: {error_msg}")
            return
        
        logger.error(f"Task {task.id} failed: {error_msg}")
        
        # Get current retry count
        status_info = await self.get_task_status(task.id)
        current_retries = int(status_info.get("retries", 0)) if status_info else 0
        
        if current_retries < task.max_retries:
            # Schedule retry
            await self._schedule_retry(task, current_retries + 1, error_msg)
        else:
            # Mark as failed permanently
            await self.redis_client.hset(
                f"pcs:tasks:meta:{task.id}",
                mapping={
                    "status": TaskStatus.FAILED.value,
                    "completed_at": datetime.utcnow().isoformat(),
                    "execution_time": str(execution_time),
                    "error": error_msg
                }
            )
            
            # Record metrics
            record_manual_metric(
                query_type="task_failed",
                execution_time=execution_time,
                rows_affected=1,
                table_name="background_tasks"
            )
    
    async def _schedule_retry(
        self,
        task: TaskDefinition,
        retry_count: int,
        error_msg: str
    ) -> None:
        """Schedule task for retry with exponential backoff."""
        # Calculate retry delay with exponential backoff
        delay = task.retry_delay * (2 ** (retry_count - 1))
        retry_time = datetime.utcnow() + timedelta(seconds=delay)
        
        # Update task metadata
        await self.redis_client.hset(
            f"pcs:tasks:meta:{task.id}",
            mapping={
                "status": TaskStatus.RETRYING.value,
                "retries": str(retry_count),
                "last_error": error_msg,
                "retry_scheduled_for": retry_time.isoformat()
            }
        )
        
        # Schedule for retry
        task_data = task.model_dump_json()
        score = retry_time.timestamp()
        await self.redis_client.zadd("pcs:tasks:scheduled", {task_data: score})
        
        logger.info(f"Scheduled task {task.id} for retry {retry_count}/{task.max_retries} at {retry_time}")
    
    def register_task(self, name: str) -> Callable:
        """
        Decorator to register a task function.
        
        Args:
            name: Task name
            
        Returns:
            Decorated function
        """
        return self.task_registry.register(name)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics and metrics."""
        if not self.redis_client:
            await self.initialize()
        
        stats = {
            "queues": {},
            "processing": 0,
            "scheduled": 0,
            "total_workers": len(self._workers)
        }
        
        # Get queue lengths for each priority
        for priority in TaskPriority:
            queue_name = f"{self.task_queue}:{priority.value}"
            length = await self.redis_client.llen(queue_name)
            stats["queues"][priority.value] = length
        
        # Get processing queue length
        stats["processing"] = await self.redis_client.llen(self.processing_queue)
        
        # Get scheduled tasks count
        stats["scheduled"] = await self.redis_client.zcard("pcs:tasks:scheduled")
        
        return stats


# Global task service instance
_background_task_service: Optional[BackgroundTaskService] = None


def get_background_task_service() -> BackgroundTaskService:
    """Get the global background task service instance."""
    global _background_task_service
    if _background_task_service is None:
        _background_task_service = BackgroundTaskService()
    return _background_task_service


@asynccontextmanager
async def background_task_lifespan():
    """Context manager for background task service lifecycle."""
    service = get_background_task_service()
    
    try:
        await service.initialize()
        await service.start_workers()
        yield service
    finally:
        await service.shutdown()


# FastAPI integration helper
async def enqueue_fastapi_task(
    background_tasks: FastAPIBackgroundTasks,
    task_name: str,
    *args,
    use_persistent_queue: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Helper to enqueue task using either FastAPI BackgroundTasks or persistent queue.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        task_name: Task name
        *args: Task arguments
        use_persistent_queue: Whether to use persistent Redis queue
        **kwargs: Task keyword arguments
        
    Returns:
        Task ID if using persistent queue, None otherwise
    """
    if use_persistent_queue:
        service = get_background_task_service()
        return await service.enqueue_task(task_name, *args, **kwargs)
    else:
        # Use FastAPI's in-memory background tasks
        service = get_background_task_service()
        task_func = service.task_registry.get_task(task_name)
        
        if not task_func:
            raise BackgroundTaskError(f"Task function '{task_name}' not found")
        
        background_tasks.add_task(task_func, *args, **kwargs)
        return None
