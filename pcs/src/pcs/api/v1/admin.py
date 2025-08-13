"""
Filepath: src/pcs/api/v1/admin.py
Purpose: Administrative API endpoints for system management, monitoring, and maintenance
Related Components: System metrics, User management, Database administration, Security
Tags: api, admin, monitoring, system-management, security, fastapi
"""

import asyncio
import psutil
import time
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel, Field, ConfigDict

from ...core.config import get_settings
from ...core.exceptions import PCSError, ValidationError
from ...models.prompts import PromptTemplate, PromptStatus
from ...models.contexts import Context, ContextType, ContextScope
from ...models.conversations import Conversation, ConversationStatus
from ...repositories.postgres_repo import PostgreSQLRepository
from ...repositories.redis_repo import RedisRepository
from ..dependencies import (
    get_database_session,
    get_current_user,
    get_current_app,
    require_admin
)

router = APIRouter(prefix="/admin", tags=["admin"])


# Pydantic schemas for admin operations
class SystemStatsResponse(BaseModel):
    """Response schema for system statistics."""
    system_info: Dict[str, Any]
    database_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    application_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    uptime_seconds: float
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class DatabaseMaintenanceRequest(BaseModel):
    """Schema for database maintenance operations."""
    operation: str = Field(..., pattern="^(analyze|vacuum|reindex|cleanup)$", description="Maintenance operation")
    tables: Optional[List[str]] = Field(None, description="Specific tables (empty for all)")
    dry_run: bool = Field(default=True, description="Perform dry run without actual changes")
    force: bool = Field(default=False, description="Force operation even if risky")


class UserManagementRequest(BaseModel):
    """Schema for user management operations."""
    action: str = Field(..., pattern="^(activate|deactivate|reset_password|change_role)$", description="User action")
    user_ids: List[str] = Field(..., min_items=1, description="Target user IDs")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Action parameters")


class CacheManagementRequest(BaseModel):
    """Schema for cache management operations."""
    operation: str = Field(..., pattern="^(flush|clear_pattern|warm|analyze)$", description="Cache operation")
    pattern: Optional[str] = Field(None, description="Pattern for selective operations")
    keys: Optional[List[str]] = Field(None, description="Specific keys to operate on")


class SystemConfigUpdate(BaseModel):
    """Schema for system configuration updates."""
    section: str = Field(..., description="Configuration section")
    settings: Dict[str, Any] = Field(..., description="Settings to update")
    apply_immediately: bool = Field(default=False, description="Apply changes immediately")


class BackupRequest(BaseModel):
    """Schema for backup operations."""
    backup_type: str = Field(..., pattern="^(full|incremental|schema_only|data_only)$", description="Backup type")
    include_cache: bool = Field(default=False, description="Include cache data")
    compress: bool = Field(default=True, description="Compress backup")
    encryption: bool = Field(default=True, description="Encrypt backup")


class MonitoringAlert(BaseModel):
    """Schema for monitoring alerts."""
    alert_id: str
    alert_type: str
    severity: str
    title: str
    description: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]
    is_resolved: bool = False


class PerformanceMetrics(BaseModel):
    """Schema for performance metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    database_connections: int
    active_requests: int
    response_times: Dict[str, float]
    error_rates: Dict[str, float]
    throughput: Dict[str, float]


# System Monitoring Endpoints

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    include_detailed: bool = Query(False, description="Include detailed metrics"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(require_admin)
) -> SystemStatsResponse:
    """
    Get comprehensive system statistics and health metrics.
    
    **Features:**
    - System resource usage
    - Database performance metrics
    - Cache statistics
    - Application performance data
    - Real-time monitoring data
    """
    try:
        start_time = time.time()
        
        # System information
        system_info = {
            "platform": psutil.platform(),
            "architecture": psutil.machine(),
            "processor": psutil.cpu_count(),
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "python_version": psutil.version_info,
        }
        
        # Resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resource_usage = {
            "cpu_percent": cpu_percent,
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_percent": (disk.used / disk.total) * 100,
        }
        
        # Database statistics
        db_stats = {}
        try:
            # Get database connection count
            result = await db.execute(text("SELECT count(*) FROM pg_stat_activity"))
            db_connections = result.scalar()
            
            # Get database size
            result = await db.execute(text("SELECT pg_database_size(current_database())"))
            db_size = result.scalar()
            
            # Get table statistics
            result = await db.execute(text("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables
                ORDER BY n_tup_ins + n_tup_upd + n_tup_del DESC
                LIMIT 10
            """))
            table_stats = [dict(row._mapping) for row in result]
            
            db_stats = {
                "connections": db_connections,
                "size_bytes": db_size,
                "table_statistics": table_stats,
            }
        except Exception as e:
            db_stats = {"error": f"Failed to get database stats: {str(e)}"}
        
        # Application statistics
        prompt_repo = PostgreSQLRepository(db, PromptTemplate)
        context_repo = PostgreSQLRepository(db, Context)
        conversation_repo = PostgreSQLRepository(db, Conversation)
        
        prompts = await prompt_repo.find_by_criteria()
        contexts = await context_repo.find_by_criteria()
        conversations = await conversation_repo.find_by_criteria()
        
        app_stats = {
            "total_prompts": len(prompts),
            "active_prompts": len([p for p in prompts if p.status == PromptStatus.ACTIVE]),
            "total_contexts": len(contexts),
            "total_conversations": len(conversations),
            "active_conversations": len([c for c in conversations if c.status == ConversationStatus.ACTIVE]),
        }
        
        # Cache statistics (simplified - would integrate with actual Redis)
        cache_stats = {
            "status": "available",
            "memory_usage": "unknown",
            "hit_rate": "unknown",
            "keys_count": "unknown"
        }
        
        # Performance metrics
        performance_metrics = {
            "response_time_ms": (time.time() - start_time) * 1000,
            "uptime_seconds": time.time() - psutil.boot_time(),
        }
        
        return SystemStatsResponse(
            system_info=system_info,
            database_stats=db_stats,
            cache_stats=cache_stats,
            application_stats=app_stats,
            performance_metrics=performance_metrics,
            resource_usage=resource_usage,
            uptime_seconds=performance_metrics["uptime_seconds"],
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )


@router.get("/health/detailed")
async def get_detailed_health(
    check_external: bool = Query(True, description="Check external dependencies"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Get detailed health check including all system components.
    
    **Features:**
    - Database connectivity and performance
    - Cache system health
    - External service availability
    - Resource availability checks
    - Security status validation
    """
    try:
        health_results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Database health check
        try:
            start_time = time.time()
            result = await db.execute(text("SELECT 1"))
            db_response_time = (time.time() - start_time) * 1000
            
            health_results["checks"]["database"] = {
                "status": "healthy",
                "response_time_ms": db_response_time,
                "details": "Database connection successful"
            }
        except Exception as e:
            health_results["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }
            health_results["status"] = "degraded"
        
        # Resource checks
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Memory check (alert if > 85% usage)
        memory_status = "healthy" if memory.percent < 85 else "warning" if memory.percent < 95 else "critical"
        health_results["checks"]["memory"] = {
            "status": memory_status,
            "usage_percent": memory.percent,
            "available_gb": memory.available / (1024**3)
        }
        
        # Disk check (alert if > 80% usage)
        disk_percent = (disk.used / disk.total) * 100
        disk_status = "healthy" if disk_percent < 80 else "warning" if disk_percent < 90 else "critical"
        health_results["checks"]["disk"] = {
            "status": disk_status,
            "usage_percent": disk_percent,
            "available_gb": disk.free / (1024**3)
        }
        
        # Update overall status based on individual checks
        check_statuses = [check["status"] for check in health_results["checks"].values()]
        if "critical" in check_statuses:
            health_results["status"] = "critical"
        elif "warning" in check_statuses:
            health_results["status"] = "warning"
        elif "unhealthy" in check_statuses:
            health_results["status"] = "degraded"
        
        return health_results
    
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Database Administration

@router.post("/database/maintenance")
async def perform_database_maintenance(
    maintenance_request: DatabaseMaintenanceRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Perform database maintenance operations.
    
    **Features:**
    - ANALYZE tables for query optimization
    - VACUUM operations for space reclamation
    - REINDEX for index optimization
    - Cleanup operations for old data
    - Dry run support for safety
    """
    try:
        operation = maintenance_request.operation
        tables = maintenance_request.tables or []
        dry_run = maintenance_request.dry_run
        
        maintenance_results = {
            "operation": operation,
            "dry_run": dry_run,
            "started_at": datetime.utcnow().isoformat(),
            "tables_processed": [],
            "errors": [],
            "warnings": []
        }
        
        if operation == "analyze":
            if not tables:
                # Analyze all user tables
                result = await db.execute(text("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                """))
                tables = [row[0] for row in result]
            
            for table in tables:
                try:
                    if not dry_run:
                        await db.execute(text(f"ANALYZE {table}"))
                    maintenance_results["tables_processed"].append({
                        "table": table,
                        "operation": "analyze",
                        "status": "completed" if not dry_run else "dry_run"
                    })
                except Exception as e:
                    maintenance_results["errors"].append({
                        "table": table,
                        "error": str(e)
                    })
        
        elif operation == "vacuum":
            if not tables:
                result = await db.execute(text("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                """))
                tables = [row[0] for row in result]
            
            for table in tables:
                try:
                    if not dry_run:
                        await db.execute(text(f"VACUUM ANALYZE {table}"))
                    maintenance_results["tables_processed"].append({
                        "table": table,
                        "operation": "vacuum",
                        "status": "completed" if not dry_run else "dry_run"
                    })
                except Exception as e:
                    maintenance_results["errors"].append({
                        "table": table,
                        "error": str(e)
                    })
        
        elif operation == "cleanup":
            # Example cleanup operations
            cleanup_operations = [
                ("conversations", "status = 'archived' AND updated_at < NOW() - INTERVAL '90 days'"),
                ("conversation_messages", "created_at < NOW() - INTERVAL '365 days'"),
            ]
            
            for table, condition in cleanup_operations:
                try:
                    if table in tables or not tables:
                        if not dry_run:
                            result = await db.execute(text(f"DELETE FROM {table} WHERE {condition}"))
                            rows_affected = result.rowcount
                        else:
                            result = await db.execute(text(f"SELECT COUNT(*) FROM {table} WHERE {condition}"))
                            rows_affected = result.scalar()
                        
                        maintenance_results["tables_processed"].append({
                            "table": table,
                            "operation": "cleanup",
                            "rows_affected": rows_affected,
                            "status": "completed" if not dry_run else "dry_run"
                        })
                except Exception as e:
                    maintenance_results["errors"].append({
                        "table": table,
                        "error": str(e)
                    })
        
        maintenance_results["completed_at"] = datetime.utcnow().isoformat()
        
        return maintenance_results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database maintenance failed: {str(e)}"
        )


@router.get("/database/query-performance")
async def get_query_performance(
    limit: int = Query(20, ge=1, le=100, description="Number of queries to return"),
    db: AsyncSession = Depends(get_database_session),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Get database query performance statistics.
    
    **Features:**
    - Slowest queries identification
    - Query execution statistics
    - Resource usage by query
    - Performance optimization suggestions
    """
    try:
        # Get slow queries from pg_stat_statements (if available)
        try:
            result = await db.execute(text("""
                SELECT 
                    query,
                    calls,
                    total_time,
                    mean_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                ORDER BY total_time DESC 
                LIMIT :limit
            """), {"limit": limit})
            
            slow_queries = [dict(row._mapping) for row in result]
        except Exception:
            # Fallback if pg_stat_statements not available
            slow_queries = []
        
        # Get current active queries
        result = await db.execute(text("""
            SELECT 
                pid,
                usename,
                application_name,
                client_addr,
                state,
                query_start,
                query
            FROM pg_stat_activity 
            WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
            ORDER BY query_start
        """))
        
        active_queries = [dict(row._mapping) for row in result]
        
        # Get database statistics
        result = await db.execute(text("""
            SELECT 
                numbackends,
                xact_commit,
                xact_rollback,
                blks_read,
                blks_hit,
                tup_returned,
                tup_fetched,
                tup_inserted,
                tup_updated,
                tup_deleted
            FROM pg_stat_database 
            WHERE datname = current_database()
        """))
        
        db_stats = dict(result.fetchone()._mapping) if result.rowcount > 0 else {}
        
        return {
            "slow_queries": slow_queries,
            "active_queries": active_queries,
            "database_stats": db_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get query performance: {str(e)}"
        )


# Cache Management

@router.post("/cache/management")
async def manage_cache(
    cache_request: CacheManagementRequest,
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Perform cache management operations.
    
    **Features:**
    - Cache flush operations
    - Pattern-based cache clearing
    - Cache warming for performance
    - Cache analytics and optimization
    """
    try:
        operation = cache_request.operation
        pattern = cache_request.pattern
        keys = cache_request.keys
        
        # This is a simplified implementation
        # In production, you would integrate with actual Redis client
        cache_results = {
            "operation": operation,
            "pattern": pattern,
            "keys_processed": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        
        if operation == "flush":
            # Flush all cache
            cache_results["keys_processed"] = "all"
            cache_results["message"] = "All cache entries flushed"
        
        elif operation == "clear_pattern":
            if not pattern:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Pattern required for clear_pattern operation"
                )
            cache_results["keys_processed"] = f"pattern:{pattern}"
            cache_results["message"] = f"Cache entries matching pattern '{pattern}' cleared"
        
        elif operation == "warm":
            # Cache warming logic would go here
            cache_results["message"] = "Cache warming initiated"
        
        elif operation == "analyze":
            # Cache analysis would go here
            cache_results["analysis"] = {
                "total_keys": 0,
                "memory_usage": "0MB",
                "hit_rate": "N/A",
                "top_patterns": []
            }
        
        return cache_results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache management failed: {str(e)}"
        )


# User Management

@router.get("/users")
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(50, ge=1, le=200, description="Page size"),
    search: Optional[str] = Query(None, description="Search users"),
    active_only: bool = Query(True, description="Show only active users"),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    List all users with administrative information.
    
    **Features:**
    - User account status
    - Activity statistics
    - Permission levels
    - Recent activity tracking
    """
    try:
        # This would integrate with your user management system
        # For now, returning a simplified response
        users = [
            {
                "id": "user_1",
                "username": "admin",
                "email": "admin@example.com",
                "is_active": True,
                "is_admin": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": datetime.utcnow().isoformat(),
                "conversation_count": 5,
                "total_tokens_used": 10000
            },
            {
                "id": "user_2", 
                "username": "user1",
                "email": "user1@example.com",
                "is_active": True,
                "is_admin": False,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "conversation_count": 3,
                "total_tokens_used": 5000
            }
        ]
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            users = [
                u for u in users
                if (search_lower in u["username"].lower() or 
                    search_lower in u["email"].lower())
            ]
        
        # Apply active filter
        if active_only:
            users = [u for u in users if u["is_active"]]
        
        # Apply pagination
        total = len(users)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_users = users[start_idx:end_idx]
        
        return {
            "users": paginated_users,
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@router.post("/users/manage")
async def manage_users(
    user_request: UserManagementRequest,
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Perform user management operations.
    
    **Features:**
    - Activate/deactivate user accounts
    - Reset passwords
    - Change user roles and permissions
    - Bulk user operations
    """
    try:
        action = user_request.action
        user_ids = user_request.user_ids
        parameters = user_request.parameters
        
        results = {
            "action": action,
            "processed_users": [],
            "errors": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for user_id in user_ids:
            try:
                # Prevent admin from deactivating themselves
                if action == "deactivate" and user_id == current_user.get("id"):
                    results["errors"].append({
                        "user_id": user_id,
                        "error": "Cannot deactivate your own account"
                    })
                    continue
                
                # Simulate user management operations
                if action == "activate":
                    results["processed_users"].append({
                        "user_id": user_id,
                        "action": "activated",
                        "status": "success"
                    })
                
                elif action == "deactivate":
                    results["processed_users"].append({
                        "user_id": user_id,
                        "action": "deactivated", 
                        "status": "success"
                    })
                
                elif action == "reset_password":
                    # Generate temporary password
                    temp_password = f"temp_{uuid4().hex[:8]}"
                    results["processed_users"].append({
                        "user_id": user_id,
                        "action": "password_reset",
                        "temporary_password": temp_password,
                        "status": "success"
                    })
                
                elif action == "change_role":
                    new_role = parameters.get("role", "user")
                    results["processed_users"].append({
                        "user_id": user_id,
                        "action": "role_changed",
                        "new_role": new_role,
                        "status": "success"
                    })
                
            except Exception as e:
                results["errors"].append({
                    "user_id": user_id,
                    "error": str(e)
                })
        
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User management failed: {str(e)}"
        )


# System Configuration

@router.get("/config")
async def get_system_config(
    section: Optional[str] = Query(None, description="Configuration section"),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Get system configuration settings.
    
    **Features:**
    - Application configuration
    - Database settings
    - Cache configuration
    - Security settings
    - Feature flags
    """
    try:
        settings = get_settings()
        
        config_data = {
            "application": {
                "app_name": settings.app_name,
                "version": settings.version,
                "environment": settings.environment,
                "debug": settings.debug,
                "api_prefix": settings.api_prefix,
            },
            "server": {
                "host": settings.host,
                "port": settings.port,
            },
            "features": {
                "enable_analytics": True,
                "enable_caching": True,
                "enable_monitoring": True,
                "max_conversation_length": 1000,
                "max_context_size": 10000,
            }
        }
        
        # Filter by section if specified
        if section:
            if section in config_data:
                return {section: config_data[section]}
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configuration section '{section}' not found"
                )
        
        return config_data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system config: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", description="Log level filter"),
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines"),
    search: Optional[str] = Query(None, description="Search in log content"),
    current_user: Dict[str, Any] = Depends(require_admin)
):
    """
    Get system logs for debugging and monitoring.
    
    **Features:**
    - Filtered log retrieval
    - Log level filtering
    - Content search
    - Real-time log streaming capability
    """
    try:
        # This would integrate with your logging system
        # For now, returning simulated logs
        
        sample_logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "INFO",
                "module": "api.v1.prompts",
                "message": "Prompt template created successfully",
                "user_id": "user_123",
                "request_id": "req_456"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
                "level": "WARNING",
                "module": "core.database",
                "message": "Database connection pool at 80% capacity",
                "details": {"pool_size": 20, "active_connections": 16}
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
                "level": "ERROR",
                "module": "services.context_service",
                "message": "Failed to merge contexts",
                "error": "Validation failed: conflicting schemas",
                "user_id": "user_789"
            }
        ]
        
        # Filter by level
        if level != "ALL":
            sample_logs = [log for log in sample_logs if log["level"] == level.upper()]
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            sample_logs = [
                log for log in sample_logs
                if search_lower in log["message"].lower()
            ]
        
        # Limit results
        sample_logs = sample_logs[:lines]
        
        return {
            "logs": sample_logs,
            "total_lines": len(sample_logs),
            "level_filter": level,
            "search_filter": search,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system logs: {str(e)}"
        )
