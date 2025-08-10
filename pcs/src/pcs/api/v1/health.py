"""
Filepath: src/pcs/api/v1/health.py
Purpose: Health check API endpoints for monitoring application status
Related Components: Database health, System monitoring, Load balancers
Tags: health-check, monitoring, api, system-status
"""

import asyncio
import platform
import psutil
import time
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import Settings
from ...core.database import get_database_manager
from ...api.dependencies import get_database_session, get_settings_dependency


router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def basic_health() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns basic service status without expensive checks.
    Used by load balancers for quick health verification.
    """
    return {
        "status": "healthy",
        "service": "PCS",
        "timestamp": time.time(),
        "version": "1.0.0"
    }


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health(
    db_session: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings_dependency)
) -> Dict[str, Any]:
    """
    Comprehensive health check with detailed system information.
    
    Includes database connectivity, system resources, and service dependencies.
    """
    start_time = time.time()
    checks = []
    overall_healthy = True
    
    # Database health check
    try:
        db_health = await _check_database_health()
        checks.append(db_health)
        if db_health["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        checks.append({
            "name": "database",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        })
        overall_healthy = False
    
    # Configuration health check
    try:
        config_health = await _check_configuration(settings)
        checks.append(config_health)
        if config_health["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        checks.append({
            "name": "configuration",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        })
        overall_healthy = False
    
    # System resources check
    try:
        system_health = await _check_system_resources()
        checks.append(system_health)
        if system_health["status"] != "healthy":
            overall_healthy = False
    except Exception as e:
        checks.append({
            "name": "system_resources",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        })
        overall_healthy = False
    
    # Calculate total response time
    total_time = time.time() - start_time
    
    result = {
        "status": "healthy" if overall_healthy else "unhealthy",
        "service": "PCS",
        "environment": settings.environment,
        "version": settings.version,
        "timestamp": time.time(),
        "response_time_ms": round(total_time * 1000, 2),
        "checks": checks,
        "summary": {
            "total_checks": len(checks),
            "healthy_checks": len([c for c in checks if c["status"] == "healthy"]),
            "unhealthy_checks": len([c for c in checks if c["status"] != "healthy"])
        }
    }
    
    # Return appropriate HTTP status
    if not overall_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result
        )
    
    return result


@router.get("/readiness", response_model=Dict[str, Any])
async def readiness_check(
    db_session: AsyncSession = Depends(get_database_session)
) -> Dict[str, Any]:
    """
    Readiness check for Kubernetes and orchestration platforms.
    
    Verifies that the service is ready to receive traffic.
    """
    start_time = time.time()
    
    try:
        # Check critical dependencies
        db_manager = await get_database_manager()
        await db_manager.health_check()
        
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "ready",
            "service": "PCS",
            "timestamp": time.time(),
            "response_time_ms": round(response_time, 2),
            "message": "Service is ready to receive traffic"
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "service": "PCS",
                "timestamp": time.time(),
                "response_time_ms": round(response_time, 2),
                "error": str(e),
                "message": "Service is not ready to receive traffic"
            }
        )


@router.get("/liveness", response_model=Dict[str, Any])
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check for Kubernetes and orchestration platforms.
    
    Simple check to verify the service is alive and responsive.
    Should not include external dependency checks.
    """
    return {
        "status": "alive",
        "service": "PCS",
        "timestamp": time.time(),
        "message": "Service is alive and responsive"
    }


async def _check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    start_time = time.time()
    
    try:
        db_manager = await get_database_manager()
        health_info = await db_manager.health_check()
        
        return {
            "name": "database",
            "status": "healthy",
            "timestamp": time.time(),
            "details": health_info
        }
        
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        
        return {
            "name": "database",
            "status": "unhealthy",
            "timestamp": time.time(),
            "response_time_ms": round(response_time, 2),
            "error": str(e)
        }


async def _check_configuration(settings: Settings) -> Dict[str, Any]:
    """Check application configuration."""
    issues = []
    
    # Check critical configuration
    if not settings.security.secret_key:
        issues.append("Missing secret key")
    
    if not settings.database.host:
        issues.append("Missing database host")
    
    # Check environment-specific settings
    if settings.is_production:
        if settings.debug:
            issues.append("Debug mode enabled in production")
        
        if not settings.security.jwt_secret_key:
            issues.append("Missing JWT secret key in production")
    
    return {
        "name": "configuration",
        "status": "healthy" if not issues else "unhealthy",
        "timestamp": time.time(),
        "details": {
            "environment": settings.environment,
            "debug": settings.debug,
            "issues": issues
        }
    }


async def _check_system_resources() -> Dict[str, Any]:
    """Check system resource availability."""
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine health based on resource usage
        issues = []
        status = "healthy"
        
        if cpu_percent > 90:
            issues.append(f"High CPU usage: {cpu_percent}%")
            status = "warning"
        
        if memory.percent > 90:
            issues.append(f"High memory usage: {memory.percent}%")
            status = "warning"
        
        if disk.percent > 90:
            issues.append(f"High disk usage: {disk.percent}%")
            status = "warning"
        
        if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
            status = "unhealthy"
        
        return {
            "name": "system_resources",
            "status": status,
            "timestamp": time.time(),
            "details": {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2)
                },
                "disk": {
                    "usage_percent": disk.percent,
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                },
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "python_version": platform.python_version()
                },
                "issues": issues
            }
        }
        
    except Exception as e:
        return {
            "name": "system_resources",
            "status": "unhealthy", 
            "timestamp": time.time(),
            "error": str(e)
        }
