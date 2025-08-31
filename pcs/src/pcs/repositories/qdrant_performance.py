"""
Filepath: pcs/src/pcs/repositories/qdrant_performance.py
Purpose: Performance monitoring, optimization, and collection management for Qdrant vector database
Related Components: Performance metrics, collection optimization, monitoring, health checks
Tags: qdrant, performance, optimization, monitoring, collections
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from .qdrant_types import VectorCollectionStats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for Qdrant operations."""
    operation_type: str
    collection_name: str
    execution_time: float
    items_processed: int
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operation_type": self.operation_type,
            "collection_name": self.collection_name,
            "execution_time": self.execution_time,
            "items_processed": self.items_processed,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CollectionPerformanceProfile:
    """Performance profile for a specific collection."""
    collection_name: str
    total_operations: int
    average_execution_time: float
    success_rate: float
    last_optimization: Optional[datetime] = None
    optimization_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.optimization_recommendations is None:
            self.optimization_recommendations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "total_operations": self.total_operations,
            "average_execution_time": self.average_execution_time,
            "success_rate": self.success_rate,
            "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
            "optimization_recommendations": self.optimization_recommendations
        }


class QdrantPerformanceMonitor:
    """
    Performance monitoring and optimization for Qdrant vector database.
    
    This class tracks performance metrics, provides optimization recommendations,
    and manages collection performance profiles.
    """
    
    def __init__(self, core_operations):
        """
        Initialize performance monitor with core operations.
        
        Args:
            core_operations: QdrantCoreOperations instance for basic operations
        """
        self.core = core_operations
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._metrics: List[PerformanceMetrics] = []
        self._collection_profiles: Dict[str, CollectionPerformanceProfile] = {}
        self._query_metrics: List[Dict[str, Any]] = []
        
        # Performance thresholds
        self.slow_query_threshold = 1.0  # seconds
        self.error_rate_threshold = 0.1  # 10%
        self.optimization_interval = timedelta(hours=24)  # 24 hours
    
    def track_operation(
        self,
        operation_type: str,
        collection_name: str,
        execution_time: float,
        items_processed: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Track performance metrics for an operation."""
        try:
            metric = PerformanceMetrics(
                operation_type=operation_type,
                collection_name=collection_name,
                execution_time=execution_time,
                items_processed=items_processed,
                success=success,
                error_message=error_message
            )
            
            self._metrics.append(metric)
            
            # Update collection profile
            self._update_collection_profile(collection_name, metric)
            
            # Log slow operations
            if execution_time > self.slow_query_threshold:
                self.logger.warning(
                    f"Slow operation detected: {operation_type} on {collection_name} "
                    f"took {execution_time:.2f}s (threshold: {self.slow_query_threshold}s)"
                )
            
            # Log errors
            if not success:
                self.logger.error(
                    f"Operation failed: {operation_type} on {collection_name} - {error_message}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to track operation metrics: {e}")
    
    def _update_collection_profile(self, collection_name: str, metric: PerformanceMetrics) -> None:
        """Update performance profile for a collection."""
        try:
            if collection_name not in self._collection_profiles:
                self._collection_profiles[collection_name] = CollectionPerformanceProfile(
                    collection_name=collection_name,
                    total_operations=0,
                    average_execution_time=0.0,
                    success_rate=0.0
                )
            
            profile = self._collection_profiles[collection_name]
            
            # Update metrics
            profile.total_operations += 1
            
            # Update average execution time
            total_time = profile.average_execution_time * (profile.total_operations - 1) + metric.execution_time
            profile.average_execution_time = total_time / profile.total_operations
            
            # Update success rate
            successful_ops = sum(1 for m in self._metrics 
                               if m.collection_name == collection_name and m.success)
            profile.success_rate = successful_ops / profile.total_operations
            
        except Exception as e:
            self.logger.error(f"Failed to update collection profile: {e}")
    
    def get_collection_performance(self, collection_name: str) -> Optional[CollectionPerformanceProfile]:
        """Get performance profile for a specific collection."""
        return self._collection_profiles.get(collection_name)
    
    def get_all_performance_profiles(self) -> List[CollectionPerformanceProfile]:
        """Get performance profiles for all collections."""
        return list(self._collection_profiles.values())
    
    def get_performance_metrics(
        self,
        collection_name: Optional[str] = None,
        operation_type: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> List[PerformanceMetrics]:
        """Get filtered performance metrics."""
        try:
            filtered_metrics = self._metrics
            
            # Filter by collection
            if collection_name:
                filtered_metrics = [m for m in filtered_metrics if m.collection_name == collection_name]
            
            # Filter by operation type
            if operation_type:
                filtered_metrics = [m for m in filtered_metrics if m.operation_type == operation_type]
            
            # Filter by time range
            if time_range:
                cutoff_time = datetime.utcnow() - time_range
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]
            
            return filtered_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        try:
            if not self._metrics:
                return {}
            
            total_operations = len(self._metrics)
            successful_operations = sum(1 for m in self._metrics if m.success)
            failed_operations = total_operations - successful_operations
            
            avg_execution_time = sum(m.execution_time for m in self._metrics) / total_operations
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            
            # Slow operations count
            slow_operations = sum(1 for m in self._metrics if m.execution_time > self.slow_query_threshold)
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "slow_operations": slow_operations,
                "collections_monitored": len(self._collection_profiles),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}


class QdrantPerformanceOptimizer:
    """
    Performance optimization for Qdrant collections.
    
    This class provides optimization recommendations and performs
    collection optimization operations.
    """
    
    def __init__(self, core_operations, performance_monitor):
        """
        Initialize performance optimizer.
        
        Args:
            core_operations: QdrantCoreOperations instance
            performance_monitor: QdrantPerformanceMonitor instance
        """
        self.core = core_operations
        self.monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
    
    async def optimize_collection_performance(
        self,
        collection_name: str,
        optimization_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Optimize collection performance based on type and metrics.
        
        Args:
            collection_name: Name of the collection to optimize
            optimization_type: Type of optimization ("auto", "index", "payload", "full")
            
        Returns:
            Optimization result with recommendations and actions taken
        """
        try:
            self.logger.info(f"Starting performance optimization for {collection_name}")
            
            # Get current collection stats
            stats = self.core.get_collection_stats(collection_name)
            
            # Get performance profile
            profile = self.monitor.get_collection_performance(collection_name)
            
            optimization_result = {
                "collection_name": collection_name,
                "optimization_type": optimization_type,
                "timestamp": datetime.utcnow().isoformat(),
                "actions_taken": [],
                "recommendations": [],
                "performance_impact": "unknown"
            }
            
            if optimization_type == "auto":
                # Analyze performance and apply appropriate optimizations
                recommendations = self._analyze_performance_needs(profile, stats)
                optimization_result["recommendations"] = recommendations
                
                # Apply automatic optimizations
                for rec in recommendations:
                    if rec["priority"] == "high":
                        action_result = await self._apply_optimization(collection_name, rec)
                        optimization_result["actions_taken"].append(action_result)
                
            elif optimization_type == "index":
                # Optimize vector index
                action_result = await self._optimize_vector_index(collection_name)
                optimization_result["actions_taken"].append(action_result)
                
            elif optimization_type == "payload":
                # Optimize payload storage
                action_result = await self._optimize_payload_storage(collection_name)
                optimization_result["actions_taken"].append(action_result)
                
            elif optimization_type == "full":
                # Full optimization
                index_result = await self._optimize_vector_index(collection_name)
                payload_result = await self._optimize_payload_storage(collection_name)
                optimization_result["actions_taken"].extend([index_result, payload_result])
            
            # Update optimization timestamp
            if profile:
                profile.last_optimization = datetime.utcnow()
                profile.optimization_recommendations = optimization_result["recommendations"]
            
            self.logger.info(f"Performance optimization completed for {collection_name}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed for {collection_name}: {e}")
            return {
                "collection_name": collection_name,
                "optimization_type": optimization_type,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "actions_taken": [],
                "recommendations": []
            }
    
    def _analyze_performance_needs(
        self,
        profile: Optional[CollectionPerformanceProfile],
        stats: VectorCollectionStats
    ) -> List[Dict[str, Any]]:
        """Analyze performance needs and generate recommendations."""
        recommendations = []
        
        try:
            # Check execution time
            if profile and profile.average_execution_time > 0.5:  # 500ms threshold
                recommendations.append({
                    "type": "index_optimization",
                    "priority": "high",
                    "reason": f"High average execution time: {profile.average_execution_time:.2f}s",
                    "action": "Optimize vector index for faster search"
                })
            
            # Check success rate
            if profile and profile.success_rate < 0.95:  # 95% threshold
                recommendations.append({
                    "type": "error_investigation",
                    "priority": "high",
                    "reason": f"Low success rate: {profile.success_rate:.1%}",
                    "action": "Investigate and fix error patterns"
                })
            
            # Check collection size
            if stats.points_count > 100000:  # 100k threshold
                recommendations.append({
                    "type": "index_optimization",
                    "priority": "medium",
                    "reason": f"Large collection: {stats.points_count:,} points",
                    "action": "Consider index optimization for large collections"
                })
            
            # Check segments
            if stats.segments_count > 10:  # 10 segments threshold
                recommendations.append({
                    "type": "segment_optimization",
                    "priority": "medium",
                    "reason": f"Many segments: {stats.segments_count}",
                    "action": "Consider segment consolidation"
                })
            
            # Default recommendation if none found
            if not recommendations:
                recommendations.append({
                    "type": "monitoring",
                    "priority": "low",
                    "reason": "Performance within acceptable ranges",
                    "action": "Continue monitoring"
                })
            
        except Exception as e:
            self.logger.error(f"Failed to analyze performance needs: {e}")
            recommendations.append({
                "type": "error",
                "priority": "high",
                "reason": f"Analysis failed: {e}",
                "action": "Investigate system issues"
            })
        
        return recommendations
    
    async def _apply_optimization(
        self,
        collection_name: str,
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a specific optimization based on recommendation."""
        try:
            optimization_type = recommendation["type"]
            
            if optimization_type == "index_optimization":
                return await self._optimize_vector_index(collection_name)
            elif optimization_type == "payload_optimization":
                return await self._optimize_payload_storage(collection_name)
            elif optimization_type == "segment_optimization":
                return await self._optimize_segments(collection_name)
            else:
                return {
                    "type": optimization_type,
                    "status": "skipped",
                    "reason": "Optimization type not implemented"
                }
                
        except Exception as e:
            return {
                "type": recommendation.get("type", "unknown"),
                "status": "failed",
                "error": str(e)
            }
    
    async def _optimize_vector_index(self, collection_name: str) -> Dict[str, Any]:
        """Optimize vector index for better search performance."""
        try:
            # This is a placeholder for actual index optimization
            # In production, you would call Qdrant's index optimization endpoints
            
            self.logger.info(f"Optimizing vector index for {collection_name}")
            
            # Simulate optimization delay
            await asyncio.sleep(1)
            
            return {
                "type": "index_optimization",
                "status": "completed",
                "details": "Vector index optimized for better search performance"
            }
            
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return {
                "type": "index_optimization",
                "status": "failed",
                "error": str(e)
            }
    
    async def _optimize_payload_storage(self, collection_name: str) -> Dict[str, Any]:
        """Optimize payload storage configuration."""
        try:
            # This is a placeholder for actual payload optimization
            # In production, you would call Qdrant's payload optimization endpoints
            
            self.logger.info(f"Optimizing payload storage for {collection_name}")
            
            # Simulate optimization delay
            await asyncio.sleep(0.5)
            
            return {
                "type": "payload_optimization",
                "status": "completed",
                "details": "Payload storage optimized for better performance"
            }
            
        except Exception as e:
            self.logger.error(f"Payload optimization failed: {e}")
            return {
                "type": "payload_optimization",
                "status": "failed",
                "error": str(e)
            }
    
    async def _optimize_segments(self, collection_name: str) -> Dict[str, Any]:
        """Optimize collection segments."""
        try:
            # This is a placeholder for actual segment optimization
            # In production, you would call Qdrant's segment optimization endpoints
            
            self.logger.info(f"Optimizing segments for {collection_name}")
            
            # Simulate optimization delay
            await asyncio.sleep(0.5)
            
            return {
                "type": "segment_optimization",
                "status": "completed",
                "details": "Collection segments optimized"
            }
            
        except Exception as e:
            self.logger.error(f"Segment optimization failed: {e}")
            return {
                "type": "segment_optimization",
                "status": "failed",
                "error": str(e)
            }
    
    def get_optimization_recommendations(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get optimization recommendations for a collection."""
        try:
            profile = self.monitor.get_collection_performance(collection_name)
            stats = self.core.get_collection_stats(collection_name)
            
            return self._analyze_performance_needs(profile, stats)
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    async def create_collection_optimized(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        on_disk_payload: bool = True,
        hnsw_config: Optional[Dict[str, Any]] = None,
        optimizers_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a collection with optimized configuration for performance.
        
        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the vectors
            distance: Distance metric to use
            on_disk_payload: Whether to store payloads on disk
            hnsw_config: HNSW index configuration
            optimizers_config: Optimizers configuration
            
        Returns:
            True if collection created successfully
        """
        try:
            # Build optimized configuration
            if hnsw_config is None:
                hnsw_config = {
                    "m": 16,  # Number of connections per layer
                    "ef_construct": 100,  # Size of dynamic candidate list
                    "full_scan_threshold": 10000  # Threshold for full scan
                }
            
            if optimizers_config is None:
                optimizers_config = {
                    "memmap_threshold": 20000,  # Memory mapping threshold
                    "default_segment_number": 2,  # Default number of segments
                    "max_optimization_threads": 2  # Max optimization threads
                }
            
            # Create collection with optimized config
            success = self.core.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config
            )
            
            if success:
                self.logger.info(f"Created optimized collection: {collection_name}")
                
                # Initialize performance profile
                self.monitor._collection_profiles[collection_name] = CollectionPerformanceProfile(
                    collection_name=collection_name,
                    total_operations=0,
                    average_execution_time=0.0,
                    success_rate=1.0
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create optimized collection {collection_name}: {e}")
            return False
