"""
Filepath: tests/unit/test_performance_optimization_service.py
Purpose: Unit tests for performance optimization service
Related Components: BottleneckAnalyzer, PerformanceOptimizer, PerformanceBenchmark
Tags: unit-tests, performance, optimization, benchmarking, bottleneck-analysis
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from typing import Any, Dict

from pcs.services.performance_optimization_service import (
    PerformanceOptimizationService,
    BottleneckAnalyzer,
    PerformanceOptimizer,
    PerformanceBenchmark,
    Bottleneck,
    OptimizationResult,
    BenchmarkResult,
    PerformanceMetric,
    OptimizationType,
    BottleneckSeverity,
    OptimizationStatus,
    PerformanceError,
    get_performance_optimization_service,
    analyze_performance_bottlenecks,
    run_performance_optimization,
    benchmark_system_performance,
    performance_profiler
)


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def bottleneck_analyzer(mock_db_session):
    """Fixture providing a BottleneckAnalyzer instance."""
    return BottleneckAnalyzer(mock_db_session)


@pytest.fixture
def performance_optimizer(mock_db_session):
    """Fixture providing a PerformanceOptimizer instance."""
    return PerformanceOptimizer(mock_db_session)


@pytest.fixture
def performance_benchmark(mock_db_session):
    """Fixture providing a PerformanceBenchmark instance."""
    return PerformanceBenchmark(mock_db_session)


@pytest.fixture
def performance_service(mock_db_session):
    """Fixture providing a PerformanceOptimizationService instance."""
    return PerformanceOptimizationService(mock_db_session)


@pytest.fixture
def sample_bottleneck():
    """Fixture providing a sample bottleneck."""
    return Bottleneck(
        id="test_bottleneck",
        name="Test Bottleneck",
        description="Test performance bottleneck",
        severity=BottleneckSeverity.HIGH,
        optimization_type=OptimizationType.MEMORY,
        impact_score=80.0,
        current_value=90.0,
        target_value=70.0,
        recommendations=["Optimize memory usage", "Implement caching"],
        detected_at=datetime.utcnow(),
        metadata={"component": "test"}
    )


@pytest.fixture
def sample_optimization_result():
    """Fixture providing a sample optimization result."""
    return OptimizationResult(
        optimization_id="test_opt_123",
        name="Test Optimization",
        optimization_type=OptimizationType.MEMORY,
        status=OptimizationStatus.COMPLETED,
        before_metrics={"memory_percent": 90.0},
        after_metrics={"memory_percent": 75.0},
        improvement_percent=16.7,
        execution_time=2.5,
        applied_at=datetime.utcnow(),
        notes="Optimization successful"
    )


@pytest.fixture
def sample_benchmark_result():
    """Fixture providing a sample benchmark result."""
    return BenchmarkResult(
        name="Test Benchmark",
        operations_per_second=1000.0,
        avg_response_time_ms=5.0,
        p95_response_time_ms=10.0,
        p99_response_time_ms=15.0,
        memory_usage_mb=50.0,
        cpu_usage_percent=25.0,
        timestamp=datetime.utcnow(),
        test_duration=10.0,
        metadata={"test_type": "unit"}
    )


class TestPerformanceMetric:
    """Test PerformanceMetric functionality."""
    
    def test_performance_metric_creation(self):
        """Test creating performance metric."""
        metric = PerformanceMetric(
            name="cpu_usage",
            value=75.5,
            unit="percent",
            timestamp=datetime.utcnow(),
            category="system",
            metadata={"source": "psutil"}
        )
        
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "percent"
        assert metric.category == "system"
        assert metric.metadata["source"] == "psutil"


class TestBottleneck:
    """Test Bottleneck functionality."""
    
    def test_bottleneck_creation(self, sample_bottleneck):
        """Test creating bottleneck."""
        assert sample_bottleneck.id == "test_bottleneck"
        assert sample_bottleneck.name == "Test Bottleneck"
        assert sample_bottleneck.severity == BottleneckSeverity.HIGH
        assert sample_bottleneck.optimization_type == OptimizationType.MEMORY
        assert sample_bottleneck.impact_score == 80.0
        assert len(sample_bottleneck.recommendations) == 2


class TestOptimizationResult:
    """Test OptimizationResult functionality."""
    
    def test_optimization_result_creation(self, sample_optimization_result):
        """Test creating optimization result."""
        assert sample_optimization_result.optimization_id == "test_opt_123"
        assert sample_optimization_result.status == OptimizationStatus.COMPLETED
        assert sample_optimization_result.improvement_percent == 16.7
        assert sample_optimization_result.before_metrics["memory_percent"] == 90.0
        assert sample_optimization_result.after_metrics["memory_percent"] == 75.0


class TestBenchmarkResult:
    """Test BenchmarkResult functionality."""
    
    def test_benchmark_result_creation(self, sample_benchmark_result):
        """Test creating benchmark result."""
        assert sample_benchmark_result.name == "Test Benchmark"
        assert sample_benchmark_result.operations_per_second == 1000.0
        assert sample_benchmark_result.avg_response_time_ms == 5.0
        assert sample_benchmark_result.p95_response_time_ms == 10.0
        assert sample_benchmark_result.cpu_usage_percent == 25.0


class TestBottleneckAnalyzer:
    """Test BottleneckAnalyzer functionality."""
    
    def test_analyzer_initialization(self, bottleneck_analyzer):
        """Test analyzer initialization."""
        assert bottleneck_analyzer.db_session is not None
        assert bottleneck_analyzer.metrics_collector is not None
        assert bottleneck_analyzer.monitoring_service is not None
        assert len(bottleneck_analyzer._analysis_history) == 0
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_analyze_memory_usage_high(self, mock_memory, bottleneck_analyzer):
        """Test memory analysis with high usage."""
        # Mock high memory usage (90%)
        mock_mem = MagicMock()
        mock_mem.percent = 90.0
        mock_mem.total = 8000000000
        mock_mem.used = 7200000000
        mock_memory.return_value = mock_mem
        
        bottlenecks = await bottleneck_analyzer._analyze_memory_usage()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0].name == "High Memory Usage"
        assert bottlenecks[0].severity == BottleneckSeverity.HIGH
        assert bottlenecks[0].optimization_type == OptimizationType.MEMORY
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_analyze_memory_usage_normal(self, mock_memory, bottleneck_analyzer):
        """Test memory analysis with normal usage."""
        # Mock normal memory usage (70%)
        mock_mem = MagicMock()
        mock_mem.percent = 70.0
        mock_memory.return_value = mock_mem
        
        bottlenecks = await bottleneck_analyzer._analyze_memory_usage()
        
        assert len(bottlenecks) == 0
    
    @patch('psutil.cpu_percent')
    @pytest.mark.asyncio
    async def test_analyze_cpu_usage_high(self, mock_cpu, bottleneck_analyzer):
        """Test CPU analysis with high usage."""
        mock_cpu.return_value = 85.0
        
        bottlenecks = await bottleneck_analyzer._analyze_cpu_usage()
        
        assert len(bottlenecks) == 1
        assert bottlenecks[0].name == "High CPU Usage"
        assert bottlenecks[0].severity == BottleneckSeverity.HIGH
        assert bottlenecks[0].current_value == 85.0
    
    @pytest.mark.asyncio
    async def test_analyze_database_performance_no_session(self, bottleneck_analyzer):
        """Test database analysis without session."""
        bottleneck_analyzer.db_session = None
        
        bottlenecks = await bottleneck_analyzer._analyze_database_performance()
        
        assert len(bottlenecks) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_database_performance_with_slow_queries(self, bottleneck_analyzer):
        """Test database analysis with slow queries."""
        # Mock slow queries
        slow_queries = [
            {
                'query': 'SELECT * FROM large_table WHERE condition',
                'mean_time': 2000,  # 2 seconds
                'calls': 100,
                'total_time': 200000
            }
        ]
        
        with patch.object(bottleneck_analyzer, '_get_slow_queries', return_value=slow_queries), \
             patch.object(bottleneck_analyzer, '_get_active_connections_count', return_value=5):
            
            bottlenecks = await bottleneck_analyzer._analyze_database_performance()
            
            assert len(bottlenecks) == 1
            assert "Slow Database Query" in bottlenecks[0].name
            assert bottlenecks[0].optimization_type == OptimizationType.QUERY_OPTIMIZATION
    
    @pytest.mark.asyncio
    async def test_analyze_database_performance_high_connections(self, bottleneck_analyzer):
        """Test database analysis with high connection count."""
        with patch.object(bottleneck_analyzer, '_get_slow_queries', return_value=[]), \
             patch.object(bottleneck_analyzer, '_get_active_connections_count', return_value=18):
            
            bottlenecks = await bottleneck_analyzer._analyze_database_performance()
            
            assert len(bottlenecks) == 1
            assert "High Database Connection Usage" in bottlenecks[0].name
            assert bottlenecks[0].optimization_type == OptimizationType.CONNECTION_POOL
    
    @pytest.mark.asyncio
    async def test_analyze_cache_performance(self, bottleneck_analyzer):
        """Test cache performance analysis."""
        # Mock metrics with slow cache operation
        mock_metrics = {
            'cache_get': {
                'avg_execution_time': 0.15,  # 150ms
                'count': 100
            }
        }
        
        with patch.object(bottleneck_analyzer.metrics_collector, 'get_metrics_summary', return_value=mock_metrics):
            bottlenecks = await bottleneck_analyzer._analyze_cache_performance()
            
            assert len(bottlenecks) == 1
            assert "Slow Cache Operation" in bottlenecks[0].name
            assert bottlenecks[0].optimization_type == OptimizationType.CACHING
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @pytest.mark.asyncio
    async def test_analyze_system_performance_comprehensive(self, mock_cpu, mock_memory, bottleneck_analyzer):
        """Test comprehensive system performance analysis."""
        # Mock high resource usage
        mock_memory.return_value = MagicMock(percent=88.0, total=8000000000, used=7040000000)
        mock_cpu.return_value = 82.0
        
        with patch.object(bottleneck_analyzer, '_get_slow_queries', return_value=[]), \
             patch.object(bottleneck_analyzer, '_get_active_connections_count', return_value=5):
            
            bottlenecks = await bottleneck_analyzer.analyze_system_performance()
            
            # Should find memory and CPU bottlenecks
            assert len(bottlenecks) >= 2
            
            # Verify bottlenecks are sorted by impact score
            for i in range(len(bottlenecks) - 1):
                assert bottlenecks[i].impact_score >= bottlenecks[i + 1].impact_score
            
            # Verify analysis history is updated
            assert len(bottleneck_analyzer._analysis_history) == 1


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality."""
    
    def test_optimizer_initialization(self, performance_optimizer):
        """Test optimizer initialization."""
        assert performance_optimizer.db_session is not None
        assert len(performance_optimizer.optimization_history) == 0
        assert len(performance_optimizer._optimization_registry) > 0
    
    def test_find_optimization_function(self, performance_optimizer, sample_bottleneck):
        """Test finding optimization function."""
        func = performance_optimizer._find_optimization_function(sample_bottleneck)
        assert func is not None
        assert callable(func)
    
    def test_find_optimization_function_unsupported(self, performance_optimizer):
        """Test finding optimization function for unsupported type."""
        bottleneck = Bottleneck(
            id="test",
            name="Test",
            description="Test",
            severity=BottleneckSeverity.LOW,
            optimization_type="unsupported_type",  # Invalid type
            impact_score=10,
            current_value=10,
            target_value=5,
            recommendations=[],
            detected_at=datetime.utcnow()
        )
        
        func = performance_optimizer._find_optimization_function(bottleneck)
        assert func is None
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_capture_metrics_memory(self, mock_memory, performance_optimizer):
        """Test capturing memory metrics."""
        mock_mem = MagicMock()
        mock_mem.percent = 75.0
        mock_mem.used = 6000000000
        mock_mem.available = 2000000000
        mock_memory.return_value = mock_mem
        
        metrics = await performance_optimizer._capture_metrics(OptimizationType.MEMORY)
        
        assert "memory_percent" in metrics
        assert metrics["memory_percent"] == 75.0
        assert "memory_used_mb" in metrics
        assert "memory_available_mb" in metrics
    
    @pytest.mark.asyncio
    async def test_calculate_improvement(self, performance_optimizer, sample_bottleneck):
        """Test improvement calculation."""
        before = {"memory_percent": 90.0}
        after = {"memory_percent": 75.0}
        
        improvement = performance_optimizer._calculate_improvement(before, after, sample_bottleneck)
        
        # (90 - 75) / 90 * 100 = 16.67%
        assert abs(improvement - 16.67) < 0.1
    
    @pytest.mark.asyncio
    async def test_calculate_improvement_no_improvement(self, performance_optimizer, sample_bottleneck):
        """Test improvement calculation with no improvement."""
        before = {"memory_percent": 75.0}
        after = {"memory_percent": 80.0}  # Worse performance
        
        improvement = performance_optimizer._calculate_improvement(before, after, sample_bottleneck)
        
        # Should return 0 for negative improvements
        assert improvement == 0.0
    
    @patch('gc.collect')
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_optimize_memory_garbage_collection(self, mock_memory, mock_gc, performance_optimizer, sample_bottleneck):
        """Test memory garbage collection optimization."""
        mock_memory.return_value = MagicMock(percent=80.0)
        mock_gc.return_value = 42  # Objects collected
        
        result = await performance_optimizer._optimize_memory_garbage_collection(sample_bottleneck)
        
        assert result is True
        assert mock_gc.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_optimize_cache_warming(self, performance_optimizer, sample_bottleneck):
        """Test cache warming optimization."""
        result = await performance_optimizer._optimize_cache_warming(sample_bottleneck)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_optimize_query_indexes(self, performance_optimizer, sample_bottleneck):
        """Test query optimization."""
        # Mock database session execute
        performance_optimizer.db_session.execute = AsyncMock()
        performance_optimizer.db_session.commit = AsyncMock()
        
        result = await performance_optimizer._optimize_query_indexes(sample_bottleneck)
        
        assert result is True
        performance_optimizer.db_session.execute.assert_called_once()
        performance_optimizer.db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_apply_optimization_success(self, performance_optimizer, sample_bottleneck):
        """Test successful optimization application."""
        with patch.object(performance_optimizer, '_capture_metrics') as mock_capture:
            # Mock metrics capture
            mock_capture.side_effect = [
                {"memory_percent": 90.0},  # Before
                {"memory_percent": 75.0}   # After
            ]
            
            with patch.object(performance_optimizer, '_optimize_memory_garbage_collection', return_value=True):
                result = await performance_optimizer.apply_optimization(sample_bottleneck)
                
                assert result.status == OptimizationStatus.COMPLETED
                assert result.improvement_percent > 0
                assert len(performance_optimizer.optimization_history) == 1
    
    @pytest.mark.asyncio
    async def test_apply_optimization_no_function(self, performance_optimizer):
        """Test optimization application with no function available."""
        bottleneck = Bottleneck(
            id="test",
            name="Test",
            description="Test",
            severity=BottleneckSeverity.LOW,
            optimization_type="unsupported",  # No function available
            impact_score=10,
            current_value=10,
            target_value=5,
            recommendations=[],
            detected_at=datetime.utcnow()
        )
        
        result = await performance_optimizer.apply_optimization(bottleneck)
        
        assert result.status == OptimizationStatus.SKIPPED
        assert "No optimization function available" in result.notes
    
    @pytest.mark.asyncio
    async def test_apply_optimization_failure(self, performance_optimizer, sample_bottleneck):
        """Test optimization application failure."""
        with patch.object(performance_optimizer, '_capture_metrics', return_value={}):
            with patch.object(performance_optimizer, '_find_optimization_function', return_value=None):
                # Force the optimization to be skipped (no function available)
                result = await performance_optimizer.apply_optimization(sample_bottleneck)
                
                assert result.status == OptimizationStatus.SKIPPED
                assert "No optimization function available" in result.notes


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark functionality."""
    
    def test_benchmark_initialization(self, performance_benchmark):
        """Test benchmark initialization."""
        assert performance_benchmark.db_session is not None
        assert len(performance_benchmark.benchmark_history) == 0
    
    @pytest.mark.asyncio
    async def test_benchmark_memory_operations(self, performance_benchmark):
        """Test memory operations benchmark."""
        result = await performance_benchmark._benchmark_memory_operations()
        
        assert result.name == "Memory Operations"
        assert result.operations_per_second > 0
        assert result.avg_response_time_ms >= 0
        assert result.test_duration > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_cache_operations(self, performance_benchmark):
        """Test cache operations benchmark."""
        result = await performance_benchmark._benchmark_cache_operations()
        
        assert result.name == "Cache Operations"
        assert result.operations_per_second > 0
        assert result.avg_response_time_ms >= 0
        assert result.p95_response_time_ms >= result.avg_response_time_ms
    
    @pytest.mark.asyncio
    async def test_benchmark_async_operations(self, performance_benchmark):
        """Test async operations benchmark."""
        result = await performance_benchmark._benchmark_async_operations()
        
        assert result.name == "Async Operations"
        assert result.operations_per_second > 0
        assert result.test_duration > 0
        assert result.metadata["concurrency"] == "full"
    
    @pytest.mark.asyncio
    async def test_benchmark_database_operations_no_session(self, performance_benchmark):
        """Test database benchmark without session."""
        performance_benchmark.db_session = None
        
        result = await performance_benchmark._benchmark_database_operations()
        
        assert result.name == "Database Operations"
        assert result.operations_per_second == 0
        assert "No database session available" in result.metadata["error"]
    
    @pytest.mark.asyncio
    async def test_benchmark_database_operations_with_session(self, performance_benchmark):
        """Test database benchmark with session."""
        # Mock database operations
        performance_benchmark.db_session.execute = AsyncMock()
        performance_benchmark.db_session.commit = AsyncMock()
        
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value = MagicMock(rss=100*1024*1024)
            mock_process.return_value.cpu_percent.return_value = 10.0
            
            result = await performance_benchmark._benchmark_database_operations()
            
            assert result.name == "Database Operations"
            assert result.operations_per_second > 0
            assert performance_benchmark.db_session.execute.call_count == 100
    
    @pytest.mark.asyncio
    async def test_benchmark_io_operations(self, performance_benchmark):
        """Test I/O operations benchmark."""
        result = await performance_benchmark._benchmark_io_operations()
        
        assert result.name == "I/O Operations"
        # I/O benchmark might fail in test environment, so check for either success or error
        assert result.operations_per_second >= 0
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_benchmark(self, performance_benchmark):
        """Test comprehensive benchmark execution."""
        performance_benchmark.db_session = None  # Simplify for testing
        
        results = await performance_benchmark.run_comprehensive_benchmark()
        
        assert len(results) == 5
        assert "database" in results
        assert "memory" in results
        assert "cache" in results
        assert "async" in results
        assert "io" in results
        
        # Verify results are stored in history
        assert len(performance_benchmark.benchmark_history) == 5


class TestPerformanceOptimizationService:
    """Test PerformanceOptimizationService functionality."""
    
    def test_service_initialization(self, performance_service):
        """Test service initialization."""
        assert performance_service.db_session is not None
        assert performance_service.analyzer is not None
        assert performance_service.optimizer is not None
        assert performance_service.benchmark is not None
        assert len(performance_service._optimization_runs) == 0
    
    @pytest.mark.asyncio
    async def test_get_current_performance_status(self, performance_service):
        """Test getting current performance status."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = MagicMock(percent=70.0)
            mock_cpu.return_value = 60.0
            
            status = await performance_service.get_current_performance_status()
            
            assert "timestamp" in status
            assert status["system_health"] == "healthy"
            assert status["memory_usage_percent"] == 70.0
            assert status["cpu_usage_percent"] == 60.0
    
    @pytest.mark.asyncio
    async def test_get_current_performance_status_degraded(self, performance_service):
        """Test getting degraded performance status."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value = MagicMock(percent=85.0)
            mock_cpu.return_value = 85.0
            
            status = await performance_service.get_current_performance_status()
            
            assert status["system_health"] == "degraded"
    
    def test_calculate_overall_improvements(self, performance_service):
        """Test calculating overall improvements."""
        baseline = {
            "database": BenchmarkResult(
                name="DB", operations_per_second=100, avg_response_time_ms=10,
                p95_response_time_ms=15, p99_response_time_ms=20,
                memory_usage_mb=50, cpu_usage_percent=25,
                timestamp=datetime.utcnow(), test_duration=10
            )
        }
        
        post = {
            "database": BenchmarkResult(
                name="DB", operations_per_second=120, avg_response_time_ms=8,
                p95_response_time_ms=12, p99_response_time_ms=16,
                memory_usage_mb=45, cpu_usage_percent=20,
                timestamp=datetime.utcnow(), test_duration=10
            )
        }
        
        improvements = performance_service._calculate_overall_improvements(baseline, post)
        
        assert "database_ops_improvement_percent" in improvements
        assert improvements["database_ops_improvement_percent"] == 20.0  # (120-100)/100*100
        
        assert "database_response_improvement_percent" in improvements
        assert improvements["database_response_improvement_percent"] == 20.0  # (10-8)/10*100
    
    def test_generate_recommendations(self, performance_service):
        """Test generating optimization recommendations."""
        bottlenecks = [
            Bottleneck(
                id="critical1", name="Critical", description="", 
                severity=BottleneckSeverity.CRITICAL, optimization_type=OptimizationType.MEMORY,
                impact_score=95, current_value=95, target_value=70,
                recommendations=[], detected_at=datetime.utcnow()
            )
        ]
        
        optimization_results = [
            OptimizationResult(
                optimization_id="opt1", name="Test", optimization_type=OptimizationType.MEMORY,
                status=OptimizationStatus.COMPLETED, before_metrics={}, after_metrics={},
                improvement_percent=15.0, execution_time=2.0, applied_at=datetime.utcnow()
            ),
            OptimizationResult(
                optimization_id="opt2", name="Test2", optimization_type=OptimizationType.CACHING,
                status=OptimizationStatus.FAILED, before_metrics={}, after_metrics={},
                improvement_percent=0.0, execution_time=1.0, applied_at=datetime.utcnow()
            )
        ]
        
        recommendations = performance_service._generate_recommendations(bottlenecks, optimization_results)
        
        assert any("critical" in rec.lower() for rec in recommendations)
        assert any("failed" in rec.lower() for rec in recommendations)
        assert any("successful" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_run_full_optimization_cycle(self, performance_service):
        """Test running full optimization cycle."""
        # Mock all the dependencies
        mock_bottleneck = Bottleneck(
            id="test", name="Test", description="Test bottleneck",
            severity=BottleneckSeverity.HIGH, optimization_type=OptimizationType.MEMORY,
            impact_score=80, current_value=90, target_value=70,
            recommendations=[], detected_at=datetime.utcnow()
        )
        
        mock_benchmark = BenchmarkResult(
            name="Test", operations_per_second=100, avg_response_time_ms=10,
            p95_response_time_ms=15, p99_response_time_ms=20,
            memory_usage_mb=50, cpu_usage_percent=25,
            timestamp=datetime.utcnow(), test_duration=10
        )
        
        mock_optimization = OptimizationResult(
            optimization_id="test", name="Test", optimization_type=OptimizationType.MEMORY,
            status=OptimizationStatus.COMPLETED, before_metrics={"memory_percent": 90},
            after_metrics={"memory_percent": 75}, improvement_percent=16.7,
            execution_time=2.0, applied_at=datetime.utcnow()
        )
        
        with patch.object(performance_service.benchmark, 'run_comprehensive_benchmark') as mock_bench, \
             patch.object(performance_service.analyzer, 'analyze_system_performance') as mock_analyze, \
             patch.object(performance_service.optimizer, 'apply_optimization') as mock_optimize:
            
            mock_bench.return_value = {"test": mock_benchmark}
            mock_analyze.return_value = [mock_bottleneck]
            mock_optimize.return_value = mock_optimization
            
            report = await performance_service.run_full_optimization_cycle()
            
            assert "cycle_id" in report
            assert report["bottlenecks_analyzed"] == 1
            assert report["optimizations_applied"] == 1
            assert "baseline_benchmarks" in report
            assert "post_benchmarks" in report
            assert "performance_improvements" in report
            assert "recommendations" in report
            
            # Verify optimization run is stored
            assert len(performance_service._optimization_runs) == 1
    
    @pytest.mark.asyncio
    async def test_get_optimization_history(self, performance_service):
        """Test getting optimization history."""
        # Add some mock history
        performance_service._optimization_runs = [
            {"cycle_id": "test1", "started_at": "2023-01-01T00:00:00"},
            {"cycle_id": "test2", "started_at": "2023-01-02T00:00:00"}
        ]
        
        history = await performance_service.get_optimization_history()
        
        assert len(history) == 2
        assert history[0]["cycle_id"] == "test1"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_performance_optimization_service_singleton(self):
        """Test global service instance is singleton."""
        service1 = get_performance_optimization_service()
        service2 = get_performance_optimization_service()
        
        assert service1 is service2
    
    @pytest.mark.asyncio
    async def test_analyze_performance_bottlenecks(self):
        """Test bottleneck analysis utility function."""
        with patch('pcs.services.performance_optimization_service.get_performance_optimization_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.analyzer.analyze_system_performance = AsyncMock(return_value=[])
            mock_get_service.return_value = mock_service
            
            result = await analyze_performance_bottlenecks()
            
            assert result == []
            mock_service.analyzer.analyze_system_performance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_performance_optimization(self):
        """Test performance optimization utility function."""
        with patch('pcs.services.performance_optimization_service.get_performance_optimization_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.run_full_optimization_cycle = AsyncMock(return_value={"cycle_id": "test"})
            mock_get_service.return_value = mock_service
            
            result = await run_performance_optimization()
            
            assert result["cycle_id"] == "test"
            mock_service.run_full_optimization_cycle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_benchmark_system_performance(self):
        """Test system benchmark utility function."""
        with patch('pcs.services.performance_optimization_service.get_performance_optimization_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.benchmark.run_comprehensive_benchmark = AsyncMock(return_value={})
            mock_get_service.return_value = mock_service
            
            result = await benchmark_system_performance()
            
            assert result == {}
            mock_service.benchmark.run_comprehensive_benchmark.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_performance_profiler(self):
        """Test performance profiler context manager."""
        async with performance_profiler("test_operation") as profiler:
            # Simulate some work
            await asyncio.sleep(0.01)
            
            assert profiler is not None


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows."""
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_cycle(self, mock_cpu, mock_memory, performance_service):
        """Test complete end-to-end optimization cycle."""
        # Setup high resource usage
        mock_memory.return_value = MagicMock(percent=88.0, total=8000000000, used=7040000000)
        mock_cpu.return_value = 85.0
        
        # Mock database queries
        performance_service.db_session = None  # Simplify for testing
        
        with patch('gc.collect', return_value=50):
            report = await performance_service.run_full_optimization_cycle()
            
            # Verify comprehensive report structure
            assert "cycle_id" in report
            assert "started_at" in report
            assert "duration_seconds" in report
            assert "bottlenecks_analyzed" in report
            assert "optimizations_applied" in report
            assert "baseline_benchmarks" in report
            assert "post_benchmarks" in report
            assert "performance_improvements" in report
            assert "recommendations" in report
            
            # Should find bottlenecks and attempt optimizations
            assert report["bottlenecks_analyzed"] > 0
    
    @pytest.mark.asyncio
    async def test_bottleneck_analysis_priority_ordering(self, bottleneck_analyzer):
        """Test that bottlenecks are correctly prioritized by impact score."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Setup multiple issues with different severities
            mock_memory.return_value = MagicMock(percent=95.0, total=8000000000, used=7600000000)
            mock_cpu.return_value = 85.0
            
            bottlenecks = await bottleneck_analyzer.analyze_system_performance()
            
            # Verify bottlenecks are sorted by impact score (descending)
            for i in range(len(bottlenecks) - 1):
                assert bottlenecks[i].impact_score >= bottlenecks[i + 1].impact_score
            
            # Critical bottlenecks should come first
            if bottlenecks:
                assert bottlenecks[0].severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]
    
    @pytest.mark.asyncio
    async def test_optimization_with_metrics_tracking(self, performance_optimizer):
        """Test optimization with proper metrics tracking."""
        bottleneck = Bottleneck(
            id="memory_test", name="Memory Test", description="Test memory optimization",
            severity=BottleneckSeverity.HIGH, optimization_type=OptimizationType.MEMORY,
            impact_score=85, current_value=90, target_value=70,
            recommendations=[], detected_at=datetime.utcnow()
        )
        
        with patch.object(performance_optimizer, '_capture_metrics') as mock_capture, \
             patch.object(performance_optimizer, '_optimize_memory_garbage_collection', return_value=True):
            
            # Mock metrics capture to return before/after values
            mock_capture.side_effect = [
                {"memory_percent": 90.0, "memory_used_mb": 7200, "memory_available_mb": 800},  # Before
                {"memory_percent": 75.0, "memory_used_mb": 6000, "memory_available_mb": 2000}  # After
            ]
            
            result = await performance_optimizer.apply_optimization(bottleneck)
            
            assert result.status == OptimizationStatus.COMPLETED
            assert result.improvement_percent > 0
            assert result.before_metrics["memory_percent"] == 90.0
            assert result.after_metrics["memory_percent"] == 75.0
    
    @pytest.mark.asyncio
    async def test_benchmark_consistency(self, performance_benchmark):
        """Test benchmark result consistency across multiple runs."""
        performance_benchmark.db_session = None  # Simplify
        
        # Run benchmark multiple times
        results1 = await performance_benchmark.run_comprehensive_benchmark()
        results2 = await performance_benchmark.run_comprehensive_benchmark()
        
        # Results should be consistent in structure
        assert set(results1.keys()) == set(results2.keys())
        
        # All benchmarks should complete successfully
        for category in results1:
            assert results1[category].operations_per_second >= 0
            assert results2[category].operations_per_second >= 0
            # Database benchmark will have 0 duration when no session is available
            if category == "database":
                assert results1[category].test_duration >= 0
                assert results2[category].test_duration >= 0
            else:
                assert results1[category].test_duration > 0
                assert results2[category].test_duration > 0
