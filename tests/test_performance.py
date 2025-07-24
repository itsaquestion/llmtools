"""
Performance Tests for Recovery Functionality

This module contains performance tests to verify that the recovery functionality
performs well under various load conditions and scales appropriately.
"""

import pytest
import time
import threading
import statistics
from unittest.mock import Mock
from typing import List, Dict, Any
import concurrent.futures

from src.llmtools.parallel_llm_processor import ParallelLLMProcessor
from src.llmtools.recovery_processor import RecoveryProcessor
from src.llmtools.database_updater import DatabaseUpdater

from tests.test_database_generator import (
    TestDatabaseGenerator,
    create_mixed_failure_database,
    create_mostly_failed_database
)


class TestPerformance:
    """Performance test suite for recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock chat function with configurable delay
        self.processing_delay = 0.01  # Default 10ms delay
        self.call_times = []
        
        def timed_mock_response(prompt: str) -> str:
            start_time = time.time()
            time.sleep(self.processing_delay)
            end_time = time.time()
            self.call_times.append(end_time - start_time)
            return f"Response for: {prompt[:30]}..."
        
        self.mock_chat_fn = Mock(side_effect=timed_mock_response)
        self.mock_chat_fn.__name__ = "timed_mock_chat_fn"
        
        # List to track created databases for cleanup
        self.test_databases = []
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Clean up test databases
        for db_path in self.test_databases:
            TestDatabaseGenerator.cleanup_database(db_path)
    
    def _track_database(self, db_path: str) -> str:
        """Track a database for cleanup and return the path."""
        self.test_databases.append(db_path)
        return db_path
    
    def _create_processor(self, num_workers: int = 4, **kwargs) -> ParallelLLMProcessor:
        """Create a processor with test configuration."""
        return ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=num_workers,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=10.0,
            **kwargs
        )
    
    # Scalability Tests
    
    def test_recovery_performance_small_database(self):
        """Test recovery performance with small database (10 records)."""
        # Create small test database
        db_path, records = create_mixed_failure_database(10)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=2)
        
        try:
            # Measure recovery time
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 10
            
            # Performance assertions
            assert recovery_time < 5.0  # Should complete within 5 seconds
            
            # Log performance metrics
            print(f"Small database recovery time: {recovery_time:.3f}s")
            
        finally:
            processor.close()
    
    def test_recovery_performance_medium_database(self):
        """Test recovery performance with medium database (100 records)."""
        # Create medium test database
        db_path, records = create_mixed_failure_database(100)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=4)
        
        try:
            # Measure recovery time
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 100
            
            # Performance assertions
            assert recovery_time < 15.0  # Should complete within 15 seconds
            
            # Log performance metrics
            print(f"Medium database recovery time: {recovery_time:.3f}s")
            
        finally:
            processor.close()
    
    def test_recovery_performance_large_database(self):
        """Test recovery performance with large database (500 records)."""
        # Create large test database
        db_path, records = create_mixed_failure_database(500)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=8)
        
        try:
            # Measure recovery time
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 500
            
            # Performance assertions
            assert recovery_time < 60.0  # Should complete within 1 minute
            
            # Log performance metrics
            print(f"Large database recovery time: {recovery_time:.3f}s")
            
        finally:
            processor.close()
    
    # Worker Scaling Tests
    
    def test_worker_scaling_performance(self):
        """Test how performance scales with different numbers of workers."""
        # Create test database
        db_path, records = create_mostly_failed_database(50)
        self._track_database(db_path)
        
        worker_counts = [1, 2, 4, 8]
        performance_results = {}
        
        for num_workers in worker_counts:
            processor = self._create_processor(num_workers=num_workers)
            
            try:
                # Reset call times for this test
                self.call_times = []
                
                # Measure recovery time
                start_time = time.time()
                results = processor.recover_from_database(db_path)
                recovery_time = time.time() - start_time
                
                # Store results
                performance_results[num_workers] = {
                    'recovery_time': recovery_time,
                    'results_count': len(results),
                    'avg_call_time': statistics.mean(self.call_times) if self.call_times else 0
                }
                
                # Verify results
                assert len(results) == 50
                
                print(f"Workers: {num_workers}, Time: {recovery_time:.3f}s")
                
            finally:
                processor.close()
        
        # Verify scaling behavior
        # More workers should generally be faster (up to a point)
        assert performance_results[2]['recovery_time'] <= performance_results[1]['recovery_time'] * 1.2
        assert performance_results[4]['recovery_time'] <= performance_results[2]['recovery_time'] * 1.2
    
    # Memory Usage Tests
    
    def test_memory_usage_large_database(self):
        """Test memory usage with large database."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large test database
        db_path, records = create_mixed_failure_database(1000)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=4)
        
        try:
            # Perform recovery
            results = processor.recover_from_database(db_path)
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Verify results
            assert len(results) == 1000
            
            # Memory usage should be reasonable (less than 100MB increase)
            assert memory_increase < 100.0
            
            print(f"Memory increase: {memory_increase:.2f}MB")
            
        finally:
            processor.close()
    
    # Database Operation Performance Tests
    
    def test_database_read_performance(self):
        """Test database read operation performance."""
        from src.llmtools.recovery_analyzer import RecoveryAnalyzer
        
        # Create large test database
        db_path, records = create_mixed_failure_database(1000)
        self._track_database(db_path)
        
        # Measure database read time
        read_times = []
        for _ in range(5):  # Multiple runs for average
            start_time = time.time()
            failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_path)
            read_time = time.time() - start_time
            read_times.append(read_time)
        
        avg_read_time = statistics.mean(read_times)
        
        # Performance assertions
        assert avg_read_time < 1.0  # Should read within 1 second
        assert len(existing_results) == 1000
        
        print(f"Average database read time: {avg_read_time:.3f}s")
    
    def test_database_write_performance(self):
        """Test database write operation performance."""
        # Create test database
        db_path, records = create_mostly_failed_database(500)
        self._track_database(db_path)
        
        # Create mock results for all records
        mock_results = {i: f"Updated result {i}" for i in range(1, 501)}
        
        # Measure database write time
        write_times = []
        for _ in range(3):  # Multiple runs for average
            start_time = time.time()
            failed_updates = DatabaseUpdater.update_results(db_path, mock_results)
            write_time = time.time() - start_time
            write_times.append(write_time)
        
        avg_write_time = statistics.mean(write_times)
        
        # Performance assertions
        assert avg_write_time < 5.0  # Should write within 5 seconds
        assert len(failed_updates) == 0  # All updates should succeed
        
        print(f"Average database write time: {avg_write_time:.3f}s")
    
    # Concurrent Access Tests
    
    def test_concurrent_recovery_performance(self):
        """Test performance with concurrent recovery operations."""
        # Create multiple test databases
        databases = []
        for i in range(3):
            db_path, records = create_mixed_failure_database(30)
            self._track_database(db_path)
            databases.append(db_path)
        
        # Create processors for concurrent operations
        processors = [self._create_processor(num_workers=2) for _ in range(3)]
        
        try:
            # Measure concurrent recovery time
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(processor.recover_from_database, db_path)
                    for processor, db_path in zip(processors, databases)
                ]
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            concurrent_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 3
            assert all(len(result) == 30 for result in results)
            
            # Performance assertion
            assert concurrent_time < 30.0  # Should complete within 30 seconds
            
            print(f"Concurrent recovery time: {concurrent_time:.3f}s")
            
        finally:
            for processor in processors:
                processor.close()
    
    # Stress Tests
    
    def test_stress_test_many_small_recoveries(self):
        """Stress test with many small recovery operations."""
        # Create many small databases
        databases = []
        for i in range(10):
            db_path, records = create_mixed_failure_database(10)
            self._track_database(db_path)
            databases.append(db_path)
        
        processor = self._create_processor(num_workers=4)
        
        try:
            # Measure total time for all recoveries
            start_time = time.time()
            
            for db_path in databases:
                results = processor.recover_from_database(db_path)
                assert len(results) == 10
            
            total_time = time.time() - start_time
            
            # Performance assertion
            assert total_time < 30.0  # Should complete within 30 seconds
            
            print(f"Many small recoveries total time: {total_time:.3f}s")
            
        finally:
            processor.close()
    
    def test_stress_test_rapid_successive_recoveries(self):
        """Stress test with rapid successive recovery operations on same database."""
        # Create test database
        db_path, records = create_mixed_failure_database(20)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=2)
        
        try:
            # Perform multiple rapid recoveries
            recovery_times = []
            
            for i in range(5):
                start_time = time.time()
                results = processor.recover_from_database(db_path)
                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)
                
                assert len(results) == 20
            
            avg_recovery_time = statistics.mean(recovery_times)
            
            # Performance assertion
            assert avg_recovery_time < 10.0  # Each recovery should be fast
            
            print(f"Average rapid recovery time: {avg_recovery_time:.3f}s")
            
        finally:
            processor.close()
    
    # Resource Utilization Tests
    
    def test_cpu_utilization_during_recovery(self):
        """Test CPU utilization during recovery operations."""
        import psutil
        import threading
        
        # Create test database
        db_path, records = create_mostly_failed_database(100)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=4)
        
        # Monitor CPU usage
        cpu_usage_samples = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_usage_samples.append(psutil.cpu_percent(interval=0.1))
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        
        try:
            # Start monitoring
            monitor_thread.start()
            
            # Perform recovery
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Stop monitoring
            monitoring = False
            monitor_thread.join()
            
            # Analyze CPU usage
            if cpu_usage_samples:
                avg_cpu = statistics.mean(cpu_usage_samples)
                max_cpu = max(cpu_usage_samples)
                
                print(f"Recovery time: {recovery_time:.3f}s")
                print(f"Average CPU usage: {avg_cpu:.1f}%")
                print(f"Peak CPU usage: {max_cpu:.1f}%")
                
                # CPU usage should be reasonable
                assert avg_cpu < 90.0  # Should not max out CPU
            
            # Verify results
            assert len(results) == 100
            
        finally:
            monitoring = False
            if monitor_thread.is_alive():
                monitor_thread.join()
            processor.close()
    
    # Throughput Tests
    
    def test_recovery_throughput(self):
        """Test recovery throughput (records processed per second)."""
        # Create test database with known incomplete records
        db_path, records = create_mostly_failed_database(200)
        self._track_database(db_path)
        
        # Get initial incomplete count
        initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        incomplete_count = initial_stats['incomplete_records']
        
        processor = self._create_processor(num_workers=6)
        
        try:
            # Measure recovery time
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Calculate throughput
            throughput = incomplete_count / recovery_time if recovery_time > 0 else 0
            
            # Verify results
            assert len(results) == 200
            
            # Throughput should be reasonable
            assert throughput > 5.0  # At least 5 records per second
            
            print(f"Recovery throughput: {throughput:.2f} records/second")
            print(f"Processed {incomplete_count} incomplete records in {recovery_time:.3f}s")
            
        finally:
            processor.close()


class TestPerformanceRegression:
    """Performance regression tests to ensure performance doesn't degrade."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chat_fn = Mock(return_value="Test response")
        self.test_databases = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for db_path in self.test_databases:
            TestDatabaseGenerator.cleanup_database(db_path)
    
    def _track_database(self, db_path: str) -> str:
        """Track a database for cleanup."""
        self.test_databases.append(db_path)
        return db_path
    
    def test_baseline_recovery_performance(self):
        """Establish baseline performance metrics for regression testing."""
        # Create standardized test database
        db_path, records = create_mixed_failure_database(100)
        self._track_database(db_path)
        
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=4,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=10.0
        )
        
        try:
            # Measure baseline performance
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            baseline_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 100
            
            # Record baseline (this would be compared against in CI/CD)
            print(f"Baseline recovery time for 100 records: {baseline_time:.3f}s")
            
            # Baseline should be reasonable
            assert baseline_time < 20.0  # Should complete within 20 seconds
            
        finally:
            processor.close()
    
    def test_memory_usage_regression(self):
        """Test for memory usage regression."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create test database
        db_path, records = create_mixed_failure_database(200)
        self._track_database(db_path)
        
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=4,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=10.0
        )
        
        try:
            # Perform recovery
            results = processor.recover_from_database(db_path)
            
            # Check memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Verify results
            assert len(results) == 200
            
            # Memory usage should not exceed baseline
            assert memory_increase < 50.0  # Should not use more than 50MB
            
            print(f"Memory usage for 200 records: {memory_increase:.2f}MB")
            
        finally:
            processor.close()