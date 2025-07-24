"""
Performance Optimization Tests for Recovery Functionality

This module contains comprehensive performance tests to verify optimizations
and ensure the system performs well under various load conditions.
"""

import pytest
import time
import threading
import statistics
import psutil
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import concurrent.futures
import sqlite3

from src.llmtools.parallel_llm_processor import ParallelLLMProcessor
from src.llmtools.recovery_processor import RecoveryProcessor
from src.llmtools.database_updater import DatabaseUpdater
from src.llmtools.recovery_analyzer import RecoveryAnalyzer

from tests.test_database_generator import (
    TestDatabaseGenerator,
    create_mixed_failure_database,
    create_mostly_failed_database
)


class TestPerformanceOptimizations:
    """Test suite for performance optimizations."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create fast mock chat function
        self.processing_delay = 0.001  # Very fast for performance testing
        self.call_times = []
        
        def fast_mock_response(prompt: str) -> str:
            start_time = time.time()
            time.sleep(self.processing_delay)
            end_time = time.time()
            self.call_times.append(end_time - start_time)
            return f"Fast response for: {prompt[:20]}..."
        
        self.mock_chat_fn = Mock(side_effect=fast_mock_response)
        self.mock_chat_fn.__name__ = "fast_mock_chat_fn"
        
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
    
    def _create_processor(self, num_workers: int = 8, **kwargs) -> ParallelLLMProcessor:
        """Create a processor with optimized configuration."""
        return ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=num_workers,
            retry_attempts=1,
            retry_delay=0.01,
            timeout=5.0,
            **kwargs
        )
    
    # Large Dataset Performance Tests
    
    def test_large_database_analysis_performance(self):
        """Test database analysis performance with large datasets."""
        # Create large test database
        db_path, records = create_mixed_failure_database(5000)
        self._track_database(db_path)
        
        # Measure analysis time
        start_time = time.time()
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_path)
        analysis_time = time.time() - start_time
        
        # Verify results
        assert len(existing_results) == 5000
        assert len(failed_records) > 0
        
        # Performance assertions
        assert analysis_time < 10.0  # Should analyze within 10 seconds
        
        # Calculate throughput
        throughput = len(existing_results) / analysis_time
        assert throughput > 500  # At least 500 records per second
        
        print(f"Large database analysis: {analysis_time:.3f}s, {throughput:.0f} records/sec")
    
    def test_large_database_update_performance(self):
        """Test database update performance with large datasets."""
        # Create large test database
        db_path, records = create_mostly_failed_database(3000)
        self._track_database(db_path)
        
        # Get failed records
        failed_records, _ = RecoveryAnalyzer.analyze_database(db_path)
        
        # Create mock results for all failed records
        mock_results = {record_id: f"Optimized result {record_id}" 
                       for record_id, _ in failed_records}
        
        # Measure update time
        start_time = time.time()
        failed_updates = DatabaseUpdater.update_results(db_path, mock_results)
        update_time = time.time() - start_time
        
        # Verify results
        assert len(failed_updates) == 0  # All updates should succeed
        
        # Performance assertions
        assert update_time < 15.0  # Should update within 15 seconds
        
        # Calculate throughput
        throughput = len(mock_results) / update_time
        assert throughput > 200  # At least 200 updates per second
        
        print(f"Large database update: {update_time:.3f}s, {throughput:.0f} updates/sec")
    
    def test_large_database_recovery_performance(self):
        """Test complete recovery performance with large datasets."""
        # Create large test database
        db_path, records = create_mixed_failure_database(2000)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=12)
        
        try:
            # Measure complete recovery time
            start_time = time.time()
            results = processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 2000
            
            # Performance assertions
            assert recovery_time < 60.0  # Should complete within 1 minute
            
            # Calculate throughput
            throughput = len(results) / recovery_time
            assert throughput > 30  # At least 30 records per second
            
            print(f"Large database recovery: {recovery_time:.3f}s, {throughput:.0f} records/sec")
            
        finally:
            processor.close()
    
    # Memory Usage Optimization Tests
    
    def test_memory_usage_optimization_large_dataset(self):
        """Test memory usage optimization with large datasets."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create very large test database
        db_path, records = create_mixed_failure_database(10000)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=8)
        
        try:
            # Monitor memory during recovery
            memory_samples = []
            
            def monitor_memory():
                for _ in range(100):  # Sample for ~10 seconds
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=monitor_memory)
            monitor_thread.start()
            
            # Perform recovery
            results = processor.recover_from_database(db_path)
            
            monitor_thread.join()
            
            # Analyze memory usage
            if memory_samples:
                peak_memory = max(memory_samples)
                avg_memory = statistics.mean(memory_samples)
                memory_increase = peak_memory - initial_memory
                
                # Verify results
                assert len(results) == 10000
                
                # Memory usage should be reasonable even for large datasets
                assert memory_increase < 200.0  # Less than 200MB increase
                
                print(f"Memory usage - Peak: {peak_memory:.1f}MB, Avg: {avg_memory:.1f}MB, Increase: {memory_increase:.1f}MB")
            
        finally:
            processor.close()
    
    def test_streaming_analysis_memory_efficiency(self):
        """Test that streaming analysis uses less memory than standard analysis."""
        # Create large database that should trigger streaming
        db_path, records = create_mixed_failure_database(15000)
        self._track_database(db_path)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Perform analysis (should use streaming for large dataset)
        start_time = time.time()
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_path)
        analysis_time = time.time() - start_time
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # Verify results
        assert len(existing_results) == 15000
        
        # Memory increase should be reasonable even for very large dataset
        assert memory_increase < 100.0  # Less than 100MB increase
        
        # Performance should still be good
        assert analysis_time < 30.0  # Should complete within 30 seconds
        
        print(f"Streaming analysis - Time: {analysis_time:.3f}s, Memory increase: {memory_increase:.1f}MB")
    
    # Concurrency Optimization Tests
    
    def test_optimal_worker_count_performance(self):
        """Test performance with different worker counts to find optimal configuration."""
        # Create test database
        db_path, records = create_mostly_failed_database(200)
        self._track_database(db_path)
        
        worker_counts = [1, 2, 4, 8, 16, 32]
        performance_results = {}
        
        for num_workers in worker_counts:
            processor = self._create_processor(num_workers=num_workers)
            
            try:
                # Reset call times
                self.call_times = []
                
                # Measure recovery time
                start_time = time.time()
                results = processor.recover_from_database(db_path)
                recovery_time = time.time() - start_time
                
                # Calculate metrics
                throughput = len(results) / recovery_time if recovery_time > 0 else 0
                avg_call_time = statistics.mean(self.call_times) if self.call_times else 0
                
                performance_results[num_workers] = {
                    'recovery_time': recovery_time,
                    'throughput': throughput,
                    'avg_call_time': avg_call_time
                }
                
                # Verify results
                assert len(results) == 200
                
                print(f"Workers: {num_workers:2d}, Time: {recovery_time:.3f}s, Throughput: {throughput:.1f} rec/sec")
                
            finally:
                processor.close()
        
        # Find optimal worker count (highest throughput)
        optimal_workers = max(performance_results.keys(), 
                            key=lambda w: performance_results[w]['throughput'])
        
        print(f"Optimal worker count: {optimal_workers}")
        
        # Verify scaling behavior
        assert performance_results[2]['throughput'] > performance_results[1]['throughput'] * 0.8
        assert performance_results[4]['throughput'] > performance_results[2]['throughput'] * 0.8
    
    def test_concurrent_database_access_performance(self):
        """Test performance with concurrent database access."""
        # Create multiple test databases
        databases = []
        for i in range(4):
            db_path, records = create_mixed_failure_database(100)
            self._track_database(db_path)
            databases.append(db_path)
        
        # Create processors for concurrent operations
        processors = [self._create_processor(num_workers=4) for _ in range(4)]
        
        try:
            # Measure concurrent recovery time
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(processor.recover_from_database, db_path)
                    for processor, db_path in zip(processors, databases)
                ]
                
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            concurrent_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 4
            assert all(len(result) == 100 for result in results)
            
            # Performance assertion
            assert concurrent_time < 45.0  # Should complete within 45 seconds
            
            # Calculate total throughput
            total_records = sum(len(result) for result in results)
            throughput = total_records / concurrent_time
            
            print(f"Concurrent access - Time: {concurrent_time:.3f}s, Throughput: {throughput:.1f} rec/sec")
            
        finally:
            for processor in processors:
                processor.close()
    
    # Database Optimization Tests
    
    def test_database_pragma_optimizations(self):
        """Test that database PRAGMA optimizations improve performance."""
        # Create test database
        db_path, records = create_mixed_failure_database(1000)
        self._track_database(db_path)
        
        # Test with standard settings
        start_time = time.time()
        failed_records_std, _ = self._analyze_database_standard(db_path)
        standard_time = time.time() - start_time
        
        # Test with optimized settings
        start_time = time.time()
        failed_records_opt, _ = RecoveryAnalyzer.analyze_database(db_path)
        optimized_time = time.time() - start_time
        
        # Verify results are the same
        assert len(failed_records_std) == len(failed_records_opt)
        
        # Optimized version should be faster or at least not significantly slower
        assert optimized_time <= standard_time * 1.2  # Allow 20% tolerance
        
        print(f"Database optimization - Standard: {standard_time:.3f}s, Optimized: {optimized_time:.3f}s")
    
    def _analyze_database_standard(self, db_file_path: str):
        """Analyze database without optimizations for comparison."""
        failed_records = []
        existing_results = []
        
        with sqlite3.connect(db_file_path, timeout=30.0) as conn:
            # No PRAGMA optimizations
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, prompt, result FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            for record_id, prompt, result in records:
                if RecoveryAnalyzer.is_result_incomplete(result):
                    failed_records.append((record_id, prompt))
                    existing_results.append(None)
                else:
                    existing_results.append(result)
        
        return failed_records, existing_results
    
    # Batch Processing Optimization Tests
    
    def test_batch_size_optimization(self):
        """Test different batch sizes to find optimal configuration."""
        # Create large test database
        db_path, records = create_mostly_failed_database(2000)
        self._track_database(db_path)
        
        # Get failed records
        failed_records, _ = RecoveryAnalyzer.analyze_database(db_path)
        mock_results = {record_id: f"Batch result {record_id}" 
                       for record_id, _ in failed_records}
        
        # Test different batch sizes
        batch_sizes = [100, 250, 500, 1000, 2000]
        batch_performance = {}
        
        for batch_size in batch_sizes:
            # Temporarily modify batch size
            original_batch_size = DatabaseUpdater.BATCH_SIZE
            DatabaseUpdater.BATCH_SIZE = batch_size
            
            try:
                # Measure update time
                start_time = time.time()
                failed_updates = DatabaseUpdater.update_results(db_path, mock_results)
                update_time = time.time() - start_time
                
                batch_performance[batch_size] = {
                    'update_time': update_time,
                    'failed_updates': len(failed_updates),
                    'throughput': len(mock_results) / update_time if update_time > 0 else 0
                }
                
                # Verify all updates succeeded
                assert len(failed_updates) == 0
                
                print(f"Batch size: {batch_size:4d}, Time: {update_time:.3f}s, Throughput: {batch_performance[batch_size]['throughput']:.1f} updates/sec")
                
            finally:
                # Restore original batch size
                DatabaseUpdater.BATCH_SIZE = original_batch_size
        
        # Find optimal batch size (highest throughput)
        optimal_batch_size = max(batch_performance.keys(), 
                               key=lambda b: batch_performance[b]['throughput'])
        
        print(f"Optimal batch size: {optimal_batch_size}")
        
        # Verify that larger batch sizes generally perform better (up to a point)
        assert batch_performance[500]['throughput'] >= batch_performance[100]['throughput'] * 0.8
    
    # End-to-End Performance Tests
    
    def test_end_to_end_performance_benchmark(self):
        """Comprehensive end-to-end performance benchmark."""
        # Create comprehensive test database
        db_path, records = create_mixed_failure_database(1500)
        self._track_database(db_path)
        
        processor = self._create_processor(num_workers=8)
        
        try:
            # Get initial statistics
            initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
            
            # Measure complete recovery with detailed timing
            overall_start = time.time()
            
            # Phase 1: Database validation and analysis
            validation_start = time.time()
            results = processor.recover_from_database(db_path)
            overall_end = time.time()
            
            # Calculate metrics
            total_time = overall_end - overall_start
            throughput = len(results) / total_time
            
            # Get final statistics
            final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
            
            # Verify results
            assert len(results) == 1500
            assert final_stats['incomplete_records'] < initial_stats['incomplete_records']
            
            # Performance benchmarks
            assert total_time < 90.0  # Should complete within 90 seconds
            assert throughput > 15.0  # At least 15 records per second
            
            # Calculate improvement metrics
            improvement = initial_stats['incomplete_records'] - final_stats['incomplete_records']
            improvement_rate = improvement / total_time if total_time > 0 else 0
            
            print(f"End-to-end benchmark:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} records/sec")
            print(f"  Records improved: {improvement}")
            print(f"  Improvement rate: {improvement_rate:.1f} improvements/sec")
            print(f"  Final completion rate: {final_stats['completion_rate']:.1f}%")
            
        finally:
            processor.close()


class TestPerformanceRegression:
    """Performance regression tests to ensure optimizations don't break functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chat_fn = Mock(return_value="Regression test response")
        self.test_databases = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for db_path in self.test_databases:
            TestDatabaseGenerator.cleanup_database(db_path)
    
    def _track_database(self, db_path: str) -> str:
        """Track a database for cleanup."""
        self.test_databases.append(db_path)
        return db_path
    
    def test_optimization_correctness_verification(self):
        """Verify that optimizations don't affect correctness of results."""
        # Create test database
        db_path, records = create_mixed_failure_database(500)
        self._track_database(db_path)
        
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=4,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=10.0
        )
        
        try:
            # Get initial state
            initial_failed, initial_results = RecoveryAnalyzer.analyze_database(db_path)
            initial_incomplete_count = len(initial_failed)
            
            # Perform recovery
            results = processor.recover_from_database(db_path)
            
            # Verify correctness
            assert len(results) == len(records)
            assert len(results) == len(initial_results)
            
            # Verify improvement
            final_failed, final_results = RecoveryAnalyzer.analyze_database(db_path)
            assert len(final_failed) <= len(initial_failed)
            
            # Verify all results are present
            assert all(result is not None for result in results)
            
        finally:
            processor.close()
    
    def test_large_dataset_stability(self):
        """Test stability with large datasets to ensure no memory leaks or crashes."""
        # Create large database
        db_path, records = create_mixed_failure_database(3000)
        self._track_database(db_path)
        
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=8,
            retry_attempts=1,
            retry_delay=0.05,
            timeout=10.0
        )
        
        try:
            # Monitor system resources
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Perform recovery
            results = processor.recover_from_database(db_path)
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Verify results
            assert len(results) == 3000
            
            # Verify memory usage is reasonable
            assert memory_increase < 300.0  # Less than 300MB increase
            
            print(f"Large dataset stability - Memory increase: {memory_increase:.1f}MB")
            
        finally:
            processor.close()
    
    def test_concurrent_access_stability(self):
        """Test stability under concurrent access conditions."""
        # Create multiple databases
        databases = []
        for i in range(6):
            db_path, records = create_mixed_failure_database(150)
            self._track_database(db_path)
            databases.append(db_path)
        
        # Create processors
        processors = [
            ParallelLLMProcessor(
                chat_fn=self.mock_chat_fn,
                num_workers=3,
                retry_attempts=1,
                retry_delay=0.05,
                timeout=10.0
            ) for _ in range(6)
        ]
        
        try:
            # Perform concurrent recoveries
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(processor.recover_from_database, db_path)
                    for processor, db_path in zip(processors, databases)
                ]
                
                results = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # Verify all recoveries completed successfully
            assert len(results) == 6
            assert all(len(result) == 150 for result in results)
            
        finally:
            for processor in processors:
                processor.close()


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])