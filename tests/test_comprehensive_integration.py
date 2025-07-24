"""
Comprehensive Integration Tests for Recovery Functionality

This module contains end-to-end integration tests that verify the complete
recovery workflow, including database validation, analysis, reprocessing,
and result updating.
"""

import pytest
import os
import time
import logging
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from concurrent.futures import TimeoutError as FutureTimeoutError

from src.llmtools.parallel_llm_processor import ParallelLLMProcessor
from src.llmtools.database_validator import DatabaseValidator
from src.llmtools.recovery_analyzer import RecoveryAnalyzer
from src.llmtools.recovery_processor import RecoveryProcessor
from src.llmtools.database_updater import DatabaseUpdater

from tests.test_database_generator import (
    TestDatabaseGenerator,
    create_mixed_failure_database,
    create_mostly_complete_database,
    create_mostly_failed_database,
    create_all_complete_database,
    create_all_failed_database,
    create_empty_database
)


class TestComprehensiveIntegration:
    """Comprehensive integration test suite for recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create mock chat function with realistic behavior
        self.mock_chat_fn = Mock()
        self.mock_chat_fn.__name__ = "mock_chat_fn"
        
        # Track call count for verification
        self.call_count = 0
        
        def mock_response(prompt: str) -> str:
            self.call_count += 1
            # Simulate some processing time
            time.sleep(0.01)
            return f"Recovery response {self.call_count} for: {prompt[:30]}..."
        
        self.mock_chat_fn.side_effect = mock_response
        
        # Create processor with test configuration
        self.processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=2,
            retry_attempts=2,
            retry_delay=0.1,
            timeout=5.0
        )
        
        # List to track created databases for cleanup
        self.test_databases = []
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Clean up test databases
        for db_path in self.test_databases:
            TestDatabaseGenerator.cleanup_database(db_path)
        
        # Close processor
        self.processor.close()
    
    def _track_database(self, db_path: str) -> str:
        """Track a database for cleanup and return the path."""
        self.test_databases.append(db_path)
        return db_path
    
    # End-to-End Integration Tests
    
    def test_complete_recovery_workflow_mixed_failures(self):
        """Test complete recovery workflow with mixed failure scenarios."""
        # Create test database with mixed failures
        db_path, original_records = create_mixed_failure_database(50)
        self._track_database(db_path)
        
        # Get initial statistics
        initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        initial_incomplete = initial_stats['incomplete_records']
        
        assert initial_incomplete > 0, "Test database should have incomplete records"
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(original_records)
        assert all(result is not None for result in results)
        
        # Verify database was updated
        final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert final_stats['incomplete_records'] < initial_incomplete
        
        # Verify chat function was called for incomplete records
        assert self.call_count == initial_incomplete
    
    def test_complete_recovery_workflow_mostly_complete(self):
        """Test recovery workflow with mostly complete database."""
        # Create test database with mostly complete records
        db_path, original_records = create_mostly_complete_database(100)
        self._track_database(db_path)
        
        # Get initial statistics
        initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        initial_incomplete = initial_stats['incomplete_records']
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(original_records)
        
        # Verify only incomplete records were processed
        assert self.call_count == initial_incomplete
        
        # Verify database improvement
        final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert final_stats['incomplete_records'] <= initial_incomplete
    
    def test_complete_recovery_workflow_mostly_failed(self):
        """Test recovery workflow with mostly failed database."""
        # Create test database with mostly failed records
        db_path, original_records = create_mostly_failed_database(30)
        self._track_database(db_path)
        
        # Get initial statistics
        initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        initial_incomplete = initial_stats['incomplete_records']
        
        assert initial_incomplete > 20, "Test database should have many incomplete records"
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(original_records)
        
        # Verify all incomplete records were processed
        assert self.call_count == initial_incomplete
        
        # Verify significant improvement
        final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert final_stats['incomplete_records'] < initial_incomplete
    
    def test_complete_recovery_workflow_all_complete(self):
        """Test recovery workflow with all complete records."""
        # Create test database with all complete records
        db_path, original_records = create_all_complete_database(25)
        self._track_database(db_path)
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(original_records)
        
        # Verify no processing was needed
        assert self.call_count == 0
        
        # Verify database unchanged
        final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert final_stats['incomplete_records'] == 0
    
    def test_complete_recovery_workflow_all_failed(self):
        """Test recovery workflow with all failed records."""
        # Create test database with all failed records
        db_path, original_records = create_all_failed_database(20)
        self._track_database(db_path)
        
        # Get initial statistics
        initial_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        # Note: Error results are considered complete by the recovery analyzer
        # so incomplete_records will be less than total records
        assert initial_stats['incomplete_records'] > 0
        assert initial_stats['incomplete_records'] < len(original_records)
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(original_records)
        
        # Verify incomplete records were processed
        assert self.call_count == initial_stats['incomplete_records']
        
        # Verify recovery improved the database
        final_stats = TestDatabaseGenerator.get_database_statistics(db_path)
        assert final_stats['incomplete_records'] == 0
    
    def test_complete_recovery_workflow_empty_database(self):
        """Test recovery workflow with empty database."""
        # Create empty test database
        db_path, original_records = create_empty_database()
        self._track_database(db_path)
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert results == []
        assert self.call_count == 0
    
    # Error Handling Integration Tests
    
    def test_recovery_with_nonexistent_database(self):
        """Test recovery with non-existent database file."""
        non_existent_path = "/path/that/does/not/exist.db"
        
        with pytest.raises(FileNotFoundError):
            self.processor.recover_from_database(non_existent_path)
    
    def test_recovery_with_invalid_database_structure(self):
        """Test recovery with database missing required table."""
        # Create database without llm_results table
        db_path = TestDatabaseGenerator.create_temporary_database()
        self._track_database(db_path)
        
        # Create database with wrong schema
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE wrong_table (id INTEGER, data TEXT)")
            conn.commit()
        
        with pytest.raises(ValueError):
            self.processor.recover_from_database(db_path)
    
    def test_recovery_with_partial_processing_failures(self):
        """Test recovery when some prompts fail during reprocessing."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(20)
        self._track_database(db_path)
        
        # Mock chat function to fail on certain prompts
        def failing_mock_response(prompt: str) -> str:
            self.call_count += 1
            if "5]" in prompt or "10]" in prompt:  # Fail on prompts with index 5 and 10
                raise Exception("Simulated API failure")
            return f"Recovery response {self.call_count} for: {prompt[:30]}..."
        
        self.mock_chat_fn.side_effect = failing_mock_response
        
        # Perform recovery (should handle failures gracefully)
        results = self.processor.recover_from_database(db_path)
        
        # Verify results were returned despite some failures
        assert len(results) == len(original_records)
        
        # Verify some processing occurred
        assert self.call_count > 0
    
    # Data Integrity Tests
    
    def test_recovery_preserves_existing_complete_results(self):
        """Test that recovery preserves existing complete results."""
        # Create test database with mixed results
        db_path, original_records = create_mixed_failure_database(30)
        self._track_database(db_path)
        
        # Get complete results before recovery
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, result FROM llm_results 
                WHERE result IS NOT NULL AND result != '' AND result != 'NA'
                ORDER BY id
            """)
            complete_before = dict(cursor.fetchall())
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Get complete results after recovery
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, result FROM llm_results 
                WHERE id IN ({})
                ORDER BY id
            """.format(','.join(map(str, complete_before.keys()))))
            complete_after = dict(cursor.fetchall())
        
        # Verify existing complete results were preserved
        for record_id, original_result in complete_before.items():
            assert record_id in complete_after
            assert complete_after[record_id] == original_result
    
    def test_recovery_maintains_record_order(self):
        """Test that recovery maintains the original record order."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(25)
        self._track_database(db_path)
        
        # Get original order
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prompt FROM llm_results ORDER BY id")
            original_order = cursor.fetchall()
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Get final order
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prompt FROM llm_results ORDER BY id")
            final_order = cursor.fetchall()
        
        # Verify order is maintained
        assert original_order == final_order
        assert len(results) == len(original_order)
    
    def test_recovery_updates_only_incomplete_records(self):
        """Test that recovery only updates incomplete records."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(20)
        self._track_database(db_path)
        
        # Get incomplete record IDs before recovery
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM llm_results 
                WHERE result IS NULL OR result = '' OR result = 'NA'
            """)
            incomplete_ids = {row[0] for row in cursor.fetchall()}
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify only incomplete records were updated
        # (This is verified by checking that the number of calls matches incomplete records)
        assert self.call_count == len(incomplete_ids)
    
    # Performance and Concurrency Tests
    
    def test_recovery_with_multiple_workers(self):
        """Test recovery with multiple worker threads."""
        # Create processor with more workers
        multi_worker_processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=4,
            retry_attempts=1,
            retry_delay=0.05,
            timeout=5.0
        )
        
        try:
            # Create test database with many incomplete records
            db_path, original_records = create_mostly_failed_database(40)
            self._track_database(db_path)
            
            # Measure recovery time
            start_time = time.time()
            results = multi_worker_processor.recover_from_database(db_path)
            recovery_time = time.time() - start_time
            
            # Verify results
            assert len(results) == len(original_records)
            
            # Verify reasonable performance (should complete within reasonable time)
            assert recovery_time < 30.0  # Should complete within 30 seconds
            
        finally:
            multi_worker_processor.close()
    
    def test_recovery_with_timeout_handling(self):
        """Test recovery with timeout scenarios."""
        # Create processor with short timeout
        timeout_processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=2,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=0.001  # Very short timeout
        )
        
        # Mock chat function to be slow
        def slow_mock_response(prompt: str) -> str:
            time.sleep(0.1)  # Longer than timeout
            return f"Slow response for: {prompt[:30]}..."
        
        timeout_processor.chat_fn = Mock(side_effect=slow_mock_response)
        
        try:
            # Create test database
            db_path, original_records = create_mixed_failure_database(5)
            self._track_database(db_path)
            
            # Perform recovery (should handle timeouts gracefully)
            results = timeout_processor.recover_from_database(db_path)
            
            # Verify results were returned despite timeouts
            assert len(results) == len(original_records)
            
        finally:
            timeout_processor.close()
    
    # Component Integration Tests
    
    def test_integration_database_validator_with_recovery(self):
        """Test integration between DatabaseValidator and recovery process."""
        # Create valid database
        db_path, _ = create_mixed_failure_database(10)
        self._track_database(db_path)
        
        # Verify validator passes
        DatabaseValidator.validate_database_for_recovery(db_path)
        
        # Verify recovery works
        results = self.processor.recover_from_database(db_path)
        assert len(results) == 10
    
    def test_integration_recovery_analyzer_with_processor(self):
        """Test integration between RecoveryAnalyzer and RecoveryProcessor."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(15)
        self._track_database(db_path)
        
        # Use analyzer to get failed records
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_path)
        
        # Use processor to reprocess failed records
        recovery_processor = RecoveryProcessor(self.processor)
        new_results = recovery_processor.process_failed_prompts(failed_records)
        
        # Verify integration
        assert len(new_results) == len(failed_records)
        assert all(record_id in new_results for record_id, _ in failed_records)
    
    def test_integration_database_updater_with_analyzer(self):
        """Test integration between DatabaseUpdater and RecoveryAnalyzer."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(12)
        self._track_database(db_path)
        
        # Get initial incomplete count
        initial_failed, _ = RecoveryAnalyzer.analyze_database(db_path)
        initial_count = len(initial_failed)
        
        # Create mock results for all failed records
        mock_results = {record_id: f"Updated result {record_id}" 
                       for record_id, _ in initial_failed}
        
        # Update database
        failed_updates = DatabaseUpdater.update_results(db_path, mock_results)
        
        # Verify updates
        assert len(failed_updates) == 0  # All updates should succeed
        
        # Verify analyzer sees the changes
        final_failed, _ = RecoveryAnalyzer.analyze_database(db_path)
        assert len(final_failed) < initial_count
    
    # Logging and Monitoring Integration Tests
    
    def test_recovery_logging_integration(self):
        """Test that recovery process generates appropriate logs."""
        # Create test database
        db_path, original_records = create_mixed_failure_database(8)
        self._track_database(db_path)
        
        # Capture logs
        with patch('src.llmtools.parallel_llm_processor.logger') as mock_logger:
            # Perform recovery
            results = self.processor.recover_from_database(db_path)
            
            # Verify logging calls were made
            assert mock_logger.info.called
            assert mock_logger.debug.called or mock_logger.info.called
            
            # Verify results
            assert len(results) == len(original_records)
    
    def test_recovery_progress_tracking_integration(self):
        """Test that recovery process tracks progress appropriately."""
        # Create test database with enough records to see progress
        db_path, original_records = create_mostly_failed_database(20)
        self._track_database(db_path)
        
        # Mock tqdm to verify progress tracking
        with patch('src.llmtools.recovery_processor.tqdm') as mock_tqdm:
            mock_progress = Mock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress
            
            # Perform recovery
            results = self.processor.recover_from_database(db_path)
            
            # Verify progress tracking was used
            assert mock_tqdm.called
            assert mock_progress.update.called
            
            # Verify results
            assert len(results) == len(original_records)


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_chat_fn = Mock(return_value="Test response")
        self.processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=1,
            retry_attempts=1,
            retry_delay=0.1,
            timeout=5.0
        )
        self.test_databases = []
    
    def teardown_method(self):
        """Clean up test fixtures."""
        for db_path in self.test_databases:
            TestDatabaseGenerator.cleanup_database(db_path)
        self.processor.close()
    
    def _track_database(self, db_path: str) -> str:
        """Track a database for cleanup."""
        self.test_databases.append(db_path)
        return db_path
    
    def test_recovery_with_single_record(self):
        """Test recovery with database containing single record."""
        # Create database with single incomplete record
        db_path, records = TestDatabaseGenerator.create_test_database_with_scenario(
            "mixed_failures", 1
        )
        self._track_database(db_path)
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == 1
    
    def test_recovery_with_large_database(self):
        """Test recovery with large database."""
        # Create large database
        db_path, records = create_mixed_failure_database(500)
        self._track_database(db_path)
        
        # Perform recovery
        start_time = time.time()
        results = self.processor.recover_from_database(db_path)
        recovery_time = time.time() - start_time
        
        # Verify results
        assert len(results) == 500
        
        # Verify reasonable performance
        assert recovery_time < 120.0  # Should complete within 2 minutes
    
    def test_recovery_with_very_long_prompts(self):
        """Test recovery with very long prompts."""
        # Create database with long prompts
        db_path = TestDatabaseGenerator.create_temporary_database()
        self._track_database(db_path)
        TestDatabaseGenerator.create_database_with_schema(db_path)
        
        # Create records with very long prompts
        long_prompt = "This is a very long prompt. " * 100  # ~2800 characters
        records = [
            (1, long_prompt, None),
            (2, long_prompt + " Second version", ""),
            (3, long_prompt + " Third version", "NA")
        ]
        
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
                records
            )
            conn.commit()
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == 3
    
    def test_recovery_with_special_characters(self):
        """Test recovery with prompts containing special characters."""
        # Create database with special character prompts
        db_path = TestDatabaseGenerator.create_temporary_database()
        self._track_database(db_path)
        TestDatabaseGenerator.create_database_with_schema(db_path)
        
        # Create records with special characters
        special_prompts = [
            "Prompt with émojis 🚀🔥💯",
            "Prompt with quotes \"and\" 'apostrophes'",
            "Prompt with SQL injection'; DROP TABLE users; --",
            "Prompt with Unicode: 你好世界 こんにちは мир",
            "Prompt with newlines\nand\ttabs"
        ]
        
        records = [(i+1, prompt, None) for i, prompt in enumerate(special_prompts)]
        
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
                records
            )
            conn.commit()
        
        # Perform recovery
        results = self.processor.recover_from_database(db_path)
        
        # Verify results
        assert len(results) == len(special_prompts)