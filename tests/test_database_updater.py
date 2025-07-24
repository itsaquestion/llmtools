"""
Tests for the DatabaseUpdater class.

This module contains comprehensive tests for the database update functionality,
including transaction handling, batch updates, concurrent access, and error scenarios.
"""

import pytest
import sqlite3
import tempfile
import os
import threading
import time
from unittest.mock import patch, MagicMock
from typing import Dict, List
from contextlib import contextmanager

from src.llmtools.database_updater import DatabaseUpdater


class TestDatabaseUpdater:
    """Test suite for DatabaseUpdater class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test database with llm_results table
        self._create_test_database()
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Remove temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def _create_test_database(self):
        """Create a test database with the expected schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def _insert_test_records(self, records: List[tuple]):
        """Insert test records into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
                records
            )
            conn.commit()
    
    def _get_all_results(self) -> Dict[int, str]:
        """Get all results from the database as a dictionary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, result FROM llm_results ORDER BY id")
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def test_update_results_empty_map(self):
        """Test updating with empty results map."""
        failed_updates = DatabaseUpdater.update_results(self.db_path, {})
        assert failed_updates == []
    
    def test_update_results_single_record(self):
        """Test updating a single record."""
        # Insert test record with NULL result
        self._insert_test_records([(1, "Test prompt", None)])
        
        # Update the record
        results_map = {1: "Updated result"}
        failed_updates = DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Verify update was successful
        assert failed_updates == []
        all_results = self._get_all_results()
        assert all_results[1] == "Updated result"
    
    def test_update_results_multiple_records(self):
        """Test updating multiple records."""
        # Insert test records with various incomplete results
        test_records = [
            (1, "Prompt 1", None),
            (2, "Prompt 2", ""),
            (3, "Prompt 3", "NA"),
            (4, "Prompt 4", "Complete result")  # This should not be updated
        ]
        self._insert_test_records(test_records)
        
        # Update incomplete records
        results_map = {
            1: "New result 1",
            2: "New result 2", 
            3: "New result 3",
            4: "Should not update"  # This should fail because result is already complete
        }
        failed_updates = DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Verify updates
        all_results = self._get_all_results()
        assert all_results[1] == "New result 1"
        assert all_results[2] == "New result 2"
        assert all_results[3] == "New result 3"
        assert all_results[4] == "Complete result"  # Should remain unchanged
        
        # Record 4 should be in failed updates because it was already complete
        assert 4 in failed_updates
    
    def test_update_results_nonexistent_record(self):
        """Test updating a record that doesn't exist."""
        # Insert one test record
        self._insert_test_records([(1, "Test prompt", None)])
        
        # Try to update non-existent record
        results_map = {999: "New result"}
        failed_updates = DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Should report failure for non-existent record
        assert 999 in failed_updates
    
    def test_update_results_batch_processing(self):
        """Test batch processing for large updates."""
        # Create many test records
        test_records = [(i, f"Prompt {i}", None) for i in range(1, 201)]  # 200 records
        self._insert_test_records(test_records)
        
        # Create results map for all records
        results_map = {i: f"Result {i}" for i in range(1, 201)}
        
        # Mock batch size to force batching
        with patch.object(DatabaseUpdater, 'BATCH_SIZE', 50):
            failed_updates = DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Verify all updates were successful
        assert failed_updates == []
        all_results = self._get_all_results()
        for i in range(1, 201):
            assert all_results[i] == f"Result {i}"
    
    def test_update_single_result_success(self):
        """Test updating a single result using convenience method."""
        # Insert test record
        self._insert_test_records([(1, "Test prompt", None)])
        
        # Update using convenience method
        success = DatabaseUpdater.update_single_result(self.db_path, 1, "New result")
        
        # Verify success
        assert success is True
        all_results = self._get_all_results()
        assert all_results[1] == "New result"
    
    def test_update_single_result_failure(self):
        """Test updating a single result that fails."""
        # Insert test record with complete result
        self._insert_test_records([(1, "Test prompt", "Complete result")])
        
        # Try to update (should fail because result is already complete)
        success = DatabaseUpdater.update_single_result(self.db_path, 1, "New result")
        
        # Verify failure
        assert success is False
        all_results = self._get_all_results()
        assert all_results[1] == "Complete result"  # Should remain unchanged
    
    def test_verify_updates_success(self):
        """Test verification of successful updates."""
        # Insert and update test records
        self._insert_test_records([(1, "Prompt 1", None), (2, "Prompt 2", "")])
        results_map = {1: "Result 1", 2: "Result 2"}
        DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Verify updates
        verified, mismatched = DatabaseUpdater.verify_updates(self.db_path, results_map)
        
        assert verified == [1, 2]
        assert mismatched == []
    
    def test_verify_updates_mismatch(self):
        """Test verification with mismatched results."""
        # Insert test record and update it
        self._insert_test_records([(1, "Test prompt", None)])
        DatabaseUpdater.update_results(self.db_path, {1: "Actual result"})
        
        # Verify with different expected result
        expected_results = {1: "Expected result"}
        verified, mismatched = DatabaseUpdater.verify_updates(self.db_path, expected_results)
        
        assert verified == []
        assert mismatched == [1]
    
    def test_verify_updates_nonexistent_record(self):
        """Test verification of non-existent record."""
        expected_results = {999: "Some result"}
        verified, mismatched = DatabaseUpdater.verify_updates(self.db_path, expected_results)
        
        assert verified == []
        assert mismatched == [999]
    
    def test_get_update_statistics(self):
        """Test getting update statistics."""
        # Insert test records with various states
        test_records = [
            (1, "Prompt 1", "Complete result"),
            (2, "Prompt 2", None),
            (3, "Prompt 3", ""),
            (4, "Prompt 4", "NA"),
            (5, "Prompt 5", "Another complete result")
        ]
        self._insert_test_records(test_records)
        
        # Get statistics
        stats = DatabaseUpdater.get_update_statistics(self.db_path)
        
        # Verify statistics
        assert stats['total_records'] == 5
        assert stats['complete_records'] == 2
        assert stats['incomplete_records'] == 3
        assert stats['null_records'] == 1
        assert stats['empty_records'] == 1
        assert stats['na_records'] == 1
        assert stats['completion_rate'] == 40.0  # 2/5 * 100
    
    def test_transaction_rollback_on_error(self):
        """Test that transactions are rolled back on errors."""
        # Insert test records
        self._insert_test_records([(1, "Prompt 1", None), (2, "Prompt 2", None)])
        
        # Mock the _update_batch method to simulate an error during transaction
        original_update_batch = DatabaseUpdater._update_batch
        
        def mock_update_batch(conn, results_map):
            # Simulate a database error during the batch update
            raise sqlite3.Error("Simulated transaction error")
        
        with patch.object(DatabaseUpdater, '_update_batch', side_effect=mock_update_batch):
            results_map = {1: "Result 1", 2: "Result 2"}
            # This should raise an exception due to the mocked error
            with pytest.raises(sqlite3.Error, match="Simulated transaction error"):
                DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Verify no updates were applied (transaction was rolled back)
        all_results = self._get_all_results()
        assert all_results[1] is None
        assert all_results[2] is None
    
    def test_concurrent_access_handling(self):
        """Test handling of concurrent database access."""
        # Insert test records
        test_records = [(i, f"Prompt {i}", None) for i in range(1, 11)]
        self._insert_test_records(test_records)
        
        results = []
        errors = []
        
        def update_worker(start_id, end_id):
            """Worker function for concurrent updates."""
            try:
                results_map = {i: f"Result {i}" for i in range(start_id, end_id + 1)}
                failed = DatabaseUpdater.update_results(self.db_path, results_map)
                results.append((start_id, end_id, failed))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads to update different ranges
        threads = []
        for i in range(0, 10, 2):
            thread = threading.Thread(target=update_worker, args=(i + 1, i + 2))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert errors == []
        
        # Verify all updates were successful
        for start_id, end_id, failed in results:
            assert failed == []
        
        # Verify final database state
        all_results = self._get_all_results()
        for i in range(1, 11):
            assert all_results[i] == f"Result {i}"
    
    def test_database_locked_retry(self):
        """Test retry mechanism for database locked errors."""
        # Insert test record
        self._insert_test_records([(1, "Test prompt", None)])
        
        # Mock connection to simulate database locked error initially
        original_connect = sqlite3.connect
        call_count = 0
        
        def mock_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            return original_connect(*args, **kwargs)
        
        with patch('sqlite3.connect', side_effect=mock_connect):
            with patch('time.sleep'):  # Speed up test by mocking sleep
                results_map = {1: "New result"}
                failed_updates = DatabaseUpdater.update_results(self.db_path, results_map)
        
        # Should succeed on retry
        assert failed_updates == []
        all_results = self._get_all_results()
        assert all_results[1] == "New result"
    
    def test_invalid_input_validation(self):
        """Test validation of invalid inputs."""
        # Test invalid results_map type
        with pytest.raises(ValueError, match="results_map must be a dictionary"):
            DatabaseUpdater.update_results(self.db_path, "not a dict")
        
        # Test invalid record_id for single update
        with pytest.raises(ValueError, match="record_id must be a positive integer"):
            DatabaseUpdater.update_single_result(self.db_path, -1, "result")
        
        with pytest.raises(ValueError, match="record_id must be a positive integer"):
            DatabaseUpdater.update_single_result(self.db_path, "not an int", "result")
        
        # Test invalid new_result type for single update
        with pytest.raises(ValueError, match="new_result must be a string"):
            DatabaseUpdater.update_single_result(self.db_path, 1, 123)
    
    def test_nonexistent_database_file(self):
        """Test handling of non-existent database file."""
        nonexistent_path = "/path/that/does/not/exist.db"
        
        with pytest.raises(sqlite3.Error):
            DatabaseUpdater.update_results(nonexistent_path, {1: "result"})
    
    def test_database_without_llm_results_table(self):
        """Test handling of database without llm_results table."""
        # Create database without the expected table
        empty_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        empty_db.close()
        
        try:
            with sqlite3.connect(empty_db.name) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE other_table (id INTEGER)")
                conn.commit()
            
            # Try to update - should handle gracefully
            results_map = {1: "result"}
            failed_updates = DatabaseUpdater.update_results(empty_db.name, results_map)
            
            # Should report failure for the record
            assert 1 in failed_updates
            
        finally:
            os.unlink(empty_db.name)
    
    def test_wal_mode_configuration(self):
        """Test that database connection works with WAL mode configuration."""
        # Insert test record
        self._insert_test_records([(1, "Test prompt", None)])
        
        # Update record - this should work with WAL mode configuration
        failed_updates = DatabaseUpdater.update_results(self.db_path, {1: "result"})
        
        # Verify update was successful (which means WAL mode didn't cause issues)
        assert failed_updates == []
        all_results = self._get_all_results()
        assert all_results[1] == "result"
        
        # Verify that the database is indeed in WAL mode after our operations
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            # WAL mode should be active (though it might show as 'wal' or 'WAL')
            assert journal_mode.upper() == 'WAL'


if __name__ == "__main__":
    pytest.main([__file__])