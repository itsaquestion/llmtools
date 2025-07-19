import unittest
import tempfile
import os
import sqlite3
import threading
import time
import concurrent.futures
from src.llmtools.database_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager table creation and initialization functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary database file for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_filename = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_filename)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connection and remove temporary file
        self.db_manager.close_connection()
        if os.path.exists(self.db_filename):
            os.unlink(self.db_filename)
    
    def test_create_table_success(self):
        """Test successful table creation."""
        result = self.db_manager.create_table()
        self.assertTrue(result)
        
        # Verify table structure
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(llm_results)")
            columns = cursor.fetchall()
            
            # Expected columns: id, prompt, result, created_at
            self.assertEqual(len(columns), 4)
            
            column_names = [col[1] for col in columns]
            self.assertIn('id', column_names)
            self.assertIn('prompt', column_names)
            self.assertIn('result', column_names)
            self.assertIn('created_at', column_names)
            
            # Verify id is primary key
            id_column = next(col for col in columns if col[1] == 'id')
            self.assertEqual(id_column[5], 1)  # pk flag should be 1
    
    def test_create_table_already_exists(self):
        """Test table creation when table already exists."""
        # Create table first time
        result1 = self.db_manager.create_table()
        self.assertTrue(result1)
        
        # Create table second time (should reuse existing)
        result2 = self.db_manager.create_table()
        self.assertTrue(result2)
        
        # Verify table still exists and has correct structure
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_results'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_initialize_database_empty_prompts(self):
        """Test database initialization with empty prompt list."""
        prompts = []
        result = self.db_manager.initialize_database(prompts)
        self.assertTrue(result)
        
        # Verify table exists but is empty
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM llm_results")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)
    
    def test_initialize_database_with_prompts(self):
        """Test database initialization with prompt list."""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing",
            "Write a Python function to sort a list"
        ]
        
        result = self.db_manager.initialize_database(prompts)
        self.assertTrue(result)
        
        # Verify records were inserted correctly
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prompt, result FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            self.assertEqual(len(records), 3)
            
            # Verify each record
            for i, (record_id, prompt, result) in enumerate(records):
                self.assertEqual(record_id, i + 1)  # id should be index + 1
                self.assertEqual(prompt, prompts[i])
                self.assertIsNone(result)  # result should be NULL initially
    
    def test_initialize_database_order_consistency(self):
        """Test that database records maintain the same order as input prompts."""
        prompts = [
            "First prompt",
            "Second prompt", 
            "Third prompt",
            "Fourth prompt",
            "Fifth prompt"
        ]
        
        result = self.db_manager.initialize_database(prompts)
        self.assertTrue(result)
        
        # Verify order is maintained
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, prompt FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            for i, (record_id, prompt) in enumerate(records):
                self.assertEqual(record_id, i + 1)
                self.assertEqual(prompt, prompts[i])
    
    def test_initialize_database_already_has_data(self):
        """Test initialization when table already contains data."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        
        # Initialize first time
        result1 = self.db_manager.initialize_database(prompts)
        self.assertTrue(result1)
        
        # Try to initialize again
        new_prompts = ["New prompt 1", "New prompt 2", "New prompt 3"]
        result2 = self.db_manager.initialize_database(new_prompts)
        self.assertTrue(result2)  # Should succeed but skip insertion
        
        # Verify original data is preserved
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM llm_results")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)  # Should still have original 2 records
    
    def test_table_structure_requirements(self):
        """Test that table structure meets all requirements."""
        self.db_manager.create_table()
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(llm_results)")
            columns = cursor.fetchall()
            
            # Convert to dict for easier testing
            column_info = {col[1]: {'type': col[2], 'notnull': col[3], 'pk': col[5]} for col in columns}
            
            # Test id column
            self.assertEqual(column_info['id']['type'], 'INTEGER')
            self.assertEqual(column_info['id']['pk'], 1)  # Primary key
            
            # Test prompt column
            self.assertEqual(column_info['prompt']['type'], 'TEXT')
            self.assertEqual(column_info['prompt']['notnull'], 1)  # NOT NULL
            
            # Test result column
            self.assertEqual(column_info['result']['type'], 'TEXT')
            self.assertEqual(column_info['result']['notnull'], 0)  # Nullable
            
            # Test created_at column
            self.assertEqual(column_info['created_at']['type'], 'TIMESTAMP')
    
    def test_created_at_default_value(self):
        """Test that created_at has proper default value."""
        prompts = ["Test prompt"]
        self.db_manager.initialize_database(prompts)
        
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT created_at FROM llm_results WHERE id = 1")
            created_at = cursor.fetchone()[0]
            
            # Should have a timestamp value (not None)
            self.assertIsNotNone(created_at)
            self.assertIsInstance(created_at, str)  # SQLite returns timestamp as string


class TestDatabaseManagerConcurrentUpdates(unittest.TestCase):
    """Test cases for thread-safe result update functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary database file for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_filename = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_filename)
        
        # Initialize with test prompts
        self.test_prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "How does backpropagation work?",
            "What is deep learning?",
            "Describe convolutional neural networks"
        ]
        self.db_manager.initialize_database(self.test_prompts)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Close database connection and remove temporary file
        self.db_manager.close_connection()
        if os.path.exists(self.db_filename):
            os.unlink(self.db_filename)
    
    def test_update_result_success(self):
        """Test successful result update for a single prompt."""
        prompt_index = 0
        result_text = "Machine learning is a subset of artificial intelligence..."
        
        success = self.db_manager.update_result(prompt_index, result_text)
        self.assertTrue(success)
        
        # Verify the update
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT result FROM llm_results WHERE id = ?", (prompt_index + 1,))
            stored_result = cursor.fetchone()[0]
            self.assertEqual(stored_result, result_text)
    
    def test_update_result_multiple_prompts(self):
        """Test updating results for multiple prompts."""
        results = [
            "ML is a subset of AI that enables computers to learn...",
            "Neural networks are computing systems inspired by biological neural networks...",
            "Backpropagation is a method used in artificial neural networks...",
            "Deep learning is part of a broader family of machine learning methods...",
            "CNNs are a class of deep neural networks most commonly applied to analyzing visual imagery..."
        ]
        
        # Update all results
        for i, result in enumerate(results):
            success = self.db_manager.update_result(i, result)
            self.assertTrue(success)
        
        # Verify all updates
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, result FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            for i, (record_id, stored_result) in enumerate(records):
                self.assertEqual(record_id, i + 1)
                self.assertEqual(stored_result, results[i])
    
    def test_update_result_invalid_index(self):
        """Test updating result with invalid prompt index."""
        # Test with index that doesn't exist
        success = self.db_manager.update_result(999, "Some result")
        self.assertFalse(success)
        
        # Test with negative index
        success = self.db_manager.update_result(-1, "Some result")
        self.assertFalse(success)
    
    def test_update_result_error_message(self):
        """Test updating result with error message."""
        prompt_index = 1
        error_message = "Error: API timeout after 60 seconds"
        
        success = self.db_manager.update_result(prompt_index, error_message)
        self.assertTrue(success)
        
        # Verify error message is stored
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT result FROM llm_results WHERE id = ?", (prompt_index + 1,))
            stored_result = cursor.fetchone()[0]
            self.assertEqual(stored_result, error_message)
    
    def test_concurrent_updates_different_records(self):
        """Test concurrent updates to different records."""
        def update_worker(prompt_index, result_text):
            """Worker function for concurrent updates."""
            return self.db_manager.update_result(prompt_index, f"Result for prompt {prompt_index}: {result_text}")
        
        # Create concurrent update tasks for different records
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(update_worker, i, f"Concurrent result {i}")
                futures.append(future)
            
            # Wait for all updates to complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All updates should succeed
        self.assertTrue(all(results))
        
        # Verify all results were stored correctly
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, result FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            for i, (record_id, stored_result) in enumerate(records):
                self.assertEqual(record_id, i + 1)
                self.assertIn(f"Concurrent result {i}", stored_result)
    
    def test_concurrent_updates_same_record(self):
        """Test concurrent updates to the same record (last writer wins)."""
        prompt_index = 2
        num_threads = 10
        results = []
        
        def update_worker(thread_id):
            """Worker function for concurrent updates to same record."""
            result_text = f"Thread {thread_id} result at {time.time()}"
            success = self.db_manager.update_result(prompt_index, result_text)
            if success:
                results.append(result_text)
            return success
        
        # Create concurrent update tasks for the same record
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_worker, i) for i in range(num_threads)]
            update_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All updates should succeed (thread-safe)
        self.assertTrue(all(update_results))
        
        # Verify that one of the results was stored (last writer wins)
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT result FROM llm_results WHERE id = ?", (prompt_index + 1,))
            stored_result = cursor.fetchone()[0]
            
            # The stored result should be one of the attempted updates
            self.assertIn("Thread", stored_result)
            self.assertIn("result at", stored_result)
    
    def test_update_result_thread_safety_stress_test(self):
        """Stress test for thread safety with many concurrent operations."""
        num_threads = 20
        updates_per_thread = 10
        
        def stress_worker(thread_id):
            """Worker function for stress testing."""
            successes = 0
            for i in range(updates_per_thread):
                prompt_index = i % len(self.test_prompts)  # Cycle through available prompts
                result_text = f"Stress test - Thread {thread_id}, Update {i}"
                if self.db_manager.update_result(prompt_index, result_text):
                    successes += 1
                # Small delay to increase chance of concurrent access
                time.sleep(0.001)
            return successes
        
        # Run stress test
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_threads)]
            success_counts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most updates should succeed (some might fail due to high concurrency)
        total_successes = sum(success_counts)
        total_attempts = num_threads * updates_per_thread
        success_rate = total_successes / total_attempts
        
        # Expect at least 80% success rate under stress
        self.assertGreater(success_rate, 0.8, f"Success rate too low: {success_rate:.2%}")
        
        # Verify database integrity
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM llm_results WHERE result IS NOT NULL")
            updated_count = cursor.fetchone()[0]
            self.assertGreater(updated_count, 0)
    
    def test_update_result_retry_mechanism(self):
        """Test the retry mechanism for database lock scenarios."""
        # This test simulates database lock by creating a long-running transaction
        prompt_index = 0
        result_text = "Test result with retry"
        
        def create_lock():
            """Create a database lock in a separate thread."""
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
                time.sleep(0.5)  # Hold lock for 500ms
                cursor.execute("ROLLBACK")
        
        # Start lock in background
        lock_thread = threading.Thread(target=create_lock)
        lock_thread.start()
        
        # Small delay to ensure lock is established
        time.sleep(0.1)
        
        # Attempt update (should retry and eventually succeed)
        start_time = time.time()
        success = self.db_manager.update_result(prompt_index, result_text, max_retries=5, retry_delay=0.1)
        end_time = time.time()
        
        # Wait for lock thread to complete
        lock_thread.join()
        
        # Update should eventually succeed
        self.assertTrue(success)
        
        # Should have taken some time due to retries
        self.assertGreater(end_time - start_time, 0.1)
        
        # Verify result was stored
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT result FROM llm_results WHERE id = ?", (prompt_index + 1,))
            stored_result = cursor.fetchone()[0]
            self.assertEqual(stored_result, result_text)
    
    def test_update_result_uninitialized_database(self):
        """Test update_result behavior when database is not initialized."""
        # Create a new database manager without initialization
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            uninit_db_manager = DatabaseManager(temp_db.name)
            # Don't call initialize_database
            
            success = uninit_db_manager.update_result(0, "Test result")
            self.assertFalse(success)
            
            uninit_db_manager.close_connection()
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)
    
    def test_update_result_data_consistency(self):
        """Test that concurrent updates maintain data consistency."""
        # Update all prompts concurrently with known results
        expected_results = {
            0: "ML Result",
            1: "NN Result", 
            2: "BP Result",
            3: "DL Result",
            4: "CNN Result"
        }
        
        def update_specific_prompt(prompt_index, result):
            """Update a specific prompt with a specific result."""
            return self.db_manager.update_result(prompt_index, result)
        
        # Execute all updates concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(update_specific_prompt, idx, result): idx 
                for idx, result in expected_results.items()
            }
            
            # Wait for completion and check results
            for future in concurrent.futures.as_completed(futures):
                self.assertTrue(future.result())
        
        # Verify data consistency
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, result FROM llm_results ORDER BY id")
            records = cursor.fetchall()
            
            for record_id, stored_result in records:
                prompt_index = record_id - 1
                expected_result = expected_results[prompt_index]
                self.assertEqual(stored_result, expected_result)


if __name__ == '__main__':
    unittest.main()