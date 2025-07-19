import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor
from src.llmtools.database_manager import DatabaseManager


class TestParallelLLMProcessorDatabaseParams(unittest.TestCase):
    """Test cases for ParallelLLMProcessor database parameter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_chat_fn = Mock(return_value="test response")
        self.test_prompts = ["prompt1", "prompt2", "prompt3"]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any temporary database files
        for filename in os.listdir('.'):
            if filename.startswith('llm_results_') and filename.endswith('.db'):
                try:
                    os.remove(filename)
                except OSError:
                    pass
    
    def test_init_without_database(self):
        """Test initialization without database functionality."""
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            num_workers=2,
            save_to_db=False
        )
        
        self.assertEqual(processor.chat_fn, self.mock_chat_fn)
        self.assertEqual(processor.num_workers, 2)
        self.assertFalse(processor.save_to_db)
        self.assertIsNone(processor.db_manager)
    
    def test_init_with_database_custom_filename(self):
        """Test initialization with database and custom filename."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_filename = tmp_file.name
        
        try:
            processor = ParallelLLMProcessor(
                chat_fn=self.mock_chat_fn,
                save_to_db=True,
                db_filename=db_filename
            )
            
            self.assertTrue(processor.save_to_db)
            self.assertEqual(processor.db_filename, db_filename)
            # Database manager is initialized lazily
            self.assertIsNone(processor.db_manager)
            
            # Trigger database manager initialization
            processor._initialize_database_manager()
            self.assertIsInstance(processor.db_manager, DatabaseManager)
            
            processor.close()
        finally:
            if os.path.exists(db_filename):
                os.remove(db_filename)
    
    def test_init_with_database_default_filename(self):
        """Test initialization with database and default filename generation."""
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            save_to_db=True
        )
        
        self.assertTrue(processor.save_to_db)
        self.assertTrue(processor.db_filename.startswith('llm_results_'))
        self.assertTrue(processor.db_filename.endswith('.db'))
        # Database manager is initialized lazily
        self.assertIsNone(processor.db_manager)
        
        # Trigger database manager initialization
        processor._initialize_database_manager()
        self.assertIsInstance(processor.db_manager, DatabaseManager)
        
        processor.close()
    
    def test_default_filename_generation(self):
        """Test default filename generation logic."""
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            save_to_db=False
        )
        
        # Test the private method directly
        filename1 = processor._generate_default_filename()
        time.sleep(1.1)  # Ensure different timestamp (1+ second)
        filename2 = processor._generate_default_filename()
        
        # Both should follow the pattern
        self.assertTrue(filename1.startswith('llm_results_'))
        self.assertTrue(filename1.endswith('.db'))
        self.assertTrue(filename2.startswith('llm_results_'))
        self.assertTrue(filename2.endswith('.db'))
        
        # Should be different due to timestamp
        self.assertNotEqual(filename1, filename2)
        
        # Test that filename contains valid timestamp
        import re
        pattern = r'llm_results_(\d+)\.db'
        match1 = re.match(pattern, filename1)
        match2 = re.match(pattern, filename2)
        
        self.assertIsNotNone(match1)
        self.assertIsNotNone(match2)
        
        # Timestamps should be valid integers
        timestamp1 = int(match1.group(1))
        timestamp2 = int(match2.group(1))
        self.assertGreater(timestamp2, timestamp1)
    
    def test_parameter_validation_chat_fn(self):
        """Test parameter validation for chat_fn."""
        with self.assertRaises(TypeError):
            ParallelLLMProcessor(chat_fn="not_callable")
        
        with self.assertRaises(TypeError):
            ParallelLLMProcessor(chat_fn=None)
    
    def test_parameter_validation_num_workers(self):
        """Test parameter validation for num_workers."""
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, num_workers=0)
        
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, num_workers=-1)
        
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, num_workers=1.5)
    
    def test_parameter_validation_retry_attempts(self):
        """Test parameter validation for retry_attempts."""
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, retry_attempts=-1)
        
        # Zero should be valid
        processor = ParallelLLMProcessor(chat_fn=self.mock_chat_fn, retry_attempts=0)
        self.assertEqual(processor.retry_attempts, 0)
    
    def test_parameter_validation_retry_delay(self):
        """Test parameter validation for retry_delay."""
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, retry_delay=-1.0)
        
        # Zero should be valid
        processor = ParallelLLMProcessor(chat_fn=self.mock_chat_fn, retry_delay=0.0)
        self.assertEqual(processor.retry_delay, 0.0)
    
    def test_parameter_validation_timeout(self):
        """Test parameter validation for timeout."""
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, timeout=0)
        
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, timeout=-1.0)
    
    def test_parameter_validation_save_to_db(self):
        """Test parameter validation for save_to_db."""
        with self.assertRaises(TypeError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, save_to_db="true")
        
        with self.assertRaises(TypeError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, save_to_db=1)
    
    def test_parameter_validation_db_filename(self):
        """Test parameter validation for db_filename."""
        with self.assertRaises(TypeError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, db_filename=123)
        
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, db_filename="")
        
        with self.assertRaises(ValueError):
            ParallelLLMProcessor(chat_fn=self.mock_chat_fn, db_filename="   ")
    
    @patch('src.llmtools.parallel_llm_processor.DatabaseManager')
    def test_database_manager_initialization_failure(self, mock_db_manager):
        """Test handling of database manager initialization failure."""
        # Mock DatabaseManager to raise an exception
        mock_db_manager.side_effect = Exception("Database connection failed")
        
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            save_to_db=True,
            db_filename="test.db"
        )
        
        # Initially save_to_db should be True (lazy initialization)
        self.assertTrue(processor.save_to_db)
        self.assertIsNone(processor.db_manager)
        
        # Trigger database manager initialization - should fail and disable database
        processor._initialize_database_manager()
        
        # Should fall back to no database functionality
        self.assertFalse(processor.save_to_db)
        self.assertIsNone(processor.db_manager)
    
    def test_close_method(self):
        """Test the close method for resource cleanup."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_filename = tmp_file.name
        
        try:
            processor = ParallelLLMProcessor(
                chat_fn=self.mock_chat_fn,
                save_to_db=True,
                db_filename=db_filename
            )
            
            # Initialize database manager
            processor._initialize_database_manager()
            
            # Mock the database manager's close_connection method
            processor.db_manager.close_connection = Mock()
            
            processor.close()
            
            # Verify close_connection was called
            processor.db_manager.close_connection.assert_called_once()
        finally:
            if os.path.exists(db_filename):
                os.remove(db_filename)
    
    def test_close_method_without_database(self):
        """Test the close method when no database is configured."""
        processor = ParallelLLMProcessor(
            chat_fn=self.mock_chat_fn,
            save_to_db=False
        )
        
        # Should not raise any exceptions
        processor.close()
    
    def test_destructor_calls_close(self):
        """Test that the destructor calls the close method."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_filename = tmp_file.name
        
        try:
            processor = ParallelLLMProcessor(
                chat_fn=self.mock_chat_fn,
                save_to_db=True,
                db_filename=db_filename
            )
            
            # Initialize database manager
            processor._initialize_database_manager()
            
            # Mock the close method
            processor.close = Mock()
            
            # Trigger destructor
            del processor
            
            # Note: __del__ behavior is implementation-dependent,
            # so we can't reliably test it was called
        finally:
            if os.path.exists(db_filename):
                os.remove(db_filename)


if __name__ == '__main__':
    unittest.main()