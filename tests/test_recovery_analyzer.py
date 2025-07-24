"""
Tests for the RecoveryAnalyzer class.

This module contains comprehensive tests for the RecoveryAnalyzer functionality,
including tests for identifying incomplete results and analyzing database records.
"""

import pytest
import sqlite3
import tempfile
import os
from typing import List, Tuple, Any
from src.llmtools.recovery_analyzer import RecoveryAnalyzer


class TestRecoveryAnalyzer:
    """Test suite for RecoveryAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)  # Close file descriptor, we'll use the path
    
    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary database file
        if os.path.exists(self.temp_db_path):
            os.unlink(self.temp_db_path)
    
    def create_test_database(self, records: List[Tuple[int, str, Any]]) -> None:
        """
        Create a test database with the specified records.
        
        Args:
            records: List of (id, prompt, result) tuples to insert
        """
        with sqlite3.connect(self.temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Create the llm_results table
            cursor.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test records
            for record_id, prompt, result in records:
                cursor.execute("""
                    INSERT INTO llm_results (id, prompt, result) 
                    VALUES (?, ?, ?)
                """, (record_id, prompt, result))
            
            conn.commit()
    
    # Tests for is_result_incomplete method
    
    def test_is_result_incomplete_with_none(self):
        """Test that None/NULL values are identified as incomplete."""
        assert RecoveryAnalyzer.is_result_incomplete(None) is True
    
    def test_is_result_incomplete_with_empty_string(self):
        """Test that empty strings are identified as incomplete."""
        assert RecoveryAnalyzer.is_result_incomplete("") is True
    
    def test_is_result_incomplete_with_na_string(self):
        """Test that 'NA' strings are identified as incomplete."""
        assert RecoveryAnalyzer.is_result_incomplete("NA") is True
    
    def test_is_result_incomplete_with_valid_string(self):
        """Test that valid non-empty strings are identified as complete."""
        assert RecoveryAnalyzer.is_result_incomplete("Valid result") is False
        assert RecoveryAnalyzer.is_result_incomplete("Error: Something went wrong") is False
        assert RecoveryAnalyzer.is_result_incomplete("0") is False  # Zero as string
        assert RecoveryAnalyzer.is_result_incomplete(" ") is False  # Whitespace
    
    def test_is_result_incomplete_with_numeric_values(self):
        """Test that numeric values are identified as complete."""
        assert RecoveryAnalyzer.is_result_incomplete(0) is False
        assert RecoveryAnalyzer.is_result_incomplete(42) is False
        assert RecoveryAnalyzer.is_result_incomplete(3.14) is False
        assert RecoveryAnalyzer.is_result_incomplete(-1) is False
    
    def test_is_result_incomplete_with_other_types(self):
        """Test that other data types are identified as complete."""
        assert RecoveryAnalyzer.is_result_incomplete([]) is False
        assert RecoveryAnalyzer.is_result_incomplete({}) is False
        assert RecoveryAnalyzer.is_result_incomplete(True) is False
        assert RecoveryAnalyzer.is_result_incomplete(False) is False
    
    # Tests for analyze_database method
    
    def test_analyze_database_with_nonexistent_file(self):
        """Test that analyzing a non-existent database file raises FileNotFoundError."""
        non_existent_path = "/path/that/does/not/exist.db"
        
        with pytest.raises(FileNotFoundError):
            RecoveryAnalyzer.analyze_database(non_existent_path)
    
    def test_analyze_database_without_llm_results_table(self):
        """Test that analyzing a database without llm_results table raises ValueError."""
        # Create database with wrong table name
        with sqlite3.connect(self.temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE wrong_table (id INTEGER, data TEXT)")
            conn.commit()
        
        with pytest.raises(ValueError, match="Database does not contain 'llm_results' table"):
            RecoveryAnalyzer.analyze_database(self.temp_db_path)
    
    def test_analyze_database_with_empty_table(self):
        """Test analyzing a database with empty llm_results table."""
        self.create_test_database([])
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        assert failed_records == []
        assert existing_results == []
    
    def test_analyze_database_with_all_complete_results(self):
        """Test analyzing a database where all results are complete."""
        test_records = [
            (1, "What is 2+2?", "4"),
            (2, "What color is the sky?", "Blue"),
            (3, "What is the capital of France?", "Paris")
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        assert failed_records == []
        assert existing_results == ["4", "Blue", "Paris"]
    
    def test_analyze_database_with_all_incomplete_results(self):
        """Test analyzing a database where all results are incomplete."""
        test_records = [
            (1, "What is 2+2?", None),
            (2, "What color is the sky?", ""),
            (3, "What is the capital of France?", "NA")
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        expected_failed = [
            (1, "What is 2+2?"),
            (2, "What color is the sky?"),
            (3, "What is the capital of France?")
        ]
        
        assert failed_records == expected_failed
        assert existing_results == [None, None, None]
    
    def test_analyze_database_with_mixed_results(self):
        """Test analyzing a database with a mix of complete and incomplete results."""
        test_records = [
            (1, "What is 2+2?", "4"),           # Complete
            (2, "What color is the sky?", None), # Incomplete (NULL)
            (3, "What is the capital of France?", "Paris"), # Complete
            (4, "What is 5*6?", ""),            # Incomplete (empty string)
            (5, "What is the moon made of?", "NA"), # Incomplete (NA)
            (6, "What is 10/2?", "5")           # Complete
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        expected_failed = [
            (2, "What color is the sky?"),
            (4, "What is 5*6?"),
            (5, "What is the moon made of?")
        ]
        
        expected_results = ["4", None, "Paris", None, None, "5"]
        
        assert failed_records == expected_failed
        assert existing_results == expected_results
    
    def test_analyze_database_maintains_order(self):
        """Test that analyze_database maintains the correct order of records."""
        # Create records with non-sequential IDs to test ordering
        test_records = [
            (3, "Third prompt", "Third result"),
            (1, "First prompt", None),
            (5, "Fifth prompt", "Fifth result"),
            (2, "Second prompt", ""),
            (4, "Fourth prompt", "NA")
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        # Results should be ordered by ID (1, 2, 3, 4, 5)
        expected_failed = [
            (1, "First prompt"),
            (2, "Second prompt"),
            (4, "Fourth prompt")
        ]
        
        expected_results = [None, None, "Third result", None, "Fifth result"]
        
        assert failed_records == expected_failed
        assert existing_results == expected_results
    
    def test_analyze_database_with_error_results(self):
        """Test that error messages in results are treated as complete."""
        test_records = [
            (1, "What is 2+2?", "Error: Timeout occurred"),
            (2, "What color is the sky?", "Error: API limit exceeded"),
            (3, "What is the capital of France?", None)  # This should be incomplete
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        expected_failed = [(3, "What is the capital of France?")]
        expected_results = ["Error: Timeout occurred", "Error: API limit exceeded", None]
        
        assert failed_records == expected_failed
        assert existing_results == expected_results
    
    # Tests for get_database_summary method
    
    def test_get_database_summary_with_mixed_data(self):
        """Test getting database summary with mixed complete and incomplete results."""
        test_records = [
            (1, "Prompt 1", "Result 1"),        # Complete
            (2, "Prompt 2", None),              # NULL
            (3, "Prompt 3", ""),                # Empty string
            (4, "Prompt 4", "NA"),              # NA
            (5, "Prompt 5", "Result 5"),        # Complete
            (6, "Prompt 6", None)               # NULL
        ]
        
        self.create_test_database(test_records)
        
        summary = RecoveryAnalyzer.get_database_summary(self.temp_db_path)
        
        assert summary['total_records'] == 6
        assert summary['complete_records'] == 2
        assert summary['incomplete_records'] == 4
        assert summary['null_results'] == 2
        assert summary['empty_string_results'] == 1
        assert summary['na_results'] == 1
        assert summary['completion_rate'] == pytest.approx(33.33, rel=1e-2)
    
    def test_get_database_summary_with_empty_database(self):
        """Test getting database summary with empty database."""
        self.create_test_database([])
        
        summary = RecoveryAnalyzer.get_database_summary(self.temp_db_path)
        
        assert summary['total_records'] == 0
        assert summary['complete_records'] == 0
        assert summary['incomplete_records'] == 0
        assert summary['null_results'] == 0
        assert summary['empty_string_results'] == 0
        assert summary['na_results'] == 0
        assert summary['completion_rate'] == 0
    
    def test_get_database_summary_with_all_complete(self):
        """Test getting database summary with all complete results."""
        test_records = [
            (1, "Prompt 1", "Result 1"),
            (2, "Prompt 2", "Result 2"),
            (3, "Prompt 3", "Result 3")
        ]
        
        self.create_test_database(test_records)
        
        summary = RecoveryAnalyzer.get_database_summary(self.temp_db_path)
        
        assert summary['total_records'] == 3
        assert summary['complete_records'] == 3
        assert summary['incomplete_records'] == 0
        assert summary['null_results'] == 0
        assert summary['empty_string_results'] == 0
        assert summary['na_results'] == 0
        assert summary['completion_rate'] == 100.0
    
    # Edge case tests
    
    def test_analyze_database_with_special_characters_in_prompts(self):
        """Test analyzing database with special characters in prompts and results."""
        test_records = [
            (1, "What's the meaning of 'life'?", "42"),
            (2, "Prompt with\nnewlines\tand\ttabs", None),
            (3, "Prompt with émojis 🚀🌟", ""),
            (4, "Prompt with quotes \"and\" apostrophes'", "Result with 'quotes'")
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        expected_failed = [
            (2, "Prompt with\nnewlines\tand\ttabs"),
            (3, "Prompt with émojis 🚀🌟")
        ]
        
        expected_results = ["42", None, None, "Result with 'quotes'"]
        
        assert failed_records == expected_failed
        assert existing_results == expected_results
    
    def test_analyze_database_with_very_long_strings(self):
        """Test analyzing database with very long prompts and results."""
        long_prompt = "A" * 10000  # Very long prompt
        long_result = "B" * 10000  # Very long result
        
        test_records = [
            (1, long_prompt, long_result),
            (2, "Short prompt", None)
        ]
        
        self.create_test_database(test_records)
        
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(self.temp_db_path)
        
        expected_failed = [(2, "Short prompt")]
        expected_results = [long_result, None]
        
        assert failed_records == expected_failed
        assert existing_results == expected_results


if __name__ == "__main__":
    pytest.main([__file__])