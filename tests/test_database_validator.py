"""
Unit tests for DatabaseValidator class.

Tests various database validation scenarios including file existence,
accessibility, and table structure validation.
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.llmtools.database_validator import DatabaseValidator


class TestDatabaseValidator:
    """Test cases for DatabaseValidator class."""
    
    def test_validate_database_file_with_valid_file(self, tmp_path):
        """Test validation with a valid SQLite database file."""
        # Create a valid SQLite database
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
        
        # Should not raise any exception
        DatabaseValidator.validate_database_file(str(db_path))
    
    def test_validate_database_file_nonexistent_file(self):
        """Test validation with non-existent database file."""
        with pytest.raises(FileNotFoundError, match="Database file does not exist"):
            DatabaseValidator.validate_database_file("/nonexistent/path/test.db")
    
    def test_validate_database_file_empty_path(self):
        """Test validation with empty or whitespace path."""
        with pytest.raises(ValueError, match="Database file path cannot be empty"):
            DatabaseValidator.validate_database_file("")
        
        with pytest.raises(ValueError, match="Database file path cannot be empty"):
            DatabaseValidator.validate_database_file("   ")
    
    def test_validate_database_file_non_string_path(self):
        """Test validation with non-string path."""
        with pytest.raises(ValueError, match="Database file path must be a string"):
            DatabaseValidator.validate_database_file(123)
        
        with pytest.raises(ValueError, match="Database file path must be a string"):
            DatabaseValidator.validate_database_file(None)
    
    def test_validate_database_file_directory_instead_of_file(self, tmp_path):
        """Test validation when path points to a directory instead of file."""
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        
        with pytest.raises(ValueError, match="Path exists but is not a file"):
            DatabaseValidator.validate_database_file(str(dir_path))
    
    def test_validate_database_file_not_readable(self, tmp_path):
        """Test validation with file that's not readable."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
        
        # Mock os.access to simulate unreadable file
        with patch('os.access') as mock_access:
            def access_side_effect(path, mode):
                if mode == os.R_OK:
                    return False
                return True
            mock_access.side_effect = access_side_effect
            
            with pytest.raises(PermissionError, match="Database file is not readable"):
                DatabaseValidator.validate_database_file(str(db_path))
    
    def test_validate_database_file_not_writable(self, tmp_path):
        """Test validation with file that's not writable."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
        
        # Mock os.access to simulate unwritable file
        with patch('os.access') as mock_access:
            def access_side_effect(path, mode):
                if mode == os.W_OK:
                    return False
                return True
            mock_access.side_effect = access_side_effect
            
            with pytest.raises(PermissionError, match="Database file is not writable"):
                DatabaseValidator.validate_database_file(str(db_path))
    
    def test_validate_database_file_invalid_sqlite_file(self, tmp_path):
        """Test validation with file that exists but is not a valid SQLite database."""
        # Create a text file instead of SQLite database
        invalid_db_path = tmp_path / "invalid.db"
        with open(invalid_db_path, 'w') as f:
            f.write("This is not a SQLite database")
        
        with pytest.raises(ValueError, match="File exists but is not a valid SQLite database"):
            DatabaseValidator.validate_database_file(str(invalid_db_path))
    
    def test_validate_table_structure_with_valid_table(self, tmp_path):
        """Test table structure validation with correct llm_results table."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Should not raise any exception
        DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_missing_table(self, tmp_path):
        """Test table structure validation when llm_results table doesn't exist."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id INTEGER)")
        
        with pytest.raises(ValueError, match="Required table 'llm_results' does not exist"):
            DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_empty_table(self, tmp_path):
        """Test table structure validation with table that has no columns."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            # Create table name but this scenario is hard to create in SQLite
            # Instead, we'll mock the PRAGMA response
            conn.execute("CREATE TABLE llm_results (id INTEGER)")
        
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock table exists but no columns
            mock_cursor.fetchone.side_effect = [('llm_results',), None]  # Table exists, no columns
            mock_cursor.fetchall.return_value = []  # No columns
            
            with pytest.raises(ValueError, match="Table 'llm_results' exists but has no columns"):
                DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_missing_columns(self, tmp_path):
        """Test table structure validation with missing required columns."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            # Create table with only some required columns
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL
                )
            """)
        
        with pytest.raises(ValueError, match="Table 'llm_results' is missing required columns"):
            DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_wrong_primary_key(self, tmp_path):
        """Test table structure validation with wrong primary key."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP
                )
            """)
        
        with pytest.raises(ValueError, match="Column 'id' should be PRIMARY KEY"):
            DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_wrong_not_null_constraint(self, tmp_path):
        """Test table structure validation with wrong NOT NULL constraint."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT,
                    result TEXT,
                    created_at TIMESTAMP
                )
            """)
        
        with pytest.raises(ValueError, match="Column 'prompt' should be NOT NULL"):
            DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_table_structure_compatible_types(self, tmp_path):
        """Test table structure validation with compatible but different types."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            # Use compatible types (VARCHAR instead of TEXT, INT instead of INTEGER)
            conn.execute("""
                CREATE TABLE llm_results (
                    id INT PRIMARY KEY,
                    prompt VARCHAR NOT NULL,
                    result CHAR,
                    created_at TEXT
                )
            """)
        
        # Should not raise exception due to type compatibility
        DatabaseValidator.validate_table_structure(str(db_path))
    
    def test_validate_database_for_recovery_success(self, tmp_path):
        """Test comprehensive validation for recovery operations - success case."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Should not raise any exception
        DatabaseValidator.validate_database_for_recovery(str(db_path))
    
    def test_validate_database_for_recovery_file_error(self):
        """Test comprehensive validation - file error propagation."""
        with pytest.raises(FileNotFoundError):
            DatabaseValidator.validate_database_for_recovery("/nonexistent/test.db")
    
    def test_validate_database_for_recovery_table_error(self, tmp_path):
        """Test comprehensive validation - table error propagation."""
        db_path = tmp_path / "test.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE other_table (id INTEGER)")
        
        with pytest.raises(ValueError, match="Required table 'llm_results' does not exist"):
            DatabaseValidator.validate_database_for_recovery(str(db_path))
    
    def test_types_compatible_exact_match(self):
        """Test type compatibility with exact matches."""
        assert DatabaseValidator._types_compatible("INTEGER", "INTEGER")
        assert DatabaseValidator._types_compatible("TEXT", "TEXT")
        assert DatabaseValidator._types_compatible("TIMESTAMP", "TIMESTAMP")
    
    def test_types_compatible_case_insensitive(self):
        """Test type compatibility is case insensitive."""
        assert DatabaseValidator._types_compatible("integer", "INTEGER")
        assert DatabaseValidator._types_compatible("TEXT", "text")
        assert DatabaseValidator._types_compatible("TimeStamp", "TIMESTAMP")
    
    def test_types_compatible_integer_variants(self):
        """Test type compatibility between integer variants."""
        integer_types = ['INTEGER', 'INT', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT']
        
        for type1 in integer_types:
            for type2 in integer_types:
                assert DatabaseValidator._types_compatible(type1, type2)
    
    def test_types_compatible_text_variants(self):
        """Test type compatibility between text variants."""
        text_types = ['TEXT', 'VARCHAR', 'CHAR', 'CHARACTER', 'CLOB']
        
        for type1 in text_types:
            for type2 in text_types:
                assert DatabaseValidator._types_compatible(type1, type2)
    
    def test_types_compatible_timestamp_as_text(self):
        """Test that TIMESTAMP can be stored as TEXT types."""
        text_types = ['TEXT', 'VARCHAR', 'CHAR', 'CHARACTER', 'CLOB']
        
        for text_type in text_types:
            assert DatabaseValidator._types_compatible("TIMESTAMP", text_type)
    
    def test_types_incompatible(self):
        """Test type incompatibility between different type families."""
        assert not DatabaseValidator._types_compatible("INTEGER", "TEXT")
        assert not DatabaseValidator._types_compatible("TEXT", "INTEGER")
        assert not DatabaseValidator._types_compatible("REAL", "INTEGER")
    
    def test_database_connection_error_handling(self, tmp_path):
        """Test handling of database connection errors."""
        db_path = tmp_path / "test.db"
        
        # Create a valid database first
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP
                )
            """)
        
        # Mock sqlite3.connect to raise an error
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = sqlite3.DatabaseError("Connection failed")
            
            with pytest.raises(sqlite3.Error, match="Database error while validating table structure"):
                DatabaseValidator.validate_table_structure(str(db_path))


# Integration test with real database scenarios
class TestDatabaseValidatorIntegration:
    """Integration tests with real database scenarios."""
    
    def test_real_database_scenario_with_data(self, tmp_path):
        """Test validation with a database that contains actual data."""
        db_path = tmp_path / "real_test.db"
        
        # Create database with the exact structure used by ParallelLLMProcessor
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert some test data
            test_data = [
                (1, "What is 2+2?", "4", "2024-01-01 12:00:00"),
                (2, "What is the capital of France?", None, "2024-01-01 12:01:00"),
                (3, "Explain quantum physics", "", "2024-01-01 12:02:00"),
                (4, "What is AI?", "NA", "2024-01-01 12:03:00"),
            ]
            
            conn.executemany("""
                INSERT INTO llm_results (id, prompt, result, created_at) 
                VALUES (?, ?, ?, ?)
            """, test_data)
        
        # Validation should succeed
        DatabaseValidator.validate_database_for_recovery(str(db_path))
    
    def test_database_created_by_database_manager(self, tmp_path):
        """Test validation with database created by DatabaseManager."""
        from src.llmtools.database_manager import DatabaseManager
        
        db_path = tmp_path / "manager_test.db"
        
        # Create database using DatabaseManager
        with DatabaseManager(str(db_path)) as db_manager:
            test_prompts = ["Test prompt 1", "Test prompt 2", "Test prompt 3"]
            db_manager.initialize_database(test_prompts)
            
            # Update some results
            db_manager.update_result(0, "Result 1")
            db_manager.update_result(2, "Result 3")
        
        # Validation should succeed
        DatabaseValidator.validate_database_for_recovery(str(db_path))