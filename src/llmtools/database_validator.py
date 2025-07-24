"""
Database validation utilities for ParallelLLMProcessor recovery functionality.

This module provides validation tools to ensure database files and table structures
are valid before attempting recovery operations.
"""

import sqlite3
import os
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseValidator:
    """
    Static utility class for validating database files and table structures.
    
    This class provides methods to validate that database files exist, are accessible,
    and contain the expected table structure for recovery operations.
    """
    
    @staticmethod
    def validate_database_file(db_file_path: str) -> None:
        """
        Validate that the database file exists and is accessible.
        
        Args:
            db_file_path: Path to the SQLite database file
            
        Raises:
            FileNotFoundError: If the database file does not exist
            ValueError: If the path is invalid or the file is not accessible
            PermissionError: If the file exists but cannot be accessed due to permissions
        """
        if not isinstance(db_file_path, str):
            raise ValueError("Database file path must be a string")
        
        if not db_file_path.strip():
            raise ValueError("Database file path cannot be empty or whitespace")
        
        # Check if file exists
        if not os.path.exists(db_file_path):
            raise FileNotFoundError(f"Database file does not exist: {db_file_path}")
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(db_file_path):
            raise ValueError(f"Path exists but is not a file: {db_file_path}")
        
        # Check if file is readable
        if not os.access(db_file_path, os.R_OK):
            raise PermissionError(f"Database file is not readable: {db_file_path}")
        
        # Check if file is writable (needed for recovery operations)
        if not os.access(db_file_path, os.W_OK):
            raise PermissionError(f"Database file is not writable: {db_file_path}")
        
        # Try to open the file as a SQLite database to verify it's a valid database
        try:
            with sqlite3.connect(db_file_path, timeout=5.0) as conn:
                # Simple query to verify it's a valid SQLite database
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                # If we get here, it's a valid SQLite database
                logger.debug(f"Database file validation successful: {db_file_path}")
        except sqlite3.DatabaseError as e:
            raise ValueError(f"File exists but is not a valid SQLite database: {db_file_path}. Error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unable to access database file: {db_file_path}. Error: {str(e)}")
    
    @staticmethod
    def validate_table_structure(db_file_path: str) -> None:
        """
        Validate that the llm_results table exists and has the expected structure.
        
        Args:
            db_file_path: Path to the SQLite database file
            
        Raises:
            ValueError: If the llm_results table doesn't exist or has incorrect structure
            sqlite3.Error: If there's an error accessing the database
        """
        try:
            with sqlite3.connect(db_file_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                
                # Check if llm_results table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='llm_results'
                """)
                
                if not cursor.fetchone():
                    raise ValueError(
                        f"Required table 'llm_results' does not exist in database: {db_file_path}"
                    )
                
                # Get table schema information
                cursor.execute("PRAGMA table_info(llm_results)")
                columns = cursor.fetchall()
                
                if not columns:
                    raise ValueError(
                        f"Table 'llm_results' exists but has no columns in database: {db_file_path}"
                    )
                
                # Expected columns: id, prompt, result, created_at
                expected_columns = {
                    'id': {'type': 'INTEGER', 'pk': True},
                    'prompt': {'type': 'TEXT', 'notnull': True},
                    'result': {'type': 'TEXT', 'notnull': False},
                    'created_at': {'type': 'TIMESTAMP', 'notnull': False}
                }
                
                # Parse column information
                actual_columns = {}
                for col_info in columns:
                    # col_info format: (cid, name, type, notnull, dflt_value, pk)
                    col_name = col_info[1]
                    col_type = col_info[2].upper()
                    col_notnull = bool(col_info[3])
                    col_pk = bool(col_info[5])
                    
                    actual_columns[col_name] = {
                        'type': col_type,
                        'notnull': col_notnull,
                        'pk': col_pk
                    }
                
                # Validate required columns exist
                missing_columns = []
                for expected_col, expected_props in expected_columns.items():
                    if expected_col not in actual_columns:
                        missing_columns.append(expected_col)
                        continue
                    
                    actual_props = actual_columns[expected_col]
                    
                    # Validate primary key constraint
                    if expected_props.get('pk', False) and not actual_props['pk']:
                        raise ValueError(
                            f"Column '{expected_col}' should be PRIMARY KEY in table 'llm_results'"
                        )
                    
                    # Validate NOT NULL constraint
                    if expected_props.get('notnull', False) and not actual_props['notnull']:
                        raise ValueError(
                            f"Column '{expected_col}' should be NOT NULL in table 'llm_results'"
                        )
                    
                    # Validate data type (allow some flexibility for SQLite type affinity)
                    expected_type = expected_props['type']
                    actual_type = actual_props['type']
                    
                    # SQLite type affinity - be flexible with type matching
                    if not DatabaseValidator._types_compatible(expected_type, actual_type):
                        logger.warning(
                            f"Column '{expected_col}' has type '{actual_type}' but expected '{expected_type}'. "
                            f"This may still work due to SQLite type affinity."
                        )
                
                if missing_columns:
                    raise ValueError(
                        f"Table 'llm_results' is missing required columns: {missing_columns} "
                        f"in database: {db_file_path}"
                    )
                
                logger.debug(f"Table structure validation successful: {db_file_path}")
                
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Database error while validating table structure: {str(e)}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Unexpected error validating table structure: {str(e)}")
    
    @staticmethod
    def _types_compatible(expected_type: str, actual_type: str) -> bool:
        """
        Check if SQLite data types are compatible considering type affinity.
        
        Args:
            expected_type: Expected column type
            actual_type: Actual column type from database
            
        Returns:
            bool: True if types are compatible, False otherwise
        """
        # Normalize types to uppercase
        expected = expected_type.upper()
        actual = actual_type.upper()
        
        # Exact match
        if expected == actual:
            return True
        
        # SQLite type affinity rules - simplified version
        integer_types = {'INTEGER', 'INT', 'TINYINT', 'SMALLINT', 'MEDIUMINT', 'BIGINT'}
        text_types = {'TEXT', 'VARCHAR', 'CHAR', 'CHARACTER', 'CLOB'}
        
        # Check if both are integer types
        if expected in integer_types and actual in integer_types:
            return True
        
        # Check if both are text types
        if expected in text_types and actual in text_types:
            return True
        
        # TIMESTAMP can be stored as TEXT in SQLite
        if expected == 'TIMESTAMP' and actual in text_types:
            return True
        
        return False
    
    @staticmethod
    def validate_database_for_recovery(db_file_path: str) -> None:
        """
        Comprehensive validation for database recovery operations.
        
        This method combines file validation and table structure validation
        to ensure the database is ready for recovery operations.
        
        Args:
            db_file_path: Path to the SQLite database file
            
        Raises:
            FileNotFoundError: If the database file does not exist
            ValueError: If the database format is invalid or table structure is incorrect
            PermissionError: If the file cannot be accessed
            sqlite3.Error: If there's a database-specific error
        """
        # First validate the file exists and is accessible
        DatabaseValidator.validate_database_file(db_file_path)
        
        # Then validate the table structure
        DatabaseValidator.validate_table_structure(db_file_path)
        
        logger.info(f"Database validation successful for recovery: {db_file_path}")