#!/usr/bin/env python3
"""
Quick test script to verify DatabaseValidator functionality.
"""

import tempfile
import sqlite3
from src.llmtools.database_validator import DatabaseValidator

def test_validator():
    """Test the DatabaseValidator with a real database."""
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Create a valid database with the expected structure
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert some test data
            conn.executemany("""
                INSERT INTO llm_results (id, prompt, result) 
                VALUES (?, ?, ?)
            """, [
                (1, "What is 2+2?", "4"),
                (2, "What is the capital of France?", None),
                (3, "Explain AI", ""),
                (4, "What is Python?", "NA")
            ])
        
        print(f"Created test database: {db_path}")
        
        # Test file validation
        print("Testing file validation...")
        DatabaseValidator.validate_database_file(db_path)
        print("✓ File validation passed")
        
        # Test table structure validation
        print("Testing table structure validation...")
        DatabaseValidator.validate_table_structure(db_path)
        print("✓ Table structure validation passed")
        
        # Test comprehensive validation
        print("Testing comprehensive validation...")
        DatabaseValidator.validate_database_for_recovery(db_path)
        print("✓ Comprehensive validation passed")
        
        print("\nAll validations successful! DatabaseValidator is working correctly.")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        raise
    finally:
        # Clean up
        import os
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"Cleaned up test database: {db_path}")

if __name__ == "__main__":
    test_validator()