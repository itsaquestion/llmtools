#!/usr/bin/env python3
"""
Example usage of the RecoveryAnalyzer class.

This script demonstrates how to use the RecoveryAnalyzer to analyze database
records and identify incomplete results that need to be reprocessed.
"""

import sqlite3
import tempfile
import os
from src.llmtools.recovery_analyzer import RecoveryAnalyzer


def create_sample_database():
    """Create a sample database with mixed complete and incomplete results."""
    # Create temporary database
    temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(temp_fd)
    
    # Sample data with mixed results
    sample_records = [
        (1, "What is 2+2?", "4"),                    # Complete
        (2, "What color is the sky?", None),         # Incomplete (NULL)
        (3, "What is the capital of France?", "Paris"), # Complete
        (4, "What is 5*6?", ""),                     # Incomplete (empty string)
        (5, "What is the moon made of?", "NA"),      # Incomplete (NA)
        (6, "What is 10/2?", "5"),                   # Complete
        (7, "Failed request", "Error: Timeout occurred"), # Complete (error message)
        (8, "Another failed prompt", None)           # Incomplete (NULL)
    ]
    
    # Create database and insert sample data
    with sqlite3.connect(temp_path) as conn:
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE llm_results (
                id INTEGER PRIMARY KEY,
                prompt TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert sample records
        for record_id, prompt, result in sample_records:
            cursor.execute("""
                INSERT INTO llm_results (id, prompt, result) 
                VALUES (?, ?, ?)
            """, (record_id, prompt, result))
        
        conn.commit()
    
    return temp_path, sample_records


def demonstrate_is_result_incomplete():
    """Demonstrate the is_result_incomplete method."""
    print("=== Testing is_result_incomplete method ===")
    
    test_cases = [
        (None, "NULL value"),
        ("", "Empty string"),
        ("NA", "NA string"),
        ("Valid result", "Valid string"),
        ("Error: Something went wrong", "Error message"),
        (42, "Numeric value"),
        ("0", "Zero as string")
    ]
    
    for value, description in test_cases:
        is_incomplete = RecoveryAnalyzer.is_result_incomplete(value)
        status = "INCOMPLETE" if is_incomplete else "COMPLETE"
        print(f"  {description:25} -> {status}")
    
    print()


def demonstrate_analyze_database():
    """Demonstrate the analyze_database method."""
    print("=== Testing analyze_database method ===")
    
    # Create sample database
    db_path, original_records = create_sample_database()
    
    try:
        # Analyze the database
        failed_records, existing_results = RecoveryAnalyzer.analyze_database(db_path)
        
        print(f"Database analysis results:")
        print(f"  Total records: {len(existing_results)}")
        print(f"  Failed records: {len(failed_records)}")
        print(f"  Complete records: {len(existing_results) - len(failed_records)}")
        print()
        
        print("Failed records (need reprocessing):")
        for record_id, prompt in failed_records:
            print(f"  ID {record_id}: {prompt}")
        print()
        
        print("All existing results (in order):")
        for i, result in enumerate(existing_results, 1):
            status = "INCOMPLETE" if result is None else "COMPLETE"
            result_display = "None" if result is None else f"'{result}'"
            print(f"  ID {i}: {result_display} ({status})")
        print()
        
    finally:
        # Clean up
        os.unlink(db_path)


def demonstrate_get_database_summary():
    """Demonstrate the get_database_summary method."""
    print("=== Testing get_database_summary method ===")
    
    # Create sample database
    db_path, original_records = create_sample_database()
    
    try:
        # Get database summary
        summary = RecoveryAnalyzer.get_database_summary(db_path)
        
        print("Database summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Complete records: {summary['complete_records']}")
        print(f"  Incomplete records: {summary['incomplete_records']}")
        print(f"  NULL results: {summary['null_results']}")
        print(f"  Empty string results: {summary['empty_string_results']}")
        print(f"  'NA' results: {summary['na_results']}")
        print(f"  Completion rate: {summary['completion_rate']:.1f}%")
        print()
        
    finally:
        # Clean up
        os.unlink(db_path)


def demonstrate_error_handling():
    """Demonstrate error handling for various edge cases."""
    print("=== Testing error handling ===")
    
    # Test with non-existent file
    try:
        RecoveryAnalyzer.analyze_database("/path/that/does/not/exist.db")
    except FileNotFoundError as e:
        print(f"  Non-existent file: {type(e).__name__} - {e}")
    
    # Test with database without llm_results table
    temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(temp_fd)
    
    try:
        # Create database with wrong table
        with sqlite3.connect(temp_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE wrong_table (id INTEGER, data TEXT)")
            conn.commit()
        
        try:
            RecoveryAnalyzer.analyze_database(temp_path)
        except ValueError as e:
            print(f"  Missing table: {type(e).__name__} - {e}")
    
    finally:
        os.unlink(temp_path)
    
    print()


def main():
    """Run all demonstrations."""
    print("RecoveryAnalyzer Example Usage")
    print("=" * 50)
    print()
    
    demonstrate_is_result_incomplete()
    demonstrate_analyze_database()
    demonstrate_get_database_summary()
    demonstrate_error_handling()
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()