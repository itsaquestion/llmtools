#!/usr/bin/env python3
"""
Example usage of the DatabaseUpdater class for updating recovery results.

This example demonstrates how to use the DatabaseUpdater to update database records
with new results while maintaining data integrity and handling various scenarios.
"""

import sqlite3
import tempfile
import os
from src.llmtools.database_updater import DatabaseUpdater


def create_example_database():
    """Create an example database with some incomplete results."""
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db_path = temp_db.name
    
    # Create database schema
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE llm_results (
                id INTEGER PRIMARY KEY,
                prompt TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert example records with various states
        test_records = [
            (1, "What is the capital of France?", "Paris"),  # Complete
            (2, "What is 2 + 2?", None),  # NULL result - needs update
            (3, "What is the largest planet?", ""),  # Empty result - needs update
            (4, "What is the speed of light?", "NA"),  # "NA" result - needs update
            (5, "What is Python?", "A programming language"),  # Complete
            (6, "What is machine learning?", None),  # NULL result - needs update
        ]
        
        cursor.executemany(
            "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
            test_records
        )
        conn.commit()
    
    return db_path


def demonstrate_basic_update():
    """Demonstrate basic database update functionality."""
    print("=== Basic Database Update Example ===")
    
    # Create example database
    db_path = create_example_database()
    
    try:
        # Get initial statistics
        print("\nInitial database state:")
        stats = DatabaseUpdater.get_update_statistics(db_path)
        print(f"Total records: {stats['total_records']}")
        print(f"Complete records: {stats['complete_records']}")
        print(f"Incomplete records: {stats['incomplete_records']}")
        print(f"Completion rate: {stats['completion_rate']:.1f}%")
        
        # Define new results for incomplete records
        new_results = {
            2: "4",  # Answer to "What is 2 + 2?"
            3: "Jupiter",  # Answer to "What is the largest planet?"
            4: "299,792,458 m/s",  # Answer to "What is the speed of light?"
            6: "A subset of AI that learns from data"  # Answer to "What is machine learning?"
        }
        
        print(f"\nUpdating {len(new_results)} records...")
        
        # Update the database
        failed_updates = DatabaseUpdater.update_results(db_path, new_results)
        
        if failed_updates:
            print(f"Failed to update records: {failed_updates}")
        else:
            print("All updates successful!")
        
        # Get final statistics
        print("\nFinal database state:")
        stats = DatabaseUpdater.get_update_statistics(db_path)
        print(f"Total records: {stats['total_records']}")
        print(f"Complete records: {stats['complete_records']}")
        print(f"Incomplete records: {stats['incomplete_records']}")
        print(f"Completion rate: {stats['completion_rate']:.1f}%")
        
        # Verify the updates
        print("\nVerifying updates...")
        verified, mismatched = DatabaseUpdater.verify_updates(db_path, new_results)
        print(f"Verified records: {verified}")
        if mismatched:
            print(f"Mismatched records: {mismatched}")
        
    finally:
        # Clean up
        os.unlink(db_path)


def demonstrate_single_update():
    """Demonstrate single record update functionality."""
    print("\n=== Single Record Update Example ===")
    
    # Create example database
    db_path = create_example_database()
    
    try:
        print("\nUpdating single record (ID 2)...")
        
        # Update a single record
        success = DatabaseUpdater.update_single_result(
            db_path, 
            record_id=2, 
            new_result="Four"
        )
        
        if success:
            print("Single update successful!")
        else:
            print("Single update failed!")
        
        # Verify the update
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prompt, result FROM llm_results WHERE id = 2")
            prompt, result = cursor.fetchone()
            print(f"Record 2: '{prompt}' -> '{result}'")
        
    finally:
        # Clean up
        os.unlink(db_path)


def demonstrate_batch_processing():
    """Demonstrate batch processing for large updates."""
    print("\n=== Batch Processing Example ===")
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    db_path = temp_db.name
    
    try:
        # Create database with many records
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE llm_results (
                    id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    result TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert 150 records with NULL results
            records = [(i, f"Prompt {i}", None) for i in range(1, 151)]
            cursor.executemany(
                "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
                records
            )
            conn.commit()
        
        print(f"Created database with 150 incomplete records")
        
        # Create results for all records
        new_results = {i: f"Result {i}" for i in range(1, 151)}
        
        print("Updating all records using batch processing...")
        
        # Update with batch processing (will be triggered automatically for large updates)
        failed_updates = DatabaseUpdater.update_results(db_path, new_results)
        
        if failed_updates:
            print(f"Failed to update {len(failed_updates)} records")
        else:
            print("All 150 records updated successfully!")
        
        # Verify final state
        stats = DatabaseUpdater.get_update_statistics(db_path)
        print(f"Final completion rate: {stats['completion_rate']:.1f}%")
        
    finally:
        # Clean up
        os.unlink(db_path)


def demonstrate_error_handling():
    """Demonstrate error handling scenarios."""
    print("\n=== Error Handling Example ===")
    
    # Create example database
    db_path = create_example_database()
    
    try:
        print("\n1. Trying to update non-existent record...")
        failed_updates = DatabaseUpdater.update_results(db_path, {999: "Non-existent"})
        print(f"Failed updates (expected): {failed_updates}")
        
        print("\n2. Trying to update already complete record...")
        failed_updates = DatabaseUpdater.update_results(db_path, {1: "Should not update"})
        print(f"Failed updates (expected): {failed_updates}")
        
        print("\n3. Testing invalid input validation...")
        try:
            DatabaseUpdater.update_results(db_path, "not a dict")
        except ValueError as e:
            print(f"Caught expected error: {e}")
        
        try:
            DatabaseUpdater.update_single_result(db_path, -1, "invalid id")
        except ValueError as e:
            print(f"Caught expected error: {e}")
        
        print("\n4. Testing non-existent database...")
        try:
            DatabaseUpdater.update_results("/non/existent/path.db", {1: "test"})
        except sqlite3.Error as e:
            print(f"Caught expected database error: {type(e).__name__}")
        
    finally:
        # Clean up
        os.unlink(db_path)


def main():
    """Run all examples."""
    print("DatabaseUpdater Examples")
    print("=" * 50)
    
    demonstrate_basic_update()
    demonstrate_single_update()
    demonstrate_batch_processing()
    demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()