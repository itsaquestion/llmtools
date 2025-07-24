#!/usr/bin/env python3
"""
Test script to verify the enhanced logging and monitoring functionality for task 6.

This script creates a test database with some incomplete records and runs the recovery
process to demonstrate the enhanced logging features.
"""

import logging
import sqlite3
import os
import sys
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from llmtools.parallel_llm_processor import ParallelLLMProcessor

# Configure logging to see all the enhanced logging output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def mock_chat_fn(prompt: str) -> str:
    """Mock LLM function that simulates processing time and occasional failures."""
    time.sleep(0.1)  # Simulate processing time
    
    # Simulate occasional failures for testing
    if "fail" in prompt.lower():
        raise Exception("Simulated processing failure")
    
    return f"Processed: {prompt}"

def create_test_database(db_path: str):
    """Create a test database with some incomplete records."""
    print(f"Creating test database: {db_path}")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database with test data
    conn = sqlite3.connect(db_path)
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
    
    # Insert test data with various incomplete states
    test_data = [
        (1, "What is 2+2?", "4"),  # Complete
        (2, "What is the capital of France?", None),  # NULL - needs recovery
        (3, "Explain quantum physics", ""),  # Empty string - needs recovery
        (4, "What is AI?", "NA"),  # NA - needs recovery
        (5, "Hello world", "Hello! How can I help you?"),  # Complete
        (6, "This will fail", None),  # NULL - will fail during recovery
        (7, "Simple math", ""),  # Empty - needs recovery
        (8, "Another question", "Another answer"),  # Complete
        (9, "Test prompt", "NA"),  # NA - needs recovery
        (10, "Final question", "Final answer"),  # Complete
    ]
    
    cursor.executemany(
        "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
        test_data
    )
    
    conn.commit()
    conn.close()
    
    print(f"✓ Test database created with {len(test_data)} records")
    print(f"  - Complete records: 5")
    print(f"  - Incomplete records: 5 (2 NULL, 2 empty, 1 NA)")

def main():
    """Main test function."""
    print("=" * 60)
    print("TESTING ENHANCED LOGGING AND MONITORING FUNCTIONALITY")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Test database path
    test_db = "test_recovery_logging.db"
    
    try:
        # Step 1: Create test database
        create_test_database(test_db)
        print()
        
        # Step 2: Initialize processor
        print("Initializing ParallelLLMProcessor...")
        processor = ParallelLLMProcessor(
            chat_fn=mock_chat_fn,
            num_workers=2,  # Use fewer workers for clearer logging
            retry_attempts=2,
            retry_delay=0.5,
            timeout=5.0
        )
        print("✓ Processor initialized")
        print()
        
        # Step 3: Run recovery with enhanced logging
        print("Starting recovery process with enhanced logging...")
        print()
        
        results = processor.recover_from_database(test_db)
        
        print()
        print("=" * 60)
        print("RECOVERY TEST COMPLETED")
        print("=" * 60)
        print(f"Returned {len(results)} results")
        
        # Display results summary
        complete_results = sum(1 for r in results if r and r != "NA" and not r.startswith("Reprocessing failed:"))
        incomplete_results = len(results) - complete_results
        
        print(f"Final results summary:")
        print(f"  - Complete results: {complete_results}")
        print(f"  - Incomplete results: {incomplete_results}")
        print(f"  - Final completion rate: {(complete_results / len(results) * 100):.1f}%")
        
        # Show sample results
        print("\nSample results:")
        for i, result in enumerate(results[:5], 1):
            status = "✓" if result and result != "NA" and not result.startswith("Reprocessing failed:") else "✗"
            print(f"  {i}. {status} {result}")
        
        if len(results) > 5:
            print(f"  ... and {len(results) - 5} more results")
        
        processor.close()
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
            print(f"\n✓ Cleaned up test database: {test_db}")
    
    print(f"\nTest completed at: {datetime.now()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())