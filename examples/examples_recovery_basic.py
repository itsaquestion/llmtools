#!/usr/bin/env python3
"""
Basic Recovery Example for ParallelLLMProcessor

This simple example demonstrates the core recovery functionality:
1. Create a database with some incomplete results (simulating interruption)
2. Use recover_from_database to complete the processing
3. Verify that recovery worked correctly

This is the simplest possible recovery example - perfect for getting started.
"""

import os
import sqlite3
import tempfile
import time
from src.llmtools.parallel_llm_processor import ParallelLLMProcessor


def simple_llm_function(prompt: str) -> str:
    """
    Simple mock LLM function for demonstration.
    In real usage, replace this with your actual LLM API call.
    """
    time.sleep(0.1)  # Simulate processing time
    return f"Answer: {prompt}"


def create_interrupted_database(db_path: str):
    """
    Create a database that looks like an interrupted processing session.
    Some results are complete, others are incomplete (NULL, empty, or "NA").
    """
    print(f"Creating interrupted database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the table (same structure as ParallelLLMProcessor uses)
    cursor.execute("""
        CREATE TABLE llm_results (
            id INTEGER PRIMARY KEY,
            prompt TEXT NOT NULL,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert test data - mix of complete and incomplete results
    test_data = [
        (1, "What is 2 + 2?", "4"),           # Complete
        (2, "What is the capital of France?", None),      # Incomplete (NULL)
        (3, "What color is the sun?", ""),               # Incomplete (empty)
        (4, "What is 5 * 6?", "30"),         # Complete
        (5, "What is the capital of Japan?", "NA"),       # Incomplete (NA)
        (6, "What is 10 - 3?", "7"),         # Complete
    ]
    
    cursor.executemany(
        "INSERT INTO llm_results (id, prompt, result) VALUES (?, ?, ?)",
        test_data
    )
    
    conn.commit()
    conn.close()
    
    print("Database created with:")
    print("  - 3 complete results")
    print("  - 3 incomplete results (1 NULL, 1 empty, 1 'NA')")
    
    return test_data


def show_database_contents(db_path: str, title: str):
    """Display the current contents of the database."""
    print(f"\n{title}")
    print("-" * len(title))
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, prompt, result FROM llm_results ORDER BY id")
    records = cursor.fetchall()
    conn.close()
    
    for record_id, prompt, result in records:
        status = "✅" if result and result not in ["", "NA"] else "❌"
        result_display = repr(result) if result is not None else "NULL"
        print(f"  {record_id}: {prompt:<30} | {result_display:<15} | {status}")


def main():
    """Demonstrate basic recovery functionality."""
    print("🔄 Basic Recovery Example")
    print("=" * 40)
    
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Step 1: Create an interrupted database
        print("\n📊 Step 1: Creating interrupted database")
        test_data = create_interrupted_database(db_path)
        show_database_contents(db_path, "Database Before Recovery")
        
        # Step 2: Perform recovery
        print("\n🔄 Step 2: Performing recovery")
        print("This will reprocess only the incomplete results...")
        
        # Create processor and recover
        processor = ParallelLLMProcessor(
            chat_fn=simple_llm_function,
            num_workers=2,  # Use 2 workers for this small example
            retry_attempts=1
        )
        
        # Perform the recovery
        results = processor.recover_from_database(db_path)
        processor.close()
        
        print(f"Recovery completed! Got {len(results)} total results.")
        
        # Step 3: Show results
        print("\n📋 Step 3: Recovery results")
        show_database_contents(db_path, "Database After Recovery")
        
        print(f"\nAll results in order:")
        for i, result in enumerate(results, 1):
            print(f"  {i}: {result}")
        
        # Step 4: Verify recovery worked
        print("\n✅ Step 4: Verification")
        
        # Check that we got the right number of results
        expected_count = len(test_data)
        if len(results) == expected_count:
            print(f"  ✓ Correct number of results: {len(results)}")
        else:
            print(f"  ❌ Wrong number of results: got {len(results)}, expected {expected_count}")
        
        # Check that previously complete results were preserved
        # Results 1, 4, and 6 should be unchanged
        preserved_results = [(0, "4"), (3, "30"), (5, "7")]  # 0-based indices
        all_preserved = True
        
        for idx, expected in preserved_results:
            if results[idx] == expected:
                print(f"  ✓ Result {idx+1} preserved: {expected}")
            else:
                print(f"  ❌ Result {idx+1} changed: expected {expected}, got {results[idx]}")
                all_preserved = False
        
        # Check that incomplete results were reprocessed
        incomplete_indices = [1, 2, 4]  # 0-based indices of originally incomplete results
        reprocessed_count = 0
        
        for idx in incomplete_indices:
            result = results[idx]
            if result and result not in ["", "NA", None] and result.startswith("Answer:"):
                reprocessed_count += 1
                print(f"  ✓ Result {idx+1} reprocessed: {result}")
        
        if reprocessed_count == len(incomplete_indices):
            print(f"  ✓ All incomplete results were reprocessed")
        else:
            print(f"  ⚠️  Only {reprocessed_count}/{len(incomplete_indices)} incomplete results were reprocessed")
        
        # Final summary
        print(f"\n🎉 Recovery Example Summary:")
        print(f"  • Started with {expected_count} total results")
        print(f"  • 3 were already complete (preserved)")
        print(f"  • 3 were incomplete (reprocessed)")
        print(f"  • Final result: {len(results)} complete results")
        
        if all_preserved and reprocessed_count == len(incomplete_indices):
            print(f"  • ✅ Recovery was completely successful!")
        else:
            print(f"  • ⚠️  Recovery had some issues - check the details above")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        raise
    finally:
        # Clean up temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)
            print(f"\n🧹 Cleaned up temporary database")


if __name__ == "__main__":
    main()